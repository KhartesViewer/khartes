import sys 
from PyQt5.QtGui import (
        QImage,
        QOpenGLVertexArrayObject,
        QOpenGLBuffer,
        QOpenGLContext,
        QOpenGLDebugLogger,
        QOpenGLDebugMessage,
        QOpenGLShader,
        QOpenGLShaderProgram,
        QOpenGLTexture,
        QSurfaceFormat,
        QTransform,
        QVector2D,
        )

from PyQt5.QtWidgets import (
        QApplication, 
        QGridLayout,
        QMainWindow,
        QOpenGLWidget,
        QWidget,
        )

from PyQt5.QtCore import (
        QFileInfo,
        QPointF,
        QSize,
        QTimer,
        )

import numpy as np
import math

import ctypes
import time

# NOTE that there is no need to include pyOpenGL

def VoidPtr(i):
    return ctypes.c_void_p(i)

vertex_code = '''
#version 410 core

in vec2 position;
in vec2 vtxt;
out vec2 ftxt;
void main()
{
  gl_Position = vec4(position, 0.0, 1.0);
  ftxt = vtxt;
}
'''

fragment_code_old = '''
#version 410 core

uniform sampler2D samplers[16];
uniform mat3 xforms[16];
uniform int ntxt;
in vec2 ftxt;
out vec4 fColor;

void main()
{
  fColor = vec4(1.0, 0.0, 0.0, 1.0);
  for (int i=0; i<ntxt; i++) {
      /*
      vec2 ttxt = (xforms[i]*vec3(ftxt, 1.)).st;
      vec4 tColor = texture(samplers[i], ttxt);
      float alpha = tColor.a;
      // alpha *= .8;
      fColor = (1-alpha)*fColor + alpha*tColor;
      */
      fColor = texture(samplers[i], ftxt);
  }
}
'''

fragment_code = '''
#version 410 core

uniform sampler2D atlas;
uniform mat3 xforms[100];
uniform vec2 tmins[100];
uniform vec2 tmaxs[100];
uniform int ncharts;
in vec2 ftxt;
out vec4 fColor;

void main()
{
  fColor = vec4(1.0, 0.0, 0.0, 1.0);
  if (ncharts == 0) {
    // fColor = vec4(0., .2, 0., 1.);
    fColor = texture(atlas, ftxt);
  } else {
    for (int i=0; i<ncharts; i++) {
    // for (int i=0; i<6; i++) {
      vec2 tmin = tmins[i];
      vec2 tmax = tmaxs[i];
      if (ftxt.s >= tmin.s && ftxt.s <= tmax.s &&
       ftxt.t >= tmin.t && ftxt.t <= tmax.t) {
        vec2 ttxt = (xforms[i]*vec3(ftxt, 1.)).st;
        fColor = texture(atlas, ttxt);
      }
    }
  }
}
'''

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.setMinimumSize(QSize(960,640))
        grid = QGridLayout()
        self.gl_widgets = []
        for i in range(6):
            # glwidget = GLWidget(self)
            # glwidget = GLTextureTest(i%2 == 0, i//2, self)
            glwidget = GLTextureTest(i, self)
            grid.addWidget(glwidget, i%2, i//2)
            self.gl_widgets.append(glwidget)
        widget = QWidget()
        widget.setLayout(grid)
        self.setCentralWidget(widget)

    def closeEvent(self, e):
        print("main close event")
        for glwidget in self.gl_widgets:
            glwidget.closeEvent(e)
        e.accept()


'''
for future reference on "bleeding" with texture atlases:
https://gamedev.stackexchange.com/questions/46963/how-to-avoid-texture-bleeding-in-a-texture-atlas
https://learn.microsoft.com/en-us/windows/win32/direct3d9/directly-mapping-texels-to-pixels?redirectedfrom=MSDN
https://pages.jh.edu/dighamm/research/2004_01_sta.pdf
https://web.cs.ucdavis.edu/~hamann/LaMarHamannJoy2000.pdf
'''

class GLTextureTest(QOpenGLWidget):
    def __init__(self, case, parent=None):
        QOpenGLWidget.__init__(self, parent)
        self.gl = None
        self.case = case
        self.linear = case%2 == 0
        self.padding = 0

        self.createSyntheticData()

        self.atlas = Atlas(self.data, (8,8))

    def createSyntheticData(self):
        width = 31
        height = 32
        cycles = 5
        wavelength = width/cycles
        # data = np.zeros((height, width), dtype=np.uint16)
        dgrid = np.mgrid[0:height, 0:width]
        # data = dgrid[0]*2*np.pi/wavelength)**2
        data = (dgrid[0].astype(np.float32)-9)**2
        data += (dgrid[1]-12)**2
        data = np.sqrt(data)
        data = np.cos(data*2*np.pi/wavelength)
        data = ((data+1.)*32767).astype(np.uint16)
        self.data = data

    def initializeGL(self):
        self.context().aboutToBeDestroyed.connect(self.destroyingContext)
        self.gl = self.context().versionFunctions()
        # Note that debug logging only takes place if the
        # surface format option "DebugContext" is set
        self.logger = QOpenGLDebugLogger()
        self.logger.initialize()
        self.logger.messageLogged.connect(self.onLogMessage)
        # synch mode is much slower
        self.logger.startLogging(1) # 0: asynchronous mode, 1: synch mode
        msg = QOpenGLDebugMessage.createApplicationMessage("test debug messaging")
        self.logger.logMessage(msg)

        self.xyuv = [
                ((-1, +1), (0., 1.)),
                ((+1, -1), (1., 0.)),
                ((-1, -1), (0., 0.)),
                ((+1, +1), (1., 1.)),
                ]
        self.vdata = np.array(self.xyuv, dtype=np.float32)
        # print("data", self.data.shape, self.data.size, self.data.itemsize)

        self.inds = [(0,1,2), (1,0,3)]
        # notice that indices must be uint8, uint16, or uint32
        self.indices = np.array(self.inds, dtype=np.uint32)
        # print("indices", self.indices.shape, self.indices.size, self.indices.itemsize)
        self.buildProgram()
        self.buildVertexBuffers()
        self.buildAtlas()
        # self.buildTextures()
        self.gl.glClearColor(.5,.5,.5,1.)

    def buildProgram(self):
        self.program = QOpenGLShaderProgram()
        ok = self.program.addShaderFromSourceCode(QOpenGLShader.Vertex, vertex_code)
        if not ok:
            print("vertex shader failed")
            exit()
        ok = self.program.addShaderFromSourceCode(QOpenGLShader.Fragment, fragment_code)
        if not ok:
            print("fragment shader failed")
            exit()
        ok = self.program.link()
        if not ok:
            print("link failed")
            exit()

    def buildVertexBuffers(self):
        self.vao = QOpenGLVertexArrayObject()
        self.vao.create()

        vloc = self.program.attributeLocation("position")
        print("vloc", vloc)
        tloc = self.program.attributeLocation("vtxt")
        print("tloc", tloc)

        self.program.bind()

        f = self.gl

        vaoBinder = QOpenGLVertexArrayObject.Binder(self.vao)

        # defaults to type=VertexBuffer, usage_pattern = Static Draw
        vbo = QOpenGLBuffer()
        vbo.create()
        vbo.bind()

        nbytes = self.vdata.size*self.vdata.itemsize
        # allocates space and writes vdata into vbo;
        # requires that vbo be bound
        vbo.allocate(self.vdata, nbytes)

        # print("buf %x %x"%(vbo.type(),vbo.usagePattern()))

        # glVertexAttribPointer attaches the currently bound vbo
        # to the vao
        f.glVertexAttribPointer(vloc, self.vdata.shape[1], int(f.GL_FLOAT), int(f.GL_FALSE), 4*self.vdata.itemsize, 0)
        f.glVertexAttribPointer(tloc, self.vdata.shape[1], int(f.GL_FLOAT), int(f.GL_FALSE), 4*self.vdata.itemsize, 2*self.vdata.itemsize)
        vbo.release()

        self.program.enableAttributeArray(vloc)
        self.program.enableAttributeArray(tloc)

        # IndexBuffer is Qt's name for GL_ELEMENT_ARRAY_BUFFER
        # In theory, the OpenGL buffer created here will be destroyed
        # when ibo goes out of scope (at the end of this function).
        # But for some reason this doesn't happen, or else
        # doesn't matter
        ibo = QOpenGLBuffer(QOpenGLBuffer.IndexBuffer)
        ibo.create()
        print("ibo", ibo.bufferId())
        ibo.bind()
        nbytes = self.indices.size*self.indices.itemsize
        ibo.allocate(self.indices, nbytes)

        # Order is important in next 2 lines.
        # Setting vaoBinder to None unbinds (releases) vao.
        # If ibo is unbound before vao is unbound, then
        # ibo will be detached from vao.  We don't want that!
        vaoBinder = None
        ibo.release()
        self.program.release()

    def buildAtlas(self):
        f = self.gl
        ncharts = 0
        self.program.bind()
        lcase = self.case // 2
        if lcase == 0:
            atlas_data = self.atlas.data.copy()
        elif lcase == 1:
            atlas_data = self.atlas.atlas_data.copy()
        elif lcase == 2:
            atlas_data = self.atlas.atlas_data.copy()
            ncharts = len(self.atlas.chunks)
            for i, chunk in enumerate(self.atlas.chunks):
                self.program.setUniformValue(
                        "xforms[%i]"%i, chunk.xform)
                self.program.setUniformValue(
                        "tmins[%i]"%i, QPointF(*chunk.tmin))
                self.program.setUniformValue(
                        "tmaxs[%i]"%i, QPointF(*chunk.tmax))

        self.program.setUniformValue("ncharts", ncharts)
        # self.xlocs = ncharts*[-1]
        # self.xforms = ncharts*[None]

        # atlas = self.data.copy()
        # if bytesperline is omitted, QImage requires
        # that data be 32-bit aligned.
        bytesperline = atlas_data.shape[1]*atlas_data.itemsize
        img = QImage(atlas_data, atlas_data.shape[1], atlas_data.shape[0],
                     bytesperline,
                     QImage.Format_Grayscale16)
        # When tex goes out of scope (at the end of this
        # function), the OpenGL texture will be destroyed.
        tex = QOpenGLTexture(img, QOpenGLTexture.DontGenerateMipMaps)
        # Persistent reference to tex, so the OpenGL texture doesn't get
        # deleted when tex goes out of scope.
        self.atlas_tex = tex
        tex.setWrapMode(
            QOpenGLTexture.DirectionS, QOpenGLTexture.ClampToBorder)
        tex.setWrapMode(
            QOpenGLTexture.DirectionT, QOpenGLTexture.ClampToBorder)
        if self.linear:
            tex.setMagnificationFilter(QOpenGLTexture.Linear)
        else:
            tex.setMagnificationFilter(QOpenGLTexture.Nearest)
        aloc = self.program.uniformLocation("atlas")
        if aloc < 0:
            print("couldn't get loc for atlas")
            return
        print("aloc", aloc)
        aunit = 1
        # tid = tex.textureId()
        f.glActiveTexture(f.GL_TEXTURE0+aunit)
        tex.bind()
        self.program.setUniformValue(aloc, aunit)
        f.glActiveTexture(f.GL_TEXTURE0)
        # tex.bind()
        tex.release()

        self.program.release()

    def buildTextures(self):
        f = self.gl
        ntxt = 1
        self.program.bind()
        # self.program.setUniformValue("ntxt", ntxt)
        self.program.setUniformValue("ncharts", 0)

        self.textures = ntxt*[None]
        self.xlocs = ntxt*[-1]
        self.xforms = ntxt*[None]
        print(len(self.textures))
        for i in range(ntxt):
            ldata = self.data.copy()
            img = QImage(ldata, ldata.shape[1], ldata.shape[0],
                         QImage.Format_Grayscale16)
            tex = QOpenGLTexture(img, QOpenGLTexture.DontGenerateMipMaps)
            '''
            tex = QOpenGLTexture(
                    QImage(root+'/images/side%d.png'%(i+1)).mirrored(), 
                    QOpenGLTexture.DontGenerateMipMaps)
            '''
            tex.setWrapMode(
                QOpenGLTexture.DirectionS, QOpenGLTexture.ClampToBorder)
            tex.setWrapMode(
                QOpenGLTexture.DirectionT, QOpenGLTexture.ClampToBorder)
            if self.linear:
                tex.setMagnificationFilter(QOpenGLTexture.Linear)
            else:
                tex.setMagnificationFilter(QOpenGLTexture.Nearest)
            self.textures[i] = tex
            xloc = -1
            '''
            xloc = self.program.uniformLocation("xforms[%d]"%i)
            if xloc < 0:
                print("couldn't get loc for xform", i)
                continue
            self.xlocs[i] = xloc
            '''

            # sloc = self.program.uniformLocation("samplers[%d]"%i)
            sloc = self.program.uniformLocation("atlas")
            if sloc < 0:
                print("couldn't get loc for sampler", i)
                continue
            tid = tex.textureId()
            f.glActiveTexture(f.GL_TEXTURE0+i)
            tex.bind()
            self.program.setUniformValue(sloc, i)
            print("xloc, sloc, i", xloc, sloc, i)

            # xform = QTransform()
            '''
            angle = 5.*t*2**i
            xform.translate(.5,.5)
            xform.rotate(angle)
            xform.translate(-.5,-.5)
            xform.translate(-(i%3), -(1.5-i//3))
            mag = 3
            xform.scale(mag, mag)
            '''
            # self.program.setUniformValue(xloc, xform)
        f.glActiveTexture(f.GL_TEXTURE0+6)

        self.program.release()
        self.gl.glClearColor(.5,.5,.5,1.)

    def paintGL(self):
        # time.sleep(1.)
        # print("paintGL")
        f = self.gl
        f.glClear(self.gl.GL_COLOR_BUFFER_BIT)
        f.glEnable(f.GL_BLEND)
        f.glBlendFunc(f.GL_SRC_ALPHA, f.GL_ONE_MINUS_SRC_ALPHA)
        vaoBinder = QOpenGLVertexArrayObject.Binder(self.vao)
        self.program.bind()
        '''
        t = time.time()-self.t0

        for i, xloc in enumerate(self.xlocs):
            xform = QTransform()
            angle = 5.*t*2**i
            xform.translate(.5,.5)
            xform.rotate(angle)
            xform.translate(-.5,-.5)
            xform.translate(-(i%3), -(1.5-i//3))
            mag = 3
            xform.scale(mag, mag)
            self.program.setUniformValue(xloc, xform)
        '''

        # Link to a discussion on how to set the last parameter
        # in glDrawElements:
        # https://stackoverflow.com/questions/61054700/how-to-specify-indices-as-void-to-gldrawelements-with-pyqt5-opengl
        # self.texture.bind()
        f.glDrawElements(f.GL_TRIANGLES, self.indices.size, f.GL_UNSIGNED_INT, None)
        self.program.release()
        vaoBinder = None

    def resizeGL(self, width, height):
        print("resize", width, height)

    def closeEvent(self, e):
        print("txtest widget close event")
        self.makeCurrent()
        self.logger.stopLogging()
        print("stopped logging")
        # e.accept()

    def destroyingContext(self):
        print("destroying context")

    def onLogMessage(self, msg):
        print("txtest log:", msg.message())

# d: data, a: atlas, c: chunk, pc: padded chunk
# : corner, k: key
# : coords, sz: size, e: single coord, r: rect
# coordinates are (x, y), i.e. (col, row);
# data value at (x, y) is data[y][x]

class Chunk:
    def __init__(self, atlas, ak, dk):
        pass
        # atlas, ak, d
        # Given atlas, atlas key, data corner,
        # copy chunk from data to atlas_data
        # compute xform

        # Atlas
        self.atlas = atlas
        # Chunk key (position) in atlas
        self.ak = ak
        # Chunk key (position) in input data
        self.dk = dk

        dcsz = atlas.dcsz
        dr = self.k2r(dk, dcsz)
        d = dr[0]

        acsz = atlas.acsz
        ar = self.k2r(ak, acsz)
        a = ar[0]

        pad = atlas.pad

        pdr = self.pad(dr, pad)
        dsz = atlas.dsz
        asz = atlas.asz
        all_dr = ((0, 0), (dsz[0], dsz[1]))
        int_dr = self.rectIntersection(pdr, all_dr)
        # print(pdr, all_dr, int_dr)

        # Compare change in pdr (padded data-chunk rectangle) 
        # due to intersection with edges of data array:
        # Difference in min corner:
        skip0 = (int_dr[0][0]-pdr[0][0], int_dr[0][1]-pdr[0][1])
        # Difference in max corner:
        skip1 = (pdr[1][0]-int_dr[1][0], pdr[1][1]-int_dr[1][1])

        # print(pdr, skip0)
        # print(ar, int_dr, skip0, skip1)
        atlas.atlas_data[
                (ar[0][1]+skip0[1]):(ar[1][1]-skip1[1]), 
                (ar[0][0]+skip0[0]):(ar[1][0]-skip1[0])
                ] = atlas.data[
                        (int_dr[0][1]):int_dr[1][1], 
                        (int_dr[0][0]):int_dr[1][0]
                        ]
        xform = QTransform()
        '''
        # xform.scale(1./asz[0], 1./asz[1])
        # xform.translate(-a[0], -a[1])
        # xform.scale(asz[0], asz[1])
        # xform.scale(1./(acsz[0]*dsz[0]), 1./(acsz[1]*dsz[1]))
        # xform.translate(-a[0]/asz[0], -a[1]/asz[1])
        # xform[0,0] = 1.
        xform.scale(1./dsz[0], 1./dsz[1])
        if ak[0] == 0 and ak[1] == 0:
            print(xform.m11(), xform.m31())
        # xform.translate(pdr[0][0]-ar[0][0], pdr[0][1]-ar[0][1])
        xform.translate(dr[0][0]-ar[0][0]-pad, dr[0][1]-ar[0][1]-pad)
        if ak[0] == 0 and ak[1] == 0:
            print(xform.m11(), xform.m31())
        xform.scale(asz[0], asz[1])
        if ak[0] == 0 and ak[1] == 0:
            print(xform.m11(), xform.m31())
            print("*", asz[0]/dsz[0], (pdr[0][0]-ar[0][0])/dsz[0])
        # self.xform = xform.inverted()
        self.xform = xform.inverted()[0]
        print(self.xform)
        '''
        xform.scale(1./asz[0], 1./asz[1])
        xform.translate(ar[0][0]+pad-dr[0][0], ar[0][1]+pad-dr[0][1])
        xform.scale(dsz[0], dsz[1])
        self.xform = xform
        self.tmin = ((dr[0][0])/dsz[0], (dr[0][1])/dsz[1])
        self.tmax = ((dr[1][0])/dsz[0], (dr[1][1])/dsz[1])
        # self.tmin = (0.01, 0.)
        # self.tmax = (1., 1.)
        if ak[0] == 0 and ak[1] == 0:
            print("tm", self.tmin, self.tmax)

    @staticmethod
    def k2r(k, csz):
        c = (k[0]*csz[0], k[1]*csz[1])
        r = (c, (c[0]+csz[0], c[1]+csz[1]))
        return r

    # padded rectangle
    @staticmethod
    def pad(rect, pad):
        return ((rect[0][0]-pad, rect[0][1]-pad), 
                (rect[1][0]+pad, rect[1][1]+pad))

    # adapted from https://stackoverflow.com/questions/25068538/intersection-and-difference-of-two-rectangles/25068722#25068722
    @staticmethod
    def rectIntersection(ra, rb):
        # if not Utils.rectIsValid(ra) or not Utils.rectIsValid(rb):
        #     return Utils.emptyRect()
        (ax1, ay1), (ax2, ay2) = ra
        (bx1, by1), (bx2, by2) = rb
        # print(ra, rb)
        x1 = max(min(ax1, ax2), min(bx1, bx2))
        y1 = max(min(ay1, ay2), min(by1, by2))
        x2 = min(max(ax1, ax2), max(bx1, bx2))
        y2 = min(max(ay1, ay2), max(by1, by2))
        if (x1<x2) and (y1<y2):
            r = ((x1, y1), (x2, y2))
            # print(r)
            return r

class Atlas:
    def __init__(self, data, dcsz=(4,4)):
        pad = 1
        # pad = 0
        self.pad = pad
        self.data = data
        self.dcsz = dcsz
        acsz = (dcsz[0]+2*pad, dcsz[1]+2*pad)
        self.acsz = acsz
        # dsz is (width, height)
        dsz = (data.shape[1], data.shape[0])
        self.dsz = dsz
        # ksz is same for data and atlas
        ksz = (self.ke(dsz[0],dcsz[0]), self.ke(dsz[1],dcsz[1]))
        self.ksz = ksz
        print("data",dsz,dcsz,ksz)
        asz = (acsz[0]*ksz[0], acsz[1]*ksz[1])
        self.asz = asz
        self.atlas_data = np.zeros((asz[1], asz[0]), dtype=data.dtype)
        print("atlas data",self.atlas_data.shape)
        self.chunks = []
        for i in range(ksz[0]):
            for j in range(ksz[1]):
                dk = (i,j)
                # flip for testing:
                # ak = (ksz[0]-i-1, j)
                ak = (ksz[0]-i-1, ksz[1]-j-1)
                # ak = (i, j)
                chunk = Chunk(self, ak, dk)
                self.chunks.append(chunk)


    # number of chunks (in 1D) given data size, chunk size
    def ke(self, e, ce):
        ke = 1 + (e-1)//ce
        return ke

    # coordinates of corner of given chunk
    def c(self, k):
        return (k[0]*self.csz[0], k[1]*self.csz[1])

    # rectangle of given chunk
    # rectangle is ((xmin,ymin),(xmax,ymax)) where {x,y}max is exclusive
    def cRect(self, kc):
        return (self.c(kc), self.c((kc[0]+1,kc[1]+1)))

    def td(self, cd, szd):
        return (cd+.5)/(szd+1)

    # texture coords given data coords
    def t(self, c):
        return(self.td(c[0], self.sz[0]), self.td(c[1], self.sz[1]))

    def xform(self, kc):
        r = self.cRect(kc)
        p = self.pad(r)
        c0 = p[0]


class GLWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        QOpenGLWidget.__init__(self, parent)
        self.gl = None
        self.first_paint = True
        self.t0 = time.time()

    def initializeGL(self):
        '''
        print("cc", cc)
        vf = cc.versionFunctions()
        self.gl = vf
        '''
        # example of using texture mapping (but Open GL 2.0!)
        # https://stackoverflow.com/questions/65032638/how-to-texture-a-sphere-in-pyqt5-qopenglwidget
        # print("ctx", self.context())
        # Not sure when this signal is ever sent
        self.context().aboutToBeDestroyed.connect(self.destroyingContext)
        self.gl = self.context().versionFunctions()
        # print(self.gl, type(self.gl))
        # print(self.gl.GL_RENDER)
        # print(self.ctx.extensions())
        # for ext in sorted(self.ctx.extensions()):
        #     print(str(ext))

        # Note that debug logging only takes place if the
        # surface format option "DebugContext" is set
        self.logger = QOpenGLDebugLogger()
        self.logger.initialize()
        self.logger.messageLogged.connect(self.onLogMessage)
        # synch mode is much slower
        self.logger.startLogging(1) # 0: asynchronous mode, 1: synch mode
        msg = QOpenGLDebugMessage.createApplicationMessage("test debug messaging")
        self.logger.logMessage(msg)

        self.xyuv = [
                ((-1, +1), (0., 1.)),
                ((+1, -1), (1., 0.)),
                ((-1, -1), (0., 0.)),
                ((+1, +1), (1., 1.)),
                ]
        self.data = np.array(self.xyuv, dtype=np.float32)
        print("data", self.data.shape, self.data.size, self.data.itemsize)

        self.inds = [(0,1,2), (1,0,3)]
        # notice that indices must be uint8, uint16, or uint32
        self.indices = np.array(self.inds, dtype=np.uint32)
        print("indices", self.indices.shape, self.indices.size, self.indices.itemsize)
        self.buildProgram()
        self.buildBuffers()
        # self.printInfo()
        # timer = QTimer(self)
        # timer.timeout.connect(self.update)
        # timer.start(20)

    def buildProgram(self):
        self.program = QOpenGLShaderProgram()
        ok = self.program.addShaderFromSourceCode(QOpenGLShader.Vertex, vertex_code)
        if not ok:
            print("vertex shader failed")
            exit()
        ok = self.program.addShaderFromSourceCode(QOpenGLShader.Fragment, fragment_code)
        if not ok:
            print("fragment shader failed")
            exit()
        ok = self.program.link()
        if not ok:
            print("link failed")
            exit()

    def buildBuffers(self):
        self.vao = QOpenGLVertexArrayObject()
        self.vao.create()

        vloc = self.program.attributeLocation("position")
        print("vloc", vloc)
        tloc = self.program.attributeLocation("vtxt")
        print("tloc", tloc)

        self.program.bind()

        f = self.gl

        vaoBinder = QOpenGLVertexArrayObject.Binder(self.vao)

        # defaults to type=VertexBuffer, usage_pattern = Static Draw
        vbo = QOpenGLBuffer()
        vbo.create()
        vbo.bind()

        nbytes = self.data.size*self.data.itemsize
        # allocates space and writes data into vbo;
        # requires that vbo be bound
        vbo.allocate(self.data, nbytes)

        print("buf %x %x"%(vbo.type(),vbo.usagePattern()))

        # glVertexAttribPointer attaches the currently bound vbo
        # to the vao, per this answer:
        # https://stackoverflow.com/questions/59892088/how-does-a-vbo-get-attached-to-a-vao
        # f.glVertexAttribPointer(loc, self.data.shape[1], int(f.GL_FLOAT), int(f.GL_FALSE), 0, 0)
        f.glVertexAttribPointer(vloc, self.data.shape[1], int(f.GL_FLOAT), int(f.GL_FALSE), 4*self.data.itemsize, 0)
        f.glVertexAttribPointer(tloc, self.data.shape[1], int(f.GL_FLOAT), int(f.GL_FALSE), 4*self.data.itemsize, 2*self.data.itemsize)
        vbo.release()

        # A few words about setAttributeArray, which is used
        # in many examples of programming in OpenGL with PyQt5.
        # The problem is that these examples work with OpenGL 3,
        # but not OpenGL 4.
        # setAttributeArray just calls glVertexAttribPointer.
        # But glVertexArrayPointer doesn't work the same way 
        # in OpenGL 4.1 as it did in OpenGL 3; it takes an integer 
        # offset into the data (which must have previously been stored
        # in a buffer) rather than a pointer to the data itself.
        # There are many examples on the internet of code like this:
        # self.program.setAttributeArray(loc, self.data)
        # with no buffer created/bound beforehand.
        # But this doesn't work in OpenGL 4.1, because of the
        # change in behavior of glVertexAttribPointer.
        # In theory, the following could be used in OpenGL 4.1,
        # after the buffer has been allocated, 
        # in place of glVertexArrayPointer,
        # but the PyQt5 argument checker won't allow 0 as the
        # second argument:
        # self.program.setAttributeArray(loc, 0)

        print("set")

        self.program.enableAttributeArray(vloc)
        self.program.enableAttributeArray(tloc)
        print("enabled")

        # https://stackoverflow.com/questions/8973690/vao-and-element-array-buffer-state
        # Qt's name for GL_ELEMENT_ARRAY_BUFFER
        ibo = QOpenGLBuffer(QOpenGLBuffer.IndexBuffer)
        ibo.create()
        print("ibo", ibo.bufferId())
        ibo.bind()
        nbytes = self.indices.size*self.indices.itemsize
        ibo.allocate(self.indices, nbytes)

        # Order is important in next 2 lines.
        # Setting vaoBinder to None unbinds (releases) vao.
        # If ibo is unbound before vao is unbound, then
        # ibo will be detached from vao.  We don't want that!
        vaoBinder = None
        ibo.release()

        root = QFileInfo(__file__).absolutePath()

        ntxt = 6
        self.program.setUniformValue("ntxt", ntxt)
        self.textures = ntxt*[None]
        self.xlocs = ntxt*[-1]
        self.xforms = ntxt*[None]
        print(len(self.textures))
        for i in range(6):
            tex = QOpenGLTexture(
                    QImage(root+'/images/side%d.png'%(i+1)).mirrored(), 
                    QOpenGLTexture.DontGenerateMipMaps)
            tex.setWrapMode(
                QOpenGLTexture.DirectionS, QOpenGLTexture.ClampToBorder)
            tex.setWrapMode(
                QOpenGLTexture.DirectionT, QOpenGLTexture.ClampToBorder)
            self.textures[i] = tex
            xloc = self.program.uniformLocation("xforms[%d]"%i)
            if xloc < 0:
                print("couldn't get loc for xform", i)
                continue
            self.xlocs[i] = xloc

            sloc = self.program.uniformLocation("samplers[%d]"%i)
            if sloc < 0:
                print("couldn't get loc for sampler", i)
                continue
            tid = tex.textureId()
            atid = i+1
            f.glActiveTexture(f.GL_TEXTURE0+atid)
            tex.bind()
            self.program.setUniformValue(sloc, atid)
            print("xloc, sloc, i, atid", xloc, sloc, i, atid)
        f.glActiveTexture(f.GL_TEXTURE0)

        self.program.release()
        self.gl.glClearColor(.5,.5,.5,1.)


    def printInfo(self):
        print("vendor", self.gl.glGetString(self.gl.GL_VENDOR))
        print("version", self.gl.glGetString(self.gl.GL_VERSION))
        # for minimum values (4.1) see:
        # https://registry.khronos.org/OpenGL/specs/gl/glspec41.core.pdf
        # starting p. 383

        # at least 16834 (4.1)
        print("max texture size", 
              self.gl.glGetIntegerv(self.gl.GL_MAX_TEXTURE_SIZE))
        # at least 2048 (4.1)
        print("max array texture size", 
              self.gl.glGetIntegerv(self.gl.GL_MAX_ARRAY_TEXTURE_LAYERS))
        # at least 2048 (4.1)
        print("max 3d texture size", 
              self.gl.glGetIntegerv(self.gl.GL_MAX_3D_TEXTURE_SIZE))
        # at least 16 (4.1)
        print("max texture image units", 
              self.gl.glGetIntegerv(self.gl.GL_MAX_TEXTURE_IMAGE_UNITS))
        print("max combined texture image units", 
              self.gl.glGetIntegerv(self.gl.GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS))
        print("max texture buffer size", 
              self.gl.glGetIntegerv(self.gl.GL_MAX_TEXTURE_BUFFER_SIZE)) 

    def paintGL(self):
        # time.sleep(1.)
        # print("paintGL")
        f = self.gl
        f.glClear(self.gl.GL_COLOR_BUFFER_BIT)
        f.glEnable(f.GL_BLEND)
        f.glBlendFunc(f.GL_SRC_ALPHA, f.GL_ONE_MINUS_SRC_ALPHA)
        vaoBinder = QOpenGLVertexArrayObject.Binder(self.vao)
        self.program.bind()
        t = time.time()-self.t0

        for i, xloc in enumerate(self.xlocs):
            xform = QTransform()
            angle = 5.*t*2**i
            xform.translate(.5,.5)
            xform.rotate(angle)
            xform.translate(-.5,-.5)
            xform.translate(-(i%3), -(1.5-i//3))
            mag = 3
            xform.scale(mag, mag)
            self.program.setUniformValue(xloc, xform)

        # Link to a discussion on how to set the last parameter
        # in glDrawElements:
        # https://stackoverflow.com/questions/61054700/how-to-specify-indices-as-void-to-gldrawelements-with-pyqt5-opengl
        # self.texture.bind()
        f.glDrawElements(f.GL_TRIANGLES, self.indices.size, f.GL_UNSIGNED_INT, None)
        self.program.release()
        vaoBinder = None

    def resizeGL(self, width, height):
        print("resize", width, height)

    def closeEvent(self, e):
        print("gl widget close event")
        self.makeCurrent()
        self.logger.stopLogging()
        print("stopped logging")
        # e.accept()

    def destroyingContext(self):
        print("destroying context")

    def onLogMessage(self, msg):
        print("gl log:", msg.message())

fmt = QSurfaceFormat()
# Note that pyQt5 only supports OpenGL versions 2.0, 2.1, and 4.1 Core :
# https://riverbankcomputing.com/pipermail/pyqt/2017-January/038640.html
# To get the latest OpenGL version, perhaps use the python opengl module?
# https://stackoverflow.com/questions/38645674/issues-with-pyqt5s-opengl-module-and-versioning-calls-for-incorrect-qopenglfu
# But for our purposes, 4.1 Core is fine, since that is the last version
# supported by MacOS.
fmt.setVersion(4, 1)
fmt.setProfile(QSurfaceFormat.CoreProfile)
fmt.setOption(QSurfaceFormat.DebugContext)
QSurfaceFormat.setDefaultFormat(fmt)

# According to https://doc.qt.io/qt-5/qopenglwidget.html
# QSurfaceFormat.setDefaultFormat needs to be called before
# constructing QApplication, for MacOS

app = QApplication(sys.argv)

widget = MainWindow()
widget.show()
app.exec_()
