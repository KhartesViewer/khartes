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
        QVector2D,
        QMatrix3x3,
        QTransform,
        )

from PyQt5.QtWidgets import (
        QApplication, QWidget,
        QOpenGLWidget)

from PyQt5.QtCore import (
        QFileInfo,
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
  gl_Position = vec4(.7*position, 0.0, 1.0);
  const float m = 1.2;
  ftxt = m*(vtxt-.5)+.5;
}
'''

fragment_code = '''
#version 410 core

uniform sampler2D sampler;
uniform sampler2D samplers[6];
uniform mat3 xform;
uniform mat3 xforms[6];
in vec2 ftxt;
out vec4 fColor;

void main()
{
  fColor = vec4(1.0, 0.0, 0.0, 1.0);
  // vec4 tColor = texture(sampler, ftxt);
  // fColor += tColor.a*tColor;
  // fColor = texture(samplers[0], ftxt);
  // fColor += texture(samplers[1], ftxt);
  // vec2 ttxt = (xform*vec3(ftxt, 1.)).st;
  for (int i=0; i<6; i++) {
      vec2 ttxt = (xforms[i]*vec3(ftxt, 1.)).st;
      vec4 tColor = texture(samplers[i], ttxt);
      // fColor += tColor.a*tColor;
      // fColor += tColor;
      float alpha = tColor.a;
      /*
      if (alpha > 0.) {
          fColor = alpha*tColor;
      }
      */
      alpha *= .8;
      fColor = (1-alpha)*fColor + alpha*tColor;
  }
  // fColor.a = .5;
}
'''

'''
for future reference on "bleeding" with texture atlases:
https://gamedev.stackexchange.com/questions/46963/how-to-avoid-texture-bleeding-in-a-texture-atlas
https://learn.microsoft.com/en-us/windows/win32/direct3d9/directly-mapping-texels-to-pixels?redirectedfrom=MSDN
https://pages.jh.edu/dighamm/research/2004_01_sta.pdf
https://web.cs.ucdavis.edu/~hamann/LaMarHamannJoy2000.pdf
'''

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

        # self.positions = [(-1, +1), (+1, -1), (-1, -1), (+1, -1)]
        # self.positions = [(-1, +1), (+1, -1), (-1, -1), (+1, +1)]
        self.xyuv = [
                ((-1, +1), (0., 1.)),
                ((+1, -1), (1., 0.)),
                ((-1, -1), (0., 0.)),
                ((+1, +1), (1.2, 1.2)),
                ]
        self.xyuv = [
                ((-1, +1), (0., 1.)),
                ((+1, -1), (1., 0.)),
                ((-1, -1), (0., 0.)),
                ((+1, +1), (1., 1.)),
                ]
        # self.data = np.array(self.positions, dtype=np.float32)
        self.data = np.array(self.xyuv, dtype=np.float32)
        print("data", self.data.shape, self.data.size, self.data.itemsize)

        # self.inds = [0,1,2, 1,2,3]
        # self.inds = [(0,1,2), (1,2,3)]
        self.inds = [(0,1,2), (1,0,3)]
        # notice that indices must be uint8, uint16, or uint32
        # self.indices = np.array(self.inds, dtype=np.uint32).reshape(-1)
        self.indices = np.array(self.inds, dtype=np.uint32)
        print("indices", self.indices.shape, self.indices.size, self.indices.itemsize)
        self.buildProgram()
        self.buildBuffers()
        # self.printInfo()
        timer = QTimer(self)
        timer.timeout.connect(self.update)
        timer.start(20)

    def buildProgram(self):
        # vshader = QOpenGLShader(QOpenGLShader.Vertex)
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

        '''
        for txt in ["sampler", "samplers", "samplers[0]", "samplers[1]", "samplers[2]", "samplers[5]", "samplers[6]"]:
            print(txt, self.program.uniformLocation(txt))
        '''

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
        # vbo.allocate(nbytes)

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
        # GL_ELEMENT_ARRAY_BUFFER
        ibo = QOpenGLBuffer(QOpenGLBuffer.IndexBuffer)
        ibo.create()
        print("ibo", ibo.bufferId())
        ibo.bind()
        nbytes = self.indices.size*self.indices.itemsize
        ibo.allocate(self.indices, nbytes)
        # ibo.allocate(self.indices.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p)).contents, nbytes)


        # Order is important in next 2 lines.
        # Setting vaoBinder to None unbinds (releases) vao.
        # If ibo is unbound before vao is unbound, then
        # ibo will be detached from vao.  We don't want that!
        vaoBinder = None
        ibo.release()

        root = QFileInfo(__file__).absolutePath()
        # print("file", root+'/images/side1.png')
        '''
        self.texture = QOpenGLTexture(
                QImage(root+'/images/side1.png').mirrored(), 
                QOpenGLTexture.DontGenerateMipMaps)
        self.texture.setWrapMode(
                QOpenGLTexture.DirectionS, QOpenGLTexture.ClampToBorder)
        self.texture.setWrapMode(
                QOpenGLTexture.DirectionT, QOpenGLTexture.ClampToBorder)
        print("filters %x %x"%self.texture.minMagFilters())

        xloc = self.program.uniformLocation("xform")
        print("xloc", xloc)
        xform = QMatrix3x3()
        xform[0,0] = 1.
        xform[1,1] = 1.
        xform[2,2] = 1.
        self.program.setUniformValue(xloc, xform)
        '''


        ntxt = 6
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
            # self.textures.append(tex)
            self.textures[i] = tex
            xloc = self.program.uniformLocation("xforms[%d]"%i)
            if xloc < 0:
                print("couldn't get loc for xform", i)
                continue
            self.xlocs[i] = xloc
            '''
            xform = QMatrix3x3()
            xform.setToIdentity()
            xform *= 3
            xform[0,2] = -(i%3)
            xform[1,2] = -(1.5-i//3)

            xform2 = QMatrix3x3()
            xform2.setToIdentity()
            angle = 10.*i
            rads = math.radians(angle)
            c = math.cos(rads)
            s = math.sin(rads)
            xform2[0,0] = c
            xform2[0,1] = s
            xform2[1,0] = -s
            xform2[1,1] = c
            xform *= xform2
            '''

            '''
            xform = QTransform()
            angle = 10.*i
            xform.translate(.5,.5)
            xform.rotate(angle)
            xform.translate(-.5,-.5)
            xform.translate(-(i%3), -(1.5-i//3))
            mag = 3
            xform.scale(mag, mag)
            '''

            # shifter = QTransform()
            # shifter.translate(-.5/mag, -.5/mag)
            # shifter2 = QTransform()
            # shifter2.translate(.5/mag, .5/mag)
            # xform.translate(.5, .5)
            # xform *= shifter
            # xform.translate(-.5, -.5)
            # mag = 3
            # xform *= mag

            # self.xforms[i] = xform
            # self.program.setUniformValue(xloc, xform)

            sloc = self.program.uniformLocation("samplers[%d]"%i)
            if sloc < 0:
                print("couldn't get loc for sampler", i)
                continue
            tid = tex.textureId()
            # print("sloc, tex id", sloc, tid)
            # tex.bind(tid)
            # tex.bind(i)
            # self.program.setUniformValue(sloc, tid)
            f.glActiveTexture(f.GL_TEXTURE0+i)
            tex.bind()
            self.program.setUniformValue(sloc, i)
            print("xloc, sloc, i", xloc, sloc, i)
            # tex.release()
        f.glActiveTexture(f.GL_TEXTURE0+6)

            # self.program.setUniformValue(sloc, tex.textureId())
            # self.program.setUniformValue(sloc, tex.textureId())

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
        print("close event")
        self.makeCurrent()
        self.logger.stopLogging()
        print("stopped logging")
        e.accept()

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

widget = GLWidget()
widget.show()
app.exec_()
