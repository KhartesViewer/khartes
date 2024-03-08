from PyQt5.QtGui import (
        QImage,
        QMatrix4x4,
        QOffscreenSurface,
        QOpenGLVertexArrayObject,
        QOpenGLBuffer,
        QOpenGLContext,
        QOpenGLDebugLogger,
        QOpenGLDebugMessage,
        QOpenGLFramebufferObject,
        QOpenGLFramebufferObjectFormat,
        QOpenGLShader,
        QOpenGLShaderProgram,
        QOpenGLTexture,
        QPixmap,
        QSurfaceFormat,
        QTransform,
        QVector2D,
        QVector4D,
        )

from PyQt5.QtWidgets import (
        QApplication, 
        QGridLayout,
        QHBoxLayout,
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

import time
from collections import OrderedDict
import numpy as np
import cv2

from utils import Utils
from data_window import DataWindow
from gl_data_window import GLDataWindowChild


class GLSurfaceWindow(DataWindow):
    def __init__(self, window):
        super(GLSurfaceWindow, self).__init__(window, 2)
        # self.clear()
        layout = QHBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        self.setLayout(layout)
        self.glw = GLSurfaceWindowChild(self)
        layout.addWidget(self.glw)
        # self.zoomMult = .5
        self.zoomMult = 2

    # see comments for this function in DataWindow
    def nodeMovementAllowedInK(self):
        return True

    def allowMouseToDragNode(self):
        return False

    def drawSlice(self):
        # the MainWindow.edit widget overlays the
        # fragment map; it was used for displaying 
        # user documentation when khartes would first 
        # start up.  We don't want it to block the gl window.
        self.window.edit.hide()
        self.window.setFocus()
        self.glw.update()

slice_code = {
    "name": "slice",

    "vertex": '''
      #version 410 core

      in vec2 position;
      in vec2 vtxt;
      out vec2 ftxt;
      void main() {
        gl_Position = vec4(position, 0.0, 1.0);
        ftxt = vtxt;
      }
    ''',

    "fragment": '''
      #version 410 core

      uniform sampler2D base_sampler;
      uniform sampler2D underlay_sampler;
      uniform sampler2D overlay_sampler;
      uniform sampler2D fragments_sampler;
      // uniform float frag_opacity = 1.;
      in vec2 ftxt;
      out vec4 fColor;

      void main()
      {
        float alpha;
        fColor = texture(base_sampler, ftxt);
        // fColor = .1*fColor + .9*vec4(.5,.5,1.,1.);

        vec4 uColor = texture(underlay_sampler, ftxt);
        alpha = uColor.a;
        fColor = (1.-alpha)*fColor + alpha*uColor;

        /*
        vec4 frColor = texture(fragments_sampler, ftxt);
        // alpha = frag_opacity*frColor.a;
        alpha = frColor.a;
        fColor = (1.-alpha)*fColor + alpha*frColor;
        */

        vec4 oColor = texture(overlay_sampler, ftxt);
        alpha = oColor.a;
        fColor = (1.-alpha)*fColor + alpha*oColor;
      }
    ''',
}

data_code = {
    "name": "data",

    "vertex": '''
      #version 410 core

      uniform mat4 xform;
      layout(location=3) in vec3 xyz;
      layout(location=4) in vec2 stxy;
      out vec3 fxyz;
      void main() {
        gl_Position = xform*vec4(stxy, 0., 1.);
        fxyz = xyz;
      }
    ''',

    "geometry": '''
      #version 410 core
      
      layout(triangles) in;
      layout(triangle_strip, max_vertices=3) out;
      // out vec3 bary;
      out vec2 bary2;

      void main() {
        for (int i=0; i<3; i++) {
          // vec3 ob = vec3(0.,0.,0.);
          vec3 ob = vec3(0.);
          ob[i] = 1.;
          vec4 pos = gl_in[i].gl_Position;
          gl_Position = pos;
          // bary = ob;
          bary2 = vec2(ob[0],ob[1]);
          EmitVertex();
        }
      }
    ''',

    "fragment": '''
      #version 410 core

      in vec3 fxyz;
      // in vec3 bary;
      in vec2 bary2;
      uniform vec4 color;
      // uniform vec3 xyzmin;
      // uniform vec3 xyzmax;
      // float dz = xyzmax.y - xyzmin.y;
      // float dx = xyzmax.x - xyzmin.x;
      out vec4 fColor;

      void main() {
        /*
        vec4 cmin = vec4(0.,1.,0.,1.);
        vec4 cmax = vec4(1.,0.,1.,1.);
        float z = (fxyz.y-xyzmin.y)/dz;
        z = floor(z*10)/10;

        fColor = z*cmin + (1.-z)*cmax;
        */
        /*
        float factor = 10;
        float z = (fxyz.y-xyzmin.y)/dz;
        z = floor(z*factor)/factor;
        float x = (fxyz.x-xyzmin.x)/dx;
        x = floor(x*factor)/factor;
        fColor = vec4(.5+.5*x, .5+.5*z, .5, 1.);
        */
        vec3 bary = vec3(bary2, 1.-bary2[0]-bary2[1]);
        if (
          bary[0]<=0. || bary[0]>=1. ||
          bary[1]<=0. || bary[1]>=1. ||
          bary[2]<=0. || bary[2]>=1.) {
            discard;
        }
        fColor = vec4(bary, 1.);
        // fColor = vec4(1.,1.,1., 1.);


        // fColor = color;
        // fColor = vec4(0.,1.,0.,1.);
      }
    ''',
}

xyz_code = {
    "name": "xyz",

    "vertex": '''
      #version 410 core

      uniform mat4 xform;
      layout(location=3) in vec3 xyz;
      layout(location=4) in vec2 stxy;
      out vec3 fxyz;
      void main() {
        gl_Position = xform*vec4(stxy, 0., 1.);
        fxyz = xyz;
      }
    ''',

    "fragment": '''
      #version 410 core

      in vec3 fxyz;
      out vec4 fColor;

      void main() {
          fColor = vec4(fxyz/65535., 1.);
      }

    ''',
}

    
class GLSurfaceWindowChild(GLDataWindowChild):
    def __init__(self, gldw, parent=None):
        super(GLSurfaceWindowChild, self).__init__(gldw, parent)

    def localInit(self):
        # This corresponds to the line in the vertex shader(s):
        # layout(location=3) in vec3 xyx;
        self.xyz_location = 3
        # This corresponds to the line in the vertex shader(s):
        # layout(location=4) in vec3 stxy;
        self.stxy_location = 4
        self.message_prefix = "sw"
        # TODO: should set prev_larr to None
        # whenever fragment shape changes
        self.prev_larr = None
        self.prev_zoom_level = None
        self.volume_view =  None
        self.volume_view_direction = -1
        self.atlas = None

    def localInitializeGL(self):
        f = self.gl
        # self.gl.glClearColor(.3,.6,.3,1.)
        f.glClearColor(.6,.3,.3,1.)

        self.active_vao = None

        self.buildPrograms()
        self.buildSliceVao()
        self.data_fbo = None
        self.xyz_fbo = None
        self.xyz_fbo_decimated = None

    def setDefaultViewport(self):
        f = self.gl
        f.glViewport(0, 0, self.vp_width, self.vp_height)
    
    def resizeGL(self, width, height):
        f = self.gl
        self.vp_width = width
        self.vp_height = height
        # print("resizeGL (surface)", width, height)
        # f.glViewport(0, 0, width, height)

        # based on https://stackoverflow.com/questions/59338015/minimal-opengl-offscreen-rendering-using-qt
        vp_size = QSize(width, height)
        fbo_format = QOpenGLFramebufferObjectFormat()
        fbo_format.setAttachment(QOpenGLFramebufferObject.CombinedDepthStencil)
        f = self.gl
        # In Qt5, QFrameworkBufferObject.toImage() creates
        # a uint8 QImage from a float32 fbo.
        # fbo_format.setInternalTextureFormat(f.GL_RGB32F)
        fbo_format.setInternalTextureFormat(f.GL_RGBA16)
        self.xyz_fbo = QOpenGLFramebufferObject(vp_size, fbo_format)
        self.xyz_fbo.bind()
        draw_buffers = (f.GL_COLOR_ATTACHMENT0,)
        f.glDrawBuffers(len(draw_buffers), draw_buffers)

        self.xyz_decimation = 4
        vp_size_decimated = QSize(width//self.xyz_decimation, height//self.xyz_decimation)
        fbo_format = QOpenGLFramebufferObjectFormat()
        fbo_format.setAttachment(QOpenGLFramebufferObject.CombinedDepthStencil)
        f = self.gl
        # In Qt5, QFrameworkBufferObject.toImage() creates
        # a uint8 QImage from a float32 fbo.
        # fbo_format.setInternalTextureFormat(f.GL_RGB32F)
        fbo_format.setInternalTextureFormat(f.GL_RGBA16)
        self.xyz_fbo_decimated = QOpenGLFramebufferObject(vp_size_decimated, fbo_format)
        self.xyz_fbo_decimated.bind()
        draw_buffers = (f.GL_COLOR_ATTACHMENT0,)
        f.glDrawBuffers(len(draw_buffers), draw_buffers)

        fbo_format = QOpenGLFramebufferObjectFormat()
        fbo_format.setAttachment(QOpenGLFramebufferObject.CombinedDepthStencil)
        f = self.gl
        fbo_format.setInternalTextureFormat(f.GL_RGBA16)
        self.data_fbo = QOpenGLFramebufferObject(vp_size, fbo_format)
        self.data_fbo.bind()
        draw_buffers = (f.GL_COLOR_ATTACHMENT0,)
        f.glDrawBuffers(len(draw_buffers), draw_buffers)

        QOpenGLFramebufferObject.bindDefault()

        self.setDefaultViewport()
        
    def paintGL(self):
        self.checkAtlas()
        if self.volume_view is None:
            return

        # print("paintGL (surface)")
        f = self.gl
        f.glClearColor(.6,.3,.6,1.)
        f.glClear(f.GL_COLOR_BUFFER_BIT)
        self.paintSlice()

    def buildPrograms(self):
        self.data_program = self.buildProgram(data_code)
        self.xyz_program = self.buildProgram(xyz_code)
        self.slice_program = self.buildProgram(slice_code)

    def checkAtlas(self):
        # if self.volume_view is None or self.atlas is None or 
        dw = self.gldw
        if dw.volume_view is None:
            self.volume_view = None
            self.volume_view_direction = -1
            self.atlas = None
            return
        if self.volume_view != dw.volume_view or self.volume_view_direction != self.volume_view.direction:
            self.volume_view = dw.volume_view
            self.volume_view_direction = self.volume_view.direction
            self.atlas = Atlas(self.volume_view)

    def paintSlice(self):
        timera = Utils.Timer()
        timera.active = False
        dw = self.gldw
        volume_view = self.volume_view
        f = self.gl

        # viewing window width
        ww = self.size().width()
        wh = self.size().height()
        # viewing window half width
        whw = ww//2
        whh = wh//2

        pv = dw.window.project_view
        mfv = pv.mainActiveFragmentView(unaligned_ok=True)
        if mfv is None:
            # print("No currently active fragment")
            return

        if self.active_vao is None or self.active_vao.fragment_view != mfv:
            self.active_vao = FragmentMapVao(
                    mfv, self.xyz_location, self.stxy_location, self.gl)

        fvao = self.active_vao

        vao = fvao.getVao()
        vao.bind()

        # self.drawXyz(self.xyz_fbo)
        # timera.time("xyz")
        self.drawXyz(self.xyz_fbo_decimated)
        timera.time("xyz 2")
        self.drawData()
        timera.time("data")

        vao.release()

        self.slice_program.bind()
        base_tex = self.data_fbo.texture()
        bloc = self.slice_program.uniformLocation("base_sampler")
        if bloc < 0:
            print("couldn't get loc for base sampler")
            return
        bunit = 1
        f.glActiveTexture(f.GL_TEXTURE0+bunit)
        f.glBindTexture(f.GL_TEXTURE_2D, base_tex)
        self.slice_program.setUniformValue(bloc, bunit)

        self.slice_program.setUniformValue(bloc, bunit)

        underlay_data = np.zeros((wh,ww,4), dtype=np.uint16)
        self.drawUnderlays(underlay_data)
        underlay_tex = self.texFromData(underlay_data, QImage.Format_RGBA64)
        uloc = self.slice_program.uniformLocation("underlay_sampler")
        if uloc < 0:
            print("couldn't get loc for underlay sampler")
            return
        uunit = 2
        f.glActiveTexture(f.GL_TEXTURE0+uunit)
        underlay_tex.bind()
        self.slice_program.setUniformValue(uloc, uunit)

        overlay_data = np.zeros((wh,ww,4), dtype=np.uint16)
        self.drawOverlays(overlay_data)
        overlay_tex = self.texFromData(overlay_data, QImage.Format_RGBA64)
        oloc = self.slice_program.uniformLocation("overlay_sampler")
        if oloc < 0:
            print("couldn't get loc for overlay sampler")
            return
        ounit = 3
        f.glActiveTexture(f.GL_TEXTURE0+ounit)
        overlay_tex.bind()
        self.slice_program.setUniformValue(oloc, ounit)

        f.glActiveTexture(f.GL_TEXTURE0)
        self.slice_vao.bind()
        self.slice_program.bind()
        f.glDrawElements(f.GL_TRIANGLES, 
                         self.slice_indices.size, f.GL_UNSIGNED_INT, None)
        self.slice_program.release()
        self.slice_vao.release()
        timera.time("combine")

        '''
        zoom_level, larr = self.getBlocks(self.xyz_fbo)
        zoom_level2, larr2 = self.getBlocks(self.xyz_fbo_decimated)
        if zoom_level >= 0:
            # print("xyz larr", zoom_level, len(larr))
            # print("xyz 2 larr", zoom_level2, len(larr2))
            # self.printBlocks(larr2)
            lset = self.blocksToSet(larr)
            lset2 = self.blocksToSet(larr2)
            dset = lset.symmetric_difference(lset2)
            # print("sym diff", dset)
            if len(dset) > 0:
                print(zoom_level, len(larr), len(larr2), dset)
        '''

        zoom_level, larr = self.getBlocks(self.xyz_fbo_decimated)
        if zoom_level >= 0 and self.atlas is not None:
            self.atlas.displayBlocks(zoom_level, larr)


    def printBlocks(self, blocks):
        for block in blocks:
            print(block)

    def blocksToSet(self, blocks):
        bset = set()
        for block in blocks:
            bset.add(tuple(block))
        return bset
        

    def getBlocks(self, fbo):
        timera = Utils.Timer()
        timera.active = False
        dw = self.gldw
        # volume_view = dw.volume_view
        f = self.gl

        # fbo = self.xyz_fbo
        im = fbo.toImage(True)
        timera.time("get image")
        # print("im format", im.format())
        # w = fbo.width()
        # h = fbo.height()
        arr = self.npArrayFromQImage(im)
        timera.time("array from image")
        # print("arr", arr.shape, arr.dtype)
        zoom = dw.getZoom()
        iscale = 1
        for izoom in range(7):
            lzoom = 1./iscale
            if lzoom < 2*zoom:
                break
            iscale *= 2

        # 1/zoom, scale
        # 0. - 2. 1
        # 2. - 4. 2
        # 4. - 8. 4
        # 8. - 16. 8
        # 16. - 32. 16
        # print("zoom", zoom, iscale)
        dv = 128*iscale
        zoom_level = izoom
        nzarr = arr[arr[:,:,3] > 0][:,:3] // dv

        if len(nzarr) == 0:
            return -1, None

        # print("nzarr", nzarr.shape, nzarr.dtype)
        nzmin = nzarr.min(axis=0)
        nzmax = nzarr.max(axis=0)
        # print(nzmin, nzmax)
        nzsarr = nzarr-nzmin
        # print(nzsarr.min(axis=0), nzsarr.max(axis=0))
        dvarr = np.zeros(nzmax-nzmin+1, dtype=np.uint32)[:,:,:,np.newaxis]
        indices = np.indices(nzmax-nzmin+1).transpose(1,2,3,0)
        # print(dvarr.shape, indices.shape)
        dvarr = np.concatenate((dvarr, indices), axis=3)
        # print(dvarr.shape)
        # print("dvarr", dvarr.shape, nzsarr.shape)
        # print((nzarr)[:5])
        # print((nzsarr)[:5])
        dvarr[nzsarr[:,0],nzsarr[:,1], nzsarr[:,2],0] = 1
        # print("dvarr", dvarr.size, dvarr.sum())
        larr = dvarr[dvarr[:,:,:,0] == 1][:,1:]+nzmin
        # print("dvarr, larr", dvarr.shape, larr.shape)
        # print(larr)
        '''
        if self.prev_zoom_level != zoom_level or self.prev_larr is None or len(self.prev_larr) != len(larr) or (self.prev_larr[:,:] != larr[:,:]).any():
            print("change", zoom_level, len(larr))
            self.prev_larr = larr
            self.prev_zoom_level = zoom_level
        '''
        # print("larr", zoom_level, len(larr))
        # if self.atlas is not None:
        #     self.atlas.displayBlocks(zoom_level, larr)

        # nzu = np.unique(nzarr//128, axis=0)
        # print(len(nzarr),len(nzu), nzu[0], nzu[-1])
        timera.time("process image")

        return zoom_level, larr

        '''
        return
        iptr = f.glReadPixels(0, 0, fbo.width(), fbo.height(), f.GL_RGBA, f.GL_FLOAT)
        # print("iptr", len(iptr), w, h, w*h*4)
        arr = np.array(iptr)
        arr.resize(h, w, 4)
        # print(arr.shape, arr.dtype)
        print(arr[0,0])
        print(arr[200,200])
        # iptr.setSize(w*h*4*4)
        # arr = np.frombuffer(iptr, dtype=np.float32)
        '''

    def stxyXform(self):
        dw = self.gldw

        ww = dw.size().width()
        wh = dw.size().height()
        volume_view = self.volume_view

        zoom = dw.getZoom()
        cij = volume_view.stxytf
        # print("cij", cij)
        mat = np.zeros((4,4), dtype=np.float32)
        # ww = dw.size().width()
        # wh = dw.size().height()
        wf = zoom/(.5*ww)
        hf = zoom/(.5*wh)
        mat[0][0] = wf
        mat[0][3] = -wf*cij[0]
        mat[1][1] = -hf
        mat[1][3] = hf*cij[1]
        mat[3][3] = 1.
        xform = QMatrix4x4(mat.flatten().tolist())
        return xform


    def drawDataOrig(self):
        f = self.gl
        dw = self.gldw

        self.data_fbo.bind()

        # Be sure to clear with alpha = 0
        # so that the slice view isn't blocked!
        f.glClearColor(0.,0.,0.,0.)
        f.glClear(f.GL_COLOR_BUFFER_BIT)

        self.data_program.bind()

        xform = self.stxyXform()
        self.data_program.setUniformValue("xform", xform)

        pv = dw.window.project_view
        mfv = pv.mainActiveFragmentView(unaligned_ok=True)
        if mfv is None:
            # print("No currently active fragment")
            return

        if self.active_vao is None or self.active_vao.fragment_view != mfv:
            self.active_vao = FragmentMapVao(
                    mfv, self.xyz_location, self.stxy_location, self.gl)

        fvao = self.active_vao
        '''
        qcolor = mfv.fragment.color
        rgba = list(qcolor.getRgbF())
        rgba[3] = 1.
        self.trgl_program.setUniformValue("color", *rgba)
        # self.trgl_program.setUniformValue("xyzmin", *mfv.xyzmin)
        # self.trgl_program.setUniformValue("xyzmax", *mfv.xyzmax)
        '''

        vao = fvao.getVao()
        vao.bind()

        # f.glPolygonMode(f.GL_FRONT_AND_BACK, f.GL_LINE)
        f.glDrawElements(f.GL_TRIANGLES, fvao.trgl_index_size,
                       f.GL_UNSIGNED_INT, None)
        vao.release()
        self.trgl_program.release()

        QOpenGLFramebufferObject.bindDefault()

    def drawData(self):
        f = self.gl
        dw = self.gldw
        fvao = self.active_vao

        self.data_fbo.bind()

        # Be sure to clear with alpha = 0
        # so that the slice view isn't blocked!
        f.glClearColor(0.,0.,0.,0.)
        f.glClear(f.GL_COLOR_BUFFER_BIT)

        self.data_program.bind()

        xform = self.stxyXform()
        self.data_program.setUniformValue("xform", xform)

        f.glDrawElements(f.GL_TRIANGLES, fvao.trgl_index_size,
                       f.GL_UNSIGNED_INT, None)
        self.data_program.release()

        QOpenGLFramebufferObject.bindDefault()

    def drawXyz(self, fbo):
        f = self.gl
        dw = self.gldw
        fvao = self.active_vao

        # self.xyz_fbo.bind()
        fbo.bind()

        # Be sure to clear with alpha = 0
        # so that the slice view isn't blocked!
        f.glClearColor(0.,0.,0.,0.)
        f.glClear(f.GL_COLOR_BUFFER_BIT)
        f.glViewport(0, 0, fbo.width(), fbo.height())

        self.xyz_program.bind()

        xform = self.stxyXform()
        self.xyz_program.setUniformValue("xform", xform)

        f.glDrawElements(f.GL_TRIANGLES, fvao.trgl_index_size,
                       f.GL_UNSIGNED_INT, None)
        self.xyz_program.release()

        QOpenGLFramebufferObject.bindDefault()
        self.setDefaultViewport()


# two attribute buffers: xyz, and stxy (st = scaled texture)
class FragmentMapVao:
    def __init__(self, fragment_view, xyz_loc, stxy_loc, gl):
        self.fragment_view = fragment_view
        self.gl = gl
        self.vao = None
        self.vao_modified = ""
        self.is_line = False
        self.xyz_loc = xyz_loc
        self.stxy_loc = stxy_loc
        self.getVao()

    def getVao(self):
        fv = self.fragment_view
        if self.vao_modified > fv.modified and self.vao_modified > fv.fragment.modified and self.vao_modified > fv.local_points_modified:
            # print("returning existing vao")
            return self.vao

        self.vao_modified = Utils.timestamp()

        if self.vao is None:
            self.vao = QOpenGLVertexArrayObject()
            self.vao.create()
            # print("creating new vao")

        # print("updating vao")
        self.vao.bind()

        f = self.gl

        self.xyz_vbo = QOpenGLBuffer()
        self.xyz_vbo.create()
        self.xyz_vbo.bind()

        xyzs = np.ascontiguousarray(fv.vpoints[:,:3], dtype=np.float32)
        self.xyzs_size = xyzs.size

        nbytes = xyzs.size*xyzs.itemsize
        self.xyz_vbo.allocate(xyzs, nbytes)

        f.glVertexAttribPointer(
                self.xyz_loc,
                xyzs.shape[1], int(f.GL_FLOAT), int(f.GL_FALSE), 
                0, 0)
        self.xyz_vbo.release()
        # This needs to be called while the current VAO is bound
        f.glEnableVertexAttribArray(self.xyz_loc)

        self.stxy_vbo = QOpenGLBuffer()
        self.stxy_vbo.create()
        self.stxy_vbo.bind()

        stxys = np.ascontiguousarray(fv.stpoints, dtype=np.float32)
        self.stxys_size = stxys.size

        nbytes = stxys.size*stxys.itemsize
        self.stxy_vbo.allocate(stxys, nbytes)
        f.glVertexAttribPointer(
                self.stxy_loc,
                stxys.shape[1], int(f.GL_FLOAT), int(f.GL_FALSE), 
                0, 0)
        self.stxy_vbo.release()
        # This needs to be called while the current VAO is bound
        f.glEnableVertexAttribArray(self.stxy_loc)

        self.ibo = QOpenGLBuffer(QOpenGLBuffer.IndexBuffer)
        self.ibo.create()
        self.ibo.bind()

        # We may have a line, not a triangulated surface.
        # Notice that indices must be uint8, uint16, or uint32
        fv_trgls = fv.trgls()
        self.is_line = False
        if fv_trgls is None:
            fv_line = fv.line
            if fv_line is not None:
                self.is_line = True
                # Despite the name "fv_trgls",
                # this contains a line strip if self.is_line is True.
                fv_trgls = fv.line[:,2]
            else:
                fv_trgls = np.zeros((0,3), dtype=np.uint32)
        
        trgls = np.ascontiguousarray(fv_trgls, dtype=np.uint32)

        self.trgl_index_size = trgls.size

        nbytes = trgls.size*trgls.itemsize
        self.ibo.allocate(trgls, nbytes)

        # print("nodes, trgls", pts3d.shape, trgls.shape)

        self.vao.release()
        
        # do not release ibo before vao is released!
        self.ibo.release()

        return self.vao



        self.makeCurrent()
        self.logger.stopLogging()
        print("stopped logging")
        # e.accept()


# The Chunk class is used by the Atlas class
# see below) to keep track of data stored in
# the 3D texture atlas.
# Variable naming conventions:
# d: data, a: atlas, c: chunk, pc: padded chunk
# _: corner, k: key, l: level
# _: coords, sz: size, e: single coord, r: rect
#
# coordinates are (x, y, z);
# the data value at (x, y, z) is accessed by data[z][y][x]

class Chunk:
    def __init__(self, atlas, ak, dk, dl):
        # atlas, ak, dk, dl
        # Given atlas, atlas key, data key, data level
        # copy chunk from data to atlas_data
        # compute xform

        # Atlas
        self.atlas = atlas
        # Chunk key (position) in atlas (3 coords)
        self.ak = ak
        # Chunk key (position) in input data (3 coords: x, y, z)

        # atlas chunk size (3 coords, usually 130,130,130)
        acsz = atlas.acsz
        # atlas rectangle
        ar = self.k2r(ak, acsz)
        self.ar = ar
        # atlas corner
        a = ar[0]

        # padding (scalar, usually 1)
        self.pad = atlas.pad

        # size of the atlas (3 coords: nx, ny, nz)
        asz = atlas.asz
        # rectangle of the entire data set

        self.setData(dk, dl)
        self.in_use = False


    def setData(self, dk, dl):
        self.dk = dk
        self.dl = dl
        if dl < 0:
            return

        # data chunk size (3 coords, usually 128,128,128)
        dcsz = self.atlas.dcsz 
        # data rectangle
        dr = self.k2r(dk, dcsz)
        # data corner
        d = dr[0]

        # padded data rectangle
        pdr = self.padRect(dr, self.pad)
        # size of the data on the data's level (3 coords: nx, ny, nz)
        dsz = self.atlas.dsz[dl]
        all_dr = ((0, 0, 0), (dsz[0], dsz[1], dsz[2]))
        # intersection of the padded data rectangle with the data
        int_dr = self.rectIntersection(pdr, all_dr)
        # print(pdr, all_dr, int_dr)

        # Compute change in pdr (padded data-chunk rectangle) 
        # due to intersection with edges of data array:
        # Difference in min corner:
        # skip0 = (int_dr[0][0]-pdr[0][0], int_dr[0][1]-pdr[0][1])
        skip0 = tuple(int_dr[0][i]-pdr[0][i] for i in range(len(int_dr[0])))
        # Difference in max corner:
        # skip1 = (pdr[1][0]-int_dr[1][0], pdr[1][1]-int_dr[1][1])
        skip1 = tuple(pdr[1][i]-int_dr[1][i] for i in range(len(pdr[1])))

        # print(pdr, skip0)
        # print(ar, int_dr, skip0, skip1)
        # TODO: copy into atlas texture
        '''
        atlas.atlas_data[
                (ar[0][1]+skip0[1]):(ar[1][1]-skip1[1]), 
                (ar[0][0]+skip0[0]):(ar[1][0]-skip1[0])
                ] = atlas.data[
                        (int_dr[0][1]):int_dr[1][1], 
                        (int_dr[0][0]):int_dr[1][0]
                        ]
        '''

        # TODO: fix errors
        xform = QMatrix4x4()
        # xform.scale(1./asz[0], 1./asz[1], 1./asz[2])
        asz = self.atlas.asz
        xform.scale(*(1./asz[i] for i in range(len(asz))))
        # xform.translate(ar[0][0]+pad-dr[0][0], ar[0][1]+pad-dr[0][1])
        xform.translate(*(self.ar[0][i]+self.pad-dr[0][i] for i in range(len(self.ar))))
        # xform.scale(dsz[0], dsz[1])
        xform.scale(*(dsz[i] for i in range(len(dsz))))
        self.xform = xform
        # self.tmin = ((dr[0][0])/dsz[0], (dr[0][1])/dsz[1])
        self.tmin = ((dr[0][i])/dsz[i] for i in range(len(dsz)))
        # self.tmax = ((dr[1][0])/dsz[0], (dr[1][1])/dsz[1])
        self.tmax = ((dr[1][i])/dsz[i] for i in range(len(dsz)))
        # self.tmin = (0.01, 0.)
        # self.tmax = (1., 1.)
        # if ak[0] == 0 and ak[1] == 0:
        #     print("tm", self.tmin, self.tmax)

        self.in_use = True

    @staticmethod
    def k2r(k, csz):
        # c = (k[0]*csz[0], k[1]*csz[1])
        # r = (c, (c[0]+csz[0], c[1]+csz[1]))
        c = tuple(k[i]*csz[i] for i in range(len(k)))
        r = (c, tuple(c[i]+csz[i] for i in range(len(c))))
        return r

    # padded rectangle
    @staticmethod
    def padRect(rect, pad):
        # return ((rect[0][0]-pad, rect[0][1]-pad), 
        #    (rect[1][0]+pad, rect[1][1]+pad))
        r = (tuple(rect[0][i]-pad for i in range(len(rect[0]))),
             tuple(rect[1][i]-pad for i in range(len(rect[1]))))
        return r

    # adapted from https://stackoverflow.com/questions/25068538/intersection-and-difference-of-two-rectangles/25068722#25068722
    @staticmethod
    def rectIntersection(ra, rb):
        # if not Utils.rectIsValid(ra) or not Utils.rectIsValid(rb):
        #     return Utils.emptyRect()
        (ax1, ay1, az1), (ax2, ay2, az2) = ra
        (bx1, by1, bz1), (bx2, by2, bz2) = rb
        # print(ra, rb)
        x1 = max(min(ax1, ax2), min(bx1, bx2))
        y1 = max(min(ay1, ay2), min(by1, by2))
        z1 = max(min(az1, az2), min(bz1, bz2))
        x2 = min(max(ax1, ax2), max(bx1, bx2))
        y2 = min(max(ay1, ay2), max(by1, by2))
        z2 = min(max(az1, az2), max(bz1, bz2))
        if (x1<x2) and (y1<y2) and (z1<z2):
            r = ((x1, y1, z1), (x2, y2, z2))
            # print(r)
            return r

# Atlas implements a 3D texture atlas.  The 3D OpenGL texture
# (the atlas) is subdivided into chunks; each atlas chunk stores
# a scroll data chunk (conventionally 128^3 in size).  
# NOTE that the data chunk size used in Atlas is NOT related
# to the zarr chunk size, if any, that is used to store
# the scroll data on disk.
# Each atlas chunk # is padded (to prevent texture bleeding) so 
# by default each atlas chunk is 130^ in size.
# The Atlas class keeps track of which data chunk is stored in
# whice atlas chunk.  As new data chunks are added, old data
# chunks are removed as needed.
# The chunks (with scroll data location and texture location)
# are stored in an OrderedDict.  In-use chunks are kept at the
# end of this dict.

class Atlas:
    def __init__(self, volume_view, tex3dsz=(2048,2048,300), dcsz=(128,128,128)):
        # self.created = Utils.timestamp()
        pad = 1
        self.pad = pad
        self.volume_view = volume_view
        self.dcsz = dcsz
        acsz = tuple(dcsz[i]+2*pad for i in range(len(dcsz)))
        self.acsz = acsz
        vol = volume_view.volume
        vdir = volume_view.direction
        is_zarr = vol.is_zarr
        dsz = []
        if not is_zarr:
            shape = vol.trdatas[vdir].shape
            dsz.append(tuple(shape[::-1]))
        else:
            for level in vol.levels:
                shape = level.trdatas[vdir].shape
                dsz.append(tuple(shape[::-1]))
        # dcsz = [(data.shape[2], data.shape[1], data.shape[0])]
        print("dsz")
        print(dsz)
        self.dsz = dsz
        ksz = []
        for l in range(len(dsz)):
            lksz = tuple(self.ke(dsz[l][i],dcsz[i]) for i in range(len(dcsz)))
            ksz.append(lksz)
        print("ksz")
        print(ksz)
        self.ksz = ksz
        # aksz = tuple(self.ke(2048, acsz[0]), self.ke(2048, acsz[1]), 2*acsz[2])
        aksz = tuple(tex3dsz[i]//acsz[i] for i in range(len(acsz)))
        self.asz = tuple(aksz[i]*acsz[i] for i in range(len(acsz)))
        print("asz", self.asz)

        self.chunks = OrderedDict()

        for k in range(aksz[2]):
            for j in range(aksz[1]):
                for i in range(aksz[0]):
                    ak = (i,j,k)
                    dk = (i,j,k)
                    dl = -1
                    chunk = Chunk(self, ak, dk, dl)
                    key = self.key(dk, dl)
                    self.chunks[key] = chunk

        # allocate 3D texture 
        tex3d = QOpenGLTexture(QOpenGLTexture.Target3D)
        tex3d.setWrapMode(QOpenGLTexture.ClampToBorder)
        tex3d.setAutoMipMapGenerationEnabled(False)
        tex3d.setMagnificationFilter(QOpenGLTexture.Linear)
        tex3d.setMinificationFilter(QOpenGLTexture.Linear)
        # width, height, depth
        tex3d.setSize(*self.asz)
        # print("immutable", tex3d.hasFeature(QOpenGLTexture.ImmutableStorage))
        # see https://stackoverflow.com/questions/23533749/difference-between-gl-r16-and-gl-r16ui
        tex3d.setFormat(QOpenGLTexture.R16_UNorm)
        tex3d.allocateStorage()
        self.tex3d = tex3d

    def key(self, dk, dl):
        return (dl, dk[2], dk[1], dk[0])

    def index(self, dk):
        dksz = self.dksz
        return (dk[2]*dksz[1] + dk[1])*dksz[0] + dk[0]

    # Number of chunks (in 1D) given data size, chunk size.
    # This gives the number of chunks needed to cover the
    # entire data set; the last chunk may stretch beyond
    # the end of the data.
    def ke(self, e, ce):
        ke = 1 + (e-1)//ce
        return ke

    # Given a list of blocks, add the blocks that are
    # not already in the atlas. 
    def addBlocks(self, zoom_level, blocks):
        for chunk in reversed(self.chunks.values()):
            if chunk.in_use == False:
                break
            chunk.in_use = False
        for block in blocks:
            key = self.key(block, zoom_level)
            chunk = self.chunks.get(key, None)
            # If the data chunk is not currently stored in the atlas:
            if chunk is None:
                # chunk = self.chunks.pop()
                # chunk = self.chunks[next(iter(self.chunks))]
                # self.chunks.pop(self.key(chunk.dk, chunk.dl))
                # Get the first Chunk in the OrderedDict: 
                _, chunk = self.chunks.popitem(last=False)
                chunk.setData(block, zoom_level)
                # print("set data", chunk.dk, chunk.dl)
                self.chunks[key] = chunk
            else: # If the data is alread in the Atlas
                # move the chunk to the end of the OrderedDict
                self.chunks.move_to_end(key)
            chunk.in_use = True

        cnt = 0
        # To get all the active chunks, search backwards from
        # the end
        for key,chunk in reversed(self.chunks.items()):
            if not chunk.in_use:
                break
            # print(chunk.dl, chunk.dk)
            cnt += 1
        # print(zoom_level, cnt, len(blocks))
            
    def displayBlocks(self, zoom_level, blocks):
        self.addBlocks(zoom_level, blocks)

