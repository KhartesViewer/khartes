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
        self.zoomMult = .5

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
          fColor = vec4(fxyz, 1.);
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

    def localInitializeGL(self):
        f = self.gl
        # self.gl.glClearColor(.3,.6,.3,1.)
        f.glClearColor(.6,.3,.3,1.)

        self.active_vao = None

        self.buildPrograms()
        self.buildSliceVao()
        self.data_fbo = None
        self.xyz_fbo = None
    
    def resizeGL(self, width, height):
        f = self.gl
        # print("resizeGL (surface)", width, height)
        # f.glViewport(0, 0, width, height)

        # based on https://stackoverflow.com/questions/59338015/minimal-opengl-offscreen-rendering-using-qt
        vp_size = QSize(width, height)
        fbo_format = QOpenGLFramebufferObjectFormat()
        fbo_format.setAttachment(QOpenGLFramebufferObject.CombinedDepthStencil)
        f = self.gl
        fbo_format.setInternalTextureFormat(f.GL_RGB32F)
        self.xyz_fbo = QOpenGLFramebufferObject(vp_size, fbo_format)
        self.xyz_fbo.bind()
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

        f.glViewport(0, 0, vp_size.width(), vp_size.height())
        
    def paintGL(self):
        if self.gldw.volume_view is None:
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

    def paintSlice(self):
        dw = self.gldw
        volume_view = dw.volume_view
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

        self.drawXyz()
        self.drawData()

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

        fbo = self.xyz_fbo
        # im = fbo.toImage(True)
        # print("im format", im.format())
        w = fbo.width()
        h = fbo.height()
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

    def stxyXform(self):
        dw = self.gldw

        ww = dw.size().width()
        wh = dw.size().height()
        volume_view = dw.volume_view

        zoom = dw.getZoom()
        cij = volume_view.stxytf
        # print("cij", cij)
        mat = np.zeros((4,4), dtype=np.float32)
        ww = dw.size().width()
        wh = dw.size().height()
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

    def drawXyz(self):
        f = self.gl
        dw = self.gldw
        fvao = self.active_vao

        self.xyz_fbo.bind()

        # Be sure to clear with alpha = 0
        # so that the slice view isn't blocked!
        f.glClearColor(0.,0.,0.,0.)
        f.glClear(f.GL_COLOR_BUFFER_BIT)

        self.xyz_program.bind()

        xform = self.stxyXform()
        self.xyz_program.setUniformValue("xform", xform)

        f.glDrawElements(f.GL_TRIANGLES, fvao.trgl_index_size,
                       f.GL_UNSIGNED_INT, None)
        self.xyz_program.release()

        QOpenGLFramebufferObject.bindDefault()


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


