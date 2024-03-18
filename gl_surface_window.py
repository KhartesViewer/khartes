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
        QVector3D,
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

    def setIjkTf(self, tf):
        oijk = self.volume_view.ijktf
        iind = self.iIndex
        jind = self.jIndex
        kind = self.kIndex
        di = tf[iind] - oijk[iind]
        dj = tf[jind] - oijk[jind]
        zoom = self.getZoom()
        dx = di*zoom
        dy = dj*zoom
        ww, wh = self.width(), self.height()
        ox, oy = ww/2, wh/2
        nx, ny = ox+dx, oy+dy
        nijk = self.xyToTijk((nx,ny))
        self.volume_view.setIjkTf(nijk)
        ostxy = self.volume_view.stxytf
        nstxy = (ostxy[0]+di, ostxy[1]+dj)
        self.volume_view.setStxyTf(nstxy)

    def setIjkOrStxyTf(self, tf):
        ostxy = self.volume_view.stxytf
        iind = self.iIndex
        jind = self.jIndex
        kind = self.kIndex
        di = tf[iind]-ostxy[iind]
        dj = tf[jind]-ostxy[jind]
        zoom = self.getZoom()
        dx = di*zoom
        dy = dj*zoom
        ww, wh = self.width(), self.height()
        ox, oy = ww/2, wh/2
        nx, ny = ox+dx, oy+dy
        nijk = self.xyToTijk((nx,ny))
        self.volume_view.setIjkTf(nijk)
        self.volume_view.setStxyTf(tf)

    def computeTfStartPoint(self):
        return self.volume_view.stxytf

    def xyToTijk(self, xy):
        # print(xy)
        # return super(GLSurfaceWindow, self).xyToTijk(xy)
        x, y = xy
        # ww, wh = self.width(), self.height()
        # wcx, wcy = ww//2, wh//2
        # dx, dy = x-wcx, y-wcy
        # dx, dy = x, y
        iind = self.iIndex
        jind = self.jIndex
        kind = self.kIndex
        xyz_arr = self.glw.xyz_arr
        if xyz_arr is None:
            print("returning vv.ijktf")
            return self.volume_view.ijktf
        # print(wcx, wcy, xyz_arr.shape)
        # idx = round(dx)
        # idy = round(dy)
        ix = round(x)
        iy = round(y)
        if iy < 0 or iy >= xyz_arr.shape[0] or ix < 0 or ix >= xyz_arr.shape[1]:
            print("error", x, y, xyz_arr.shape)
            return self.volume_view.ijktf
        xyza = xyz_arr[iy, ix]
        if xyza[3] == 0:
            return self.volume_view.ijktf
        i = xyza[iind]
        j = xyza[jind]
        k = xyza[kind]
        # print("ijk", i,j,k)
        return (i,j,k)

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
      in vec2 ftxt;
      out vec4 fColor;

      void main()
      {
        float alpha;
        fColor = texture(base_sampler, ftxt);

        vec4 uColor = texture(underlay_sampler, ftxt);
        alpha = uColor.a;
        fColor = (1.-alpha)*fColor + alpha*uColor;

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
      out vec2 bary2;

      void main() {
        for (int i=0; i<3; i++) {
          vec3 ob = vec3(0.);
          ob[i] = 1.;
          vec4 pos = gl_in[i].gl_Position;
          gl_Position = pos;
          bary2 = vec2(ob[0],ob[1]);
          EmitVertex();
        }
      }
    ''',

    "fragment": '''
      #version 410 core

      in vec3 fxyz;
      in vec2 bary2;
      uniform vec4 color;
      out vec4 fColor;

      void main() {
        vec3 bary = vec3(bary2, 1.-bary2[0]-bary2[1]);
        if (
          bary[0]<=0. || bary[0]>=1. ||
          bary[1]<=0. || bary[1]>=1. ||
          bary[2]<=0. || bary[2]>=1.) {
            discard;
        }
        fColor = vec4(bary, 1.);
      }
    ''',
}

trgl_id_code_no_longer_used = {
    "name": "trgl_id",

    "vertex": '''
      #version 410 core

      uniform mat4 xform;
      // layout(location=3) in vec3 xyz;
      layout(location=4) in vec2 stxy;
      // in vec2 stxy;
      // out vec3 fxyz;
      void main() {
        gl_Position = xform*vec4(stxy, 0., 1.);
        // fxyz = xyz;
      }
    ''',

    "fragment": '''
      #version 410 core

      out vec4 fColor;

      void main() {
        uint lsid = gl_PrimitiveID & 0xffff;
        uint msid = (gl_PrimitiveID>>16) & 0xffff;
        fColor = vec4(float(msid)/65535., float(lsid)/65535., 0., 1.);
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
        # self.prev_larr = None
        # self.prev_zoom_level = None
        # Cache these so we can recalculate the atlas 
        # whenever volume_view or volume_view.direction change
        self.volume_view =  None
        self.volume_view_direction = -1
        self.atlas = None
        self.active_vao = None
        self.data_fbo = None
        # self.xyz_fbo = None
        self.trgl_id_fbo = None
        self.xyz_fbo_decimated = None
        # self.trgl_ids = None
        self.xyz_arr = None

    def localInitializeGL(self):
        f = self.gl
        # self.gl.glClearColor(.3,.6,.3,1.)
        f.glClearColor(.6,.3,.3,1.)
        self.buildPrograms()
        self.buildSliceVao()

    def setDefaultViewport(self):
        f = self.gl
        f.glViewport(0, 0, self.vp_width, self.vp_height)
    
    def resizeGL(self, width, height):
        f = self.gl
        self.vp_width = width
        self.vp_height = height
        vp_size = QSize(width, height)
        # print("resizeGL (surface)", width, height)

        # fbo where xyz positions are drawn; this information is used
        # to determine which data chunks to load.
        # The fbo is decimated relative to the display window,
        # in order to speed up some parts of this, which are done
        # (at the moment) in the CPU rather than the GPU.
        # based on https://stackoverflow.com/questions/59338015/minimal-opengl-offscreen-rendering-using-qt
        # self.xyz_decimation = 4
        self.xyz_decimation = 1
        vp_size_decimated = QSize(width//self.xyz_decimation, height//self.xyz_decimation)
        fbo_format = QOpenGLFramebufferObjectFormat()
        fbo_format.setAttachment(QOpenGLFramebufferObject.CombinedDepthStencil)
        # We would prefer to store the xyz information as floats.
        # However, in Qt5, QFrameworkBufferObject.toImage() creates
        # a uint8 QImage from a float32 fbo.  uint8 is too low
        # a resolution for our purposes!
        # The uint16 format can store xyz at a resolution of 1
        # pixel, which is good enough for our purposes.
        # fbo_format.setInternalTextureFormat(f.GL_RGB32F)
        fbo_format.setInternalTextureFormat(f.GL_RGBA16)
        self.xyz_fbo_decimated = QOpenGLFramebufferObject(vp_size_decimated, fbo_format)
        self.xyz_fbo_decimated.bind()
        draw_buffers = (f.GL_COLOR_ATTACHMENT0,)
        f.glDrawBuffers(len(draw_buffers), draw_buffers)

        '''
        # fbo where the trgl_id's will be drawn
        fbo_format = QOpenGLFramebufferObjectFormat()
        fbo_format.setAttachment(QOpenGLFramebufferObject.CombinedDepthStencil)
        fbo_format.setInternalTextureFormat(f.GL_RGBA16)
        self.trgl_id_fbo = QOpenGLFramebufferObject(vp_size, fbo_format)
        self.trgl_id_fbo.bind()
        draw_buffers = (f.GL_COLOR_ATTACHMENT0,)
        f.glDrawBuffers(len(draw_buffers), draw_buffers)
        '''

        # fbo where the data will be drawn
        fbo_format = QOpenGLFramebufferObjectFormat()
        fbo_format.setAttachment(QOpenGLFramebufferObject.CombinedDepthStencil)
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
        # self.data_program = self.buildProgram(data_code)
        self.xyz_program = self.buildProgram(xyz_code)
        # self.trgl_id_program = self.buildProgram(trgl_id_code)
        self.slice_program = self.buildProgram(slice_code)

    # Rebuild atlas if volume_view or volume_view.direction
    # changes
    def checkAtlas(self):
        dw = self.gldw
        if dw.volume_view is None:
            self.volume_view = None
            self.volume_view_direction = -1
            self.atlas = None
            return
        if self.volume_view != dw.volume_view or self.volume_view_direction != self.volume_view.direction:
            self.volume_view = dw.volume_view
            self.volume_view_direction = self.volume_view.direction
            self.atlas = Atlas(self.volume_view, self.gl)

    def paintSlice(self):
        timera = Utils.Timer()
        timera.active = False
        timerb = Utils.Timer()
        # timerb.active = False
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

        # timera.time("xyz")
        # self.drawXyz(self.xyz_fbo_decimated)
        self.drawTrgls(self.xyz_fbo_decimated, self.xyz_program)
        timera.time("xyz 2")

        ''''''
        zoom_level, larr, self.xyz_arr = self.getBlocks(self.xyz_fbo_decimated)
        if zoom_level >= 0 and self.atlas is not None:
            if len(larr) >= self.atlas.max_nchunks-1:
                larr = larr[:self.atlas.max_nchunks-1]
            self.atlas.addBlocks(zoom_level, larr)
        ''''''

        '''
        self.drawTrgls(self.trgl_id_fbo, self.trgl_id_program)
        timera.time("trgl ids")
        self.trgl_ids = self.getTrglIds(self.trgl_id_fbo)
        # print(self.trgl_ids.shape, self.trgl_ids.min(), self.trgl_ids.max())
        '''

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

        # The texture atlas is updated at the end (here),
        # in order to avoid a round trip to the GPU.
        # However, the scroll data is drawn onto data_fbo
        # further up, even though this means that the data will be
        # drawn using a texture atlas that is out of date by one frame.
        '''
        zoom_level, larr = self.getBlocks(self.xyz_fbo_decimated)
        if zoom_level >= 0 and self.atlas is not None:
            if len(larr) >= self.atlas.max_nchunks-1:
                larr = larr[:self.atlas.max_nchunks-1]
            self.atlas.addBlocks(zoom_level, larr)
        '''
        timerb.time("done")

    def printBlocks(self, blocks):
        for block in blocks:
            print(block)

    def blocksToSet(self, blocks):
        bset = set()
        for block in blocks:
            bset.add(tuple(block))
        return bset
        
    '''
    def getTrglIds(self, fbo):
        dw = self.gldw
        f = self.gl

        im = fbo.toImage(True)
        im_arr = self.npArrayFromQImage(im)
        w = fbo.width()
        h = fbo.height()
        trgl_ids = np.full((h,w), -2, dtype=np.int32)
        # print("im_arr[3]", im_arr[:,:,3].min(), im_arr[:,:,3].max())
        non_zeros = im_arr[:,:,3] > 0
        msid = im_arr[:,:,0]
        lsid = im_arr[:,:,1]
        tid = msid*65536 + lsid
        trgl_ids[non_zeros] = tid[:,:][non_zeros]
        # print("trgl_ids", trgl_ids.shape)
        return trgl_ids
    '''

    def getBlocks(self, fbo):
        timera = Utils.Timer()
        timera.active = False
        dw = self.gldw
        f = self.gl

        im = fbo.toImage(True)
        timera.time("get image")
        # print("im format", im.format())
        # w = fbo.width()
        # h = fbo.height()
        # arr = self.npArrayFromQImage(im)
        # decimation factor
        # arr = self.npArrayFromQImage(im)
        farr = self.npArrayFromQImage(im)
        df = 4
        arr = farr[::df,::df,:]
        # print(farr.shape, arr.shape)
        timera.time("array from image")
        # print("arr", arr.shape, arr.dtype)
        zoom = dw.getZoom()
        # fuzz = 1.0 for full resolution; smaller fuzz values
        # give less resolution
        fuzz = .75
        iscale = 1
        for izoom in range(7):
            lzoom = 1./iscale
            if lzoom < 2*zoom*fuzz:
                break
            iscale *= 2

        # 1/zoom, scale (assuming fuzz = 1.0)
        # 0. - 2. 1
        # 2. - 4. 2
        # 4. - 8. 4
        # 8. - 16. 8
        # 16. - 32. 16
        # print("zoom", zoom, iscale)
        dv = 128*iscale
        zoom_level = izoom
        # look for xyz values where alpha is not zero
        nzarr = arr[arr[:,:,3] > 0][:,:3] // dv

        if len(nzarr) == 0:
            return -1, None, farr

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
        # print("larr", zoom_level, len(larr))

        timera.time("process image")

        return zoom_level, larr, farr

    def stxyXform(self):
        dw = self.gldw

        ww = dw.size().width()
        wh = dw.size().height()
        volume_view = self.volume_view

        zoom = dw.getZoom()
        cij = volume_view.stxytf
        # print("cij", cij)
        mat = np.zeros((4,4), dtype=np.float32)
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

        # self.data_program.bind()

        # xform = self.stxyXform()
        # self.data_program.setUniformValue("xform", xform)

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

        f.glDrawElements(f.GL_TRIANGLES, fvao.trgl_index_size,
                       f.GL_UNSIGNED_INT, None)
        vao.release()
        self.trgl_program.release()

        QOpenGLFramebufferObject.bindDefault()

    # TODO: drawData should call self.atlas.displayData
    # with arguments data_fbo, fvao, stxy_xform.
    # data_program should belong to self.atlas, not self.
    def drawData(self):
        if self.atlas is None:
            return
        stxy_xform = self.stxyXform()
        self.atlas.displayBlocks(self.data_fbo, self.active_vao, stxy_xform)

    def drawTrgls(self, fbo, program):
        f = self.gl
        dw = self.gldw
        fvao = self.active_vao

        fbo.bind()

        # Be sure to clear with alpha = 0
        # so that the slice view isn't blocked!
        f.glClearColor(0.,0.,0.,0.)
        f.glClear(f.GL_COLOR_BUFFER_BIT)
        f.glViewport(0, 0, fbo.width(), fbo.height())

        program.bind()

        xform = self.stxyXform()
        program.setUniformValue("xform", xform)
        # print("fbo w h", fbo.width(), fbo.height())
        # print("xform", xform)
        # print("tis", fvao.trgl_index_size)

        f.glDrawElements(f.GL_TRIANGLES, fvao.trgl_index_size,
                       f.GL_UNSIGNED_INT, None)
        program.release()

        QOpenGLFramebufferObject.bindDefault()
        self.setDefaultViewport()

    def drawXyz(self, fbo):
        f = self.gl
        dw = self.gldw
        fvao = self.active_vao

        fbo.bind()

        # Be sure to clear with alpha = 0
        # so that the slice view isn't blocked!
        f.glClearColor(0.,0.,0.,0.)
        f.glClear(f.GL_COLOR_BUFFER_BIT)
        f.glViewport(0, 0, fbo.width(), fbo.height())

        self.xyz_program.bind()

        xform = self.stxyXform()
        self.xyz_program.setUniformValue("xform", xform)
        # print("fbo w h", fbo.width(), fbo.height())
        # print("xform", xform)
        # print("tis", fvao.trgl_index_size)

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
        print("set data", self.ak, dk, dl)
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
        skip0 = tuple(int_dr[0][i]-pdr[0][i] for i in range(len(int_dr[0])))
        # Difference in max corner:
        skip1 = tuple(pdr[1][i]-int_dr[1][i] for i in range(len(pdr[1])))

        # print(pdr, skip0)
        # print(ar, int_dr, skip0, skip1)
        # print(skip0, skip1)
        # print(dr, int_dr)
        # TODO: copy into atlas texture
        acsz = self.atlas.acsz
        # buf = np.zeros((acsz[2], acsz[1], acsz[0], 4), np.uint16)
        buf = np.zeros((acsz[2], acsz[1], acsz[0]), np.uint16)
        c0 = skip0
        c1 = tuple(acsz[i]-skip1[i] for i in range(len(acsz)))
        data = self.atlas.datas[dl]

        self.atlas.volume_view.volume.setImmediateDataMode(True)
        # buf[c0[2]:c1[2], c0[1]:c1[1], c0[0]:c1[0], :3] = data[int_dr[0][2]:int_dr[1][2], int_dr[0][1]:int_dr[1][1], int_dr[0][0]:int_dr[1][0]][:,:,:,np.newaxis]
        buf[c0[2]:c1[2], c0[1]:c1[1], c0[0]:c1[0]] = data[int_dr[0][2]:int_dr[1][2], int_dr[0][1]:int_dr[1][1], int_dr[0][0]:int_dr[1][0]]
        self.atlas.volume_view.volume.setImmediateDataMode(False)
        # TODO: for testing:
        # buf[c0[2]:c1[2], c0[1]:c1[1], c0[0]:c1[0], :3] = 32000
        # buf[c0[2]:c1[2], c0[1]:c1[1], c0[0]:c1[0], 3] = 65535
        # print("buf", buf.min(), buf.max())
        a = self.ar[0]
        # print(a, acsz)
        # self.atlas.tex3d.setData(a[0], a[1], a[2], acsz[0], acsz[1], acsz[2], QOpenGLTexture.RGBA, QOpenGLTexture.UInt16, buf.tobytes())
        self.atlas.tex3d.setData(a[0], a[1], a[2], acsz[0], acsz[1], acsz[2], QOpenGLTexture.Red, QOpenGLTexture.UInt16, buf.tobytes())

        asz = self.atlas.asz

        xform = QMatrix4x4()
        xform.scale(*(1./asz[i] for i in range(len(asz))))
        xform.translate(*(self.ar[0][i]+self.pad-dr[0][i] for i in range(len(self.ar[0]))))
        xform.scale(*(dsz[i] for i in range(len(dsz))))
        self.xform = xform
        # print("dsz", dsz)
        # print("xform", self.xform)

        self.tmin = tuple((dr[0][i])/dsz[i] for i in range(len(dsz)))
        self.tmax = tuple((dr[1][i])/dsz[i] for i in range(len(dsz)))

        # print("xtmin", self.xform*QVector4D(*self.tmin, 1.))
        # print("xtmax", self.xform*QVector4D(*self.tmax, 1.))

        # if ak[0] == 0 and ak[1] == 0:
        #     print("tm", self.tmin, self.tmax)
        self.atlas.program.bind()
        ind = self.atlas.index(self.ak)
        # print("setting xform", ind)
        self.atlas.program.setUniformValue("xforms[%d]"%ind, self.xform)
        # print("unf loc", ind, self.atlas.program.uniformLocation("xforms[%d]"%ind))
        # print("unf loc", ind, self.atlas.program.uniformLocation("tmins[%d]"%ind))
        # print("unf loc", ind, self.atlas.program.uniformLocation("tmaxs[%d]"%ind))
        # print("setting tmin", ind)
        self.atlas.program.setUniformValue("tmins[%d]"%ind, QVector3D(*self.tmin))
        # print("setting tmax", ind)
        self.atlas.program.setUniformValue("tmaxs[%d]"%ind, QVector3D(*self.tmax))

        self.in_use = True

    @staticmethod
    def k2r(k, csz):
        c = tuple(k[i]*csz[i] for i in range(len(k)))
        r = (c, tuple(c[i]+csz[i] for i in range(len(c))))
        return r

    # padded rectangle
    @staticmethod
    def padRect(rect, pad):
        r = (tuple(rect[0][i]-pad for i in range(len(rect[0]))),
             tuple(rect[1][i]+pad for i in range(len(rect[1]))))
        return r

    # adapted from https://stackoverflow.com/questions/25068538/intersection-and-difference-of-two-rectangles/25068722#25068722
    @staticmethod
    def rectIntersection(ra, rb):
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

atlas_data_code = {
    "name": "atlas_data",

    "vertex": '''
      #version 410 core

      uniform mat4 stxy_xform;
      uniform mat4 xyz_xform;
      layout(location=3) in vec3 xyz;
      layout(location=4) in vec2 stxy;
      out vec4 fxyz;
      void main() {
        gl_Position = stxy_xform*vec4(stxy, 0., 1.);
        fxyz = xyz_xform*vec4(xyz, 1.);
      }
    ''',

    "fragment_template": '''
      #version 410 core

      uniform sampler3D atlas;
      uniform mat4 xforms[{max_nchunks}];
      uniform vec3 tmins[{max_nchunks}];
      uniform vec3 tmaxs[{max_nchunks}];
      uniform int chart_ids[{max_nchunks}];
      uniform int ncharts;

      in vec4 fxyz;
      out vec4 fColor;

      void main() {{
        fColor = vec4(.5,0.,.5,1.);
        for (int i=0; i<ncharts; i++) {{
            int id = chart_ids[i];
            vec3 tmin = tmins[id];
            vec3 tmax = tmaxs[id];
            if (fxyz.x >= tmin.x && fxyz.x <= tmax.x &&
             fxyz.y >= tmin.y && fxyz.y <= tmax.y &&
             fxyz.z >= tmin.z && fxyz.z <= tmax.z) {{
              vec3 txyz = (xforms[id]*fxyz).xyz;
              fColor = texture(atlas, txyz);
              fColor.g = fColor.r;
              fColor.b = fColor.r;
              fColor.a = 1.;
            }}
        }}

      }}
    ''',
}

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
    def __init__(self, volume_view, gl, tex3dsz=(2048,1500,150), dcsz=(128,128,128)):
        # self.created = Utils.timestamp()
        self.gl = gl
        pad = 1
        self.pad = pad
        self.volume_view = volume_view
        self.dcsz = dcsz
        acsz = tuple(dcsz[i]+2*pad for i in range(len(dcsz)))
        self.acsz = acsz
        vol = volume_view.volume
        vdir = volume_view.direction
        is_zarr = vol.is_zarr

        datas = []
        if not is_zarr:
            data = vol.trdatas[vdir]
            datas.append(data)
        else:
            for level in vol.levels:
                data = level.trdatas[vdir]
                datas.append(data)
        dsz = []
        for data in datas:
            shape = data.shape
            dsz.append(tuple(shape[::-1]))
        # print("dsz")
        # print(dsz)
        self.datas = datas
        self.dsz = dsz
        # number of data chunks in each direction
        ksz = []
        for l in range(len(dsz)):
            lksz = tuple(self.ke(dsz[l][i],dcsz[i]) for i in range(len(dcsz)))
            ksz.append(lksz)
        # print("ksz")
        # print(ksz)
        self.ksz = ksz
        # number of atlas chunks in each direction
        aksz = tuple(tex3dsz[i]//acsz[i] for i in range(len(acsz)))
        # size of atlas in each direction
        self.asz = tuple(aksz[i]*acsz[i] for i in range(len(acsz)))
        self.aksz = aksz
        # print("asz", self.asz)

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

        max_nchunks = aksz[0]*aksz[1]*aksz[2]
        print("max_nchunks", max_nchunks)
        self.max_nchunks = max_nchunks
        atlas_data_code["fragment"] = atlas_data_code["fragment_template"].format(max_nchunks = max_nchunks)
        self.program = GLDataWindowChild.buildProgram(atlas_data_code)
        self.program.bind()
        xyz_xform = self.xyzXform(dsz[0])
        self.program.setUniformValue("xyz_xform", xyz_xform)
        pid = self.program.programId()
        print("program id", pid)

        # allocate 3D texture 
        tex3d = QOpenGLTexture(QOpenGLTexture.Target3D)
        tex3d.setWrapMode(QOpenGLTexture.ClampToBorder)
        tex3d.setAutoMipMapGenerationEnabled(False)
        tex3d.setMagnificationFilter(QOpenGLTexture.Linear)
        tex3d.setMinificationFilter(QOpenGLTexture.Linear)
        # width, height, depth
        tex3d.setSize(*self.asz)
        # see https://stackoverflow.com/questions/23533749/difference-between-gl-r16-and-gl-r16ui
        tex3d.setFormat(QOpenGLTexture.R16_UNorm)
        # tex3d.setFormat(QOpenGLTexture.RGBA16_UNorm)
        tex3d.allocateStorage()
        self.tex3d = tex3d
        aunit = 4
        gl.glActiveTexture(gl.GL_TEXTURE0+aunit)
        tex3d.bind()
        self.program.setUniformValue("atlas", aunit)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        tex3d.release()


    def xyzXform(self, data_size):
        mat = np.zeros((4,4), dtype=np.float32)
        mat[0][0] = 1./data_size[0]
        mat[1][1] = 1./data_size[1]
        mat[2][2] = 1./data_size[2]
        mat[3][3] = 1.
        xform = QMatrix4x4(mat.flatten().tolist())
        return xform

    def key(self, dk, dl):
        return (dl, dk[2], dk[1], dk[0])

    # given an atlas chunk location, return a key
    def index(self, ak):
        aksz = self.aksz
        # print("aksz", aksz)
        # print("ak", ak)
        return (ak[2]*aksz[1] + ak[1])*aksz[0] + ak[0]

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
                # Get the first Chunk in the OrderedDict: 
                _, chunk = self.chunks.popitem(last=False)
                chunk.setData(block, zoom_level)
                # print("set data", chunk.dk, chunk.dl)
                self.chunks[key] = chunk
            else: # If the data is already in the Atlas
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
            
    # displayBlocks is in a separate operation
    # than addBlocks, because addBlocks needs to be called later
    # than displayBlocks, to prevent GPU round trips
    def displayBlocks(self, data_fbo, fvao, stxy_xform):
        # self.addBlocks(zoom_level, blocks)
        # dw = self.gldw
        gl = self.gl

        data_fbo.bind()

        # Be sure to clear with alpha = 0
        # so that the slice view isn't blocked!
        gl.glClearColor(0.,0.,0.,0.)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        self.program.bind()

        self.program.setUniformValue("stxy_xform", stxy_xform)

        nchunks = 0
        for key,chunk in reversed(self.chunks.items()):
            if not chunk.in_use:
                break
            ak = chunk.ak
            ind = self.index(ak)
            self.program.setUniformValue("chart_ids[%d]"%nchunks, ind)
            # print(nchunks, ind)
            nchunks += 1
        print("nchunks", nchunks)
        self.program.setUniformValue("ncharts", nchunks)

        gl.glDrawElements(gl.GL_TRIANGLES, fvao.trgl_index_size,
                       gl.GL_UNSIGNED_INT, None)
        self.program.release()

        QOpenGLFramebufferObject.bindDefault()

