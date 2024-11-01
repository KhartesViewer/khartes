from PyQt5.QtGui import (
        QColor,
        QImage,
        QMatrix4x4,
        QOffscreenSurface,
        QOpenGLContext,
        QPixmap,
        QSurfaceFormat,
        QTransform,
        QVector2D,
        QVector3D,
        QVector4D,
        )

# from PySide6.QtOpenGL import (
from PyQt5.QtGui import (
        QOpenGLVertexArrayObject,
        QOpenGLBuffer,
        QOpenGLDebugLogger,
        QOpenGLDebugMessage,
        QOpenGLFramebufferObject,
        QOpenGLFramebufferObjectFormat,
        QOpenGLShader,
        QOpenGLShaderProgram,
        QOpenGLTexture,
        )

from PyQt5.QtWidgets import (
        QApplication, 
        QGridLayout,
        QHBoxLayout,
        QMainWindow,
        QWidget,
        )

# from PySide6.QtOpenGLWidgets import (
from PyQt5.QtWidgets import (
        QOpenGLWidget,
        )

from PyQt5.QtCore import (
        QFileInfo,
        QPointF,
        QSize,
        QTimer,
        )

import time
from collections import OrderedDict
import enum
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue
import numpy as np
import cv2
# OpenGL error checking is set/unset in gl_data_windows.py,
# since that is loaded first
from OpenGL import GL as pygl
# from shiboken6 import VoidPtr
import ctypes
def VoidPtr(i):
    return ctypes.c_void_p(i)

from utils import Utils
from data_window import DataWindow
from project import ProjectView
from gl_data_window import (
        GLDataWindowChild, 
        ColormapTexture,
        fragment_trgls_code, 
        common_offset_code, 
        # UniBuf
        )


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

    def ctrlArrowKey(self, direction):
        # print("cak", direction)
        if direction[1] == 0:
            return
        offset = self.window.getNormalOffsetOnCurrentFragment()
        if offset is None:
            return
        offset += direction[1]
        self.window.setNormalOffsetOnCurrentFragment(offset)
        # print("offset", self.window.getNormalOffsetOnCurrentFragment())

    def setIjkTf(self, tf):
        # ij = self.tijkToIj(tf)
        stxy = self.ijkToStxy(tf)
        # print("tf, xy, stxy", tf, xy, stxy)
        if stxy is not None:
            self.volume_view.setStxyTf(stxy)
        self.volume_view.setIjkTf(tf)

    def addPoint(self, stxy):
        # print("glsw add point", tijk)
        # stxy = self.ijkToStxy(tijk)
        # print("stxy", stxy)
        tijk = self.stxyToTijk(stxy, True)
        self.window.addPointToCurrentFragment(tijk, stxy)

    def computeTfStartPoint(self):
        stxy = self.volume_view.stxytf
        if stxy is None:
            return None
        return (stxy[0], stxy[1], 0)
        # print("tfs", len(tfs), tfs)
        # return tfs

    def setTf(self, tf):
        tf = tf[:2]
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

    def ijkToStxy(self, ijk):
        xyz_arr = self.glw.xyz_arr
        if xyz_arr is None:
            return None
        dxyz = (xyz_arr[:,:,:3] - ijk).astype(np.float32)
        dxyz *= dxyz
        dsq = dxyz.sum(axis=2)
        dsq[xyz_arr[:,:,3] == 0] = 2**30
        minindex = np.unravel_index(dsq.argmin(), dsq.shape)
        # print("shapes", xyz_arr.shape, dsq.shape)
        # print("minindex", minindex, xyz_arr[*minindex,:3], ijk)
        iy,ix = minindex
        zoom = self.getZoom()
        ratio = self.screen().devicePixelRatio()
        x = ix/ratio
        y = iy/ratio
        w = self.width()
        h = self.height()
        hw = w/2
        hh = h/2
        dx = (x-hw)/zoom
        dy = (y-hh)/zoom
        ostxy = self.volume_view.stxytf
        nstxy = (ostxy[0]+dx, ostxy[1]+dy)
        return nstxy

    # given mouse position xy, return stxy position
    def xyToT(self, xy):
        x, y = xy
        ww, wh = self.width(), self.height()
        wcx, wcy = ww//2, wh//2
        dx, dy = x-wcx, y-wcy
        cij = self.volume_view.stxytf
        if cij is None:
            return None
        # print("tf", tijk)
        zoom = self.getZoom()
        # i = cij[0] + int(dx/zoom)
        # j = cij[1] + int(dy/zoom)
        i = cij[0] + dx/zoom
        j = cij[1] + dy/zoom
        return (i, j)

    def stxyToOglPixel(self, ij):
        zoom = self.getZoom()
        cij = self.volume_view.stxytf
        if cij is None:
            return None
        ci = cij[0]
        cj = cij[1]
        ww, wh = self.width(), self.height()
        wcx, wcy = ww//2, wh//2

        # note that the values are floats:
        x, y = (wcx+zoom*(ij[0]-ci), wcy+zoom*(ij[1]-cj))

        ratio = self.screen().devicePixelRatio()
        ix = round(x*ratio)
        iy = round(y*ratio)
        return (ix, iy)

    def stxyToTijk(self, ij, return_none_if_outside=False):
        if return_none_if_outside:
            outside_value = None
        else:
            outside_value = self.volume_view.ijktf

        if ij is None:
            return None

        xyz_arr = self.glw.xyz_arr
        if xyz_arr is None:
            # print("xyz_arr is None; returning vv.ijktf")
            # return None
            # print("None a")
            return outside_value

        ixy = self.stxyToOglPixel(ij)
        if ixy is None:
            # print("None b")
            return outside_value

        ix, iy = ixy

        if iy < 0 or iy >= xyz_arr.shape[0] or ix < 0 or ix >= xyz_arr.shape[1]:
            # print("error", x, y, xyz_arr.shape)
            # return self.volume_view.ijktf
            # print("None c")
            return outside_value

        xyza = xyz_arr[iy, ix]
        if xyza[3] == 0:
            # print("None d")
            return outside_value

        iind = self.iIndex
        jind = self.jIndex
        kind = self.kIndex

        i = xyza[iind]
        j = xyza[jind]
        k = xyza[kind]
        return (i,j,k)

    def xyToTijk(self, xy, return_none_if_outside=False):
        ij = self.xyToT(xy)
        return self.stxyToTijk(ij, return_none_if_outside)

    def ijToTijk(self, ij):
        return (ij[0], ij[1], 0)

    def getTrackingCursorXy(self):
        stxyz = self.window.cursor_stxyz
        # print("gtxy", stxyz)
        if stxyz is None:
            return None
        xy = self.stxyToWindowXy(stxyz[:2])
        # print("gtxy", xy)
        return xy

    def getTrackingCursorHeight(self):
        stxyz = self.window.cursor_stxyz
        if stxyz is None:
            return None
        # print("gth", stxyz[2])
        return stxyz[2]

    def stxyToWindowXy(self, ij):
        zoom = self.getZoom()
        cij = self.volume_view.stxytf
        if cij is None:
            return None
        ci = cij[0]
        cj = cij[1]
        ww, wh = self.width(), self.height()
        wcx, wcy = ww//2, wh//2

        # note that the values are floats:
        xy = (wcx+zoom*(ij[0]-ci), wcy+zoom*(ij[1]-cj))
        return xy

    def setNearbyNodeIjk(self, ijk, update_xyz, update_st):
        # print("snnijk", ijk)
        # NOTE that the two flags are switched below, because the
        # flags have opposite values when passed to
        # this window.
        update_xyz, update_st = update_st, update_xyz
        stxys = self.cur_frag_pts_stxy
        xyijks = self.cur_frag_pts_xyijk
        nearbyNode = self.localNearbyNodeIndex
        if nearbyNode < 0 or stxys.shape[0] == 0:
            return
        fv = self.glw.active_vao.fragment_view
        vv = fv.cur_volume_view
        stxy = stxys[nearbyNode, 0:2]
        oijk = xyijks[nearbyNode, 2:5]
        gijk = vv.transposedIjkToGlobalPosition(ijk)
        ogijk = vv.transposedIjkToGlobalPosition(oijk)

        # shift in transposed ijk coordinates:
        dijk = [ijk[i]-oijk[i] for i in range(3)]

        # shift in global coordinates
        dgijk = [gijk[i]-ogijk[i] for i in range(3)]

        # print(dijk, dgijk)
        # Use convention that ^ is outwards
        index = int(stxys[nearbyNode, 2])

        axes = fv.localStAxes(index)

        if axes is None:
            print("GLSurfaceWindow.setNearbyNodeIjk: could not compute axes")
            return

        # Use transposed ijk coordinates to calculate
        # the shift, since dijk represents the original user
        # input in the map-view plane (up, down, right, left, in, out)
        shift = axes@dijk

        # apply the shift to the global coordinates
        ngijk = ogijk + shift

        # And then transform the shifted result back to ijk coordinates:
        nijk = vv.globalPositionToTransposedIjk(ngijk)
        # print(ngijk, nijk)

        # This eventually ends up calling movePoint(), which
        # is defined in both FragmentView and TrglFragmentView.
        super(GLSurfaceWindow, self).setNearbyNodeIjk(nijk, update_xyz, update_st)

    def stxyWindowBounds(self):
        stxy = self.volume_view.stxytf
        if stxy is None:
            return ((0.,0.), (-1.,-1.))
        zoom = self.getZoom()
        ww, wh = self.width(), self.height()
        hw, hh = ww/2, wh/2
        dx,dy = hw/zoom, hh/zoom
        return ((stxy[0]-dx,stxy[1]-dy),(stxy[0]+dx,stxy[1]+dy))

    # overrides version in DataWindow
    def setMapImage(self, fv):
        print("GLSW setMapImage")
        if fv is None:
            return
        fv.map_image = None
        fv.map_corners = None
        if fv != self.glw.active_fragment:
            return
        if self.volume_view.stxytf is None:
            return
        fbo = self.glw.data_fbo
        if fbo is None:
            return
        # This is a QImage
        image = fbo.toImage()
        imarr = self.glw.npArrayFromQImage(image)
        # cv2.imwrite("test.png", imarr)
        fv.map_image = imarr

        fv.map_corners = self.stxyWindowBounds()
        # print("corners", fv.map_corners)

    def drawSlice(self):
        # print("gsw drawSlice")
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
      // uniform int use_colormap_sampler = 0;
      // NOTE: base_alpha is not currently used
      uniform float base_alpha;
      uniform int base_colormap_sampler_size = 0;
      uniform sampler2D base_colormap_sampler;
      uniform int base_uses_overlay_colormap = 0;

      uniform sampler2D overlay_samplers[2];
      uniform float overlay_alphas[2];
      uniform int overlay_colormap_sampler_sizes[2];
      uniform sampler2D overlay_colormap_samplers[2];
      uniform int overlay_uses_overlay_colormaps[2];
      
      uniform sampler2D underlay_sampler;
      uniform sampler2D top_label_sampler;
      uniform sampler2D trgls_sampler;
      in vec2 ftxt;
      out vec4 fColor;

      void colormapper(in vec4 pixel, in int uoc, in int css, in sampler2D colormap, out vec4 result) {
        if (uoc > 0) {
            float fr = pixel[0];
            uint ir = uint(fr*65535.);
            if ((ir & uint(32768)) == 0) {
                // pixel *= 2.;
                float gray = pixel[0]*2.;
                result = vec4(gray, gray, gray, 1.);
            } else {
                uint ob = ir & uint(31);
                // ob = ir & uint(31);
                ir >>= 5;
                uint og = ir & uint(31);
                ir >>= 5;
                uint or = ir & uint(31);
                result[0] = float(or) / 31.;
                result[1] = float(og) / 31.;
                result[2] = float(ob) / 31.;
                result[3] = 1.;
            }
        } else if (css > 0) {
            float fr = pixel[0];
            float sz = float(css);
            // adjust to allow for peculiarities of texture coordinates
            fr = .5/sz + fr*(sz-1)/sz;
            vec2 ftx = vec2(fr, .5);
            result = texture(colormap, ftx);
        } else {
            float fr = pixel[0];
            result = vec4(fr, fr, fr, 1.);
        }
      }

      void main()
      {
        float alpha;
        fColor = texture(base_sampler, ftxt);
        /*
        if (base_uses_overlay_colormap > 0) {
            float fr = fColor[0];
            uint ir = uint(fr*65535.);
            if ((ir & uint(32768)) == 0) {
                // fColor *= 2.;
                float gray = fColor[0]*2.;
                fColor = vec4(gray, gray, gray, 1.);
            } else {
                uint ob = ir & uint(31);
                // ob = ir & 31;
                ir >>= 5;
                uint og = ir & uint(31);
                ir >>= 5;
                uint or = ir & uint(31);
                fColor[0] = float(or) / 31.;
                fColor[1] = float(og) / 31.;
                fColor[2] = float(ob) / 31.;
                fColor[3] = 1.;
            }

        } else if (base_colormap_sampler_size > 0) {
            float fr = fColor[0];
            float sz = float(base_colormap_sampler_size);
            fr = .5/sz + fr*(sz-1)/sz;
            vec2 ftx = vec2(fr, .5);
            fColor = texture(base_colormap_sampler, ftx);
        }
        */

        vec4 result;
        colormapper(fColor, base_uses_overlay_colormap, base_colormap_sampler_size, base_colormap_sampler, result);
        fColor = result;

        for (int i=0; i<2; i++) {
            float oalpha = overlay_alphas[i];
            if (oalpha == 0.) continue;
            vec4 oColor = texture(overlay_samplers[i], ftxt);
            vec4 result;
            colormapper(oColor, overlay_uses_overlay_colormaps[i], overlay_colormap_sampler_sizes[i], overlay_colormap_samplers[i], result);
            oalpha *= result[3];
            fColor = (1.-oalpha)*fColor + oalpha*result;
        }
        
        vec4 uColor = texture(underlay_sampler, ftxt);
        alpha = uColor.a;
        fColor = (1.-alpha)*fColor + alpha*uColor;

        vec4 frColor = texture(trgls_sampler, ftxt);
        alpha = frColor.a;
        // alpha = 0.;
        fColor = (1.-alpha)*fColor + alpha*frColor;

        vec4 oColor = texture(top_label_sampler, ftxt);
        alpha = oColor.a;
        fColor = (1.-alpha)*fColor + alpha*oColor;
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
      layout(location=5) in vec3 normal;
      uniform float normal_offset;
      out vec3 fxyz;
      void main() {
        gl_Position = xform*vec4(stxy, 0., 1.);
        fxyz = xyz + normal_offset*normal;
      }
    ''',

    "fragment": '''
      #version 410 core

      in vec3 fxyz;
      out vec4 fColor;

      void main() {
          // fColor = vec4(fxyz/65535., 1.);
          fColor = vec4(fxyz, 1.);
      }

    ''',
}

trgls_code = {
    "name": "trgls",

    "vertex": '''
      #version 410 core

      uniform mat4 xform;
      // layout(location=3) in vec3 xyz;
      layout(location=4) in vec2 stxy;
      // out vec3 fxyz;
      void main() {
        gl_Position = xform*vec4(stxy, 0., 1.);
        // fxyz = xyz;
      }
    ''',

    "geometry_template": '''
      #version 410 core

      uniform float thickness;
      uniform vec2 window_size;

      layout(triangles) in;
      // 42 = 14*3
      layout(triangle_strip, max_vertices=42) out;

      %s

      void draw_line(vec4 pcs[2]);

      void main()
      {
        vec2 xys[3];
        bool xslo = true; // true if all xs are less than -limit
        bool xshi = true; // true if all xs are greater than limit
        bool yslo = true; // true if all ys are less than -limit
        bool yshi = true; // true if all ys are greater than limit
        float limit = 1.1;
        for (int i=0; i<3; i++) {
          vec2 xy = gl_in[i].gl_Position.xy;
          xys[i] = xy;
          if (xy.x > -limit) xslo = false;
          if (xy.x < limit) xshi = false;
          if (xy.y > -limit) yslo = false;
          if (xy.y < limit) yshi = false;
        }
        if (xslo || xshi || yslo || yshi) return;
        /*
        for (int i=0; i<4; i++) {
          int ii = i%%3;
          gl_Position = vec4(xys[ii], 0., 1.);
          EmitVertex();
          gl_Position = vec4(xys[ii], 0., 1.);
          EmitVertex();
        }
        */
        /*
        for (int i=0; i<3; i++) {
          gl_Position = vec4(xys[i], 0., 1.);
          EmitVertex();
        }
        */
        for (int i=0; i<3; i++) {
          int ip1 = (i+1)%%3;
          vec4 pcs[2];
          pcs[0] = vec4(xys[i], 0., 1.);
          pcs[1] = vec4(xys[ip1], 0., 1.);
          draw_line(pcs);
        }
      }

      void draw_line(vec4 pcs[2]) {
        int vcount = 4;
        if (thickness < 5) {
          vcount = 4;
        } else {
           vcount = 10;
        }

        vec2 tan = (pcs[1]-pcs[0]).xy;
        if (tan.x == 0 && tan.y == 0) {
          tan.x = 1.;
          tan.y = 0.;
        }
        tan = normalize(tan);
        vec2 norm = vec2(-tan.y, tan.x);
        vec2 factor = vec2(1./window_size.x, 1./window_size.y);
        vec4 offsets[9];
        for (int i=0; i<9; i++) {
          // trig contains cosine and sine of angle i*45 degrees
          vec2 trig = trig_table[i];
          vec2 raw_offset = -trig.x*tan + trig.y*norm;
          vec4 scaled_offset = vec4(factor*raw_offset, 0., 0.);
          offsets[i] = scaled_offset;
        }
        ivec2 vs[10];
        if (vcount == 10) {
          vs = v10;
        } else if (vcount == 4) {
          vs = v4;
        }

        for (int i=0; i<vcount; i++) {
          ivec2 iv = vs[i];
          gl_Position = pcs[iv.x] + thickness*offsets[iv.y];
          EmitVertex();
        }
        EndPrimitive();
      }

    ''',

    "fragment": '''
      #version 410 core

      // in vec3 fxyz;
      uniform vec4 frag_color;
      out vec4 fColor;

      void main() {
          // fColor = vec4(fxyz/65535., 1.);
          // fColor = vec4(.2,.8,.2,.8);
          fColor = frag_color;
      }

    ''',
}

trgl_pts_code = {
    "name": "trgl_pts",

    "vertex": '''
      #version 410 core

      uniform vec4 node_color;
      uniform vec4 highlight_node_color;
      uniform int nearby_node_id;
      out vec4 color;
      uniform mat4 xform;
      layout(location=4) in vec2 stxy;
      void main() {
        if (gl_VertexID == nearby_node_id) {
          color = highlight_node_color;
        } else {
          color = node_color;
        }
        gl_Position = xform*vec4(stxy, 0.0, 1.0);
      }

    ''',
    "fragment": '''
      #version 410 core

      in vec4 color;
      out vec4 fColor;

      void main()
      {
        fColor = color;
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
        self.normal_location = 5
        self.message_prefix = "sw"
        # Cache these so we can recalculate the atlas 
        # whenever volume_view or volume_view.direction change
        self.volume_view =  None
        self.volume_view_direction = -1
        self.active_fragment = None
        self.atlas = None
        self.overlay_atlases = ProjectView.overlay_count*[None]
        self.active_vao = None
        self.data_fbo = None
        self.xyz_fbo = None
        self.xyz_pbo = None
        self.xyz_arr = None
        self.trgls_fbo = None
        # self.atlas_chunk_size = 254
        self.atlas_chunk_size = 126
        # self.atlas_chunk_size = 62

    def localInitializeGL(self):
        f = self.gl
        # Color when no project is loaded
        f.glClearColor(.6,.3,.3,1.)
        self.buildPrograms()
        self.buildSliceVao()
        # self.printInfo()

    def setDefaultViewport(self):
        f = self.gl
        f.glViewport(0, 0, self.vp_width, self.vp_height)
    
    def resizeGL(self, width, height):
        f = self.gl

        # See https://doc.qt.io/qt-6/highdpi.html for why
        # this is needed when working with OpenGL.
        # I would prefer to set the size based on the size of
        # the default framebuffer (or viewport), but because of 
        # the PySide6 bug mentioned above, this does not seem
        # to be possible.
        ratio = self.screen().devicePixelRatio()
        width = int(ratio*width)
        height = int(ratio*height)
        
        self.vp_width = width
        self.vp_height = height
        vp_size = QSize(width, height)
        # print("resizeGL (surface)", width, height, vp_size)

        # fbo where xyz positions are drawn; this information is used
        # to determine which data chunks to load.
        # (at the moment) in the CPU rather than the GPU.
        # based on https://stackoverflow.com/questions/59338015/minimal-opengl-offscreen-rendering-using-qt
        vp_size = QSize(width, height)
        fbo_format = QOpenGLFramebufferObjectFormat()
        fbo_format.setAttachment(QOpenGLFramebufferObject.CombinedDepthStencil)
        fbo_format.setInternalTextureFormat(pygl.GL_RGBA32F)
        self.xyz_fbo = QOpenGLFramebufferObject(vp_size, fbo_format)
        self.xyz_fbo.bind()
        draw_buffers = (pygl.GL_COLOR_ATTACHMENT0,)
        f.glDrawBuffers(len(draw_buffers), draw_buffers)
        self.xyz_pbo = QOpenGLBuffer(QOpenGLBuffer.PixelPackBuffer)
        self.xyz_pbo.create()
        self.xyz_pbo.bind()
        pbo_size = width*height*4*4
        self.xyz_pbo.allocate(pbo_size)
        self.xyz_pbo.release()

        # fbo where the data will be drawn
        fbo_format = QOpenGLFramebufferObjectFormat()
        fbo_format.setAttachment(QOpenGLFramebufferObject.CombinedDepthStencil)
        fbo_format.setInternalTextureFormat(pygl.GL_RGBA16)
        self.data_fbo = QOpenGLFramebufferObject(vp_size, fbo_format)
        self.data_fbo.bind()
        draw_buffers = (pygl.GL_COLOR_ATTACHMENT0,)
        f.glDrawBuffers(len(draw_buffers), draw_buffers)

        # fbo where vertices and wireframe triangles will be drawn 
        fbo_format = QOpenGLFramebufferObjectFormat()
        fbo_format.setAttachment(QOpenGLFramebufferObject.CombinedDepthStencil)
        fbo_format.setInternalTextureFormat(pygl.GL_RGBA16)
        self.trgls_fbo = QOpenGLFramebufferObject(vp_size, fbo_format)
        self.trgls_fbo.bind()
        draw_buffers = (pygl.GL_COLOR_ATTACHMENT0,)
        f.glDrawBuffers(len(draw_buffers), draw_buffers)

        QOpenGLFramebufferObject.bindDefault()

        self.setDefaultViewport()
        
    def paintGL(self):
        self.checkAtlases()
        if self.volume_view is None:
            return

        # print("paintGL (surface)")
        f = self.gl
        # This is the color users will see when
        # no fragment is active.
        # If a fragment is active but not in range,
        # this is NOT the color that will be shone.
        # f.glClearColor(.6,.3,.6,1.)
        f.glClearColor(.1,.1,.1,1.)
        f.glClear(pygl.GL_COLOR_BUFFER_BIT)
        self.paintSlice()

    def buildPrograms(self):
        self.xyz_program = self.buildProgram(xyz_code)
        self.slice_program = self.buildProgram(slice_code)
        trgls_code["geometry"] = trgls_code["geometry_template"] % common_offset_code
        self.trgls_program = self.buildProgram(trgls_code)
        self.trgl_pts_program = self.buildProgram(trgl_pts_code)
        self.fragment_trgls_program = self.buildProgram(fragment_trgls_code)

    # Rebuild atlas if volume_view or volume_view.direction
    # changes
    def checkAtlases(self):
        dw = self.gldw
        if dw.volume_view is None:
            self.volume_view = None
            self.volume_view_direction = -1
            self.active_fragment = None
            # TODO: for testing only!
            # self.atlas = None
            # if self.atlas is set to None without
            # calling setVolumeView first, the old volume_view's
            # memory will not be released until after the new volume_view
            # is loaded!
            if self.atlas is not None:
                self.atlas.setVolumeView(None)
            return
        pv = dw.window.project_view
        mfv = None
        if pv is not None:
            mfv = pv.mainActiveFragmentView(unaligned_ok=True)
        if self.active_fragment != mfv:
            self.active_fragment = mfv
        if self.volume_view != dw.volume_view or self.volume_view_direction != self.volume_view.direction :
            self.volume_view = dw.volume_view
            self.volume_view_direction = self.volume_view.direction
            # TODO: for testing!
            self.atlas = None
            if self.atlas is None:
                aw,ah,ad = (2048,2048,400)
                # TODO: for testing
                # ad = 150
                if self.atlas_chunk_size < 65:
                    ad = 70
                # Loop to determine how much GPU memory can
                # be allocated by Atlas.  If initial allocation
                # fails, keep reducing the dimensions until
                # it fits into memory.
                while True:
                    print("creating atlas with dimensions",aw,ah,ad)
                    self.atlas = Atlas(self.volume_view, self.gl, self.logger, tex3dsz=(aw,ah,ad), chunk_size=self.atlas_chunk_size)
                    if self.atlas.valid:
                        break
                    aw = (aw*3)//4
            else:
                self.atlas.setVolumeView(self.volume_view)


    def drawUnderlays(self, data):
        dw = self.gldw
        volume_view = dw.volume_view

        ww = dw.size().width()
        wh = dw.size().height()
        opacity = dw.getDrawOpacity("overlay")
        bw = dw.getDrawWidth("borders")
        if bw > 0:
            bwh = (bw-1)//2
            axis_color = dw.axisColor(dw.axis)
            alpha = 1.
            if dw.getDrawApplyOpacity("borders"):
                alpha = opacity
            alpha16 = int(alpha*65535)
            axis_color[3] = alpha16
            cv2.rectangle(data, (bwh,bwh), (ww-bwh-1,wh-bwh-1), axis_color, bw)
            cv2.rectangle(data, (0,0), (ww-1,wh-1), (0,0,0,alpha*65535), 1)


    # TODO: is map-view scale bar correct?
    def drawOverlays(self, data, label_text):
        dw = self.gldw
        volume_view = dw.volume_view
        opacity = dw.getDrawOpacity("overlay")

        lw = dw.getDrawWidth("labels")
        alpha = 1.
        if dw.getDrawApplyOpacity("labels"):
            alpha = opacity
        alpha16 = int(alpha*65535)
        dww = dw.window
        
        if lw > 0:
            txt = label_text
            org = (10,20)
            size = 1.
            m = 16000
            gray = (m,m,m,alpha16)
            white = (65535,65535,65535,alpha16)
            
            cv2.putText(data, txt, org, cv2.FONT_HERSHEY_PLAIN, size, gray, 3)
            cv2.putText(data, txt, org, cv2.FONT_HERSHEY_PLAIN, size, white, 1)
            dw.drawScaleBar(data, alpha16)
            dw.drawTrackingCursor(data, alpha16)

    def paintSlice(self):
        timera = Utils.Timer()
        timera.active = False
        timerb = Utils.Timer()
        timerb.active = False
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
        mfv = self.active_fragment
        if mfv is None:
            # print("No fragment visible")
            return

        if self.active_vao is None or self.active_vao.fragment_view != mfv:
            self.active_vao = FragmentMapVao(
                    mfv, self.xyz_location, self.stxy_location, self.normal_location, self.gl)

        fvao = self.active_vao

        vao = fvao.getVao()

        # NOTE that drawTrgls, drawTrglXyzs, and drawData all
        # asssume that self.active_vao has been bound; they
        # don't bind it themselves.
        vao.bind()

        # timera.time("xyz")
        self.drawTrglXyzs(self.xyz_fbo, self.xyz_program)
        timera.time("xyz")

        xform = self.stxyXform()
        self.xyz_arr = None
        overlay_label_text = "%s  Offset: %g" % (mfv.fragment.name, mfv.normal_offset)
        if xform is not None:
            # NOTE that getBlocks reads from xyz_fbo, which has
            # just been written to
            larr, self.xyz_arr, zoom_level = self.getBlocks(self.xyz_fbo)
            timera.time("get blocks")
            if len(larr) > 0 and self.atlas is not None:
                if len(larr) >= self.atlas.max_nchunks-1:
                    larr = larr[:self.atlas.max_nchunks-1]
                self.atlas.addBlocks(larr, dw.window.zarrFutureDoneCallback)
                overlay_label_text += "  Zoom Level: %d  Chunks: %d"%(zoom_level, len(larr))
                timera.time("add blocks")

        self.drawAxes(self.trgls_fbo, self.fragment_trgls_program)
        timera.time("axes")
        self.drawTrgls(self.trgls_fbo, self.trgls_program)
        timera.time("trgls")

        # This is part of the addBlocks process, but it has been
        # moved here to give time for chunk PBOs to be loaded
        # from RAM, in the background.  See comments in addBlocks
        # for more details
        self.atlas.loadTexturesFromPbos(dw.window.zarrFutureDoneCallback)
        timera.time("load textures")

        # NOTE that drawData uses the blocks added in addBlocks;
        # xyToTijk uses self.xyz_arr, which is created by getBlocks 
        self.drawData()
        timera.time("data")

        vao.release()

        self.slice_program.bind()
        base_tex = self.data_fbo.texture()
        '''
        bloc = self.slice_program.uniformLocation("base_sampler")
        if bloc < 0:
            print("couldn't get loc for base sampler")
            return
        tunit = 1
        # bunit = 1
        f.glActiveTexture(pygl.GL_TEXTURE0+tunit)
        f.glBindTexture(pygl.GL_TEXTURE_2D, base_tex)
        self.slice_program.setUniformValue(bloc, tunit)

        uoc = 0
        if volume_view.volume.uses_overlay_colormap:
            uoc = 1
        self.slice_program.setUniformValue("base_uses_overlay_colormap", uoc)

        cmtex = self.getColormapTexture(volume_view)
        if cmtex is None:
            self.slice_program.setUniformValue("base_colormap_sampler_size", 0)
        else:
            cloc = self.slice_program.uniformLocation("base_colormap_sampler")
            tunit += 1
            f.glActiveTexture(pygl.GL_TEXTURE0+tunit)
            cmtex.bind()
            self.slice_program.setUniformValue(cloc, tunit)
            # print("using colormap sampler")
            self.slice_program.setUniformValue("base_colormap_sampler_size", cmtex.width())
        '''
        tunit = 1
        tunit = self.setTextureOfSlice(base_tex, volume_view, tunit, "base", "")

        underlay_data = np.zeros((wh,ww,4), dtype=np.uint16)
        self.drawUnderlays(underlay_data)
        underlay_tex = self.texFromData(underlay_data, QImage.Format_RGBA64)
        uloc = self.slice_program.uniformLocation("underlay_sampler")
        if uloc < 0:
            print("couldn't get loc for underlay sampler")
            return
        tunit += 1
        # uunit = 2
        f.glActiveTexture(pygl.GL_TEXTURE0+tunit)
        underlay_tex.bind()
        self.slice_program.setUniformValue(uloc, tunit)

        top_label_data = np.zeros((wh,ww,4), dtype=np.uint16)
        self.drawOverlays(top_label_data, overlay_label_text)
        top_label_tex = self.texFromData(top_label_data, QImage.Format_RGBA64)
        oloc = self.slice_program.uniformLocation("top_label_sampler")
        if oloc < 0:
            print("couldn't get loc for top_label sampler")
            return
        # ounit = 3
        tunit += 1
        f.glActiveTexture(pygl.GL_TEXTURE0+tunit)
        top_label_tex.bind()
        self.slice_program.setUniformValue(oloc, tunit)

        tloc = self.slice_program.uniformLocation("trgls_sampler")
        if tloc < 0:
            print("couldn't get loc for trgls sampler")
            return
        tunit += 1
        f.glActiveTexture(pygl.GL_TEXTURE0+tunit)
        tex_ids = self.trgls_fbo.textures()
        trgls_tex_id = tex_ids[0]
        f.glBindTexture(pygl.GL_TEXTURE_2D, trgls_tex_id)
        self.slice_program.setUniformValue(tloc, tunit)

        f.glActiveTexture(pygl.GL_TEXTURE0)
        self.slice_vao.bind()
        self.slice_program.bind()
        f.glDrawElements(pygl.GL_TRIANGLES, 
                         self.slice_indices.size, pygl.GL_UNSIGNED_INT, VoidPtr(0))
        self.slice_program.release()

        self.slice_vao.release()
        timera.time("combine")

        timerb.time("done")
        # print()

    def printBlocks(self, blocks):
        for block in blocks:
            print(block)

    def blocksToSet(self, blocks):
        bset = set()
        for block in blocks:
            bset.add(tuple(block))
        return bset

    def getDrawnData(self):
        f = self.gl
        fbo = self.data_fbo
        fbo.bind()
        w = fbo.width()



    def getBlocks(self, fbo):
        timera = Utils.Timer()
        timera.active = False
        dw = self.gldw
        f = self.gl

        self.xyz_fbo.bind()
        w = fbo.width()
        h = fbo.height()
        self.xyz_pbo.bind()
        raw_uint8_data = f.glGetBufferSubData(f.GL_PIXEL_PACK_BUFFER, 0, h*w*4*4)
        self.xyz_pbo.release()
        # https://stackoverflow.com/questions/34637222/glgetbuffersubdata-pyopengl-random-result-and-segfault
        farr = raw_uint8_data.view('<f4')
        # print("whf", w, h, farr.shape, farr.dtype)
        farr = farr.reshape((h,w,4))
        farr = farr[::-1, :, :]
        QOpenGLFramebufferObject.bindDefault()
        # print("farr", farr.shape, farr.dtype)
        timera.time("get image")
        # print("im format", im.format())
        # print("farr", farr.shape, farr.dtype)
        # print(farr)
        # df is decimation factor
        df = 4
        arr = farr[::df,::df,:]
        # print(farr.shape, arr.shape)
        timera.time("array from image")
        # print("arr", arr.shape, arr.dtype)
        zoom = dw.getZoom()
        vol = dw.volume_view.volume
        if vol.is_zarr:
            nlevels = len(vol.levels)
        else:
            nlevels = 1
        # TODO: testing only!
        # nlevels = 1

        # fuzz = 1.0 for full resolution; smaller fuzz values
        # give less resolution
        fuzz = .75
        iscale = 1
        for izoom in range(nlevels):
            lzoom = 1./iscale
            if lzoom < 2*zoom*fuzz or izoom == nlevels-1:
                break
            iscale *= 2

        # 1/zoom, scale (assuming fuzz = 1.0)
        # 0. - 2. 1
        # 2. - 4. 2
        # 4. - 8. 4
        # 8. - 16. 8
        # 16. - 32. 16
        # print("zoom", zoom, iscale)
        dv = self.atlas_chunk_size*iscale
        zoom_level = izoom
        # look for xyz values where alpha is not zero
        nzarr = (arr[arr[:,:,3] > 0][:,:3]).astype(np.int32) // dv
        # print("nzarr", nzarr.shape, nzarr.dtype)

        if len(nzarr) == 0:
            # print("zero-length nzarr")
            # print("arr", arr.shape, arr.dtype)
            return [], farr, zoom_level

        nzmin = nzarr.min(axis=0)
        nzmax = nzarr.max(axis=0)
        nzsarr = nzarr-nzmin
        dvarr = np.zeros(nzmax-nzmin+1, dtype=np.uint32)[:,:,:,np.newaxis]
        indices = np.indices(nzmax-nzmin+1).transpose(1,2,3,0)
        dvarr = np.concatenate((dvarr, indices), axis=3)
        dvarr[nzsarr[:,0],nzsarr[:,1], nzsarr[:,2],0] = 1
        larr = dvarr[dvarr[:,:,:,0] == 1][:,1:]+nzmin
        larr = np.concatenate((larr, np.full((len(larr),1), izoom)), axis=1)
        # print("larr shape", larr.shape, larr.dtype)

        cur_larr = larr[:,:3].copy()
        for izoom in range(zoom_level+1, nlevels):
            nxyzs = cur_larr // 2
            cur_larr = np.unique(nxyzs, axis=0)
            # print("cur_larr shape", cur_larr.shape, cur_larr.dtype)
            clarr = np.concatenate((cur_larr, np.full((len(cur_larr),1), izoom)), axis=1)
            larr = np.concatenate((clarr, larr), axis=0)
            # print("new larr shape", izoom, larr.shape, larr.dtype)

        timera.time("process image")

        return larr, farr, zoom_level

    def stxyXform(self):
        dw = self.gldw

        ww = dw.size().width()
        wh = dw.size().height()
        volume_view = self.volume_view

        zoom = dw.getZoom()
        cij = volume_view.stxytf
        if cij is None:
            return None
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

    def drawData(self):
        if self.atlas is None:
            return
        stxy_xform = self.stxyXform()
        # if stxy_xform is None:
        #     return
        self.atlas.displayBlocks(self.data_fbo, self.active_vao, stxy_xform)

    # pts are in form stxy.x, stxy.y, index
    def getPointsInStxyWindow(self, fv, xywindow):
        pts = fv.stpoints
        matches = ((pts > xywindow[0]) & (pts < xywindow[1])).all(axis=1).nonzero()[0]
        mpts = pts[matches]
        # print("m", xywindow, len(pts), matches.shape, mpts.shape)
        # print("m", len(pts), matches.shape, mpts.shape)
        opts = np.concatenate((mpts, matches[:,np.newaxis]), axis=1)
        return opts


    # TODO: should this be in parent widget?
    def stxysToWindowXys(self, ijs):
        dw = self.gldw
        zoom = dw.getZoom()
        cij = self.volume_view.stxytf
        ci = cij[0]
        cj = cij[1]
        ww, wh = self.width(), self.height()
        wcx, wcy = ww//2, wh//2
        cij = np.array(cij)
        wc = np.array((wcx,wcy))
        xys = np.rint(zoom*(ijs-cij)+wc).astype(np.int32)
        return xys

    def drawTrgls(self, fbo, trgls_program):
        # bind program, bind fbo, assume vao is already bound
        f = self.gl
        dw = self.gldw
        fvao = self.active_vao

        fbo.bind()

        # Be sure to clear with alpha = 0
        # so that the slice view isn't blocked!
        f.glClearColor(0.,0.,0.,0.)
        # f.glClear(pygl.GL_COLOR_BUFFER_BIT)
        xform = self.stxyXform()
        if xform is None:
            QOpenGLFramebufferObject.bindDefault()
            return

        f.glViewport(0, 0, fbo.width(), fbo.height())

        opacity = dw.getDrawOpacity("overlay")
        apply_line_opacity = dw.getDrawApplyOpacity("mesh")
        line_alpha = 1.
        if apply_line_opacity:
            line_alpha = opacity
        line_thickness = dw.getDrawWidth("mesh")
        line_thickness = (3*line_thickness)//2
        fv = fvao.fragment_view
        pv = dw.window.project_view

        if fv.visible and line_thickness != 0 and line_alpha != 0:
            trgls_program.bind()
            trgls_program.setUniformValue("xform", xform)

            wsize = QVector2D(fbo.width(), fbo.height())
            trgls_program.setUniformValue("window_size", wsize)

            tloc = self.trgls_program.uniformLocation("thickness")
            f.glUniform1f(tloc, 1.*line_thickness)

            qcolor = fv.fragment.color
            rgba = list(qcolor.getRgbF())
            rgba[3] = line_alpha
            self.trgls_program.setUniformValue("frag_color", *rgba)

            f.glDrawElements(pygl.GL_TRIANGLES, fvao.trgl_index_size,
                       pygl.GL_UNSIGNED_INT, VoidPtr(0))

            trgls_program.release()

        apply_node_opacity = dw.getDrawApplyOpacity("node")
        node_alpha = 1.
        if apply_node_opacity:
            node_alpha = opacity
        default_node_thickness = dw.getDrawWidth("node")
        free_node_thickness = dw.getDrawWidth("free_node")
        node_thickness = default_node_thickness
        if not fv.mesh_visible:
            node_thickness = free_node_thickness
        node_thickness *= 2
        
        dw.cur_frag_pts_xyijk = None
        dw.cur_frag_pts_fv = []
        xyptslist = []
        dw.nearbyNode = -1

        if fv.visible and node_thickness != 0 and node_alpha != 0:
            self.trgl_pts_program.bind()
            self.trgl_pts_program.setUniformValue("xform", xform)
            highlight_node_color = [c/65535 for c in dw.highlightNodeColor]
            highlight_node_color[3] = node_alpha
            self.trgl_pts_program.setUniformValue("highlight_node_color", *highlight_node_color)
            color = dw.nodeColor
            if not fv.active:
                color = dw.inactiveNodeColor
            if not fv.mesh_visible:
                color = fv.fragment.cvcolor
            rgba = [c/65535 for c in color]
            rgba[3] = node_alpha
            self.trgl_pts_program.setUniformValue("node_color", *rgba)

            nearby_node_id = 2**30
            xywindow = dw.stxyWindowBounds()
            # pts are in form stxy.x, stxy.y, index
            pts = self.getPointsInStxyWindow(fv, xywindow)
            xys = self.stxysToWindowXys(pts[:,:2])
            xyzs = fv.vpoints[np.int32(pts[:,2])]
            xypts = np.concatenate((xys, xyzs), axis=1)
            if len(xypts) > 0:
                dw.cur_frag_pts_xyijk = xypts
                dw.cur_frag_pts_stxy = pts
            else:
                dw.cur_frag_pts_xyijk = np.zeros((0,5), dtype=np.float32)
                dw.cur_frag_pts_stxy = np.zeros((0,3), dtype=np.float32)
            dw.cur_frag_pts_fv = [fv]*len(xypts)

            if fv == pv.nearby_node_fv:
                ind = pv.nearby_node_index
                nz = np.nonzero(pts[:,2] == ind)[0]
                if len(nz) > 0:
                    ind = nz[0]
                    self.nearbyNode = ind
                    nearby_node_id = int(pts[ind,2])

            # figure out highlighted node and set nearby_node_id
            nniloc = self.trgl_pts_program.uniformLocation("nearby_node_id")
            self.trgl_pts_program.setUniformValue(nniloc, int(nearby_node_id))
                
            f.glPointSize(node_thickness)
            # print("fvao count", fvao.stxys_count)
            f.glDrawArrays(pygl.GL_POINTS, 0, fvao.stxys_count)
            self.trgl_pts_program.release()

        QOpenGLFramebufferObject.bindDefault()

    def drawAxes(self, fbo, fragment_trgls_program):
        # bind program, bind fbo, assume vao is already bound
        f = self.gl
        dw = self.gldw
        fvao = self.active_vao

        fbo.bind()

        # Be sure to clear with alpha = 0
        # so that the slice view isn't blocked!
        # f.glClearColor(0.,0.,0.,0.)
        f.glClear(pygl.GL_COLOR_BUFFER_BIT)
        stxy_xform = self.stxyXform()
        if stxy_xform is None:
            QOpenGLFramebufferObject.bindDefault()
            return

        f.glViewport(0, 0, fbo.width(), fbo.height())

        opacity = dw.getDrawOpacity("overlay")
        apply_line_opacity = dw.getDrawApplyOpacity("axes")
        line_alpha = 1.
        if apply_line_opacity:
            line_alpha = opacity
        line_thickness = dw.getDrawWidth("axes")
        line_thickness = (3*line_thickness)//2
        # fv = fvao.fragment_view
        pv = dw.window.project_view

        # if fv.visible and line_thickness != 0 and line_alpha != 0:
        if line_thickness != 0 and line_alpha != 0:
            fragment_trgls_program.bind()
            fragment_trgls_program.setUniformValue("stxform", stxy_xform)
            fragment_trgls_program.setUniformValue("flag", 1)

            vv = dw.volume_view
            wsize = QVector2D(fbo.width(), fbo.height())
            fragment_trgls_program.setUniformValue("window_size", wsize)

            tloc = self.fragment_trgls_program.uniformLocation("thickness")
            f.glUniform1f(tloc, 1.*line_thickness)

            for axis in range(3):
                iind, jind = vv.volume.ijIndexesInPlaneOfSlice(axis)
                kind = axis
                # qcolor = fv.fragment.color
                # rgba = list(qcolor.getRgbF())
                irgb = dw.axisColor(axis)
                rgba = (irgb[0]/65535, irgb[1]/65535, irgb[2]/65535, line_alpha)
                # rgba[3] = line_alpha
                # rgba[axis] = 255.
                self.fragment_trgls_program.setUniformValue("gcolor", *rgba)
                mat = np.zeros((4,4), dtype=np.float32)
                # ww = dw.size().width()
                # wh = dw.size().height()
                # print("w h", ww, wh)
                # TODO: 20000 was hardwired in; this should actually
                # be twice the size of the scroll (in pixels) in the i axis
                # and j axis directions?
                ww = 20000.
                wh = 20000.
                zoom = 1.
                wf = zoom/(.5*ww)
                hf = zoom/(.5*wh)
                df = 1/.5
                cijk = vv.ijktf
                mat[0][iind] = wf
                mat[0][3] = -wf*cijk[iind]
                mat[1][jind] = -hf
                mat[1][3] = hf*cijk[jind]
                mat[2][kind] = df
                mat[2][3] = -df*cijk[kind]
                mat[3][3] = 1.
                xyz_xform = QMatrix4x4(mat.flatten().tolist())
                fragment_trgls_program.setUniformValue("xform", xyz_xform)

                f.glDrawElements(pygl.GL_TRIANGLES, fvao.trgl_index_size,
                       pygl.GL_UNSIGNED_INT, VoidPtr(0))

            fragment_trgls_program.release()

    def drawTrglXyzs(self, fbo, program):
        f = self.gl
        dw = self.gldw
        fvao = self.active_vao


        fbo.bind()

        # Be sure to clear with alpha = 0
        # so that the slice view isn't blocked!
        f.glClearColor(0.,0.,0.,0.)
        f.glClear(pygl.GL_COLOR_BUFFER_BIT)
        xform = self.stxyXform()
        if xform is None:
            QOpenGLFramebufferObject.bindDefault()
            return

        f.glViewport(0, 0, fbo.width(), fbo.height())

        program.bind()

        program.setUniformValue("xform", xform)

        f.glDrawElements(pygl.GL_TRIANGLES, fvao.trgl_index_size,
                       pygl.GL_UNSIGNED_INT, VoidPtr(0))
        program.release()

        self.setDefaultViewport()
        self.xyz_pbo.bind()
        w = fbo.width()
        h = fbo.height()
        f.glReadPixels(0, 0, w, h, f.GL_RGBA, f.GL_FLOAT, 0)
        self.xyz_pbo.release()
        QOpenGLFramebufferObject.bindDefault()

# two attribute buffers: xyz, and stxy (st = scaled texture)
class FragmentMapVao:
    def __init__(self, fragment_view, xyz_loc, stxy_loc, normal_loc, gl):
        self.fragment_view = fragment_view
        self.gl = gl
        self.vao = None
        self.vao_modified = ""
        self.is_line = False
        self.xyz_loc = xyz_loc
        self.stxy_loc = stxy_loc
        self.normal_loc = normal_loc
        self.getVao()

    def getVao(self):
        fv = self.fragment_view
        if fv is not None and self.vao_modified > fv.modified and self.vao_modified > fv.fragment.modified and self.vao_modified > fv.local_points_modified:
            # print("returning existing vao")
            return self.vao

        self.vao_modified = Utils.timestamp()
        # print("modifying vao")

        if self.vao is None:
            self.vao = QOpenGLVertexArrayObject()
            self.vao.create()
            # print("creating new vao")

        if fv is None:
            return self.vao

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
                xyzs.shape[1], int(pygl.GL_FLOAT), int(pygl.GL_FALSE), 
                0, VoidPtr(0))
        self.xyz_vbo.release()
        # This needs to be called while the current VAO is bound
        f.glEnableVertexAttribArray(self.xyz_loc)


        self.stxy_vbo = QOpenGLBuffer()
        self.stxy_vbo.create()
        self.stxy_vbo.bind()

        if fv.stpoints is None:
            stxys = np.zeros((0,2))
        else:
            stxys = np.ascontiguousarray(fv.stpoints, dtype=np.float32)
        self.stxys_size = stxys.size
        self.stxys_count = stxys.shape[0]

        nbytes = stxys.size*stxys.itemsize
        self.stxy_vbo.allocate(stxys, nbytes)
        f.glVertexAttribPointer(
                self.stxy_loc,
                stxys.shape[1], int(pygl.GL_FLOAT), int(pygl.GL_FALSE), 
                0, VoidPtr(0))
        self.stxy_vbo.release()
        # This needs to be called while the current VAO is bound
        f.glEnableVertexAttribArray(self.stxy_loc)


        self.normal_vbo = QOpenGLBuffer()
        self.normal_vbo.create()
        self.normal_vbo.bind()

        normals = np.ascontiguousarray(fv.normals, dtype=np.float32)
        self.normals_size = normals.size

        nbytes = normals.size*normals.itemsize
        self.normal_vbo.allocate(normals, nbytes)
        f.glVertexAttribPointer(
                self.normal_loc,
                normals.shape[1], int(pygl.GL_FLOAT), int(pygl.GL_FALSE), 
                0, VoidPtr(0))
        self.normal_vbo.release()
        # This needs to be called while the current VAO is bound
        f.glEnableVertexAttribArray(self.normal_loc)


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

# gl is the OpenGL function holder
# arr is the numpy array
# uniform_index is the location of the uniform block in the shader
# binding_point is the binding point
# To use: modify values in the data member, then call setBuffer().
class UniBuf:
    def __init__(self, gl, arr, binding_point):
        gl = pygl
        self.gl = gl
        self.binding_point = binding_point
        self.data = arr
        self.buffer_id = gl.glGenBuffers(1)
        gl.glBindBufferBase(gl.GL_UNIFORM_BUFFER, self.binding_point, self.buffer_id)
        self.setBuffer()

    def bindToShader(self, shader_id, uniform_index):
        gl = self.gl
        gl.glUniformBlockBinding(shader_id, uniform_index, self.binding_point)

    def setBuffer(self):
        gl = self.gl
        gl.glBindBufferBase(gl.GL_UNIFORM_BUFFER, self.binding_point, self.buffer_id)
        byte_size = self.data.size * self.data.itemsize
        gl.glBufferData(gl.GL_UNIFORM_BUFFER, byte_size, self.data, gl.GL_STATIC_DRAW)
        gl.glBindBuffer(gl.GL_UNIFORM_BUFFER, 0)

    def setSubBuffer(self, cnt):
        gl = self.gl
        # cnt = 0
        if cnt == 0:
            return
        full_size = self.data.size * self.data.itemsize
        cnt_size = abs(cnt)*self.data.shape[1] * self.data.itemsize
        if cnt < 0:
            offset = full_size - cnt_size
            subdata = self.data[offset:]
        else:
            offset = 0
            subdata = self.data[:cnt_size]
        # print(cnt, full_size, cnt_size, offset, subdata.shape)
        # print("about to bind buffer", self.buffer_id)
        gl.glBindBuffer(pygl.GL_UNIFORM_BUFFER, self.buffer_id)
        # print("about to set buffer", self.data.shape, self.data.dtype)
        gl.glBufferSubData(gl.GL_UNIFORM_BUFFER, offset, cnt_size, subdata)
        # print("buffer has been set")
        gl.glBindBuffer(gl.GL_UNIFORM_BUFFER, 0)


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
        self.pbo = None
        # Chunk key (position) in atlas (3 coords)
        self.ak = ak
        # Chunk key (position) in input data (3 coords: x, y, z)

        # atlas chunk size (3 coords, usually 128,128,128)
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

        # in_use: one of the currently-displayed blocks
        self.in_use = False

        # misses: status of reading the data from disk to core.
        # 0 means fully in core.  
        # > 0 means reading has started, but parts are still not read.
        # < 0 means reading has not started.
        self.misses = -1

        self.status = Chunk.Status.UNINITIALIZED

        self.data_bytes = None

    class Status(Enum):
        UNINITIALIZED = enum.auto()
        INITIALIZED = enum.auto()
        LOADING_FROM_DISK = enum.auto()
        PARTIALLY_LOADED_FROM_DISK = enum.auto()
        LOADED_FROM_DISK = enum.auto()
        LOADED_TO_PBO = enum.auto()
        LOADED_TO_TEXTURE = enum.auto()
        IGNORE = enum.auto()

    def initialize(self, dk, dl):
        self.dk = dk
        self.dl = dl
        self.status = Chunk.Status.INITIALIZED
        self.data_bytes = None
        ind = self.atlas.index(self.ak)
        self.atlas.tmin_ubo.data[ind, 3] = False
        self.pbo = None
        self.misses = -1

    def getDataFromDisk(self):
        if self.status not in [Chunk.Status.INITIALIZED, Chunk.Status.PARTIALLY_LOADED_FROM_DISK]:
            # print("returning")
            return
        self.status = Chunk.Status.LOADING_FROM_DISK

        # print("set data", self.ak, dk, dl)
        dk = self.dk
        dl = self.dl
        if dl < 0:
            return False

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
        if int_dr is None:
            self.status = Chunk.Status.IGNORE
            return False
        # TODO:
        int_dr4 = (
                (0, int_dr[0][0], int_dr[0][1], int_dr[0][2]),
                (1, int_dr[1][0], int_dr[1][1], int_dr[1][2]))
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
        acsz = self.atlas.acsz
        c0 = skip0
        c1 = tuple(acsz[i]-skip1[i] for i in range(len(acsz)))
        # adata = self.atlas.datas[dl]
        adata = self.atlas.datas[dl]
        # TODO:
        buf = np.zeros((acsz[2], acsz[1], acsz[0], adata.shape[3]), adata.dtype)

        timera = Utils.Timer()
        timera.active = False

        thread = threading.current_thread()
        # print("thread ident", thread.ident)
        # See the extensive comment in volume_zarr.py, just beforek
        # the KhartesThreadedLRUCache.__getitems__() function,
        # for an explanation of what the next line does.  Its
        # effect is that the line buf[...] = adata[...] below will block
        # until the data required by adata has been loaded from disk.
        thread.immediate_data_mode = True
        # The reshape at the end is needed because under some
        # circumstances (when a fragment goes off the end of the
        # data array) the indexing causes a dimension to be lost.
        # For instance, shape (128, 1, 128) is auto-flattened
        # to shape (128, 128).  For some reason, this dimension loss 
        # occurs with adata but not buf.
        # The reshape restores the lost dimension.
        # buf[c0[2]:c1[2], c0[1]:c1[1], c0[0]:c1[0]] = adata[int_dr[0][2]:int_dr[1][2], int_dr[0][1]:int_dr[1][1], int_dr[0][0]:int_dr[1][0]].reshape([int_dr[1][i]-int_dr[0][i] for i in reversed(range(3))])
        buf[c0[2]:c1[2], c0[1]:c1[1], c0[0]:c1[0]] = adata[int_dr[0][2]:int_dr[1][2], int_dr[0][1]:int_dr[1][1], int_dr[0][0]:int_dr[1][0], :].reshape([int_dr4[1][i]-int_dr4[0][i] for i in reversed(range(4))])
        '''
        # Trying to figure out the dropped-dimension problem...
        try:
            # buf[c0[2]:c1[2], c0[1]:c1[1], c0[0]:c1[0]] = adata[int_dr[0][2]:int_dr[1][2], int_dr[0][1]:int_dr[1][1], int_dr[0][0]:int_dr[1][0]]
            buf[c0[2]:c1[2], c0[1]:c1[1], c0[0]:c1[0]] = adata[int_dr[0][2]:int_dr[1][2], int_dr[0][1]:int_dr[1][1], int_dr[0][0]:int_dr[1][0]].reshape([int_dr[1][i]-int_dr[0][i] for i in reversed(range(3))])
        except:
            print("shape problem", buf.shape, adata.shape)
            print("c0, c1", c0, c1)
            print("dc", [c1[i]-c0[i] for i in range(3)])
            print("int_dr", int_dr)
            print("d int_dr", [int_dr[1][i]-int_dr[0][i] for i in range(3)])
            z1 = buf[c0[2]:c1[2], c0[1]:c1[1], c0[0]:c1[0]] 
            z2 = adata[int_dr[0][2]:int_dr[1][2], int_dr[0][1]:int_dr[1][1], int_dr[0][0]:int_dr[1][0]]
            print("z1, z2", z1.shape, z2.shape)
            y1 = buf[c0[2]:c1[2], c0[1]:c1[1], c0[0]:c1[0]] 
            # y2 = adata[int_dr[0][2]:int_dr[1][2], int_dr[0][1]:int_dr[1][1], int_dr[0][0]:int_dr[1][0]]
            # y2 = adata[range(int_dr[0][2],int_dr[1][2]), range(int_dr[0][1],int_dr[1][1]), range(int_dr[0][0],int_dr[1][0])]
            y2 = adata[int_dr[0][2]:int_dr[1][2], int_dr[0][1]:int_dr[1][1], int_dr[0][0]:int_dr[1][0]].reshape([int_dr[1][i]-int_dr[0][i] for i in range(3)])
            print("y1, y2", y1.shape, y2.shape)
        '''
        # print("from disk", self.dk, self.dl, "*")
        misses = 0

        self.misses = misses

        if misses == 0:
            self.data_bytes = buf.tobytes()
            self.status = Chunk.Status.LOADED_FROM_DISK
        else:
            self.status = Chunk.Status.PARTIALLY_LOADED_FROM_DISK

        ind = self.atlas.index(self.ak)
        self.atlas.tmin_ubo.data[ind, 3] = False

    # usual sequence: copyRamDataToPbo, then copyPboDataToTexmap
    def copyRamDataToPbo(self):
        if self.misses != 0:
            print("do not call this if misses is not 0", self.misses)
            return
        if self.status == Chunk.Status.LOADED_TO_TEXTURE:
            print("do not call this if texture is set")
            return

        # print("to pbo", self.dk, self.dl)

        acsz = self.atlas.acsz
        a = self.ar[0]
        dk = self.dk
        dl = self.dl
        # data chunk size (3 coords, usually 128,128,128)
        dcsz = self.atlas.dcsz 
        # data rectangle
        dr = self.k2r(dk, dcsz)
        dsz = self.atlas.dsz[dl]
        asz = self.atlas.asz

        self.pbo = self.atlas.getPbo()

        pbo_size = len(self.data_bytes)
        self.pbo.bind()
        # print("calling pbo.write", pbo_size, self.pbo.bufferId())
        self.pbo.write(0, self.data_bytes, pbo_size)
        # print("called")
        self.pbo.release()
        self.data_bytes = None

        # Don't set uniforms here, need to wait until
        # tex3d is set

        self.status = Chunk.Status.LOADED_TO_PBO

    # alternate sequence (not currently used):
    # copyRamDataToTexmap directly, without using PBOs
    def copyRamDataToTexmap(self):
        if self.misses != 0:
            print("do not call this if misses is not 0", self.misses)
            return
        if self.status == Chunk.Status.LOADED_TO_TEXTURE:
            print("do not call this if texture is set")
            return

        # print("to texmap", self.dk, self.dl)

        acsz = self.atlas.acsz
        a = self.ar[0]
        dk = self.dk
        dl = self.dl
        # data chunk size (3 coords, usually 128,128,128)
        dcsz = self.atlas.dcsz 
        # data rectangle
        dr = self.k2r(dk, dcsz)
        dsz = self.atlas.dsz[dl]
        asz = self.atlas.asz

        # print("a",a,"acsz",acsz, "db", len(self.data_bytes))

        self.atlas.tex3d.setData(a[0], a[1], a[2], acsz[0], acsz[1], acsz[2], QOpenGLTexture.Red, QOpenGLTexture.UInt16, self.data_bytes)

        # self.texture_status = 2
        self.status = Chunk.Status.LOADED_TO_TEXTURE
        # print("loaded")
        # self.data_bytes = None

        xform = QMatrix4x4()
        xform.scale(*(1./asz[i] for i in range(len(asz))))
        xform.translate(*(self.ar[0][i]+self.pad-dr[0][i] for i in range(len(self.ar[0]))))
        xform.scale(*(dsz[i] for i in range(len(dsz))))
        self.xform = xform

        # self.atlas.program.bind()
        ind = self.atlas.index(self.ak)

        self.tmin = tuple((dr[0][i])/dsz[i] for i in range(len(dsz)))
        self.tmax = tuple((dr[1][i])/dsz[i] for i in range(len(dsz)))
        self.atlas.tmax_ubo.data[ind, :3] = self.tmax
        self.atlas.tmin_ubo.data[ind, :3] = self.tmin
        self.atlas.tmin_ubo.data[ind, 3] = 1.

        xformarr = np.array(xform.transposed().copyDataTo(), dtype=np.float32).reshape(4,4)
        self.atlas.xform_ubo.data[ind, :, :] = xformarr

    def copyPboDataToTexmap(self):
        if self.misses != 0:
            print("do not call this if misses is not 0", self.misses)
            return
        if self.status == Chunk.Status.LOADED_TO_TEXTURE:
            print("do not call this if texture is set")
            return

        # print("to texmap", self.dk, self.dl)

        acsz = self.atlas.acsz
        a = self.ar[0]
        dk = self.dk
        dl = self.dl
        # data chunk size (3 coords, usually 128,128,128)
        dcsz = self.atlas.dcsz 
        # data rectangle
        dr = self.k2r(dk, dcsz)
        dsz = self.atlas.dsz[dl]
        asz = self.atlas.asz

        # print("a",a,"acsz",acsz, "db", len(self.data_bytes))

        self.pbo.bind()

        self.atlas.tex3d.setData(a[0], a[1], a[2], acsz[0], acsz[1], acsz[2], QOpenGLTexture.Red, QOpenGLTexture.UInt16, 0)
        self.pbo.release()

        self.status = Chunk.Status.LOADED_TO_TEXTURE
        self.atlas.releasePbo(self.pbo)
        self.pbo = None
        self.data_bytes = None

        xform = QMatrix4x4()
        xform.scale(*(1./asz[i] for i in range(len(asz))))
        xform.translate(*(self.ar[0][i]+self.pad-dr[0][i] for i in range(len(self.ar[0]))))
        xform.scale(*(dsz[i] for i in range(len(dsz))))
        self.xform = xform

        ind = self.atlas.index(self.ak)

        self.tmin = tuple((dr[0][i])/dsz[i] for i in range(len(dsz)))
        self.tmax = tuple((dr[1][i])/dsz[i] for i in range(len(dsz)))
        self.atlas.tmax_ubo.data[ind, :3] = self.tmax
        self.atlas.tmin_ubo.data[ind, :3] = self.tmin
        self.atlas.tmin_ubo.data[ind, 3] = 1.

        xformarr = np.array(xform.transposed().copyDataTo(), dtype=np.float32).reshape(4,4)
        self.atlas.xform_ubo.data[ind, :, :] = xformarr

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
      uniform float normal_offset;
      layout(location=3) in vec3 xyz;
      layout(location=4) in vec2 stxy;
      layout(location=5) in vec3 normal;
      out vec4 fxyz;
      void main() {
        gl_Position = stxy_xform*vec4(stxy, 0., 1.);
        // fxyz = xyz_xform*vec4(xyz, 1.);
        fxyz = xyz_xform*vec4(xyz+normal_offset*normal, 1.);
      }
    ''',

    # double braces ({{ and }}) because this code is
    # templated, to allow max_nchunks to be varied
    "fragment_template": '''
      #version 410 core

      const int max_nchunks = {max_nchunks};
      // NOTE: On an XPS 9320 running PyQt5 and OpenGL 4.1,
      // the uniform buffers below MUST be in alphabetical
      // order!!
      layout (std140) uniform TMaxs {{
        vec4 tmaxs[max_nchunks];
      }};
      layout (std140) uniform TMins {{
        vec4 tmins[max_nchunks];
      }};
      layout (std140) uniform XForms {{
        mat4 xforms[max_nchunks];
      }};
      layout (std140) uniform ChartIds {{
        int chart_ids[max_nchunks];
      }};
      uniform sampler3D atlas;
      uniform int ncharts;

      in vec4 fxyz;
      out vec4 fColor;

      void main() {{
        // fColor = vec4(.7,.5,.5,1.);
        fColor = vec4(.5,.5,.5,1.);
        for (int i=0; i<ncharts; i++) {{
        // for (int i=ncharts-1; i>=0; i--) {{
            int id = chart_ids[i];
            vec4 tmin = tmins[id];
            vec4 tmax = tmaxs[id];
            if (tmin.w != 0. && fxyz.x >= tmin.x && fxyz.x <= tmax.x &&
             fxyz.y >= tmin.y && fxyz.y <= tmax.y &&
             fxyz.z >= tmin.z && fxyz.z <= tmax.z) {{
              mat4 xform = xforms[id];
              vec3 txyz = (xform*fxyz).xyz;
              fColor = texture(atlas, txyz);
              fColor.g = fColor.r;
              fColor.b = fColor.r;
              // fColor.r = float(id+1)/10.;
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

    class MiniLogger:
        def __init__(self, logger):
            self.message_count = 0
            self.logger = logger
            self.connection = self.logger.messageLogged.connect(self.onLogMessage)
        def onLogMessage(self, msg):
            self.message_count += 1
        def close(self):
            # self.logger.messageLogged.disconnect(self.connection)
            self.logger.disconnect(self.connection)


    def __init__(self, volume_view, gl, logger, tex3dsz=(2048,2048,300), chunk_size=126):
        print("Creating atlas")
        dcsz = (chunk_size, chunk_size, chunk_size)
        self.gl = gl
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.pbo_queue = Queue()
        self.pbo_pool = Queue()
        pad = 1
        self.pad = pad
        self.dcsz = dcsz
        acsz = tuple(dcsz[i]+2*pad for i in range(len(dcsz)))
        self.acsz = acsz
        if chunk_size < 65:
            self.max_textures_set = 10
        else:
            self.max_textures_set = 3
            self.max_textures_set = 8

        # number of atlas chunks in each direction
        aksz = tuple(tex3dsz[i]//acsz[i] for i in range(len(acsz)))
        # size of atlas in each direction
        self.asz = tuple(aksz[i]*acsz[i] for i in range(len(acsz)))
        self.aksz = aksz

        self.chunks = OrderedDict()

        max_nchunks = aksz[0]*aksz[1]*aksz[2]
        print("max_nchunks", max_nchunks)
        self.max_nchunks = max_nchunks
        atlas_data_code["fragment"] = atlas_data_code["fragment_template"].format(max_nchunks = max_nchunks)
        self.program = GLDataWindowChild.buildProgram(atlas_data_code)

        # self.setVolumeView(volume_view)
        
        self.program.bind()
        # for var in ["atlas", "xyz_xform", "tmins", "tmaxs", "TMins", "TMaxs", "XForms", "ZChartIds", "chart_ids", "ncharts"]:
        #     print(var, self.program.uniformLocation(var))
        pid = self.program.programId()
        # print("program id", pid)

        # for var in ["TMaxs", "TMins", "XForms", "ZChartIds"]:
        #    print(var, gl.glGetUniformBlockIndex(pid, var))
        self.tmax_ubo = UniBuf(gl, np.zeros((max_nchunks, 4), dtype=np.float32), 0)
        loc = gl.glGetUniformBlockIndex(pid, "TMaxs")
        self.tmax_ubo.bindToShader(pid, loc)

        self.tmin_ubo = UniBuf(gl, np.zeros((max_nchunks, 4), dtype=np.float32), 1)
        loc = gl.glGetUniformBlockIndex(pid, "TMins")
        self.tmin_ubo.bindToShader(pid, loc)

        self.xform_ubo = UniBuf(gl, np.zeros((max_nchunks, 4, 4), dtype=np.float32), 2)
        loc = gl.glGetUniformBlockIndex(pid, "XForms")
        self.xform_ubo.bindToShader(pid, loc)

        # even though data in this case could be listed as a 1D
        # array of ints, UBO layout rules require that the ints
        # be aligned every 16 bytes.
        self.chart_id_ubo = UniBuf(gl, np.zeros((max_nchunks, 4), dtype=np.int32), 3)
        loc = gl.glGetUniformBlockIndex(pid, "ChartIds")
        self.chart_id_ubo.bindToShader(pid, loc)

        # allocate 3D texture 
        tex3d = QOpenGLTexture(QOpenGLTexture.Target3D)
        tex3d.setWrapMode(QOpenGLTexture.ClampToBorder)
        # Useful for debugging:
        # tex3d.setBorderColor(QColor(100,100,200,255))
        tex3d.setAutoMipMapGenerationEnabled(False)

        '''
        uoc = volume_view.volume.uses_overlay_colormap
        print("uoc", uoc)
        if uoc:
            tex3d.setMagnificationFilter(QOpenGLTexture.Nearest)
            tex3d.setMinificationFilter(QOpenGLTexture.Nearest)
        else:
            tex3d.setMagnificationFilter(QOpenGLTexture.Linear)
            tex3d.setMinificationFilter(QOpenGLTexture.Linear)
        '''
        # width, height, depth
        tex3d.setSize(*self.asz)
        # TODO: set format based on volume_view information
        # see https://stackoverflow.com/questions/23533749/difference-between-gl-r16-and-gl-r16ui
        tex3d.setFormat(QOpenGLTexture.R16_UNorm)

        self.valid = False
        # MiniLogger will detect if the Qt OpenGL error logger
        # receives any error messages
        ml = self.MiniLogger(logger)
        try:
            # This will fail if there is not enough GPU memory
            tex3d.allocateStorage()
            self.tex3d = tex3d
            aunit = 1
            # If the OpenGL module is allowed to throw exceptions
            # (the default; this can be changed at the top of
            # gl_data_window.py), the out-of-memory exception
            # will be thrown at the next line, rather than at
            # the allocateStorage() call above
            gl.glActiveTexture(pygl.GL_TEXTURE0+aunit)
            tex3d.bind()
            aloc = self.program.uniformLocation("atlas")
            self.program.setUniformValue(aloc, aunit)
            gl.glActiveTexture(pygl.GL_TEXTURE0)
            tex3d.release()
            self.valid = True
        except:
            print("exception!")
            ml.close()
            pass
        if ml.message_count > 0: 
            # The Qt OpenGL error logging system detected an error message
            print("error message!")
            self.valid = False
        ml.close()
        self.setVolumeView(volume_view)

    def setVolumeView(self, volume_view):
        print("Atlas.setVolumeView", volume_view.volume.name if volume_view else "(None)")
        self.clearData()

        self.volume_view = volume_view

        if volume_view is None:
            # Need to clear out self.datas as soon as possible
            # when a volume's data is released, in order to make
            # sure the data's memory is released.
            self.datas = None
            # print("Atlas.setVolumeView: self.datas set to None")
            return

        vol = volume_view.volume
        vdir = volume_view.direction
        is_zarr = vol.is_zarr
        dcsz = self.dcsz

        # TODO: use volume_view.colormap_is_indicator
        ind = volume_view.colormap_is_indicator
        # print("svv ind", ind)
        tex3d = self.tex3d
        if ind:
            tex3d.setMagnificationFilter(QOpenGLTexture.Nearest)
            tex3d.setMinificationFilter(QOpenGLTexture.Nearest)
        else:
            tex3d.setMagnificationFilter(QOpenGLTexture.Linear)
            tex3d.setMinificationFilter(QOpenGLTexture.Linear)

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
            # print("data shape", data.shape)
            # TODO
            # shape = data.shape
            shape = data.shape[:3]
            dsz.append(tuple(shape[::-1]))
        self.datas = datas
        self.dsz = dsz
        # number of data chunks in each direction
        ksz = []
        for l in range(len(dsz)):
            lksz = tuple(self.ke(dsz[l][i],dcsz[i]) for i in range(len(dcsz)))
            ksz.append(lksz)
        self.ksz = ksz
        self.program.bind()
        xyz_xform = self.xyzXform(dsz[0])
        self.program.setUniformValue("xyz_xform", xyz_xform)
        # self.program.release()

    def clearData(self):
        # print("clearing atlas data")
        aksz = self.aksz
        self.chunks.clear()
        for k in range(aksz[2]):
            for j in range(aksz[1]):
                for i in range(aksz[0]):
                    ak = (i,j,k)
                    dk = (i,j,k)
                    dl = -1
                    chunk = Chunk(self, ak, dk, dl)
                    key = self.key(dk, dl)
                    self.chunks[key] = chunk
        self.pbo_queue = Queue()

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
        return (ak[2]*aksz[1] + ak[1])*aksz[0] + ak[0]

    # Number of chunks (in 1D) given data size, chunk size.
    # This gives the number of chunks needed to cover the
    # entire data set; the last chunk may stretch beyond
    # the end of the data.
    def ke(self, e, ce):
        ke = 1 + (e-1)//ce
        return ke

    def getPbo(self):
        if self.pbo_pool.empty():
            acsz = self.acsz
            pbo = QOpenGLBuffer(QOpenGLBuffer.PixelUnpackBuffer)
            pbo.create()
            pbo.bind()
            # Assumes UInt16
            pbo_size = acsz[0]*acsz[1]*acsz[2]*2
            pbo.allocate(pbo_size)
            pbo.release()
            # print("created pbo", pbo.bufferId(), pbo_size)
            self.pbo_pool.put(pbo)
        return self.pbo_pool.get()

    def releasePbo(self, pbo):
        self.pbo_pool.put(pbo)

    def initializeChunks(self, zblocks):
        for chunk in reversed(self.chunks.values()):
            if not chunk.in_use:
                break
            chunk.in_use = False

        # reverse to make lowest-resolution blocks
        # are loaded first
        for zblock in reversed(zblocks):
            block = zblock[:3]
            zoom_level = zblock[3]
            key = self.key(block, zoom_level)
            chunk = self.chunks.get(key, None)
            # If the data chunk is not currently stored in the atlas:
            if chunk is None:
                # Get the first Chunk in the OrderedDict: 
                _, chunk = self.chunks.popitem(last=False)
                chunk.initialize(block, zoom_level)
                self.chunks[key] = chunk
            else:
                self.chunks.move_to_end(key)
            chunk.in_use = True

    def loadChunks(self, in_progress_cb=None):
        chunks_loading = 0
        for key,chunk in reversed(self.chunks.items()):
            if not chunk.in_use:
                break
            if chunks_loading >= 2*self.max_textures_set:
                break
            if chunk.status == Chunk.Status.INITIALIZED or chunk.status == Chunk.Status.PARTIALLY_LOADED_FROM_DISK:
                # print("request", chunk.dk, chunk.dl)
                future = self.executor.submit(chunk.getDataFromDisk)
                # TODO: need a callback?
                if in_progress_cb is not None:
                    future.add_done_callback(lambda x: self.futureCallback(in_progress_cb, x))
                chunks_loading += 1

    def futureCallback(self, in_progress_cb, future):
        # We don't care about the result, but this will throw
        # an exception if the thread had an exception; without
        # this, the thread would quietly die without a trace
        result = future.result()
        if in_progress_cb is not None:
            in_progress_cb()

    def loadPbos(self):
        # To get all the active chunks, search backwards from
        # the end
        num_pbos_set = 0
        for key,chunk in reversed(self.chunks.items()):
            if not chunk.in_use:
                break
            if self.pbo_queue.qsize() >= 8*self.max_textures_set:
                break
            if num_pbos_set >= 1.5*self.max_textures_set:
                break
            if chunk.status == Chunk.Status.LOADED_FROM_DISK:
                chunk.copyRamDataToPbo()
                self.pbo_queue.put(chunk)
                num_pbos_set += 1
            # print(chunk.dl, chunk.dk)
        # print(zoom_level, cnt, len(blocks))
        # print("cnt", cnt)

    def loadTexturesFromRam(self, in_progress_cb=None):
        # To get all the active chunks, search backwards from
        # the end
        num_textures_set = 0
        for key,chunk in reversed(self.chunks.items()):
            if not chunk.in_use:
                break
            if num_textures_set >= self.max_textures_set:
                break
            if chunk.status == Chunk.Status.LOADED_FROM_DISK:
                chunk.copyRamDataToTexmap()
                num_textures_set += 1

        if num_textures_set > 0:
            self.tmin_ubo.setBuffer()
            self.tmax_ubo.setBuffer()
            self.xform_ubo.setBuffer()

        if num_textures_set >= self.max_textures_set and in_progress_cb is not None:
            in_progress_cb()


    def loadTexturesFromPbos(self, in_progress_cb=None):
        # To get all the active chunks, search backwards from
        # the end
        num_textures_set = 0
        while not self.pbo_queue.empty() and num_textures_set < self.max_textures_set:
            chunk = self.pbo_queue.get()
            if chunk.status == Chunk.Status.LOADED_TO_PBO:
                chunk.copyPboDataToTexmap()
                num_textures_set += 1

        if num_textures_set > 0:
            self.tmin_ubo.setBuffer()
            self.tmax_ubo.setBuffer()
            self.xform_ubo.setBuffer()

        if num_textures_set >= self.max_textures_set and in_progress_cb is not None:
            in_progress_cb()

    def addBlocks(self, zblocks, in_progress_cb=None):
        timer = Utils.Timer()
        timer.active = False
        self.initializeChunks(zblocks)
        timer.time(" init")

        # Load data from disk to RAM
        # NOTE that each chunk will be loaded in a
        # separate thread
        self.loadChunks(in_progress_cb)
        timer.time(" from disk")

        # 3 options for getting the data from RAM into 
        # the Atlas 3D texture:

        # 1) Load data from RAM into a PBO (one per chunk);
        # data will later (outside of addBlocks) be
        # loaded from each PBO into the Atlas 3D texture.

        # 2) Load data from RAM into a PBO (one per chunk);
        # then immediately load data from each PBO into the
        # Atlas 3D texture (inside addBlocks).

        # 3) Load data from RAM directly into the
        # Atlas 3D texture.

        # Here Option 1 is used, because delaying the loading
        # from the PBOs into the 3D texture allows more time
        # for the GPU to get the data from RAM to the PBOs
        # (PBOs are designed to allow transferring data from
        # RAM to GPU in the background)

        ''''''
        # Option 1
        self.loadPbos()
        timer.time(" to pbos")
        ''''''
        '''
        # Option 2
        self.loadPbos()
        timer.time(" to pbos")
        self.loadTexturesFromPbos(in_progress_cb)
        timer.time(" to textures")
        '''
        '''
        # Option 3
        self.loadTexturesFromRam(in_progress_cb)
        timer.time(" to textures")
        '''

    # displayBlocks is in a separate operation
    # from addBlocks, because addBlocks may need to be called later
    # than displayBlocks, to prevent GPU round trips
    def displayBlocks(self, data_fbo, fvao, stxy_xform):
        gl = self.gl

        data_fbo.bind()

        # Be sure to clear with alpha = 0
        # so that the slice view isn't blocked!
        gl.glClearColor(0.,0.,0.,0.)
        gl.glClear(pygl.GL_COLOR_BUFFER_BIT)

        if stxy_xform is None:
            QOpenGLFramebufferObject.bindDefault()
            return

        self.program.bind()

        self.program.setUniformValue("stxy_xform", stxy_xform)
        normal_offset = fvao.fragment_view.normal_offset
        self.program.setUniformValue("normal_offset", normal_offset)

        uchunks = []
        for key,chunk in reversed(self.chunks.items()):
            if not chunk.in_use:
                break
            uchunks.append(chunk)
        uchunks.sort(reverse=True, key=lambda chunk: chunk.dl)
        nchunks = 0
        for chunk in uchunks:
            ak = chunk.ak
            ind = self.index(ak)
            self.chart_id_ubo.data[nchunks,0] = ind
            nchunks += 1

        nloc = self.program.uniformLocation("ncharts")
        # print("nloc, nchunks", nloc, nchunks)
        # print("nchunks", nchunks)
        gl.glUniform1i(nloc, nchunks)

        self.chart_id_ubo.setSubBuffer(nchunks)

        # print("db de")
        gl.glDrawElements(pygl.GL_TRIANGLES, fvao.trgl_index_size,
                       pygl.GL_UNSIGNED_INT, VoidPtr(0))
        # print("db de finished")
        self.program.release()

        QOpenGLFramebufferObject.bindDefault()

