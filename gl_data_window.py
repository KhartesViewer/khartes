from PyQt5.QtGui import (
        QImage,
        QMatrix4x4,
        QOffscreenSurface,
        QOpenGLContext,
        QPixmap,
        QSurfaceFormat,
        QTransform,
        QVector2D,
        QVector4D,
        )

# from PyQt5.QtOpenGL import (
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
        # QOpenGLVersionFunctionsFactory,
        )

from PyQt5.QtWidgets import (
        QApplication, 
        QGridLayout,
        QHBoxLayout,
        QMainWindow,
        QWidget,
        )

# from PyQt5.QtOpenGLWidgets import (
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
import collections
import traceback

import numpy as np
import numpy.linalg as npla
import cv2
import OpenGL
# Uncomment this line to turn off OpenGL error checking
# OpenGL.ERROR_CHECKING = False
from OpenGL import GL as pygl
# from shiboken6 import VoidPtr
import ctypes
def VoidPtr(i):
    return ctypes.c_void_p(i)

from utils import Utils
from data_window import DataWindow


class GLDataWindow(DataWindow):
    def __init__(self, window, axis):
        super(GLDataWindow, self).__init__(window, axis)
        layout = QHBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        self.setLayout(layout)
        self.glw = GLDataWindowChild(self)
        layout.addWidget(self.glw)
        self.main_active_fragment_view = None

    def drawSlice(self):
        self.window.setFocus()
        self.glw.update()
        if self.volume_view is not None:
            pv = self.window.project_view
            mfv = pv.mainActiveFragmentView(unaligned_ok=True)
            if mfv != self.main_active_fragment_view:
                self.main_active_fragment_view = mfv
                self.volume_view.setStxyTf(None)
            if self.volume_view.stxytf is None:
                self.setStxyTfFromIjkTf()

    # Overrides DataWindow.fvsInBounds
    # Returns a set of all FragmentViews whose cross-section
    # line on the slice passes through the bounding box given
    # by xymin and xymax, where xymin and xymax are Qt window 
    # coordinates (not OpenGL window coordinates)
    def fvsInBounds(self, xymin, xymax):
        dw = self.glw
        # print("xy", (xymax[0]+xymin[0])/2, (xymax[1]+xymin[1])/2)
        # print("xy", xymin, xymax)
        xyfvs = dw.xyfvs
        indexed_fvs = dw.indexed_fvs
        fvs = set()
        if xyfvs is None or indexed_fvs is None:
            return fvs

        # matches = ((xyfvs[:,:2] >= xymin).all(axis=1) & (xyfvs[:,:2] <= xymax).all(axis=1)).nonzero()[0]
        matches = ((xyfvs[:,:2] >= xymin) & (xyfvs[:,:2] <= xymax)).all(axis=1).nonzero()[0]
        # print("xyfvs", xymin, xymax, xyfvs.shape)
        # print("matches", matches.shape)
        if len(matches) == 0:
            return fvs
        # print("matches", matches.shape, xyfvs[matches[0]])
        uniques = np.unique(xyfvs[matches][:,2])
        # print(uniques)
        for ind in uniques:
            if ind < 0 or ind >= len(indexed_fvs):
                continue
            fv = indexed_fvs[ind]
            fvs.add(fv)
        return fvs

    '''
    # slice ij position to window xy position
    def ijToGLXy(self, ij):
        i,j = ij
        ratio = self.screen().devicePixelRatio()
        zoom = self.getZoom()
        zoom *= ratio
        tijk = self.volume_view.ijktf
        ci = tijk[self.iIndex]
        cj = tijk[self.jIndex]
        ww, wh = ratio*self.width(), ratio*self.height()
        wcx, wcy = ww//2, wh//2
        x = int(zoom*(i-ci)) + wcx
        y = int(zoom*(j-cj)) + wcy
        print(i,j,ci,cj)
        return (x,y)
    '''

    def setCursorPosition(self, tijk):
        ij = self.tijkToIj(tijk)
        xy = self.ijToXy(ij)
        d = 10
        xyl = (xy[0]-d, xy[1]-d)
        xyg = (xy[0]+d, xy[1]+d)

        # stxyz = self.stxyzInBounds(xyl, xyg, tijk)
        stxyz = self.stxyzInRange(tijk, d)
        # print("tf, xy, stxy", tf, xy, stxy)
        self.window.setCursorPosition(self, tijk, stxyz)

    def addPoint(self, tijk):
        # print("gldw add point", tijk)
        d = 100
        stxyz = self.stxyzInRange(tijk, d)
        print("stxyz", stxyz)
        maxz = 10
        if stxyz is None:
            stxy = None
        else:
            stxy = stxyz[:2]
            if abs(stxyz[2]) > maxz:
                stxy = None
        self.window.addPointToCurrentFragment(tijk, stxy)

    def setStxyTfFromIjkTf(self):
        # dw = self.glw
        vv = self.volume_view
        if vv is None:
            return
        tf = vv.ijktf
        ij = self.tijkToIj(tf)
        xy = self.ijToXy(ij)
        d = 10
        xyl = (xy[0]-d, xy[1]-d)
        xyg = (xy[0]+d, xy[1]+d)

        # stxy = self.stxyInBounds(xyl, xyg, tf)
        stxy = self.stxyInRange(tf, d)
        # print("tf, xy, stxy", tf, xy, stxy)
        # if stxy is not None:
        #     self.volume_view.setStxyTf(stxy)
        self.volume_view.setStxyTf(stxy)

    def setIjkTf(self, tf):
        self.volume_view.setIjkTf(tf)
        self.setStxyTfFromIjkTf()

    def ptToBary(self, ijk, vs):
        v01 = vs[1]-vs[0]
        v02 = vs[2]-vs[0]
        tnorm = np.cross(v01,v02)
        tnormlen = npla.norm(tnorm)
        if tnormlen != 0:
            tnorm /= tnormlen
        vcs = vs - ijk
        bs = np.inner(tnorm, 
                      np.cross(
                          np.roll(vcs, 2, axis=0), 
                          np.roll(vcs, 1, axis=0)))/tnormlen
        return bs

    def ptInTrgl(self, ijk, tindex):
        pv = self.window.project_view
        mfv = pv.mainActiveFragmentView(unaligned_ok=True)
        trgl = mfv.trgls()[tindex]
        vs = mfv.vpoints[trgl][:,:3]
        bs = self.ptToBary(ijk, vs)
        # print("bs", bs, bs.sum())
        return (bs>=0).all() and (bs<=1).all()

    @staticmethod
    def tetVolume(v0, v1, v2, v3):
        v = np.inner(v3-v0, np.cross(v1-v0, v2-v0)) / 6.
        return v

    # height of v3 above the plane formed by v0,v1,v2
    @staticmethod
    def tetHeight(v0, v1, v2, v3):
        cr = np.cross(v1-v0, v2-v0)
        l2 = np.sqrt(np.inner(cr,cr))
        if l2 != 0:
            cr /= l2
        h = np.inner(cr, v3-v0)
        return h

    # Note that this only looks for trgls on the
    # main active fragment view
    def stxyzInRange(self, ijk, maxd):
        dw = self.glw
        ij = self.tijkToIj(ijk)
        xy0 = self.ijToXy(ij)
        xymin = (xy0[0]-maxd, xy0[1]-maxd)
        xymax = (xy0[0]+maxd, xy0[1]+maxd)
        # x y fragment_view_id trgl_id
        xyfvs = dw.xyfvs
        # list of fragment views (to go from fragment_view_id
        # to fragment_view)
        indexed_fvs = dw.indexed_fvs
        pv = self.window.project_view
        mfvi = -1
        mfv = pv.mainActiveFragmentView(unaligned_ok=True)
        if mfv is None:
            return None
        if mfv is not None and dw.indexed_fvs is not None and mfv in indexed_fvs:
            mfvi = indexed_fvs.index(mfv)
        if mfvi < 0:
            return None
        # indexes of rows where fragment_view matches mfvi
        # matches = (xyfvs[:,2] == mfvi).nonzero()[0]
        # need a lot of parentheses because the & operator
        # has higher precedence than comparison operators
        matches = ((xyfvs[:,2] == mfvi) & (xyfvs[:,:2] >= xymin).all(axis=1) & (xyfvs[:,:2] <= xymax).all(axis=1)).nonzero()[0]
        if len(matches) == 0:
            # print("no matches")
            return None

        mxyft = xyfvs[matches]

        # dels = mxyft[:,:2] - np.array(xy0)[:,np.newaxis]
        dels = mxyft[:,:2] - xy0
        # d2s = np.inner(dels, dels)
        d2s = (dels*dels).sum(axis=1)
        minindex = np.argmin(d2s)
        # print("d2s", mxyft.shape, xy0, dels.shape, d2s.shape, d2s[minindex])
        mind = np.sqrt(d2s[minindex])
        if mind > maxd:
            return None

        trgl_index = mxyft[minindex, 3]
        trgl = mfv.trgls()[trgl_index]
        vpts = mfv.vpoints[trgl][:,:3]

        bs = self.ptToBary(ijk, vpts)
        sts = mfv.stpoints[trgl]
        # print(sts)
        # print(bs)
        stxy = (sts*bs.reshape(-1,1)).sum(axis=0)
        z = self.tetHeight(*vpts, ijk)
        stxyz = np.concatenate((stxy, [z]))
        return stxyz

    def stxyInRange(self, ijk, maxd):
        stxyz = self.stxyzInRange(ijk, maxd)
        if stxyz is None:
            return None
        return stxyz[:2]

    # Note that this only looks for trgls on the
    # main active fragment view
    def stxyzInBounds(self, xymin, xymax, ijk):
        timera = Utils.Timer()
        fvs = set()
        # ratio = self.screen().devicePixelRatio()
        # xymin = (round(xymin[0]*ratio), round(xymin[1]*ratio))
        # xymax = (round(xymax[0]*ratio), round(xymax[1]*ratio))
        xycenter = ((xymin[0]+xymax[0])//2, (xymin[1]+xymax[1])//2)
        dw = self.glw
        xyfvs = dw.xyfvs
        indexed_fvs = dw.indexed_fvs
        pv = self.window.project_view
        mfvi = -1
        mfv = pv.mainActiveFragmentView(unaligned_ok=True)
        if mfv is None:
            return None
        if mfv is not None and dw.indexed_fvs is not None and mfv in dw.indexed_fvs:
            mfvi = dw.indexed_fvs.index(mfv)
        if mfvi < 0:
            return None
        # need a lot of parentheses because the & operator
        # has higher precedence than comparison operators
        matches = ((xyfvs[:,2] == mfvi) & (xyfvs[:,:2] >= xymin).all(axis=1) & (xyfvs[:,:2] <= xymax).all(axis=1)).nonzero()[0]
        if len(matches) == 0:
            # print("no matches")
            return None

        mrows = xyfvs[matches]

        trgls = mrows[:,3]
        uniques = np.unique(trgls)
        # print("u", uniques)
        inbounds = []
        for trgl in uniques:
            if self.ptInTrgl(ijk, trgl):
                inbounds.append(trgl)
        mirows = mrows[np.isin(trgls, inbounds)]
        if len(mirows) == 0:
            # print("no pts in trgls")
            return None
        # print("mirows", mirows)
        xys = mirows[:,:2]
        ds = npla.norm(xys-xycenter, axis=1)
        imin = np.argmin(ds)
        # print(mrows[imin])
        trgl_index = int(mirows[imin, 3])
        # print("trgl_index", trgl_index)
        trgl = mfv.trgls()[trgl_index]
        # print(trgl_index, trgl)
        vpts = mfv.vpoints[trgl][:,:3]
        # print(ijk)
        # print(vpts)
        # print(trgl_index)
        # print(self.ptInTrgl(ijk, trgl_index))


        bs = self.ptToBary(ijk, vpts)
        sts = mfv.stpoints[trgl]
        # print(sts)
        # print(bs)
        stxy = (sts*bs.reshape(-1,1)).sum(axis=0)
        z = self.tetHeight(*vpts, ijk)
        stxyz = np.concatenate((stxy, [z]))
        # print(stxy)
        timera.time("old")
        nstxyz = self.stxyzInRange(ijk, 10)
        timera.time("new")
        print(stxyz)
        print(" ", nstxyz)
        return stxyz

    def stxyInBounds(self, xymin, xymax, ijk):
        stxyz = self.stxyzInBounds(xymin, xymax, ijk)
        if stxyz is None:
            return None
        return stxyz[:2]


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

        vec4 uColor = texture(underlay_sampler, ftxt);
        alpha = uColor.a;
        fColor = (1.-alpha)*fColor + alpha*uColor;

        vec4 frColor = texture(fragments_sampler, ftxt);
        // alpha = frag_opacity*frColor.a;
        alpha = frColor.a;
        fColor = (1.-alpha)*fColor + alpha*frColor;

        vec4 oColor = texture(overlay_sampler, ftxt);
        alpha = oColor.a;
        fColor = (1.-alpha)*fColor + alpha*oColor;
      }
    ''',
}

common_offset_code = '''

    const float angles[] = float[8](
      radians(0), radians(45), radians(90), radians(135), 
      radians(180), radians(225), radians(270), radians(315));
    const vec2 trig_table[] = vec2[9](
      vec2(cos(angles[0]), sin(angles[0])),
      vec2(cos(angles[1]), sin(angles[1])),
      vec2(cos(angles[2]), sin(angles[2])),
      vec2(cos(angles[3]), sin(angles[3])),
      vec2(cos(angles[4]), sin(angles[4])),
      vec2(cos(angles[5]), sin(angles[5])),
      vec2(cos(angles[6]), sin(angles[6])),
      vec2(cos(angles[7]), sin(angles[7])),
      vec2(0., 0.));

  // all arrays need to be the same size
  // so the correct one can be copied into "vs"
  const ivec2 v10[] = ivec2[10](
    ivec2(0, 0),
    ivec2(0, 1),
    ivec2(0, 7),
    ivec2(0, 2),
    ivec2(0, 6),
    ivec2(1, 2),
    ivec2(1, 6),
    ivec2(1, 3),
    ivec2(1, 5),
    ivec2(1, 4)
  );
  const ivec2 v4[] = ivec2[10](
    ivec2(0, 2),
    ivec2(0, 6),
    ivec2(1, 2),
    ivec2(1, 6),
    ivec2(-1, -1),
    ivec2(-1, -1),
    ivec2(-1, -1),
    ivec2(-1, -1),
    ivec2(-1, -1),
    ivec2(-1, -1)
  );
'''

fragment_pts_code = {
    "name": "fragment_pts",

    "vertex": '''
      #version 410 core

      uniform vec4 node_color;
      uniform vec4 highlight_node_color;
      uniform int nearby_node_id;
      out vec4 color;
      uniform mat4 xform;
      layout(location=3) in vec3 position;
      void main() {
        if (gl_VertexID == nearby_node_id) {
          color = highlight_node_color;
        } else {
          color = node_color;
        }
        gl_Position = xform*vec4(position, 1.0);
      }
    ''',

    "geometry_hide": '''
      #version 410 core
  
      layout(points) in;
      in vec4 color[1];
      out vec4 gcolor;
      layout(points, max_vertices = 1) out;

      void main() {
        vec4 pos = gl_in[0].gl_Position;
        if (pos.x < -1. || pos.x > 1. ||
            pos.y < -1. || pos.y > 1. ||
            pos.z < -1. || pos.z > 1.) return;

        gl_Position = pos;
        gcolor = color[0];
        EmitVertex();
    }
    ''',

    "fragment": '''
      #version 410 core

      // in vec4 gcolor;
      in vec4 color;
      out vec4 fColor;

      void main()
      {
        // fColor = gcolor;
        fColor = color;
      }
    ''',
}

fragment_lines_code = {
    "name": "fragment_lines",

    "vertex": '''
      #version 410 core

      uniform mat4 xform;
      layout(location=3) in vec3 position;
      void main() {
        gl_Position = xform*vec4(position, 1.0);
      }
    ''',
    "geometry": '''
      #version 410 core
  
      uniform float thickness;
      uniform vec2 window_size;
  
      layout(lines) in;
      // max_vertices = 10+4 (10 for thick line, 4 for pick line)
      layout(triangle_strip, max_vertices = 14) out;
      flat out int trgl_type;
  
      %s
  
      void main() {
        float dist[2];
        float sgn[2]; // sign(float) returns float
        float sig = 0; // signature
        float m = 1;

        for (int i=0; i<2; i++) {
          dist[i] = gl_in[i].gl_Position.z;
          sgn[i] = sign(dist[i]);
          sig += m*(1+sgn[i]);
          m *= 3;
        }
        if (sig == 0 || sig == 8) return;
        vec4 pcs[2];
        if (sig == 4) {
          pcs[0] = gl_in[0].gl_Position;
          pcs[1] = gl_in[1].gl_Position;
        } else {
          float da = dist[0];
          float db = dist[1];
  
          vec4 pa = gl_in[0].gl_Position;
          vec4 pb = gl_in[1].gl_Position;
          float fa = abs(da);
          float fb = abs(db);
          vec4 pc = pa;
          if (fa > 0 || fb > 0) pc = (fa * pb + fb * pa) / (fa + fb);
          pcs[0] = pc;
          pcs[1] = pc;
        }

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
          trgl_type = 0;
          EmitVertex();
        }
        EndPrimitive();

        for (int i=0; i<4; i++) {
          ivec2 iv = v4[i];
          gl_Position = pcs[iv.x] + 1.*offsets[iv.y];
          trgl_type = 1;
          EmitVertex();
        }
      }

    ''' % common_offset_code,

    "fragment": '''
      #version 410 core

      uniform vec4 gcolor;
      uniform vec4 icolor;
      layout(location = 0) out vec4 frag_color;
      layout(location = 1) out vec4 pick_color;
      // The most important thing about empty_color
      // is that alpha = 0., so with blending enabled,
      // empty_color is effectively not drawn
      const vec4 empty_color = vec4(0.,0.,0.,0.);
      flat in int trgl_type;

      void main()
      {
        // in both clauses of the if statement, need to
        // set both frag_color and pick_color.  If either
        // is not set, it will be drawn in an undefined color.
        if (trgl_type == 0) {
          frag_color = gcolor;
          pick_color = empty_color;
        } else {
          frag_color = empty_color;
          pick_color = icolor;
        }
      }
    ''',
}

fragment_trgls_code = {
    "name": "fragment_trgls",

    "vertex": '''
      #version 410 core

      uniform mat4 xform;
      uniform mat4 stxform;
      // uniform int flag;
      layout(location=3) in vec3 xyz;
      layout(location=4) in vec2 stxy;
      out vec4 stxyt;
      void main() {
        gl_Position = xform*vec4(xyz, 1.0);
        stxyt = stxform*vec4(stxy, 0., 1.0);
      }
    ''',

    # modified from https://stackoverflow.com/questions/16884423/geometry-shader-producing-gaps-between-lines/16886843
    "geometry": '''
      #version 410 core
  
      uniform float thickness;
      uniform vec2 window_size;

      // flag=0: Used by GLDataWindow to draw fragment cross sections
      // on data slices.
      // flag=1: Used by GLSurfaceWindow to draw axes on map view
      uniform int flag;
  
      layout(triangles) in;
      // max_vertices = 10+4 (10 for thick line, 4 for pick line)
      layout(triangle_strip, max_vertices = 14) out;
      flat out int trgl_type;
      in vec4 stxyt[];
  
      %s
  
      void main()
      {
        float dist[3];
        float sgn[3]; // sign(float) returns float
        float sig = 0; // signature
        float m = 1;

        for (int i=0; i<3; i++) {
          dist[i] = gl_in[i].gl_Position.z;
          sgn[i] = sign(dist[i]);
          sig += m*(1+sgn[i]);
          m *= 3;
        }

        // These correspond to the cases where there are
        // no intersections (---, 000, +++):
        if (sig == 0 || sig == 13 || sig == 26) return;
  
        // Have to go through nodes in the correct order.
        // Imagine a triangle a,b,c, with distances
        // a = -1, b = 0, c = 1.  In this case, there
        // are two intersections: one at point b, and one on
        // the line between a and c.
        // All three lines (ab, bc, ca) will have intersections,
        // the lines ab and bc will both have the same intersection,
        // at point b.
        // If the lines are scanned in that order, and only the first
        // two detected intersections are stored, then the two detected
        // intersections will both be point b!
        // There are various ways to detect and avoid this problem,
        // but the method below seems the least convoluted.

        // General note: much of the code below could be replaced with
        // a lookup table based on the sig (signature) computed above.
        // This rewrite can wait until a later time, though, since 
        // the existing code works, and seems fast enough.
        
        ivec3 ijk = ivec3(0, 1, 2); // use swizzle to permute the indices

        // Let each vertex of the triangle be denoted by +, -, or 0,
        // depending on the sign (sgn) of its distance from the plane.
        // 
        // We want to rotate any given triangle so that
        // its ordered sgn values match one of these:
        // ---  000  +++  (no intersections)
        // 0++  -0-       (one intersection)
        // 0+0  -00       (two intersections)
        // 0+-  -+0       (two intersections)
        // -++  -+-       (two intersections)
        // Every possible triangle can be cyclically reordered into
        // one of these orderings.
        // In the two-intersection cases above, the intersections
        // computed from the first two segments (ignoring 00 segments)
        // will be unique, and in a consistent orientation,
        // given these orderings.
        // In most cases, the test sgn[ijk.x] < sgn[ijk.y] is
        // sufficient to ensure this order.  But there is
        // one ambiguous case: 0+- and -0+ are two orderings
        // of the same triangle, and both pass the test.
        // But only the 0+- ordering will allow the first two
        // segments to yield two intersections in the correct order
        // (the -0+ ordering will yield the same location twice!).
        // So an additional test is needed to avoid this case:
        // sgn[ijk.y] >= sgn[ijk.z]
        // Thus the input triangle needs to be rotated until
        // the following condition holds:
        // sgn[ijk.x] < sgn[ijk.y] && sgn[ijk.y] >= sgn[ijk.z]
        // So the condition for continuing to rotate is that the
        // condition above not be true, in other words:
        // !(sgn[ijk.x] < sgn[ijk.y] && sgn[ijk.y] >= sgn[ijk.z])
        // Rewrite, so the condition to continue to rotate is:
        // sgn[ijk.x] >= sgn[ijk.y] || sgn[ijk.y] < sgn[ijk.z]>0;

        // Continue to rotate the triangle so long as the above condition is
        // met:
        for (int i=0; 
             i<3 // stop after 3 iterations
             && (sgn[ijk.x] >= sgn[ijk.y] || sgn[ijk.y] < sgn[ijk.z]);
             ijk=ijk.yzx, i++);
        // At this point, ijk has been set to rotate the triangle 
        // to the correct order.

        vec4 pcs[2];
        int j = 0;
        for (int i=0; i<3 && j<2; ijk=ijk.yzx, i++) {
          float da = dist[ijk.x];
          float db = dist[ijk.y];
          if (da*db > 0 || (da == 0 && db == 0)) continue;
  
          vec4 pa;
          vec4 pb;
          if (flag == 0) {
            pa = gl_in[ijk.x].gl_Position;
            pb = gl_in[ijk.y].gl_Position;
          } else if (flag == 1) {
            pa = stxyt[ijk.x];
            pb = stxyt[ijk.y];
          }
          float fa = abs(da);
          float fb = abs(db);
          vec4 pc = pa;
          if (fa > 0 || fb > 0) pc = (fa * pb + fb * pa) / (fa + fb);
          pcs[j++] = pc;
        }

        if (j<2) return;
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
          trgl_type = 0;
          EmitVertex();
        }
        EndPrimitive();

        if (flag == 0) {
          for (int i=0; i<4; i++) {
            ivec2 iv = v4[i];
            gl_Position = pcs[iv.x] + 1.*offsets[iv.y];
            trgl_type = 1;
            gl_PrimitiveID = gl_PrimitiveIDIn;
            EmitVertex();
          }
        }
      }
    ''' % common_offset_code,

    "fragment": '''
      #version 410 core

      uniform vec4 gcolor;
      uniform vec4 icolor;

      // uniform int flag;

      // out vec4 fColor;
      layout(location = 0) out vec4 frag_color;
      layout(location = 1) out vec4 pick_color;
      // The most important thing about empty_color
      // is that alpha = 0., so with blending enabled,
      // empty_color is effectively not drawn
      const vec4 empty_color = vec4(0.,0.,0.,0.);
      flat in int trgl_type;

      void main()
      {
        // in both clauses of the if statement, need to
        // set both frag_color and pick_color.  If either
        // is not set, it will be drawn in an undefined color.
        if (trgl_type == 0) {
          frag_color = gcolor;
          pick_color = empty_color;
        } else {
          frag_color = empty_color;
          uint lsid = gl_PrimitiveID & 0xffff;
          uint msid = (gl_PrimitiveID>>16) & 0xffff;
          // remember alpha must = 1!
          // vec4 ocolor = vec4(icolor.r, float(msid)/65536., float(lsid)/65536., 1.);
          vec4 ocolor = vec4(icolor.r, float(msid)/65535., float(lsid)/65535., 1.);
          // vec4 ocolor = icolor;
          pick_color = ocolor;
        }
      }
    ''',
}

class GLDataWindowChild(QOpenGLWidget):
    def __init__(self, gldw, parent=None):
        super(GLDataWindowChild, self).__init__(parent)
        self.gldw = gldw
        self.setMouseTracking(True)
        self.fragment_vaos = {}

        # synchronous mode is said to be much slower
        # self.logging_mode = QOpenGLDebugLogger.SynchronousLogging
        self.logging_mode = QOpenGLDebugLogger.AsynchronousLogging
        # self.common_offset_code = common_offset_code
        self.localInit()

    def localInit(self):
        self.xyfvs = None
        self.indexed_fvs = None
        # Location of "position" variable in vertex shaders.
        # This is specified by the shader line:
        # layout(location=3) in vec3 postion;
        self.position_location = 3
        self.message_prefix = "dw"

    def dwKeyPressEvent(self, e):
        self.gldw.dwKeyPressEvent(e)

    def initializeGL(self):
        print(self.message_prefix, "initializeGL")
        self.context().aboutToBeDestroyed.connect(self.destroyingContext)
        # self.gl = self.context().versionFunctions()
        self.gl = pygl
        # self.gl = QOpenGLVersionFunctionsFactory.get()
        self.main_context = self.context()
        # Note that debug logging only takes place if the
        # surface format option "DebugContext" is set
        self.logger = QOpenGLDebugLogger()
        self.logger.initialize()
        self.logger.messageLogged.connect(lambda m: self.onLogMessage(self.message_prefix, m))
        self.logger.startLogging(self.logging_mode)
        msg = QOpenGLDebugMessage.createApplicationMessage("test debug messaging")
        self.logger.logMessage(msg)
        self.localInitializeGL()

        # self.printInfo()

    def localInitializeGL(self):
        f = self.gl
        f.glClearColor(.6,.3,.3,1.)

        self.buildPrograms()
        self.buildSliceVao()

        self.fragment_fbo = None

    def resizeGL(self, width, height):
        # print("resize", width, height)
        # based on https://stackoverflow.com/questions/59338015/minimal-opengl-offscreen-rendering-using-qt
        # vp_size = QSize(width, height)
        f = self.gl
        # f.glViewport(0, 0, vp_size.width(), vp_size.height())
        # BUG in PySide6: can't seem to get vector info from glGet
        # fbdims = f.glGetIntegerv(pygl.GL_VIEWPORT)
        # fbdims = f.glGetIntegeri_v(pygl.GL_VIEWPORT, 0)
        # fbdims = f.glGetFloatv(pygl.GL_VIEWPORT)
        # fbdims = f.glGetFloati_v(pygl.GL_VIEWPORT, 0)
        # fbdims = f.glGetFloati_v(pygl.GL_DEPTH_RANGE, 2)

        # fbdims = pygl.glGetIntegerv(pygl.GL_VIEWPORT)
        # print("fbdims", width, height, fbdims)

        # See https://doc.qt.io/qt-6/highdpi.html for why
        # this is needed when working with OpenGL.
        # I would prefer to set the size based on the size of
        # the default framebuffer (or viewport), but because of 
        # the PySide6 bug mentioned above, this does not seem
        # to be possible.
        ratio = self.screen().devicePixelRatio()
        vp_size = QSize(int(ratio*width), int(ratio*height))
        # print("resize", width, height, ratio, vp_size)
        fbo_format = QOpenGLFramebufferObjectFormat()
        fbo_format.setAttachment(QOpenGLFramebufferObject.CombinedDepthStencil)
        fbo_format.setInternalTextureFormat(pygl.GL_RGBA16)
        self.fragment_fbo = QOpenGLFramebufferObject(vp_size, fbo_format)
        self.fragment_fbo.bind()

        self.fragment_fbo.addColorAttachment(vp_size.width(), vp_size.height(), pygl.GL_RGBA16)
        draw_buffers = (pygl.GL_COLOR_ATTACHMENT0, pygl.GL_COLOR_ATTACHMENT0+1)
        f.glDrawBuffers(len(draw_buffers), draw_buffers)
        # f.glViewport(0, 0, vp_size.width(), vp_size.height())
        # print("max vp", f.glGetIntegerv(pygl.GL_MAX_VIEWPORT_DIMS))

        QOpenGLFramebufferObject.bindDefault()

    def paintGL(self):
        # print("paintGL")
        volume_view = self.gldw.volume_view
        if volume_view is None :
            return
        
        f = self.gl
        f.glClearColor(.6,.3,.3,1.)
        f.glClear(pygl.GL_COLOR_BUFFER_BIT)
        self.paintSlice()

    # assumes the image is from fragment_fbo, and that
    # fragment_fbo was created with the RGBA16 format
    def npArrayFromQImage(self, im):
        # Because fragment_fbo was created with an
        # internal texture format of RGBA16 (see the code
        # where fragment_fbo was created), the QImage
        # created by toImage is in QImage format 27, which is 
        # "a premultiplied 64-bit halfword-ordered RGBA format (16-16-16-16)"
        # The "premultiplied" means that the RGB values have already
        # been multiplied by alpha.
        # This comment is based on:
        # https://doc.qt.io/qt-5/qimage.html
        # https://doc.qt.io/qt-5/qopenglframebufferobject.html

        # conversion to numpy array based on
        # https://stackoverflow.com/questions/19902183/qimage-to-numpy-array-using-pyside
        # print("im format", im.format())
        iw = im.width()
        ih = im.height()
        iptr = im.constBits()
        iptr.setsize(im.sizeInBytes())
        # make copy because the buffer from im will be deleted
        # at some point
        arr = np.frombuffer(iptr, dtype=np.uint16).copy()
        arr.resize(ih, iw, 4)
        return arr

    def drawFragments(self):
        # print("entering draw fragments")
        timera = Utils.Timer()
        timera.active = False
        self.fragment_fbo.bind()
        f = self.gl

        # Be sure to clear with alpha = 0
        # so that the slice view isn't blocked!
        f.glClearColor(0.,0.,0.,0.)
        f.glClear(pygl.GL_COLOR_BUFFER_BIT)

        '''
        # Aargh!  PyQt5 does not define glClearBufferfv!
        # And it isn't clear how to call it in PySide6
        # f.glClearBufferfv(int(pygl.GL_COLOR), int(0), [.3, .6, .3, 1.])
        seq = [float(.3), float(.6), float(.3), float(1.)]
        # seq = collections.abc.Sequence([.3, .6, .3, 1.])
        print("seq type", type(seq), type(collections.abc.Sequence[float]))
        print("isinstance", isinstance(seq, collections.abc.Sequence))
        f.glClearBufferfv(int(pygl.GL_COLOR), int(0), seq)
        '''

        dw = self.gldw
        axstr = "(%d) "%dw.axis
        ww = dw.size().width()
        wh = dw.size().height()
        opacity = dw.getDrawOpacity("overlay")
        volume_view = dw.volume_view
        xform = QMatrix4x4()

        iind = dw.iIndex
        jind = dw.jIndex
        kind = dw.kIndex
        zoom = dw.getZoom()
        cijk = volume_view.ijktf

        zf = .8
        # zoom *= .8

        # Convert tijk coordinates to OpenGL clip-window coordinates.
        # Note that the matrix converts the axis coordinate such that
        # only points within .5 voxel width on either side are
        # in the clip-window range -1. < z < 1.
        mat = np.zeros((4,4), dtype=np.float32)
        ww = dw.size().width()
        wh = dw.size().height()
        # print("w h", ww, wh)
        wf = zoom/(.5*ww)
        hf = zoom/(.5*wh)
        df = 1/.5
        ''''''
        mat[0][iind] = wf
        mat[0][3] = -wf*cijk[iind]
        mat[1][jind] = -hf
        mat[1][3] = hf*cijk[jind]
        '''
        mat[0][iind] = wf
        mat[0][3] = -wf*cijk[iind]
        mat[1][jind] = -hf
        mat[1][3] = hf*cijk[jind]
        '''
        mat[2][kind] = df
        mat[2][3] = -df*cijk[kind]
        mat[3][3] = 1.
        xform = QMatrix4x4(mat.flatten().tolist())

        '''
        for i in range(4):
            print(xform.row(i))
        '''

        apply_line_opacity = dw.getDrawApplyOpacity("line")
        line_alpha = 1.
        if apply_line_opacity:
            line_alpha = opacity
        line_thickness = dw.getDrawWidth("line")
        line_thickness = (3*line_thickness)//2

        self.fragment_trgls_program.bind()
        self.fragment_trgls_program.setUniformValue("xform", xform)
        self.fragment_trgls_program.setUniformValue("window_size", dw.size())
        # self.fragment_trgls_program.setUniformValue("thickness", float(1.*line_thickness))
        tloc = self.fragment_trgls_program.uniformLocation("thickness")
        # test whether glGetUniformBlockIndex is available
        # pid = self.fragment_trgls_program.programId()
        # bloc = f.glGetUniformBlockIndex(pid, "asdf")
        # print("tloc", tloc)
        # BUG in PySide6: calls glUniform1i instead of glUniform1f
        # self.fragment_trgls_program.setUniformValue(tloc, (1.*line_thickness,))
        f.glUniform1f(tloc, 1.*line_thickness)
        # print("tloc2")
        timera.time(axstr+"setup")
        new_fragment_vaos = {}
        self.indexed_fvs = []
        lines = []
        for fv in dw.fragmentViews():
            if not fv.visible:
                continue
            if line_thickness == 0 or line_alpha == 0:
                continue
            # self.fragment_trgls_program.setUniformValue("icolor", 1.,0.,0.,1.)
            if fv not in self.fragment_vaos:
                fvao = FragmentVao(fv, self.position_location, self.gl)
                self.fragment_vaos[fv] = fvao
            fvao = self.fragment_vaos[fv]
            new_fragment_vaos[fv] = fvao
            self.indexed_fvs.append(fv)

            if fvao.is_line:
                lines.append(fv)
                continue
            qcolor = fv.fragment.color
            rgba = list(qcolor.getRgbF())
            rgba[3] = line_alpha
            iindex = len(self.indexed_fvs)
            findex = iindex/65535.
            self.fragment_trgls_program.setUniformValue("gcolor", *rgba)
            self.fragment_trgls_program.setUniformValue("icolor", findex,0.,0.,1.)
            vao = fvao.getVao()
            vao.bind()

            f.glDrawElements(pygl.GL_TRIANGLES, fvao.trgl_index_size, 
                             pygl.GL_UNSIGNED_INT, VoidPtr(0))
            vao.release()
        self.fragment_trgls_program.release()

        if len(lines) > 0:
            self.fragment_lines_program.bind()
            self.fragment_lines_program.setUniformValue("xform", xform)
            self.fragment_lines_program.setUniformValue("window_size", dw.size())
            self.fragment_lines_program.setUniformValue("thickness", 1.*line_thickness)
            for fv in lines:
                fvao = self.fragment_vaos[fv]
    
                qcolor = fv.fragment.color
                rgba = list(qcolor.getRgbF())
                rgba[3] = line_alpha
                iindex = self.indexed_fvs.index(fv)
                findex = iindex/65535.
                self.fragment_trgls_program.setUniformValue("gcolor", *rgba)
                self.fragment_trgls_program.setUniformValue("icolor", findex,0.,0.,1.)
                vao = fvao.getVao()
                vao.bind()
    
                f.glDrawElements(pygl.GL_LINE_STRIP, fvao.trgl_index_size, 
                                 pygl.GL_UNSIGNED_INT, VoidPtr(0))
                vao.release()

            self.fragment_lines_program.release()

        timera.time(axstr+"draw lines")

        apply_node_opacity = dw.getDrawApplyOpacity("node")
        node_alpha = 1.
        if apply_node_opacity:
            node_alpha = opacity
        default_node_thickness = dw.getDrawWidth("node")
        free_node_thickness = dw.getDrawWidth("free_node")
        # node_thickness *= 2
        # node_thickness = int(node_thickness)

        self.fragment_pts_program.bind()
        self.fragment_pts_program.setUniformValue("xform", xform)
        highlight_node_color = [c/65535 for c in dw.highlightNodeColor]
        highlight_node_color[3] = node_alpha
        self.fragment_pts_program.setUniformValue("highlight_node_color", *highlight_node_color)

        dw.cur_frag_pts_xyijk = None
        dw.cur_frag_pts_fv = []
        xyptslist = []
        # if node_thickness > 0:
        pv = dw.window.project_view
        # nearby_node = (pv.nearby_node_fv, pv.nearby_node_index)
        dw.nearbyNode = -1
        i0 = 0
        for fv in dw.fragmentViews():
            if not fv.visible:
                continue
            node_thickness = default_node_thickness
            if not fv.mesh_visible:
                node_thickness = free_node_thickness
            # in OpenCV, node_thickness is the radius
            node_thickness *= 2

            if node_thickness == 0 or node_alpha == 0:
                continue
            f.glPointSize(node_thickness)

            color = dw.nodeColor
            if not fv.active:
                color = dw.inactiveNodeColor
            if not fv.mesh_visible:
                color = fv.fragment.cvcolor
            rgba = [c/65535 for c in color]
            rgba[3] = node_alpha
            # print(color, rgba)
            self.fragment_pts_program.setUniformValue("node_color", *rgba)

            nearby_node_id = 2**30
            pts = fv.getPointsOnSlice(dw.axis, dw.positionOnAxis())
            # print(fv.fragment.name, pts.shape)
            if fv == pv.nearby_node_fv:
                ind = pv.nearby_node_index
                nz = np.nonzero(pts[:,3] == ind)[0]
                if len(nz) > 0:
                    ind = nz[0]
                    self.nearbyNode = i0 + ind
                    nearby_node_id = int(pts[ind,3])
                    # print("nearby node", len(nz), nz, self.nearbyNode, pts[nz, 3])

            i0 += len(pts)
            # print("nni", self.fragment_pts_program.uniformLocation("nearby_node_id"))
            # Can't use because of a BUG in PySide6
            # self.fragment_pts_program.setUniformValue("nearby_node_id", nearby_node_id)
            nniloc = self.fragment_pts_program.uniformLocation("nearby_node_id")
            # print("nniloc", nniloc, nearby_node_id)
            self.fragment_pts_program.setUniformValue(nniloc, int(nearby_node_id))

            ijs = dw.tijksToIjs(pts)
            xys = dw.ijsToXys(ijs)
            # print(pts.shape, xys.shape)
            xypts = np.concatenate((xys, pts), axis=1)
            xyptslist.append(xypts)
            dw.cur_frag_pts_fv.extend([fv]*len(pts))

            if fv not in self.fragment_vaos:
                fvao = FragmentVao(fv, self.position_location, self.gl)
                self.fragment_vaos[fv] = fvao
            fvao = self.fragment_vaos[fv]
            new_fragment_vaos[fv] = fvao
            vao = fvao.getVao()
            vao.bind()

            # print("drawing", node_thickness, fvao.pts_size)
            f.glDrawArrays(pygl.GL_POINTS, 0, fvao.pts_size)
            vao.release()

        self.fragment_pts_program.release()
        if len(xyptslist) > 0:
            dw.cur_frag_pts_xyijk = np.concatenate(xyptslist, axis=0)
        else:
            dw.cur_frag_pts_xyijk = np.zeros((0,5), dtype=np.float32)
        # print("pts", len(dw.cur_frag_pts_xyijk))

        timera.time(axstr+"draw points")
        self.fragment_vaos = new_fragment_vaos

        ''''''
        QOpenGLFramebufferObject.bindDefault()
        # self.getPicks()
        # self.frag_last_change = time.time()

    # The toImage() call in this routine can be time-consuming,
    # since it requires the GPU to pause and export data.
    # But the result is not needed after every drawSlice call;
    # it is sufficient to call getPicks once a second or so.
    # So call getPicks from a QTimer instead of from inside 
    # drawFragments.
    def getPicks(self):
        if self.fragment_fbo is None:
            return
        # if self.frag_last_change < self.frag_last_check:
        #     return
        # self.frag_last_check = time.time()
        # print(self.frag_last_check)
        dw = self.gldw
        f = self.gl
        timerb = Utils.Timer()
        timerb.active = False
        axstr = "(%d) "%dw.axis
        self.fragment_fbo.bind()
        # "True" means that the image should be flipped to convert
        # from OpenGl's y-upwards convention to QImage's y-downwards
        # convention.
        # "1" means use drawing-attachment 1, which is the
        # texture containing icolor (index) information
        im = self.fragment_fbo.toImage(True, 1)
        timerb.time(axstr+"get image")

        arr = self.npArrayFromQImage(im)

        # In the loop above, findex (iindex/65535) is stored in 
        # the red color component (element 0), thus the 0 here.
        pts = np.nonzero(arr[:,:,0] > 0)
        nonzeros = np.concatenate(
                (pts[1].reshape(-1,1), 
                 pts[0].reshape(-1,1), 
                 arr[pts[0], pts[1]]), 
                axis=1)

        self.xyfvs = np.zeros((nonzeros.shape[0], 4), nonzeros.dtype)
        ratio = self.screen().devicePixelRatio()
        # print("arr", arr.shape, self.width(), self.height(), ratio)
        self.xyfvs[:,0:2] = nonzeros[:,0:2].astype(np.float32)/ratio
        # Subtract 1 from value in nonzeros, 
        # because the stored iindex value starts at 1, not 0.
        self.xyfvs[:,2] = nonzeros[:,2] - 1
        msid = nonzeros[:,3]
        lsid = nonzeros[:,4]
        trglid = msid*65536 + lsid
        self.xyfvs[:,3] = trglid
        # print("nz", nonzeros.shape, self.xyfvs.shape)

        # print("ijv", ijv.shape)
        # print(frag_points[0], frag_points[-1])

        # print("pa max",pick_array.max())
        QOpenGLFramebufferObject.bindDefault()
        # print("leaving drawFragments")
        timerb.time(axstr+"done")


    # Create a texture map from data (a numpy arrary), by first
    # converting it to the QImage format specified by qiformat,
    # and then creating a texture map from the QImage.
    # On of the main purposes of this function is to set
    # defaults that are suitable for this program.
    def texFromData(self, data, qiformat):
        bytesperline = (data.size*data.itemsize)//data.shape[0]
        img = QImage(data, data.shape[1], data.shape[0],
                     bytesperline, qiformat)
        # mirror image vertically because of different y direction conventions
        tex = QOpenGLTexture(img.mirrored(), 
                             QOpenGLTexture.DontGenerateMipMaps)
        tex.setWrapMode(QOpenGLTexture.DirectionS, 
                        QOpenGLTexture.ClampToBorder)
        tex.setWrapMode(QOpenGLTexture.DirectionT, 
                        QOpenGLTexture.ClampToBorder)
        tex.setMagnificationFilter(QOpenGLTexture.Nearest)
        return tex

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
        aw = dw.getDrawWidth("axes")
        if aw > 0:
            axis_color = dw.axisColor(dw.axis)
            fij = dw.tijkToIj(volume_view.ijktf)
            fx,fy = dw.ijToXy(fij)
            alpha = 1.
            if dw.getDrawApplyOpacity("axes"):
                alpha = opacity
            alpha16 = int(alpha*65535)
            icolor = dw.axisColor(dw.iIndex)
            icolor[3] = alpha16
            cv2.line(data, (fx,0), (fx,wh), icolor, aw)
            jcolor = dw.axisColor(dw.jIndex)
            jcolor[3] = alpha16
            cv2.line(data, (0,fy), (ww,fy), jcolor, aw)

    def areVolBoxesVisible(self):
        return self.gldw.window.getVolBoxesVisible()

    def drawOverlays(self, data):
        dw = self.gldw
        volume_view = dw.volume_view
        opacity = dw.getDrawOpacity("overlay")

        lw = dw.getDrawWidth("labels")
        alpha = 1.
        if dw.getDrawApplyOpacity("labels"):
            alpha = opacity
        alpha16 = int(alpha*65535)
        dww = dw.window
        # if dww.getVolBoxesVisible():
        if self.areVolBoxesVisible():
            cur_vol_view = dww.project_view.cur_volume_view
            cur_vol = dww.project_view.cur_volume
            for vol, vol_view in dww.project_view.volumes.items():
                if vol == cur_vol:
                    continue
                gs = vol.corners()
                minxy, maxxy, intersects_slice = dw.cornersToXY(gs)
                if not intersects_slice:
                    continue
                color = vol_view.cvcolor
                color[3] = alpha16
                cv2.rectangle(data, minxy, maxxy, color, 2)
        tiff_corners = dww.tiff_loader.corners()
        if tiff_corners is not None:
            # print("tiff corners", tiff_corners)

            minxy, maxxy, intersects_slice = dw.cornersToXY(tiff_corners)
            if intersects_slice:
                # tcolor is a string
                tcolor = dww.tiff_loader.color()
                qcolor = QColor(tcolor)
                rgba = qcolor.getRgbF()
                cvcolor = [int(65535*c) for c in rgba]
                cvcolor[3] = alpha16
                cv2.rectangle(outrgbx, minxy, maxxy, cvcolor, 2)
        
        if lw > 0:
            label = dw.sliceGlobalLabel()
            gpos = dw.sliceGlobalPosition()
            # print("label", self.axis, label, gpos)
            txt = "%s: %d" % (label, gpos)
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
        dw = self.gldw
        volume_view = dw.volume_view
        f = self.gl
        self.slice_program.bind()

        # viewing window width
        ww = self.size().width()
        wh = self.size().height()
        # viewing window half width
        whw = ww//2
        whh = wh//2

        data_slice = np.zeros((wh,ww), dtype=np.uint16)
        zarr_max_width = self.gldw.getZarrMaxWidth()
        paint_result = volume_view.paintSlice(
                data_slice, self.gldw.axis, volume_view.ijktf, 
                self.gldw.getZoom(), zarr_max_width)

        base_tex = self.texFromData(data_slice, QImage.Format_Grayscale16)
        bloc = self.slice_program.uniformLocation("base_sampler")
        if bloc < 0:
            print("couldn't get loc for base sampler")
            return
        # print("bloc", bloc)
        bunit = 1
        f.glActiveTexture(pygl.GL_TEXTURE0+bunit)
        base_tex.bind()
        self.slice_program.setUniformValue(bloc, bunit)

        underlay_data = np.zeros((wh,ww,4), dtype=np.uint16)
        self.drawUnderlays(underlay_data)
        underlay_tex = self.texFromData(underlay_data, QImage.Format_RGBA64)
        uloc = self.slice_program.uniformLocation("underlay_sampler")
        if uloc < 0:
            print("couldn't get loc for underlay sampler")
            return
        uunit = 2
        f.glActiveTexture(pygl.GL_TEXTURE0+uunit)
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
        f.glActiveTexture(pygl.GL_TEXTURE0+ounit)
        overlay_tex.bind()
        self.slice_program.setUniformValue(oloc, ounit)

        # In the fragment shader of the fragment_trgls_code program, 
        # each fragment is written to two textures.  But we only
        # want each given fragment to be drawn onto one particular texture,
        # not on both.  So when drawing to the texture that we don't
        # really want to draw on, we draw a dummy fragment with alpha = 0.
        # So that this dummy fragment is effectively ignored, we
        # need to use alpha blending.
        # I'm not sure whether enabling alpha blending affects
        # only the current fbo, or whether it affects every drawing
        # operation everywhere from now on.
        f.glEnable(pygl.GL_BLEND)
        f.glBlendFunc(pygl.GL_SRC_ALPHA, pygl.GL_ONE_MINUS_SRC_ALPHA)
        self.drawFragments()
        # f.glDisable(pygl.GL_BLEND)

        self.slice_program.bind()
        floc = self.slice_program.uniformLocation("fragments_sampler")
        if floc < 0:
            print("couldn't get loc for fragments sampler")
            return
        funit = 4
        f.glActiveTexture(pygl.GL_TEXTURE0+funit)
        # only valid if texture is created using
        # addColorAttachment()
        tex_ids = self.fragment_fbo.textures()
        # print("textures", tex_ids)
        # The 0 below means to use color attachment 0 of the
        # fbo, which corresponds to the texture containing the
        # cross-section of the fragments
        fragments_tex_id = tex_ids[0]
        # fragments_tex_id = self.pick_tex.textureId()
        # testing:
        # fragments_tex_id = tex_ids[1]
        f.glBindTexture(pygl.GL_TEXTURE_2D, fragments_tex_id)
        self.slice_program.setUniformValue(floc, funit)

        f.glActiveTexture(pygl.GL_TEXTURE0)
        vaoBinder = QOpenGLVertexArrayObject.Binder(self.slice_vao)
        self.slice_program.bind()
        f.glDrawElements(pygl.GL_TRIANGLES, 
                         self.slice_indices.size, pygl.GL_UNSIGNED_INT, VoidPtr(0))
        self.slice_program.release()
        vaoBinder = None
        # Putting getPicks here seems to fix the problems
        # caused by synching between GPU and CPU
        self.getPicks()

    def closeEvent(self, e):
        print("glw widget close event")

    def destroyingContext(self):
        print("glw destroying context")
    def onLogMessage(self, head, msg):
        print(head, "log:", msg.message())
        # traceback.print_stack()

    @staticmethod
    def buildProgram(sdict):
        edict = {
            "vertex": QOpenGLShader.Vertex,
            "fragment": QOpenGLShader.Fragment,
            "geometry": QOpenGLShader.Geometry,
            "tessellation_control": QOpenGLShader.TessellationControl,
            "tessellation_evaluation": QOpenGLShader.TessellationEvaluation,
            }
        name = sdict["name"]
        program = QOpenGLShaderProgram()
        for key, code in sdict.items():
            if key not in edict:
                continue
            enum = edict[key]
            ok = program.addShaderFromSourceCode(enum, code)
            if not ok:
                print(name, key, "shader failed")
                exit()
        ok = program.link()
        if not ok:
            print(name, "link failed")
            exit()
        print("program", name, program.programId())
        return program

    def buildPrograms(self):
        self.slice_program = self.buildProgram(slice_code)
        # self.borders_program = self.buildProgram(borders_code)
        self.fragment_trgls_program = self.buildProgram(fragment_trgls_code)
        self.fragment_lines_program = self.buildProgram(fragment_lines_code)
        self.fragment_pts_program = self.buildProgram(fragment_pts_code)

    def buildSliceVao(self):
        self.slice_vao = QOpenGLVertexArrayObject()
        self.slice_vao.create()

        vloc = self.slice_program.attributeLocation("position")
        # print("vloc", vloc)
        tloc = self.slice_program.attributeLocation("vtxt")
        # print("tloc", tloc)

        self.slice_program.bind()

        f = self.gl

        vaoBinder = QOpenGLVertexArrayObject.Binder(self.slice_vao)

        # defaults to type=VertexBuffer, usage_pattern = Static Draw
        vbo = QOpenGLBuffer()
        vbo.create()
        vbo.bind()

        xyuvs_list = [
                ((-1, +1), (0., 1.)),
                ((+1, -1), (1., 0.)),
                ((-1, -1), (0., 0.)),
                ((+1, +1), (1., 1.)),
                ]
        xyuvs = np.array(xyuvs_list, dtype=np.float32)

        nbytes = xyuvs.size*xyuvs.itemsize
        # allocates space and writes xyuvs into vbo;
        # requires that vbo be bound
        vbo.allocate(xyuvs, nbytes)
        
        f.glVertexAttribPointer(
                vloc,
                xyuvs.shape[1], int(pygl.GL_FLOAT), int(pygl.GL_FALSE), 
                4*xyuvs.itemsize, VoidPtr(0))
        f.glVertexAttribPointer(
                tloc, 
                xyuvs.shape[1], int(pygl.GL_FLOAT), int(pygl.GL_FALSE), 
                4*xyuvs.itemsize, VoidPtr(2*xyuvs.itemsize))
        vbo.release()
        self.slice_program.enableAttributeArray(vloc)
        self.slice_program.enableAttributeArray(tloc)
        # print("enabled")

        # https://stackoverflow.com/questions/8973690/vao-and-element-array-buffer-state
        # Qt's name for GL_ELEMENT_ARRAY_BUFFER
        ibo = QOpenGLBuffer(QOpenGLBuffer.IndexBuffer)
        ibo.create()
        # print("ibo", ibo.bufferId())
        ibo.bind()

        indices_list = [(0,1,2), (1,0,3)]
        # notice that indices must be uint8, uint16, or uint32
        self.slice_indices = np.array(indices_list, dtype=np.uint32)
        nbytes = self.slice_indices.size*self.slice_indices.itemsize
        ibo.allocate(self.slice_indices, nbytes)

        # Order is important in next 2 lines.
        # Setting vaoBinder to None unbinds (releases) vao.
        # If ibo is unbound before vao is unbound, then
        # ibo will be detached from vao.  We don't want that!
        vaoBinder = None
        ibo.release()

    def printInfo(self):
        print("vendor", pygl.glGetString(pygl.GL_VENDOR))
        print("version", pygl.glGetString(pygl.GL_VERSION))
        # for minimum values (4.1) see:
        # https://registry.khronos.org/OpenGL/specs/gl/glspec41.core.pdf
        # starting p. 383

        # at least 16834 (4.1)
        print("max texture size", 
              pygl.glGetIntegerv(pygl.GL_MAX_TEXTURE_SIZE))
        # at least 2048 (4.1)
        print("max array texture size", 
              pygl.glGetIntegerv(pygl.GL_MAX_ARRAY_TEXTURE_LAYERS))
        # at least 2048 (4.1)
        print("max 3d texture size", 
              pygl.glGetIntegerv(pygl.GL_MAX_3D_TEXTURE_SIZE))
        # at least 16 (4.1)
        print("max texture image units", 
              pygl.glGetIntegerv(pygl.GL_MAX_TEXTURE_IMAGE_UNITS))
        print("max combined texture image units", 
              pygl.glGetIntegerv(pygl.GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS))
        print("max texture buffer size", 
              pygl.glGetIntegerv(pygl.GL_MAX_TEXTURE_BUFFER_SIZE)) 
        print("max fragment uniform blocks", 
              pygl.glGetIntegerv(pygl.GL_MAX_FRAGMENT_UNIFORM_BLOCKS)) 
        print("max uniform block size", 
              pygl.glGetIntegerv(pygl.GL_MAX_UNIFORM_BLOCK_SIZE)) 


class FragmentVao:
    def __init__(self, fragment_view, position_location, gl):
        self.fragment_view = fragment_view
        self.gl = gl
        self.vao = None
        self.vao_modified = ""
        self.is_line = False
        self.position_location = position_location
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

        self.vbo = QOpenGLBuffer()
        self.vbo.create()
        self.vbo.bind()
        pts3d = np.ascontiguousarray(fv.vpoints[:,:3], dtype=np.float32)
        self.pts_size = pts3d.size

        nbytes = pts3d.size*pts3d.itemsize
        self.vbo.allocate(pts3d, nbytes)

        vloc = self.position_location
        f = self.gl
        f.glVertexAttribPointer(
                vloc,
                pts3d.shape[1], int(pygl.GL_FLOAT), int(pygl.GL_FALSE), 
                0, VoidPtr(0))
        self.vbo.release()

        # This needs to be called while the current VAO is bound
        f.glEnableVertexAttribArray(vloc)

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
