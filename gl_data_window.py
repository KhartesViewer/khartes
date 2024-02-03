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

import numpy as np
import cv2
from utils import Utils

from data_window import DataWindow

class FragmentVao:
    def __init__(self, fragment_view, fragment_program, gl):
        self.fragment_view = fragment_view
        self.gl = gl
        self.vao = None
        self.vao_modified = ""
        self.fragment_program = fragment_program
        self.getVao()

    def getVao(self):
        if self.vao_modified >= self.fragment_view.modified:
            return self.vao

        self.fragment_program.bind()

        if self.vao is None:
            self.vao = QOpenGLVertexArrayObject()
            self.vao.create()
        self.vao.bind()
        # vaoBinder = QOpenGLVertexArrayObject.Binder(self.vao)

        self.vbo = QOpenGLBuffer()
        self.vbo.create()
        self.vbo.bind()
        fv = self.fragment_view
        pts3d = np.ascontiguousarray(fv.vpoints[:,:3], dtype=np.float32)

        '''
        xys_list = [
                ((-1, +1, 0.)),
                ((+1, -1, 0.)),
                ((-1, -1, 0.)),
                ((+1, +1, 0.)),
                ]
        pts3d = np.array(xys_list, dtype=np.float32)
        pts3d *= .5
        '''

        # pts3d /= 3000.
        '''
        pts3d[:,0] -= 300
        pts3d[:,1] -= 1200
        pts3d[:,2] -= 300
        pts3d /= 500.
        '''
        # print(pts3d.min(axis=0))
        # print(pts3d.max(axis=0))

        nbytes = pts3d.size*pts3d.itemsize
        self.vbo.allocate(pts3d, nbytes)

        vloc = self.fragment_program.attributeLocation("position")
        print("vloc", vloc)
        f = self.gl
        f.glVertexAttribPointer(
                vloc,
                pts3d.shape[1], int(f.GL_FLOAT), int(f.GL_FALSE), 
                0, 0)
        self.vbo.release()

        self.fragment_program.enableAttributeArray(vloc)

        self.ibo = QOpenGLBuffer(QOpenGLBuffer.IndexBuffer)
        self.ibo.create()
        self.ibo.bind()

        trgls = np.ascontiguousarray(fv.trgls(), dtype=np.uint32)

        '''
        indices_list = [(0,1,2), (1,0,3)]
        # notice that indices must be uint8, uint16, or uint32
        trgls = np.array(indices_list, dtype=np.uint32)
        '''

        self.trgl_index_size = trgls.size

        nbytes = trgls.size*trgls.itemsize
        self.ibo.allocate(trgls, nbytes)

        print("nodes, trgls", pts3d.shape, trgls.shape)

        self.vao_modified = Utils.timestamp()
        self.vao.release()
        # vaoBinder = None
        
        # do not release ibo before vao is released!
        self.ibo.release()

        return self.vao


class GLDataWindow(DataWindow):
    def __init__(self, window, axis):
        super(GLDataWindow, self).__init__(window, axis)
        layout = QHBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        self.setLayout(layout)
        self.glw = GLDataWindowChild(self)
        layout.addWidget(self.glw)

    def drawSlice(self):
        # print("drawSlice", self.size())
        # super(GLDataWindow, self).drawSlice()

        # The next 3 lines keep the window from shrinking too much
        # img = QImage(self.size(), QImage.Format_ARGB32)
        # pixmap = QPixmap.fromImage(img)
        # self.setPixmap(pixmap)

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
      uniform sampler2D overlay_sampler;
      uniform sampler2D fragments_sampler;
      in vec2 ftxt;
      out vec4 fColor;

      void main()
      {
        fColor = texture(base_sampler, ftxt);
        vec4 frColor = texture(fragments_sampler, ftxt);
        float alpha = frColor.a;
        fColor = (1.-alpha)*fColor + alpha*frColor;
        vec4 oColor = texture(overlay_sampler, ftxt);
        alpha = oColor.a;
        fColor = (1.-alpha)*fColor + alpha*oColor;
      }
    ''',
}

fragment_code = {
    "name": "fragment",

    "vertex": '''
      #version 410 core

      uniform mat4 xform;
      in vec3 position;
      void main() {
        gl_Position = xform*vec4(position, 1.0);
        // gl_Position = vec4(position, 1.0);
      }
    ''',

    # from https://stackoverflow.com/questions/16884423/geometry-shader-producing-gaps-between-lines/16886843
    "geometry_orig": '''
    #version 410 core

    layout(triangles) in;
    layout(line_strip, max_vertices = 3) out;

    void emitIntersection(in vec4 a, in float distA, in vec4 b, in float distB)
    {
      if (sign(distA) * sign(distB) <= 0.0f && !(sign(distA) == 0 && sign(distB) == 0))
      {
        float fa = abs(distA);
        float fb = abs(distB);
        gl_Position = (fa * b + fb * a) / (fa + fb);
        EmitVertex();
      }
    }

    void main()
    {
      float dist[3];
      for (int i=0; i<3; i++)
        // dist[i] = dot(gl_in[i].gl_Position, plane);
        dist[i] = gl_in[i].gl_Position.z;

      // Find the smallest i where vertex i is below and vertex i+1 is above the plane.
      ivec3 ijk = ivec3(0, 1, 2); // use swizzle to permute the indices
      for (int i=0; i < 3 && (dist[ijk.x] > 0 || dist[ijk.y] < 0); ijk=ijk.yzx, i++);

      emitIntersection(gl_in[ijk.x].gl_Position, dist[ijk.x], gl_in[ijk.y].gl_Position, dist[ijk.y]);
      emitIntersection(gl_in[ijk.y].gl_Position, dist[ijk.y], gl_in[ijk.z].gl_Position, dist[ijk.z]);
      emitIntersection(gl_in[ijk.z].gl_Position, dist[ijk.z], gl_in[ijk.x].gl_Position, dist[ijk.x]);
    }
    ''',

    # modified from https://stackoverflow.com/questions/16884423/geometry-shader-producing-gaps-between-lines/16886843
    "geometry": '''
    #version 410 core

    uniform float thickness;
    uniform vec2 size;
    uniform int vcount = 10;

    layout(triangles) in;
    // layout(line_strip, max_vertices = 3) out;
    // layout(triangle_strip, max_vertices = 18) out;
    layout(triangle_strip, max_vertices = 18) out;
    // layout(triangle_strip, max_vertices = 4) out;

    /*
    void emitIntersection(in vec4 a, in float distA, in vec4 b, in float distB)
    {
      if (sign(distA) * sign(distB) <= 0.0f && !(sign(distA) == 0 && sign(distB) == 0))
      {
        float fa = abs(distA);
        float fb = abs(distB);
        gl_Position = (fa * b + fb * a) / (fa + fb);
        EmitVertex();
      }
    }
    */

    void main()
    {
      float dist[3];
      for (int i=0; i<3; i++)
        // dist[i] = dot(gl_in[i].gl_Position, plane);
        dist[i] = gl_in[i].gl_Position.z;
      int j = 0;
      // Have to go through nodes in the correct order.
      // Imagine a triangle a,b,c, with distances
      // a = -1, b = 0, c = 1.  In this case, there
      // are two intersections: one at point b, and one on
      // the line between a and c.
      // All three lines (ab, bc, ca) will have intersections,
      // with the lines ab and bc both having the same intersection,
      // at point b.
      // If the lines are scanned in that order, and only the first
      // two detected intersections are stored, then the two detected
      // intersections will both be point b!
      // There are various ways to detect and avoid this problem,
      // but the method below seems the least convoluted.
      
      // Find the smallest i where vertex i and vertex i+1 
      // are on opposite sides of the plane
      ivec3 ijk = ivec3(0, 1, 2); // use swizzle to permute the indices
      for (int i=0; i < 3 && (dist[ijk.x]*dist[ijk.y] >= 0); ijk=ijk.yzx, i++);

      vec4 pcs[2];
      // int zcnt = 0;
      for (int i=0; i<3 && j<2; ijk=ijk.yzx, i++) {
      // for (int i=0; i<3; ijk=ijk.yzx, i++) {
        float da = dist[ijk.x];
        float db = dist[ijk.y];
        // float prod = da*db;
        if (da*db > 0 || (da == 0 && db == 0)) continue;
        // if (prod > 0 ) continue;
        /*
        if (prod == 0) {
          if (zcnt > 0) continue;
          zcnt++;
        }
        */
        vec4 pa = gl_in[ijk.x].gl_Position;
        vec4 pb = gl_in[ijk.y].gl_Position;
        float fa = abs(da);
        float fb = abs(db);
        vec4 pc = (fa * pb + fb * pa) / (fa + fb);
        pcs[j++] = pc;
        /*
        if (j>0) {
          vec4 prev = pcs[j-1];
          vec2 norm = (pc-pcs[1]).xy;
          norm = normalize(vec2(-norm.y, norm.x));
          vec4 offset = thickness*vec4(norm.x/size.x, norm.y/size.y, 0., 0.);
        }
        */
        // gl_Position = pc;
        // EmitVertex();

      }
      if (j<2) return;
      vec2 tan = (pcs[1]-pcs[0]).xy;
      tan = normalize(tan);
      vec2 norm = vec2(-tan.y, tan.x);
      float xfactor = thickness/size.x;
      float yfactor = thickness/size.y;
      if (vcount == 18) {
        vec4 offsets[8];
        for (int i=0; i<8; i++) {
          float deg = i*45;
          float rad = radians(deg);
          vec2 raw_offset = -cos(rad)*tan + sin(rad)*norm;
          vec4 scaled_offset = 
            vec4(xfactor*raw_offset.x, yfactor*raw_offset.y, 0., 0.);
          offsets[i] = scaled_offset;
        }
        vec4 opts[18];
        int i = 0;
        opts[i++] = pcs[0];
        opts[i++] = pcs[0]+offsets[6];
        opts[i++] = pcs[0]+offsets[7];
        opts[i++] = pcs[0];
        opts[i++] = pcs[0]+offsets[0];
        opts[i++] = pcs[0]+offsets[1];
        opts[i++] = pcs[0];
        opts[i++] = pcs[0]+offsets[2];
        opts[i++] = pcs[1];
        opts[i++] = pcs[1]+offsets[2];
        opts[i++] = pcs[1]+offsets[3];
        opts[i++] = pcs[1];
        opts[i++] = pcs[1]+offsets[4];
        opts[i++] = pcs[1]+offsets[5];
        opts[i++] = pcs[1];
        opts[i++] = pcs[1]+offsets[6];
        opts[i++] = pcs[0];
        opts[i++] = pcs[0]+offsets[6];
        for (int i=0; i<18; i++) {
          gl_Position = opts[i];
          EmitVertex();
        }
      } else if (vcount == 10) {
        vec4 offsets[8];
        for (int i=0; i<8; i++) {
          float deg = i*45;
          float rad = radians(deg);
          vec2 raw_offset = -cos(rad)*tan + sin(rad)*norm;
          vec4 scaled_offset = 
            vec4(xfactor*raw_offset.x, yfactor*raw_offset.y, 0., 0.);
          offsets[i] = scaled_offset;
        }
        vec4 opts[10];
        int i = 0;
        opts[i++] = pcs[0]+offsets[0];
        opts[i++] = pcs[0]+offsets[1];
        opts[i++] = pcs[0]+offsets[7];
        opts[i++] = pcs[0]+offsets[2];
        opts[i++] = pcs[0]+offsets[6];
        opts[i++] = pcs[1]+offsets[2];
        opts[i++] = pcs[1]+offsets[6];
        opts[i++] = pcs[1]+offsets[3];
        opts[i++] = pcs[1]+offsets[5];
        opts[i++] = pcs[1]+offsets[4];
        for (int i=0; i<10; i++) {
          gl_Position = opts[i];
          EmitVertex();
        }
      } else if (vcount == 4) {

          vec4 offset = vec4(xfactor*norm.x, yfactor*norm.y, 0., 0.);
          // vec4 offset = thickness*vec4(norm.x/size.x, norm.y/size.y, 0., 0.);
          gl_Position = pcs[0] + offset;
          EmitVertex();
          gl_Position = pcs[0] - offset;
          EmitVertex();
          gl_Position = pcs[1] + offset;
          EmitVertex();
          gl_Position = pcs[1] - offset;
          EmitVertex();
      }
      /*
      gl_Position = pcs[0];
      EmitVertex();
      gl_Position = pcs[1]+vec4(.01,0.,0.,1.);
      EmitVertex();
      gl_Position = pcs[0]+vec4(.01,0.,0.,1.);
      EmitVertex();
      gl_Position = pcs[1];
      EmitVertex();
      */


      /*
      // Find the smallest i where vertex i is below and vertex i+1 is above the plane.
      ivec3 ijk = ivec3(0, 1, 2); // use swizzle to permute the indices
      for (int i=0; i < 3 && (dist[ijk.x] > 0 || dist[ijk.y] < 0); ijk=ijk.yzx, i++);

      emitIntersection(gl_in[ijk.x].gl_Position, dist[ijk.x], gl_in[ijk.y].gl_Position, dist[ijk.y]);
      emitIntersection(gl_in[ijk.y].gl_Position, dist[ijk.y], gl_in[ijk.z].gl_Position, dist[ijk.z]);
      emitIntersection(gl_in[ijk.z].gl_Position, dist[ijk.z], gl_in[ijk.x].gl_Position, dist[ijk.x]);
      */
    }
    ''',

    "fragment": '''
      #version 410 core

      uniform vec4 color;
      out vec4 fColor;

      void main()
      {
        fColor = color;
        // fColor = vec4(1.,1.,0.,1.);
      }
    ''',
}

"""
borders_code = {
    "name": "borders",
    "vertex": '''
      #version 410 core

      in vec2 position;
      void main() {
        gl_Position = vec4(position, 0.0, 1.0);
      }
    ''',
    "fragment": '''
      #version 410 core

      uniform vec4 color;
      out vec4 fColor;

      void main()
      {
        fColor = color;
      }
    ''',
}
"""

class GLDataWindowChild(QOpenGLWidget):
    def __init__(self, gldw, parent=None):
        super(GLDataWindowChild, self).__init__(parent)
        self.gldw = gldw
        self.setMouseTracking(True)
        self.fragment_vaos = {}
        # 0: asynchronous mode, 1: synch mode
        # synch mode is much slower
        self.logging_mode = 1
        # self.logging_mode = 0

    def dwKeyPressEvent(self, e):
        self.gldw.dwKeyPressEvent(e)

    def initializeGL(self):
        print("initializeGL")
        self.context().aboutToBeDestroyed.connect(self.destroyingContext)
        self.gl = self.context().versionFunctions()
        self.main_context = self.context()
        # Note that debug logging only takes place if the
        # surface format option "DebugContext" is set
        self.logger = QOpenGLDebugLogger()
        self.logger.initialize()
        self.logger.messageLogged.connect(lambda m: self.onLogMessage("dc", m))
        self.logger.startLogging(self.logging_mode)
        msg = QOpenGLDebugMessage.createApplicationMessage("test debug messaging")
        self.logger.logMessage(msg)
        self.buildPrograms()
        self.buildSliceVao()
        # self.buildBordersVao()

        self.createGLSurfaces()
        
        f = self.gl
        # self.gl.glClearColor(.3,.6,.3,1.)
        f.glClearColor(.6,.3,.3,1.)

    def createGLSurfaces(self):
        self.fragment_context = QOpenGLContext()
        self.fragment_context.create()
        self.fragment_surface = QOffscreenSurface()
        self.fragment_surface.create()
        # Make new context current; need to undo this
        # before leaving the function
        self.fragment_context.makeCurrent(self.fragment_surface)
        # Note that debug logging only takes place if the
        # surface format option "DebugContext" is set
        self.frag_logger = QOpenGLDebugLogger()
        self.frag_logger.initialize()
        # self.frag_logger.messageLogged.connect(self.onLogMessage)
        self.frag_logger.messageLogged.connect(lambda m: self.onLogMessage("fc", m))
        self.frag_logger.startLogging(self.logging_mode)
        msg = QOpenGLDebugMessage.createApplicationMessage("test debug messaging")
        self.logger.logMessage(msg)
        self.fragment_gl = self.fragment_context.versionFunctions()
        self.fragment_program = self.buildProgram(fragment_code)
        # Restore default context
        self.makeCurrent()

    def resizeGL(self, width, height):
        # pass
        # self.buildBordersVao()
        # print("resize", width, height)
        # based on https://stackoverflow.com/questions/59338015/minimal-opengl-offscreen-rendering-using-qt
        self.fragment_context.makeCurrent(self.fragment_surface)
        vp_size = QSize(width, height)
        fbo_format = QOpenGLFramebufferObjectFormat()
        fbo_format.setAttachment(QOpenGLFramebufferObject.CombinedDepthStencil)
        ff = self.fragment_gl
        fbo_format.setInternalTextureFormat(ff.GL_RGBA16)
        self.fragment_fbo = QOpenGLFramebufferObject(vp_size, fbo_format)
        ff.glViewport(0, 0, vp_size.width(), vp_size.height())

    def paintGL(self):
        # print("paintGL")
        volume_view = self.gldw.volume_view
        if volume_view is None :
            return
        
        f = self.gl
        f.glClear(f.GL_COLOR_BUFFER_BIT)
        self.paintSlice()
        # self.paintFragments()
        # self.paintBorders()

    # def paintFragments(self):
    def drawFragments(self, fragments_overlay):
        # change current context; undo this before
        # leaving the function
        self.fragment_context.makeCurrent(self.fragment_surface)
        f = self.fragment_gl
        f.glClear(f.GL_COLOR_BUFFER_BIT)
        dw = self.gldw
        ww = dw.size().width()
        wh = dw.size().height()
        opacity = dw.getDrawOpacity("overlay")
        apply_line_opacity = dw.getDrawApplyOpacity("line")
        line_alpha = 1.
        if apply_line_opacity:
            line_alpha = opacity
        thickness = dw.getDrawWidth("line")
        thickness = (3*thickness)//2
        volume_view = dw.volume_view
        xform = QMatrix4x4()
        # xform.scale(1./12000.)
        '''
        pts3d[:,0] -= 300
        pts3d[:,1] -= 1200
        pts3d[:,2] -= 300
        pts3d /= 500.
        '''
        # xform.scale(1./500.)
        # xform.translate(-300, -1200, -300)
        iind = dw.iIndex
        jind = dw.jIndex
        kind = dw.kIndex
        zoom = dw.getZoom()
        cijk = volume_view.ijktf
        '''
        xform.setRow(0, QVector4D(.002,0,0,-.6))
        xform.setRow(1, QVector4D(0,.002,0,-2.4))
        xform.setRow(2, QVector4D(0,0,.002,-.6))
        '''
        '''
        mat = np.array((
            (.002,0,0,-.6),
            (0,.002,0,-2.4),
            (0,0,.002,-.6),
            (0,0,0,1.)
            ))
        '''
        mat = np.zeros((4,4), dtype=np.float32)
        ww = dw.size().width()
        wh = dw.size().height()
        wf = zoom/(.5*ww)
        hf = zoom/(.5*wh)
        df = 1/.5
        mat[0][iind] = wf
        mat[0][3] = -wf*cijk[iind]
        mat[1][jind] = -hf
        mat[1][3] = hf*cijk[jind]
        mat[2][kind] = df
        mat[2][3] = -df*cijk[kind]
        mat[3][3] = 1.
        xform = QMatrix4x4(mat.flatten().tolist())

        '''
        for i in range(4):
            print(xform.row(i))
        '''
        self.fragment_program.bind()
        self.fragment_program.setUniformValue("xform", xform)
        self.fragment_program.setUniformValue("size", dw.size())
        self.fragment_program.setUniformValue("thickness", 1.*thickness)
        colors = []
        colors.append((0.,0.,0.,0.))
        for fv in dw.fragmentViews():
            if not fv.visible:
                continue
            qcolor = fv.fragment.color
            rgba = list(qcolor.getRgbF())
            # print("rgba", rgba, [65535*c for c in rgba])
            cvcolor = [int(65535*c) for c in rgba]
            cvcolor[3] = int(line_alpha*65535)
            # rgba[3] = 1.
            findex = len(colors)/65536.
            colors.append(cvcolor)
            # self.fragment_program.bind()
            self.fragment_program.setUniformValue("color", findex,0.,0.,1.)
            if fv not in self.fragment_vaos:
                fvao = FragmentVao(fv, self.fragment_program, self.gl)
                self.fragment_vaos[fv] = fvao
            fvao = self.fragment_vaos[fv]
            vao = fvao.getVao()
            vao.bind()
            # vaoBinder = QOpenGLVertexArrayObject.Binder(vao)
            self.fragment_program.setUniformValue("xform", xform)

            # print("tis", fvao.trgl_index_size)
            f.glDrawElements(f.GL_TRIANGLES, fvao.trgl_index_size, 
                             f.GL_UNSIGNED_INT, None)
            vao.release()
            # vaoBinder = None
            # self.fragment_program.release()
        self.fragment_program.release()
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
        im = self.fragment_fbo.toImage()
        # print("image format, size", im.format(), im.size(), im.sizeInBytes())
        # im.save("test.png")

        # conversion to numpy array based on
        # https://stackoverflow.com/questions/19902183/qimage-to-numpy-array-using-pyside
        iw = im.width()
        ih = im.height()
        iptr = im.constBits()
        iptr.setsize(im.sizeInBytes())
        arr = np.frombuffer(iptr, dtype=np.uint16)
        arr.resize(ih, iw, 4)
        # farr = arr.flatten()
        # am = farr.argmax()
        # print("arr", arr.shape, arr.dtype, farr[0:16], farr[am-4:am+16])
        # self.fragment_lines_arr = np.zeros_like(arr)
        acolors = np.array(colors)
        # self.fragment_lines_arr[:,:] = acolors[arr[:,:,0]]
        fragments_overlay[:,:] = acolors[arr[:,:,0]]
        # farr = self.fragment_lines_arr.flatten()
        # am = farr.argmax()
        # print("farr", farr.dtype, farr[0:16], farr[am-4:am+16])

        # restore default context
        self.makeCurrent()

    def texFromData(self, data, qiformat):
        bytesperline = (data.size*data.itemsize)//data.shape[0]
        img = QImage(data, data.shape[1], data.shape[0],
                     bytesperline, qiformat)
        # When tex goes out of scope (at the end of this
        # function), the OpenGL texture will be destroyed.
        # mirror image vertically because of different y direction conventions
        tex = QOpenGLTexture(img.mirrored(), 
                             QOpenGLTexture.DontGenerateMipMaps)
        tex.setWrapMode(QOpenGLTexture.DirectionS, 
                        QOpenGLTexture.ClampToBorder)
        tex.setWrapMode(QOpenGLTexture.DirectionT, 
                        QOpenGLTexture.ClampToBorder)
        tex.setMagnificationFilter(QOpenGLTexture.Nearest)
        return tex

    def drawOverlays(self, data):
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
        lw = dw.getDrawWidth("labels")
        alpha = 1.
        if dw.getDrawApplyOpacity("labels"):
            alpha = opacity
        alpha16 = int(alpha*65535)
        dww = dw.window
        if dww.getVolBoxesVisible():
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
                cv2.rectangle(outrgbx, minxy, maxxy, color, 2)
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
        f.glActiveTexture(f.GL_TEXTURE0+bunit)
        base_tex.bind()
        self.slice_program.setUniformValue(bloc, bunit)


        overlay_data = np.zeros((wh,ww,4), dtype=np.uint16)
        self.drawOverlays(overlay_data)
        overlay_tex = self.texFromData(overlay_data, QImage.Format_RGBA64)
        oloc = self.slice_program.uniformLocation("overlay_sampler")
        if oloc < 0:
            print("couldn't get loc for overlay sampler")
            return
        ounit = 2
        f.glActiveTexture(f.GL_TEXTURE0+ounit)
        overlay_tex.bind()
        self.slice_program.setUniformValue(oloc, ounit)


        fragments_data = np.zeros((wh,ww,4), dtype=np.uint16)
        self.drawFragments(fragments_data)
        fragments_tex = self.texFromData(fragments_data, QImage.Format_RGBA64)
        floc = self.slice_program.uniformLocation("fragments_sampler")
        if floc < 0:
            print("couldn't get loc for fragments sampler")
            return
        funit = 3
        f.glActiveTexture(f.GL_TEXTURE0+funit)
        fragments_tex.bind()
        self.slice_program.setUniformValue(floc, funit)


        f.glActiveTexture(f.GL_TEXTURE0)
        # base_tex.release()
        # overlay_tex.release()
        vaoBinder = QOpenGLVertexArrayObject.Binder(self.slice_vao)
        self.slice_program.bind()
        f.glDrawElements(f.GL_TRIANGLES, 
                         self.slice_indices.size, f.GL_UNSIGNED_INT, None)
        self.slice_program.release()
        vaoBinder = None

    '''
    def paintBorders(self):
        f = self.gl
        dw = self.gldw

        f.glEnable(f.GL_BLEND)
        f.glBlendFunc(f.GL_SRC_ALPHA, f.GL_ONE_MINUS_SRC_ALPHA)

        ww = dw.size().width()
        wh = dw.size().height()
        opacity = dw.getDrawOpacity("overlay")
        apply_borders_opacity = dw.getDrawApplyOpacity("borders")
        bw = dw.getDrawWidth("borders")
        bwh = (bw-1)//2
        bwhf = (bw)/2

        cmin = (bwh+2, bwh+2)
        cmax = (ww-bwh-1, wh-bwh-1)
        cmm = np.array((cmin,cmax), dtype=np.float32)

        sgns = np.array(((0,0),(0,1),(1,0),(1,1)))

        cxys = np.zeros((4,2), dtype=np.float32)
        cxys[:,0] = cmm[:,0][sgns[:,0]]
        cxys[:,1] = cmm[:,1][sgns[:,1]]

        axys = cxys+(sgns*2-1)*bwhf
        bxys = cxys-(sgns*2-1)*bwhf

        xys = np.zeros((8,2), dtype=np.float32)
        xys[0:4] = axys
        xys[4:8] = bxys

        xys[:,0] = 2*xys[:,0]/ww - 1
        xys[:,1] = 2*xys[:,1]/wh - 1

        self.slice_vbo.bind()
        self.slice_vbo.write(0, xys, xys.size*xys.itemsize)
        
        # f.glEnable(f.GL_LINE_SMOOTH)
        # print("lw", f.glGetFloatv(f.GL_SMOOTH_LINE_WIDTH_RANGE))
        # Oops, deprecated!!
        # f.glLineWidth(2.)

        vaoBinder = QOpenGLVertexArrayObject.Binder(self.borders_vao)
        self.borders_program.bind()

        axis_color = self.fAxisColor(self.gldw.axis)
        alpha = 1.
        if apply_borders_opacity:
            alpha = opacity
        axis_color[3] = alpha
        self.borders_program.setUniformValue("color", *axis_color)

        # f.glDrawElements(f.GL_LINE_LOOP, 
        # TODO: testing
        f.glDrawElements(f.GL_TRIANGLES, 
                         self.borders_indices.size, f.GL_UNSIGNED_INT, None)
        self.slice_program.release()
        vaoBinder = None
    '''

    def closeEvent(self, e):
        print("glw widget close event")

    def destroyingContext(self):
        print("glw destroying context")

    def onLogMessage(self, head, msg):
        print(head, "log:", msg.message())

    def buildProgram(self, sdict):
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
        return program

    def buildPrograms(self):
        self.slice_program = self.buildProgram(slice_code)
        # self.borders_program = self.buildProgram(borders_code)
        # self.fragment_program = self.buildProgram(fragment_code)

    """
    def buildBordersVao(self):
        self.borders_vao = QOpenGLVertexArrayObject()
        self.borders_vao.create()
        vloc = self.borders_program.attributeLocation("position")
        self.borders_program.bind()
        f = self.gl
        vaoBinder = QOpenGLVertexArrayObject.Binder(self.borders_vao)

        # defaults to type=VertexBuffer, usage_pattern = Static Draw
        vbo = QOpenGLBuffer()
        vbo.create()
        vbo.bind()

        '''
        xys_list = [
                (-1, -1),
                (-1, +1),
                (+1, -1),
                (+1, +1),
                ]
        '''
        '''
        xys_list = [
                (-.9999, -.9999),
                (-.9999, +1),
                (+1, -.9999),
                (+1, +1),
                ]
        '''
        # xys = np.array(xys_list, dtype=np.float32)
        # TODO: testing
        # xys *= .5
        xys = np.zeros((8, 2), dtype=np.float32)
        nbytes = xys.size*xys.itemsize
        vbo.allocate(xys, nbytes)
        f.glVertexAttribPointer(
                vloc,
                xys.shape[1], int(f.GL_FLOAT), int(f.GL_FALSE), 
                0, 0)
        vbo.release()
        self.slice_vbo = vbo
        # self.slice_xys = xys
        self.borders_program.enableAttributeArray(vloc)

        ibo = QOpenGLBuffer(QOpenGLBuffer.IndexBuffer)
        ibo.create()
        ibo.bind()
        # TODO: testing
        # indices_list = [0,1,3,2]
        # indices_list = [0,1,3,3,0,2]
        indices_list = (
                (0,1,4), (4,1,5),
                (1,3,5), (5,3,7),
                (3,2,7), (7,2,6),
                (2,0,6), (6,0,4)
                )
        self.borders_indices = np.array(indices_list, dtype=np.uint32)
        nbytes = self.borders_indices.size*self.borders_indices.itemsize
        ibo.allocate(self.borders_indices, nbytes)
        vaoBinder = None
        ibo.release()
    """


    '''
    def fAxisColor(self, axis):
        color = self.gldw.axisColor(axis)
        fcolor = [c/65535 for c in color]
        return fcolor
    '''

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
                xyuvs.shape[1], int(f.GL_FLOAT), int(f.GL_FALSE), 
                4*xyuvs.itemsize, 0)
        f.glVertexAttribPointer(
                tloc, 
                xyuvs.shape[1], int(f.GL_FLOAT), int(f.GL_FALSE), 
                4*xyuvs.itemsize, 2*xyuvs.itemsize)
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


