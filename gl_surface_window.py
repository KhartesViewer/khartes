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


class GLSurfaceWindow(DataWindow):
    def __init__(self, window):
        super(GLSurfaceWindow, self).__init__(window, 2)
        # self.clear()
        layout = QHBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        self.setLayout(layout)
        self.glw = GLSurfaceWindowChild(self)
        layout.addWidget(self.glw)

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


trgl_code = {
    "name": "trgl",

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
      uniform vec3 xyzmin;
      uniform vec3 xyzmax;
      float dz = xyzmax.y - xyzmin.y;
      float dx = xyzmax.x - xyzmin.x;
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

    
class GLSurfaceWindowChild(QOpenGLWidget):
    def __init__(self, glsw, parent=None):
        super(GLSurfaceWindowChild, self).__init__(parent)
        self.glsw = glsw
        self.setMouseTracking(True)
        self.fragment_vaos = {}

        # 0: asynchronous mode, 1: synch mode
        # synch mode is said to be much slower
        self.logging_mode = 1

        # This corresponds to the line in the vertex shader(s):
        # layout(location=3) in vec3 xyx;
        self.xyz_location = 3
        # This corresponds to the line in the vertex shader(s):
        # layout(location=4) in vec3 stxy;
        self.stxy_location = 4

    def dwKeyPressEvent(self, e):
        self.glsw.dwKeyPressEvent(e)

    def initializeGL(self):
        print("initializeGL (surface)")
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
        # self.buildBordersVao()

        # self.createGLSurfaces()
        
        f = self.gl
        # self.gl.glClearColor(.3,.6,.3,1.)
        f.glClearColor(.6,.3,.3,1.)

        self.active_vao = None

        self.buildPrograms()
    
    def resizeGL(self, width, height):
        f = self.gl
        # print("resizeGL (surface)", width, height)
        f.glViewport(0, 0, width, height)

    def paintGL(self):
        if self.glsw.volume_view is None:
            return

        # print("paintGL (surface)")
        f = self.gl
        f.glClearColor(.6,.3,.6,1.)
        f.glClear(f.GL_COLOR_BUFFER_BIT)
        self.drawTrgls()

    def closeEvent(self, e):
        print("glsw widget close event")

    def destroyingContext(self):
        print("glsw destroying context")

    def onLogMessage(self, head, msg):
        print(head, "s log:", msg.message())

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
        self.trgl_program = self.buildProgram(trgl_code)

    def drawTrgls(self):
        f = self.gl

        # Be sure to clear with alpha = 0
        # so that the slice view isn't blocked!
        f.glClearColor(0.,0.,0.,0.)
        f.glClear(f.GL_COLOR_BUFFER_BIT)

        dw = self.glsw

        ww = dw.size().width()
        wh = dw.size().height()
        volume_view = dw.volume_view
        xform = QMatrix4x4()

        iind = dw.iIndex
        jind = dw.jIndex
        kind = dw.kIndex
        zoom = dw.getZoom()
        cijk = volume_view.ijktf

        # Convert tijk coordinates to OpenGL clip-window coordinates.
        # Note that the matrix converts the axis coordinate such that
        # only points within .5 voxel width on either side are
        # in the clip-window range -1. < z < 1.
        mat = np.zeros((4,4), dtype=np.float32)
        ww = dw.size().width()
        wh = dw.size().height()
        wf = zoom/(.5*ww)
        hf = zoom/(.5*wh)
        # df = 1/.5
        df = 0
        mat[0][iind] = wf
        mat[0][3] = -wf*cijk[iind]
        mat[1][jind] = -hf
        mat[1][3] = hf*cijk[jind]
        mat[2][kind] = df
        mat[2][3] = -df*cijk[kind]
        mat[3][3] = 1.
        xform = QMatrix4x4(mat.flatten().tolist())
        
        self.trgl_program.bind()
        self.trgl_program.setUniformValue("xform", xform)

        pv = dw.window.project_view
        mfv = pv.mainActiveFragmentView(unaligned_ok=True)
        if mfv is None:
            print("No currently active fragment")
            return

        if self.active_vao is None or self.active_vao.fragment_view != mfv:
            self.active_vao = FragmentMapVao(
                    mfv, self.xyz_location, self.stxy_location, self.gl)

        fvao = self.active_vao
        qcolor = mfv.fragment.color
        rgba = list(qcolor.getRgbF())
        rgba[3] = 1.
        self.trgl_program.setUniformValue("color", *rgba)
        self.trgl_program.setUniformValue("xyzmin", *mfv.xyzmin)
        self.trgl_program.setUniformValue("xyzmax", *mfv.xyzmax)

        vao = fvao.getVao()
        vao.bind()

        # f.glPolygonMode(f.GL_FRONT_AND_BACK, f.GL_LINE)
        f.glDrawElements(f.GL_TRIANGLES, fvao.trgl_index_size,
                       f.GL_UNSIGNED_INT, None)
        vao.release()
        self.trgl_program.release()



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


