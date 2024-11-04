import time
import random
import datetime
import re
import numpy as np
import cmap
from PyQt5.QtGui import QColorConstants as QCC
from PyQt5.QtGui import QColor
# import PySide6.QtGuiQColor.SVG as QtSVG

c1 = QCC.Svg.skyblue


class Utils:

    class Timer():

        def __init__(self, active=True):
            self.t0 = time.time()
            self.active = active

        def time(self, msg=""):
            t = time.time()
            if self.active:
                print("%.3f %s"%(t-self.t0, msg))
            self.t0 = t


    def timestamp():
        t = datetime.datetime.utcnow()
        txt = t.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        # round to hundredths of a second
        txt2 = txt[:-5]+'Z'
        # print("timer", txt, txt2)
        return txt2

    def timestampToVc(ts):
        try:
            dt = datetime.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%fZ")
        except:
            print("Couldn't parse %s as Utils timestamp"%ts)
            return None
        odt = dt.strftime("%Y%m%d%H%M%S")
        return odt

    def vcToTimestamp(vc):
        try:
            dt = datetime.datetime.strptime(vc, "%Y%m%d%H%M%S")
        except:
            print("Couldn't parse %s as VC timestamp"%vc)
            return None
        odt = dt.strftime("%Y-%m-%dT%H:%M:%S.00Z")
        return odt

    def getNextColor():
        h = random.randrange(359)
        s = random.randrange(128,255)
        l = 128
        v = 255
        color = QColor()
        # color.setHsl(h,s,l)
        color.setHsv(h,s,v)
        # rgba = color.getRgbF()
        # print(color.name())
        # cvcolor = [int(65535*c) for c in rgba]
        return color


    # TODO deal intelligently with nested '-copy'
    def nextName(name, inc):
        # print(name, inc)
        m = re.match(r'^(.*\D)(\d*)$', name)
        base = ""
        i = 0
        if m is None:
            # print("  (None)")
            i = int(name)
        else:
            gs = m.groups()
            # print("  %s|%s|"%(gs[0], gs[1]))
            base = gs[0]
            if gs[1] != "":
                i = int(gs[1])
        i += inc
        return "%s%d"%(base,i)

        
    # https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
    def updateDict(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = Utils.updateDict(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    # https://stackoverflow.com/questions/64414944/hot-to-get-the-set-difference-of-two-2d-numpy-arrays-or-equivalent-of-np-setdif
    # returns indices of rows in a that do not appear in b
    def setDiff2DIndex(a, b):
        nrows, ncols = a.shape
        # print("rows, cols",nrows, ncols)
        adt = a.dtype
        # print("a type", adt)
        # dtype is a tuple of the elements in a single row
        dtype={'names':['f{}'.format(i) for i in range(ncols)], 'formats':ncols * [adt]}
        # print("dtype", dtype)
        sa = a.copy().view(dtype)
        sb = b.copy().view(dtype)
        # print("sb", sb)
        sc = np.setdiff1d(sa, sb)
        # print("sc", sc)
        # c = sc.view(adt).reshape(-1, ncols)
        # print("c", c)
        fc = np.isin(sa, sc)
        # print("fc", fc)
        nzc = fc.nonzero()[0]
        # print(nzc)
        return nzc

    # adapted from https://stackoverflow.com/questions/25068538/intersection-and-difference-of-two-rectangles/25068722#25068722
    # The C++ version of OpenCV provides operations, including intersection,
    # on rectangles, but the Python version doesn't.
    def rectIntersection(ra, rb):
        if not Utils.rectIsValid(ra) or not Utils.rectIsValid(rb):
            return Utils.emptyRect()
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

    def rectIsValid(r):
        if r is None:
            return False
        (x1, y1), (x2, y2) = r
        if x1 >= x2:
            return False
        if y1 >= y2:
            return False
        return True

    def emptyRect():
        return (0,0),(0,0)

    def rectUnion(ra, rb):
        if not Utils.rectIsValid(ra):
            return rb
        if not Utils.rectIsValid(rb):
            return ra
        (ax1, ay1), (ax2, ay2) = ra
        (bx1, by1), (bx2, by2) = rb
        ru = ((min(ax1,bx1), min(ay1,by1)),
              (max(ax2,bx2), max(ay2,by2)))
        return ru


    class ColorMap():

        # index_range (if not None) is a two-element tuple,
        # giving the range (min and max) of the color map,
        # with min and max between 0.0 and 1.0.
        # Values outside of the range will be set to
        # transparent.
        def __init__(self, cmap_name, dtype, alpha, index_range=None):
            self.cmap_name = cmap_name
            if cmap_name == "":
                # cmap_name = "matlab:gray"
                # index_range = None
                self.lut = None
                return
            '''
            # assumes dtype is unsigned int (uint8 or uint16)
            dtmax = np.iinfo(dtype).max
            lutsize = dtmax+1
            vmin = 0
            vmax = dtmax
            '''
            # wanted lutsize to be based on the dtype,
            # but for OpenGL reasons (limitations on the
            # max size of a texture map), cannot have
            # a lut with size 65536
            lutsize = 256

            self.lut = np.zeros((lutsize, 4), dtype=np.float32)
            vmin = 0.
            vmax = 1.
            if index_range is not None:
                if index_range[0] is not None:
                    vmin = index_range[0]
                if index_range[1] is not None:
                    vmax = index_range[1]
            # imin = int(vmin*lutsize+.5)
            # imax = int(vmax*lutsize-.5)
            imin = int(vmin*(lutsize-1)+.5)
            imax = int(vmax*(lutsize-1)+.5)
            '''
            if cmap_name == "kh_encoded_555":
                self.lut[:32768, 0:3] = np.linspace(0., 1., 32768)[:, np.newaxis]
                rng = np.arange(0, 32768)
                r = (rng >> 10) & 31
                g = (rng >> 5) & 31
                b = rng & 31
                self.lut[32768:,0] = r / 31.
                self.lut[32768:,1] = g / 31.
                self.lut[32768:,2] = b / 31.
                self.lut[:,3] = alpha
            else:
                cm = cmap.Colormap(cmap_name)
                cmlut = cm.lut(vmax-vmin+1)
                self.lut[vmin:(vmax+1)] = cmlut
                self.lut[vmin:(vmax+1), 3] = alpha
            '''
            # would prefer "nearest" for indicator values,
            # but providing "nearest" returns an incorrectly
            # sized lut for certain colormaps (those that only
            # contain a few colors).
            cm = cmap.Colormap(cmap_name, interpolation="linear")
            n = imax-imin+1
            # cm.lut() is a bit buggy; it sometimes returns
            # an array of a different shape than requested!
            # cmlut = cm.lut(N=n)
            colors = np.linspace(0.,1.,n)
            cmlut = cm(colors)
            # print(imin,imax,n,cmlut.shape)
            # print(cmlut[0], cmlut[-1])
            # print(cmlut)
            self.lut[imin:(imax+1)] = cmlut
            self.lut[imin:(imax+1), 3] = alpha

    def getNextColorOld():
        Utils.colorCounter = (Utils.colorCounter+1)%len(Utils.colors)
        color = Utils.colors[Utils.colorCounter]
        color = Utils.colors[random.randrange(len(Utils.colors))]
        rgba = color.getRgbF()
        return [int(65535*c) for c in rgba]


    colorCounter = 7
    colors= [
    QCC.Svg.aliceblue,
    QCC.Svg.aqua,
    QCC.Svg.aquamarine,
    QCC.Svg.azure,
    QCC.Svg.beige,
    QCC.Svg.bisque,
    QCC.Svg.black,
    QCC.Svg.blanchedalmond,
    QCC.Svg.blue,
    QCC.Svg.blueviolet,
    QCC.Svg.brown,
    QCC.Svg.burlywood,
    QCC.Svg.cadetblue,
    QCC.Svg.chartreuse,
    QCC.Svg.chocolate,
    QCC.Svg.coral,
    QCC.Svg.cornflowerblue,
    QCC.Svg.cornsilk,
    QCC.Svg.crimson,
    QCC.Svg.cyan,
    QCC.Svg.darkblue,
    QCC.Svg.darkcyan,
    QCC.Svg.darkgoldenrod,
    QCC.Svg.darkgray,
    QCC.Svg.darkgreen,
    QCC.Svg.darkkhaki,
    QCC.Svg.darkmagenta,
    QCC.Svg.darkolivegreen,
    QCC.Svg.darkorange,
    QCC.Svg.darkorchid,
    QCC.Svg.darkred,
    QCC.Svg.darksalmon,
    QCC.Svg.darkseagreen,
    QCC.Svg.darkslateblue,
    QCC.Svg.darkslategray,
    QCC.Svg.darkturquoise,
    QCC.Svg.darkviolet,
    QCC.Svg.deeppink,
    QCC.Svg.deepskyblue,
    QCC.Svg.dimgray,
    QCC.Svg.dodgerblue,
    QCC.Svg.firebrick,
    QCC.Svg.forestgreen,
    QCC.Svg.fuchsia,
    QCC.Svg.gainsboro,
    QCC.Svg.gold,
    QCC.Svg.goldenrod,
    QCC.Svg.gray,
    QCC.Svg.green,
    QCC.Svg.greenyellow,
    QCC.Svg.honeydew,
    QCC.Svg.hotpink,
    QCC.Svg.indianred,
    QCC.Svg.indigo,
    QCC.Svg.ivory,
    QCC.Svg.khaki,
    QCC.Svg.lavender,
    QCC.Svg.lavenderblush,
    QCC.Svg.lawngreen,
    QCC.Svg.lemonchiffon,
    QCC.Svg.lightblue,
    QCC.Svg.lightcoral,
    QCC.Svg.lightcyan,
    QCC.Svg.lightgoldenrodyellow,
    QCC.Svg.lightgray,
    QCC.Svg.lightgreen,
    QCC.Svg.lightpink,
    QCC.Svg.lightsalmon,
    QCC.Svg.lightseagreen,
    QCC.Svg.lightskyblue,
    QCC.Svg.lightslategray,
    QCC.Svg.lightsteelblue,
    QCC.Svg.lightyellow,
    QCC.Svg.lime,
    QCC.Svg.limegreen,
    QCC.Svg.linen,
    QCC.Svg.magenta,
    QCC.Svg.maroon,
    QCC.Svg.mediumaquamarine,
    QCC.Svg.mediumblue,
    QCC.Svg.mediumorchid,
    QCC.Svg.mediumpurple,
    QCC.Svg.mediumseagreen,
    QCC.Svg.mediumslateblue,
    QCC.Svg.mediumspringgreen,
    QCC.Svg.mediumturquoise,
    QCC.Svg.mediumvioletred,
    QCC.Svg.midnightblue,
    QCC.Svg.mintcream,
    QCC.Svg.mistyrose,
    QCC.Svg.moccasin,
    QCC.Svg.navy,
    QCC.Svg.oldlace,
    QCC.Svg.olive,
    QCC.Svg.olivedrab,
    QCC.Svg.orange,
    QCC.Svg.orangered,
    QCC.Svg.orchid,
    QCC.Svg.palegoldenrod,
    QCC.Svg.palegreen,
    QCC.Svg.paleturquoise,
    QCC.Svg.palevioletred,
    QCC.Svg.papayawhip,
    QCC.Svg.peachpuff,
    QCC.Svg.peru,
    QCC.Svg.pink,
    QCC.Svg.plum,
    QCC.Svg.powderblue,
    QCC.Svg.purple,
    QCC.Svg.red,
    QCC.Svg.rosybrown,
    QCC.Svg.royalblue,
    QCC.Svg.saddlebrown,
    QCC.Svg.salmon,
    QCC.Svg.sandybrown,
    QCC.Svg.seagreen,
    QCC.Svg.seashell,
    QCC.Svg.sienna,
    QCC.Svg.silver,
    QCC.Svg.skyblue,
    QCC.Svg.slateblue,
    QCC.Svg.slategray,
    QCC.Svg.snow,
    QCC.Svg.springgreen,
    QCC.Svg.steelblue,
    QCC.Svg.tan,
    QCC.Svg.teal,
    QCC.Svg.thistle,
    QCC.Svg.tomato,
    QCC.Svg.turquoise,
    QCC.Svg.violet,
    QCC.Svg.wheat,
    QCC.Svg.yellow,
    QCC.Svg.yellowgreen,
    ]
    
