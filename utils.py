import time
import random
import datetime
import re
import numpy as np
from PyQt5.QtGui import QColorConstants as QCC
from PyQt5.QtGui import QColor
# import PyQt5.QtGuiQColor.SVG as QtSVG

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
    QCC.Svg.green,
    QCC.Svg.greenyellow,
    QCC.Svg.honeydew,
    QCC.Svg.hotpink,
    QCC.Svg.indianred,
    QCC.Svg.indigo,
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

# High intensity colors for volume annotations
COLORLIST = [
    QCC.Svg.hotpink,
    QCC.Svg.blue,
    QCC.Svg.red,
    QCC.Svg.magenta,
    QCC.Svg.green,
    QCC.Svg.blueviolet,
    QCC.Svg.chocolate,
    QCC.Svg.crimson,
    QCC.Svg.cyan,
    QCC.Svg.darkgoldenrod,
    QCC.Svg.darkgreen,
    QCC.Svg.darkmagenta,
    QCC.Svg.darkorange,
    QCC.Svg.darkseagreen,
    QCC.Svg.gold,
    QCC.Svg.greenyellow,
    QCC.Svg.yellow,
    QCC.Svg.yellowgreen,
    ]