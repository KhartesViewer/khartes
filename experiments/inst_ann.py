import argparse
from pathlib import Path
import datetime

import numpy as np
import cv2
import nrrd

def timestamp():
    t = datetime.datetime.utcnow()
    txt = t.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    # round to hundredths of a second
    txt2 = txt[:-5]+'Z'
    # print("timer", txt, txt2)
    return txt2

def main():
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="Create khartes-compatible NRRD file from annotated instance files")
    parser.add_argument(
            "input_directory", 
            help="Directory of annotated-instance files")
    parser.add_argument(
            "output_nrrd_file", 
            help="NRRD file to be created")
    parser.add_argument(
            "--alpha", 
            default=.25,
            type=float,
            help="Alpha value (opacity) of overlay (range 0. to 1.)")

    args = parser.parse_args()

    idir = Path(args.input_directory)
    ofile = Path(args.output_nrrd_file)
    alpha = args.alpha

    vols = idir.glob("*_volume.nrrd")
    # WARNING: calling list(vols) in the print statement
    # will "empty out" vols
    # print("vols", list(vols))
    zyxmin = np.full([3], 1000000)
    zyxmax = np.full([3], -1)

    dzyxs = []

    for vfile in vols:
        print("reading", vfile)
        data, header = nrrd.read(vfile)
        # print("header")
        # print(header)
        zyx0 = header["space origin"].astype(np.int32)
        zyxn = header["sizes"]
        # print(" ", zyx0, zyxn, data.shape)
        zyxmin = np.minimum(zyxmin, zyx0)
        zyxmax = np.maximum(zyxmax, zyx0+zyxn)
        dzyxs.append([data, zyx0, zyxn])
    # print("zyx min", zyxmin)
    # print("zyx max", zyxmax)


    masks = idir.glob("*_mask.nrrd")

    mzyxs = []

    for mfile in masks:
        print("reading", mfile)
        data, header = nrrd.read(mfile)
        # print("header")
        # print(header)
        zyx0 = header["space origin"].astype(np.int32)
        zyxn = header["sizes"]
        # print(" ", zyx0, zyxn, data.shape)
        # zyxmin = np.minimum(zyxmin, zyx0)
        # zyxmax = np.maximum(zyxmax, zyx0+zyxn)
        mzyxs.append([data, zyx0, zyxn, header])

    gdata = np.zeros((zyxmax-zyxmin), dtype=np.uint16)
    print("output volume shape:", gdata.shape)
    for data, zyx0, zyxn in dzyxs:
        mult = 1
        if data.dtype == np.uint8:
            mult = 256
        zyx0a = zyx0-zyxmin
        zyx1a = zyx0a+zyxn
        gdata[zyx0a[0]:zyx1a[0], zyx0a[1]:zyx1a[1], zyx0a[2]:zyx1a[2]] = mult*data

    if len(mzyxs) > 0:
        has_overlay = True
        d = zyxmax-zyxmin
        frgba = np.zeros((d[0], d[1], d[2], 4), dtype=np.float32)
        for data, zyx0, zyxn, header in mzyxs:
            ind2id = {}
            ind2rgb = {}
            maxid = 0
            maxind = 0
            for key, value in header.items():
                # print(key, value)
                if not key.startswith("Segment"):
                    continue
                ws = key.split('_')
                if len(ws) != 2:
                    continue 
                if len(ws[0]) < 8:
                    continue
                try:
                    ind = int(ws[0][7:])
                except:
                    continue
                maxind = max(maxind, ind)
                if ws[1] == "ID":
                    vs = value.split('_')
                    if len(vs) != 2:
                        continue
                    id = int(vs[1])
                    maxid = max(maxid, id)
                    ind2id[ind] = id
                if ws[1] == "Color":
                    vs = value.split(' ')
                    rgb = (float(vs[0]), float(vs[1]), float(vs[2]))
                    ind2rgb[ind] = rgb
            # print(ind2id)
            # print(ind2rgb)
            fcmap = np.zeros((maxid+1, 4), dtype=np.float32)
            for ind in range(maxind+1):
                if ind in ind2id and ind in ind2rgb:
                    id = ind2id[ind]
                    rgb = ind2rgb[ind]
                    fcmap[id, :3] = rgb
                    fcmap[id, 3] = alpha
            # print(fcmap)
            zyx0a = zyx0-zyxmin
            zyx1a = zyx0a+zyxn
            frgba[zyx0a[0]:zyx1a[0], zyx0a[1]:zyx1a[1], zyx0a[2]:zyx1a[2], :] = fcmap[data]
            alpha_args = np.nonzero(frgba[:,:,:,3] != 0)
            odata = gdata.copy()
            # clear out bit 15,
            # so now uint16 in the range 0 to 32767
            odata >>= 1
            # floats in the range 0. to 1.
            rgba = frgba[alpha_args[0], alpha_args[1], alpha_args[2]]
            # floats in the range 0. to 1.
            a = rgba[:,3]
            # floats in the range 0. to 255.
            rgb = rgba[:,:3]*31.
            # if a = 0, this will produce floats in the range 0. to 31.
            # we only need to shift by 10 bits, because we already shifted
            # by one bit above
            gray = (odata[alpha_args] >> 10) * (1.-a)
            # This produces uint16's in the range 0. to 31.
            combi = (rgb*a[:, np.newaxis] + gray[:, np.newaxis]).astype(np.uint16)
            r = combi[:,0]
            g = combi[:,1]
            b = combi[:,2]
            odata[alpha_args] = 32768 + (r << 10) + (g << 5) + b
    else:
        has_overlay = False
        odata = gdata

    ts = timestamp()
    header = {
            "khartes_xyz_starts": "%d %d %d"%(zyxmin[2], zyxmin[1], zyxmin[0]),
            "khartes_xyz_steps": "1 1 1",
            "khartes_version": "1.0",
            "khartes_created": ts,
            "khartes_modified": ts,
            "khartes_from_vc_render": False,
            "khartes_uses_overlay_colormap": has_overlay,
            "encoding": "raw",
            }

    print("Saving", ofile)
    nrrd.write(str(ofile), odata, header, index_order='C')

if __name__ == '__main__':
    exit(main())
