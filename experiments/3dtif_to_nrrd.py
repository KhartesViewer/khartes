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

def readTiff(ifile):
    result = cv2.imreadmulti(str(ifile), flags=cv2.IMREAD_UNCHANGED)
    # print(im[0])
    if not result[0]:
        print("Problem reading", ifile)
        return None
    imt = result[1]
    print(" ", len(imt), "images in file")
    print("  image shape", imt[0].shape, "data type", imt[0].dtype)
    im = np.stack(imt, axis=0)
    # print(im.shape, im.dtype)
    return im

def main():
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="Create khartes-compatible NRRD file from 3D TIFF file")
    parser.add_argument(
            "input_3d_tiff_file", 
            help="3D TIFF file (grayscale) to be converted")
    parser.add_argument(
            "output_nrrd_file", 
            help="NRRD file to be created")
    parser.add_argument(
            "--overlay", 
            default=None,
            help="Optional 3D TIFF file (RGB or RGBA) to use as overlay")
    parser.add_argument(
            "--alpha", 
            default=None,
            type=float,
            help="Override overlay alpha values (range 0. to 1.)")

    args = parser.parse_args()

    ifile = Path(args.input_3d_tiff_file)
    ofile = Path(args.output_nrrd_file)
    overlay = args.overlay
    alpha_override = args.alpha


    '''
    result = cv2.imreadmulti(str(ifile), flags=cv2.IMREAD_UNCHANGED)
    # print(im[0])
    if not result[0]:
        print("Problem reading", ifile)
    imt = result[1]
    print(len(imt), "images in file")
    print(imt[0].shape, imt[0].dtype)
    im = np.stack(imt, axis=0)
    print(im.shape, im.dtype)
    '''
    print("Reading", ifile)
    im = readTiff(ifile)
    if im is None:
        return
    sh = im.shape
    if len(im.shape) != 3:
        print("file", ifile, "is", len(im.shape), "dimensions instead of the expected 3")
        return
    dtype = im.dtype
    # print(im.min(), im.max())
    if dtype == np.uint8:
        im = im.astype(np.uint16)
        im *= 256
    elif dtype == np.uint16:
        pass
    else:
        print("cannot handle dtype", dtype, "in file", ifile)
        return
    # print(im.min(), im.max())

    has_overlay = False
    if overlay is not None:
        overlay_file = Path(overlay)
        print("Reading overlay", overlay_file)
        overlay_im = readTiff(overlay_file)
        if overlay_im is None:
            print("could not read overlay file", overlay_file)
            return
        osh = overlay_im.shape
        print("  overlay shape", osh)
        if len(osh) != 4 or osh[0] != sh[0] or osh[1] != sh[1] or osh[2] != sh[2] or osh[3] != 4:
            print("overlay file", overlay_file,"has unexpected shape",osh,"expected",sh)
            return None
        if overlay_im.dtype != np.uint8:
            print("overlay file", overlay_file, "has", overlay_im.dtype, "expected", np.uint8)
        has_overlay = True

    # print("has overlay:", has_overlay)

    if has_overlay:
        print("Applying overlay")
        im //= 2
        alpha = overlay_im[:,:,:,3]
        alpha_args = np.nonzero(alpha != 0)
        # print(alpha_args.shape)
        # rgba = overlay_im[alpha_args]
        rgba = overlay_im[alpha_args[0], alpha_args[1], alpha_args[2], :]
        # print(rgba.shape)
        af = rgba[:, 3] / 255
        if alpha_override is not None:
            af = alpha_override
        # af[:] = 1.
        gray = (im[alpha_args] >> 7) * (1.-af)
        # gray[:] = 0
        r = (af*rgba[:,0] + gray).astype(np.uint16) >> 3
        r = np.clip(r, 0, 31)
        # r = 4
        # r = 0
        g = (af*rgba[:,1] + gray).astype(np.uint16) >> 3
        g = np.clip(g, 0, 31)
        # g = 0
        # g = 31
        b = (af*rgba[:,2] + gray).astype(np.uint16) >> 3
        b = np.clip(b, 0, 31)
        # b = 0
        # b = 31
        # r = 31
        # g = 31
        # b = 31
        im[alpha_args] = 32768 + (r << 10) + (g << 5) + b
        # print("im[alpha_args][0]", im[alpha_args][0])

    ts = timestamp()
    header = {
            "khartes_xyz_starts": "0 0 0",
            "khartes_xyz_steps": "1 1 1",
            "khartes_version": "1.0",
            "khartes_created": ts,
            "khartes_modified": ts,
            "khartes_from_vc_render": False,
            "khartes_uses_overlay_colormap": has_overlay,
            "encoding": "raw",
            }

    print("Saving", ofile)
    nrrd.write(str(ofile), im, header, index_order='C')

if __name__ == '__main__':
    exit(main())
