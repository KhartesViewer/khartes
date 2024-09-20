import argparse
from pathlib import Path

import numpy as np
import cv2
from cmap import Colormap

def readTiff(ifile):
    result = cv2.imreadmulti(str(ifile), flags=cv2.IMREAD_UNCHANGED)
    # print(im[0])
    if not result[0]:
        print("Problem reading", ifile)
        return None
    imt = result[1]
    print(" ", len(imt), "images in file")
    print("  image shape",imt[0].shape, "data type", imt[0].dtype)
    im = np.stack(imt, axis=0)
    # print(im.shape, im.dtype)
    return im


def main():
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="Convert indicator 3D TIFF file to 3D RGB tiff file")
    parser.add_argument(
            "input_3d_tiff_file", 
            help="3D TIFF file (grayscale), to be converted")
    parser.add_argument(
            "output_3d_tiff_file", 
            help="3D RGBA tiff file, to be created")
    parser.add_argument(
            "--colormap", 
            default="bmr_3c",
            help="indicator colormap; see https://cmap-docs.readthedocs.io/en/stable/catalog/")
    parser.add_argument(
            "--alpha", 
            default=1.,
            type=float,
            help="alpha value of indicator colors (range 0.0 to 1.0)")

    args = parser.parse_args()

    ifile = Path(args.input_3d_tiff_file)
    ofile = Path(args.output_3d_tiff_file)
    colormap = args.colormap
    alpha = args.alpha

    print("Reading", ifile)
    im = readTiff(ifile)
    if im is None:
        return
    if len(im.shape) != 3:
        print("expected 3 dimensions, got", len(im.shape))
        return
    # print(im.shape, im.dtype)
    # print(im.min(), im.max())
    maxind = im.max()
    if maxind == 0:
        print("No indicators in file")
        return
    cmap = Colormap(colormap, under=[0,0,0,0])
    # im = im[0:200,:,:]
    # print(im.shape)
    den = maxind-1
    if maxind == 0:
        den = 1
    print("Converting indicators to colors")
    oimb = cmap((im.astype(np.float32)-1)/den, bytes=True)
    # print(oimb.shape, oimb.dtype, oimb.min(), oimb.max())
    oimb[:,:,:,3] = (oimb[:,:,:,3]*alpha).astype(oimb.dtype)

    toim = tuple(oimb)
    # print(len(toim))
    # print(toim[0].shape)
    print("Saving", ofile)
    cv2.imwritemulti(str(ofile), toim)






if __name__ == '__main__':
    exit(main())

