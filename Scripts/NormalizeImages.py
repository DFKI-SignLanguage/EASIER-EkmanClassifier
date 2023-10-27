import argparse

# See https://github.com/ipazc/mtcnn
from mtcnn import MTCNN

from utils.img import normalize_image

import PIL
import PIL.Image
from PIL.Image import Image

import os

from typing import Optional

IMAGE_FORMATS = set(('.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG'))

# If True, some visual information (dots, lines) are painted on the output image for visual debug purposes
DEBUG_DRAW = False


def normalize_images(in_dir: str, out_dir: str,
                     tolerant: bool,
                     normalize_color: bool,
                     square: bool, bbox_scale: Optional[float],
                     rotate: bool, rot_filter: int = PIL.Image.NEAREST) -> None:
    """Scans files in a directory.
    For each image ending in a recognized format, detect the position of a face, crop the image,
    and save the cropped result in the destination directory.

    :param square: If True, the face bounds will be extended to be squared.
    :param tolerant: If true, the iteration will continue on warnings, instead of stopping.
    :param out_dir: The output directory. Created if not existing.
    :param in_dir: Directory to scan. Will not be recursed.
    :param bbox_scale: A float number scaling the edges of the cropping rectangle around its center.
    Values <1 will shink the bbox, =1 has no effect, >1 will expand teh bbox.
    :param rotate: Wether the image should be rotate to bring the eyes at the horizonal lavel.
    :param rot_filter: The rotation interpolation filter. Values (int) taken from the PIL library.
    """

    detector = MTCNN(min_face_size=50)

    files = os.listdir(in_dir)

    for i, f in enumerate(files):
        print("Processing file {} '{}'".format(i, f))

        _, ext = os.path.splitext(f)
        if ext not in IMAGE_FORMATS:
            print("Not an image. skipping...")
            continue

        #
        # Load the image
        f_path = os.path.join(in_dir, f)
        img: Image = PIL.Image.open(f_path)
        # print(img.size)

        try:
            img_cropped = normalize_image(img=img, mtcnn_face_detector=detector,
                                 normalize_color=normalize_color,
                                 square=square,
                                 bbox_scale=bbox_scale,
                                 rotate=rotate,
                                 rot_filter=rot_filter)
        except Exception as e:

            if tolerant:
                # If tolerant, just use the full image
                print("WARNING: ", e)
                img_cropped = img
            else:
                # Else, forward the exception
                raise e

        out_name = os.path.join(out_dir, f)
        img_cropped.save(out_name)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Apply the MTCNN face detection to images in a directory.')
    parser.add_argument('-t', '--tolerant', action='store_true', default=False, required=False,
                        help='If tolerant, do NOT stop each time there is a warning'
                             ' (e.g., more faces in a pic, no face in the pic, ...)')
    parser.add_argument('-nc', '--normalize-color', action='store_true', default=False, required=False,
                        help="If specified, the image color is normalized by centering and scaling the color histogram"
                             "separately for each of the RGB color channels")
    parser.add_argument('-s', '--square', action='store_true', default=False, required=False,
                        help='If selected, the output cropped region will be forced to have 1:1 ratio'
                             ' by extending by the same amount of pixels in both directions'
                             ' (up and down, or left and right)')
    parser.add_argument('-bbs', '--bbox-scale', default=None, type=float, required=False,
                        help='scaling factor for the bounding box around the face.'
                             ' Values > 1.0 will extend the box (zoom-out of the camera).'
                             ' Values < 1.0 will shrink the box (zoom-in of the camera).')
    parser.add_argument('-r', '--rotate', action='store_true', default=False, required=False,
                        help='If selected, the face, after being cropped, zoomed and squared, will be rotated '
                             ' so that the eyes are horizontally aligned.')
    parser.add_argument('-bl', '--rot_filter_bilinear', action='store_true', default=False, required=False,
                        help='If selected, the rotation filter will be a bi-linear sampling,'
                             ' instead of the default nearest-neighbour.')
    parser.add_argument('-i', '--input', default=None, type=str, required=True,
                        help='path to a directory of images to analyse')
    parser.add_argument('-o', '--output', default=None, type=str, required=True,
                        help='path to a directory where to save the images to.')

    args = parser.parse_args()

    indir = args.input
    outdir = args.output
    tolerant = args.tolerant
    norm_color = args.normalize_color
    square = args.square
    bbox_scale = args.bbox_scale
    rotate = args.rotate

    if args.rot_filter_bilinear:
        rot_filter = PIL.Image.Resampling.BILINEAR
    else:
        rot_filter = PIL.Image.Resampling.NEAREST

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    normalize_images(in_dir=indir, out_dir=outdir,
                     tolerant=tolerant,
                     normalize_color=norm_color,
                     square=square, bbox_scale=bbox_scale, rotate=rotate, rot_filter=rot_filter)

    print("Done.")
