import argparse

# See https://github.com/ipazc/mtcnn
from mtcnn import MTCNN

import PIL
import PIL.Image
from PIL.Image import Image
import numpy as np

import os

from typing import Tuple
from typing import Optional

IMAGE_FORMATS = set(('.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG'))


def convert_images(in_dir: str, out_dir: str, tolerant: bool, square: bool, bbox_scale: Optional[float]) -> None:
    """Scans files in a directory.
    For each image ending in a recognized format, detect the position of a face, crop the image,
    and save the cropped result in the destination directory.

    :param square: If True, the face bounds will be extended to be squared.
    :param tolerant: If true, the iteration will continue on warnings, instead of stopping.
    :param out_dir: The output directory. Created if not existing.
    :param in_dir: Directory to scan. Will not be recursed.
    """

    detector = MTCNN(min_face_size=50)

    files = os.listdir(in_dir)

    for i, f in enumerate(files):
        print("Processing file {} '{}'".format(i, f))

        if not f[-4:] in IMAGE_FORMATS:
            print("Not an image. skipping...")
            continue

        #
        # Load the image
        f_path = os.path.join(in_dir, f)
        img: Image = PIL.Image.open(f_path)
        # print(img.size)

        #
        # Detect the face (requires numpy array format)
        img_np = np.asarray(img)
        # Check for the presence of alpha channel
        # In case, remove it because the face detector doesn't support it.
        depth = img_np.shape[2]
        if depth == 4:
            # Drop the alpha channel
            print("WARNING: Dropping alpha channel...")
            img_np = img_np[:, :, :3]
            # PIL.Image.fromarray(img_np, 'RGB').save("after_alpha.png")
        elif depth == 3:
            pass
        else:
            assert False

        # print(img_np.shape)
        face_list = detector.detect_faces(img_np)
        # print(face_list)

        # No faces?
        if len(face_list) == 0:
            print("WARNING: no face detected!")
            if tolerant:
                # Just use the whole image
                img_cropped = img
            else:
                break
        else:
            # More faces?
            if len(face_list) > 1:
                print("WARNING: more than one face detected: {}. Using the first...".format(len(face_list)))
                if not tolerant:
                    break

            # Take the first face by default
            face = face_list[0]
            bbox = face['box']
            # bbox format is [x, y, width, height]
            x, y, width, height = bbox

            # Do we want a squared output?
            if square:
                if width > height:
                    # extends up and down
                    dy = width - height
                    top_dy = int(dy / 2)
                    y = y - top_dy
                    height = height + dy
                elif width < height:
                    # extends left and right
                    dx = height - width
                    left_dx = int(dx / 2)
                    x = x - left_dx
                    width = width + dx
                else:
                    pass

                assert width == height

            #
            # Scale the box
            if bbox_scale is not None:
                x, y, width, height = scale_bbox(x, y, width, height, bbox_scale)
                x = round(x)
                y = round(y)
                width = round(width)
                height = round(height)

            img_cropped = img.crop((x, y, x + width, y + height))

            # Beware! If the image crop area is outside of the visible area, PIL (at least Pillow==8.3.1)
            # adds an alpha channel and sets the out bounds to black color with 0 on the alpha channel.
            # Hence, we test again if there is an alpha channel and possibly remove it.
            if img_cropped.mode == 'RGBA':
                img_cropped = img_cropped.convert('RGB')

        assert img_cropped is not None

        out_name = os.path.join(out_dir, f)
        img_cropped.save(out_name)


def scale_bbox(x: float, y: float, width: float, height: float, scale: float) -> Tuple[float, float, float, float]:
    """
    Scales a bounding box around its center

    :param x: top-left x coordinate
    :param y: top-up y coordinate
    :param width: width of the box
    :param height: height of the box
    :param scale: scaling factor. < 1 shrinks, > 1 expands
    :return: the scaled box parameters, in order: x, y, width, height
    """

    # Compute the bottom-right corner coords
    x2 = x + width - 1
    y2 = y + height - 1

    cx = (x + x2) / 2
    cy = (y + y2) / 2

    # Prepare the transformation matrices

    # The matrix to extend the box
    scaling_matrix = np.array([[scale, 0,     0],
                               [0,     scale, 0],
                               [0,     0,     1]])

    # The matrix to center the box around the center, before scaling
    to_origin_matrix = np.array([[1, 0, -cx],
                                 [0, 1, -cy],
                                 [0, 0, 1]])

    back_translation_matrix = np.linalg.inv(to_origin_matrix)

    # compose the matrix that is scaling the box around its center
    t = back_translation_matrix @ scaling_matrix @ to_origin_matrix

    # Transform the box coordinates
    nx, ny, o = t @ (x, y, 1)
    nx2, ny2, o2 = t @ (x2, y2, 1)

    nwidth = nx2 - nx + 1
    nheight = ny2 - ny + 1

    return nx, ny, nwidth, nheight


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Apply the MTCNN face detection to images in a directory.')
    parser.add_argument('-t', '--tolerant', action='store_true', default=False, required=False,
                        help='If tolerant, do NOT stop each time there is a warning'
                             ' (e.g., more faces in a pic, no face in the pic, ...)')
    parser.add_argument('-s', '--square', action='store_true', default=False, required=False,
                        help='If selected, the output cropped region will be forced to have 1:1 ratio'
                             ' by extending by the same amount of pixels in both directions'
                             ' (up and down, or left and right)')
    parser.add_argument('-bbs', '--bbox-scale', default=None, type=float, required=False,
                        help='scaling factor for the bounding box around the face.'
                             ' Values > 1.0 will extend the box (zoom-out of the camera).'
                             ' Values < 1.0 will shrink the box (zoom-in of the camera).')
    parser.add_argument('-i', '--input', default=None, type=str, required=True,
                        help='path to a directory of images to analyse')
    parser.add_argument('-o', '--output', default=None, type=str, required=True,
                        help='path to a directory where to save the images to.')

    args = parser.parse_args()

    indir = args.input
    outdir = args.output
    tolerant = args.tolerant
    square = args.square
    bbox_scale = args.bbox_scale

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    convert_images(in_dir=indir, out_dir=outdir, tolerant=tolerant, square=square, bbox_scale=bbox_scale)

    print("Done.")
