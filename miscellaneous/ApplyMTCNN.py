import argparse

# See https://github.com/ipazc/mtcnn
from mtcnn import MTCNN

import PIL
import PIL.Image
from PIL.Image import Image
import numpy as np

import os

IMAGE_FORMATS = set(('.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG'))


def convert_images(in_dir: str, out_dir: str, tolerant: bool) -> None:

    detector = MTCNN(min_face_size=50)

    files = os.listdir(in_dir)

    for f in files:
        print("Processing '{}'".format(f))

        if not f[-4:] in IMAGE_FORMATS:
            print("Not an image. skipping...")
            continue

        ext = f[-4:]

        f_path = os.path.join(in_dir, f)
        img: Image = PIL.Image.open(f_path)
        img_np = np.asarray(img)
        # print(img.size)
        # print(img_np.shape)
        face_list = detector.detect_faces(img_np)
        # print(face_list)

        if len(face_list) == 0:
            print("Warning: no face detected!")
            if tolerant:
                continue
            else:
                break

        if len(face_list) > 1:
            print("Warning: more than one face detected: {}".format(len(face_list)))
            if not tolerant:
                break

        face = face_list[0]
        bbox = face['box']
        # [x, y, width, height]
        x, y, width, height = bbox

        img_cropped = img.crop((x, y, x + width, y + height))

        out_name = os.path.join(outdir, f)
        img_cropped.save(out_name)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Apply the MTCNN face detection to images in a directory.')
    parser.add_argument('-t', '--tolerant', action='store_true', default=False, required=False,
                      help='If tolerant, do NOT stop each time there is a warning'
                           ' (e.g., more faces in a pic, no face in the pic, ...)')
    parser.add_argument('-i', '--input', default=None, type=str, required=True,
                      help='path to a directory of images to analyse')
    parser.add_argument('-o', '--output', default=None, type=str, required=True,
                      help='path to a directory where to save the images to.')

    args = parser.parse_args()

    indir = args.input
    outdir = args.output
    tolerant = args.tolerant

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    convert_images(in_dir=indir, out_dir=outdir, tolerant=tolerant)

    print("Done.")
