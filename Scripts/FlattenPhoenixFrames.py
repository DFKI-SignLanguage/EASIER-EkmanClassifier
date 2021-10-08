import argparse
import os

import PIL.Image
import numpy as np

parser = argparse.ArgumentParser(
    description='Python wrapper for ffmpeg to convert Phoenix dataset image frames to video')
parser.add_argument("-i", "--in_dir", help="input folder containing folders with frames", required=True)
parser.add_argument("-o", "--out_dir", help="folder in which all frames will be saved", required=True)
parser.add_argument("-c", "--out_csv", help="CSV file containing the list of all frames", required=True)
parser.add_argument("-r", "--resize", help="New size of the frames in WxH format (e.g., 210x300) ", required=False)

args = parser.parse_args()

frames_dir = args.in_dir
out_dir = args.out_dir
out_csv = args.out_csv

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

if args.resize is not None:
    w, h = args.resize.split('x')
    w = int(w)
    h = int(h)
    resize = True
else:
    w, h = None, None
    resize = False


# Will be filled with the list of all frames
flat_list = []

# Iterate over the directories
sorted_dirs = sorted([d for d in os.listdir(frames_dir) if not d.startswith(".")])
for i, curr_img_set_dir in enumerate(sorted_dirs):

    print("Scanning dir {} - {}".format(i, curr_img_set_dir))
    # e.g.: 01April_2010_Thursday_heute-6697

    files = os.listdir(os.path.join(frames_dir, curr_img_set_dir))
    sorted_frame_files = sorted([image_file for image_file in files if image_file.endswith(".png")])
    # e.g.: images0001.png, images0002.png, ...
    print("{} frames".format(len(sorted_frame_files)))

    for frame_file in sorted_frame_files:

        frame_full_path = os.path.join(frames_dir, curr_img_set_dir, frame_file)

        # Load the image
        img = PIL.Image.open(frame_full_path)

        # Get rid of alpha
        if img.mode != 'RGB':
            if img.mode == 'RGBA':
                print("WARNING: Dropping alpha channel...")
                img_np = np.asarray(img)
                # Check for the presence of alpha channel
                assert img_np.shape[2] == 4
                img_np = img_np[:, :, :3]
                img = PIL.Image.fromarray(img_np, 'RGB')
            else:
                raise Exception("Mode '{}' not supported.".format(img.mode))

        # Resize, if asked
        if resize:
            assert w is not None
            assert h is not None
            img = img.resize((w, h))

        # Save
        # Compose the name concatenating the video dir with the frame file
        frame_new_name = curr_img_set_dir + "-" + frame_file
        # e.g., 01April_2010_Thursday_heute-6697-images0001.png
        image_save_path = os.path.join(out_dir, frame_new_name)
        img.save(image_save_path)

        flat_list.append(frame_new_name)


print("Writing '{}' file...".format(out_csv))
with open(out_csv, "w") as out_csv_file:
    out_csv_file.write("ImageName\n")

    for f in flat_list:
        out_csv_file.write(f+"\n")

print("Done.")
