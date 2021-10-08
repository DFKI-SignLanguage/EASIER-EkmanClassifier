import re
import os
import argparse
import subprocess
# from subprocess import DEVNULL
import shlex

INPUT_FORMAT = "image2"
VCODEC = "mpeg4"


parser = argparse.ArgumentParser(
    description='Python wrapper for ffmpeg to convert Phoenix dataset image frames to video')
parser.add_argument("-r", "--in_frame_rate", help="frame rate of input sequence")
parser.add_argument("-s", "--in_frame_size", help="input frame size. Ex: 210x260")
parser.add_argument("-i", "--in_dir", help="input folder containing folders with frames", required=True)
parser.add_argument("-crf", "--out_crf", help="constant rate factor for codec")
parser.add_argument("-y", "--out_overwrite", help="overwrite files in output folder or not", action='store_true')
parser.add_argument("-o", "--out_dir", help="folder in which output is to be saved", required=True)

args = parser.parse_args()

frames_dir = args.in_dir
videos_out_dir = args.out_dir

framerate = args.in_frame_rate if args.in_frame_rate else "25"
constant_rate_factor = framerate

args_in_frame_size = args.in_frame_size if args.in_frame_size else "210x260"
args_out_overwrite = "-y" if args.out_overwrite else ""

ffmpeg_cmd = "ffmpeg -r {in_framerate} -f {in_force_format} -s {in_frame_size} " \
             "-i {input_file_url} " \
             "-vcodec {out_vcodec} -crf {out_constant_rate_factor} " \
             "{out_overwrite} " \
             "{output_file_url}"

# The regular expression to catch the sentence number
r = re.compile("([a-zA-Z]+)([0-9]+)")
DEVNULL = open(os.devnull, 'wb')
for i, curr_img_set_dir in enumerate(os.listdir(frames_dir)):

    if i % 10 == 0:
        print("Videos processed:" + str(i))

    first_img_name = os.listdir(os.path.join(frames_dir, curr_img_set_dir))[0]
    img_name_in_ffmpeg_style = r.match(first_img_name).group(1) + "%0" + str(
        len(r.match(first_img_name).group(2))) + "d." + first_img_name.split(".")[1]
    img_name_in_ffmpeg_style = os.path.join(frames_dir, curr_img_set_dir, img_name_in_ffmpeg_style)
    out_vid_filename = os.path.join(videos_out_dir,
                                    curr_img_set_dir + ".mp4")

    p = subprocess.Popen(
        shlex.split(
            ffmpeg_cmd.format(
                in_framerate=framerate,
                in_force_format=INPUT_FORMAT,
                in_frame_size=args_in_frame_size,
                input_file_url=img_name_in_ffmpeg_style,
                out_vcodec=VCODEC,
                out_constant_rate_factor=constant_rate_factor,
                out_overwrite=args_out_overwrite,
                output_file_url=out_vid_filename
            )
        ),
        stdin=subprocess.PIPE, stdout=DEVNULL, stderr=DEVNULL
    )
    output = p.communicate()[0]
DEVNULL.close()
print("Converted")
