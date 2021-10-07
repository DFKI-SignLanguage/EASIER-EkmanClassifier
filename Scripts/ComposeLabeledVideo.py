import argparse
import os

import pandas

import cv2

FPS = 25
VIDEO_OUT_SIZE = (224, 224)

# Constants for video output
TEXT_COLOR = (220, 10, 10)
TEXT_POSITION = (10, 20)


def compose_video(predictions: str, frames_dir: str, out_video: str) -> None:

    predictions_df = pandas.read_csv(filepath_or_buffer=predictions)  # type: pandas.DataFrame

    video_writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*'mp4v'), FPS, VIDEO_OUT_SIZE)

    for img_name, label in zip(predictions_df['ImageName'], predictions_df['ClassName']):

        img_path = os.path.join(frames_dir, img_name)

        print("Loading {} ({})...".format(img_path, label))

        if not os.path.exists(img_path):
            raise Exception("Cannot find file '{}'".format(img_path))

        img = cv2.imread(img_path)

        frame = cv2.resize(src=img, dsize=VIDEO_OUT_SIZE, interpolation=cv2.INTER_AREA)

        cv2.putText(img=frame, text=label, org=TEXT_POSITION,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                    color=TEXT_COLOR, thickness=3)

        # Write the resulting frame
        video_writer.write(frame)

    # Close the video stream
    video_writer.release()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Compose a video by concatenating the frames specified in a'
                                                 ' prediction CSV file and applying the labels as overlay text.')
    parser.add_argument('-p', '--predictions', default=None, type=str, required=True,
                        help='path to the CSV file with the prediction labels.')
    parser.add_argument('-d', '--dir', default=None, type=str, required=True,
                        help='path to the directory containing the frames.')
    parser.add_argument('-o', '--output', default=None, type=str, required=True,
                        help='path to the name of the video output file.')

    args = parser.parse_args()

    predictions_path = args.predictions
    framesdir_path = args.dir
    outvideo_path = args.output

    if not os.path.exists(predictions_path):
        raise Exception("Missing '{}' file".format(predictions_path))

    if not os.path.exists(framesdir_path):
        raise Exception("Missing '{}' directory".format(framesdir_path))

    compose_video(predictions=predictions_path, frames_dir=framesdir_path, out_video=outvideo_path)

    print("Done.")
