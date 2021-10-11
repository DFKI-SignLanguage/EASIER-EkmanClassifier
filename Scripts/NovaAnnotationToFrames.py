import argparse
import os

import pandas
import numpy as np

AFFNET_CLASSES = ["neutral", "happy", "sad", "surprise", "fear", "disgust", "anger", "contempt"]
AFFNET_CLASS_TO_ID = {l: i for i, l in enumerate(AFFNET_CLASSES)}

EASIER_TO_AFFNET_LABEL = {
                "Happiness": "happy",
                "Sadness": "sad",
                "Surprise": "surprise",
                "Fear": "fear",
                "Disgust": "disgust",
                "Anger": "anger",
                "Contempt": "contempt"}
                # "Other" -- not supported
                # default: "neutral"

EASIER_CLASSES = ["Happiness", "Sadness", "Surprise", "Fear", "Anger", "Disgust", "Contempt", "Other"]
EASIER_CLASS_TO_ID = {l: i for i, l in enumerate(EASIER_CLASSES)}


OUTPUT_HEADER = ['ImageName', 'ClassName', 'Class']


def convert_annotation(annotations: pandas.DataFrame,
                       frames_list: pandas.DataFrame,
                       fps: float = 25) -> pandas.DataFrame:

    #
    # Prepare the output
    # default_data = [ [i, 1, 0, 0, 0, 0, 0, 0, 0, "neutral", 0] for i in range(max_frames)]
    out = pandas.DataFrame(columns=OUTPUT_HEADER)
    # out = pandas.DataFrame.from_records(columns=OUTPUT_HEADER, data=default_data)

    # Copy the name of the frames
    out['ImageName'] = frames_list['ImageName']

    # By default, when there is NO ANNOTATION, the face is neutral.
    out['ClassName'] = AFFNET_CLASSES[0]
    out['Class'] = 0

    # Reference to the current frame we are outputting
    out_frame = 0

    # For each annotated line, we will likely generate several several frames
    for annotation_line in annotations.itertuples(index=True, name=None):
        # Each line in the annotation is a time range associated to label number and confidence
        i = annotation_line[0]
        start_time = annotation_line[1]
        stop_time = annotation_line[2]
        label = annotation_line[3]  # EASIER class id
        confidence = annotation_line[4]

        # From id to class name
        easier_class = EASIER_CLASSES[label]

        # Convert into Affect Net
        if easier_class in EASIER_TO_AFFNET_LABEL:
            affnet_class = EASIER_TO_AFFNET_LABEL[easier_class]
        else:
            print("Class {} cannot be converted...".format(label))
            affnet_class = None

        affnet_class_id = AFFNET_CLASS_TO_ID[affnet_class]

        #
        # Stat filling the time range
        out_frame_time = out_frame / fps

        # Skip frames until we enter the range
        while out_frame_time < start_time:
            out_frame += 1
            out_frame_time = out_frame / fps

        assert out_frame_time >= start_time

        # Fill lines for the annotated range
        while out_frame_time < stop_time:
            out.loc[out_frame, 'ClassName'] = affnet_class
            out.loc[out_frame, 'Class'] = affnet_class_id

            out_frame += 1
            out_frame_time = out_frame / fps

    #
    # TODO --  transform the label in 1-hot
    # numpy.eye(number of classes)[vector containing the labels]
    one_hot_cols = np.eye(len(AFFNET_CLASSES))[out['Class']].astype(int)
    one_hot_df = pandas.DataFrame(columns=AFFNET_CLASSES, data=one_hot_cols)

    # Concatnate the two
    out = pandas.concat([out, one_hot_df], axis=1)

    # Change the order of the columns to match the prediction output
    out = out[['ImageName'] + AFFNET_CLASSES + ['ClassName', 'Class']]

    return out


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Converts an annotation file from the NOVA format (.annotation~)'
                    ' into a dataframe where each line is a frame with timestamp and associated label')
    parser.add_argument('-i', '--input', default=None, type=str, required=True,
                        help='path to the .annotation~ file with the NOVA annotations.')
    parser.add_argument('-fl', '--frame-list', default=None, type=str, required=True,
                        help='path to the .csv file with the image list (Must have a column with ImageName).')
    parser.add_argument('-o', '--output', default=None, type=str, required=True,
                        help='path to the name of the CSV output file.')
    #parser.add_argument('-d', '--dir', default=None, type=str, required=True,
    #                    help='path to the directory containing the frames.')

    args = parser.parse_args()

    annotation_filepath = args.input
    frame_list_filepath = args.frame_list
    dataframe_filepath = args.output

    if not os.path.exists(annotation_filepath):
        raise Exception("Missing '{}' file".format(annotation_filepath))

    annotation_df = pandas.read_csv(filepath_or_buffer=annotation_filepath, sep=";", header=None)

    frames_df = pandas.read_csv(filepath_or_buffer=frame_list_filepath)

    out_df = convert_annotation(annotations=annotation_df, frames_list=frames_df)

    out_df.to_csv(path_or_buf=dataframe_filepath, header=True, index=False)

    print("Done.")
