# Script to extract the labels from the AffectNet dataformat into an easy-to-read CSV file.
# See: http://mohammadmahoor.com/affectnet/

import argparse
import os
import glob

import numpy as np
import pandas

REQUIRED_COLUMNS = ["ImageName", "ClassName", "Class"]


AFFECTNET_LABELS_LIST = [
    "neutral",   # 0
    "happy",     # 1
    "sad",       # 2
    "surprise",  # 3
    "fear",      # 4
    "disgust",   # 5
    "anger",     # 6
    "contempt"   # 7
    ]

parser = argparse.ArgumentParser(description='Extract AffectNet labels and outputs them into our CSV format,'
                                             ' with three columns: ImageName, ClassName, Class.')
parser.add_argument('-d', '--dir', default=None, type=str, required=True,
                    help='path to the source directory, containing the "annotations" and "images" subdirectories.')
parser.add_argument('-o', '--output', default=None, type=str, required=True,
                    help='path for the destination CSV.')

args = parser.parse_args()

source_dir = args.dir
destination_csv = args.output


# Prepare the output dataframe
dest_df = pandas.DataFrame(columns=REQUIRED_COLUMNS)


#
# Iterate over the annotation data
annotation_dir = os.path.join(source_dir, "annotations")

for entry in glob.glob(annotation_dir + "/*_exp.npy", recursive=False):
    # print(entry)

    # Get the label number and name
    label_np = np.load(entry)
    assert type(label_np) == np.ndarray
    label_int = int(label_np)
    assert 0 <= label_int < len(AFFECTNET_LABELS_LIST)
    label = AFFECTNET_LABELS_LIST[label_int]

    # Compose the image filename
    file_name = os.path.basename(entry)
    assert file_name.endswith("_exp.npy")
    image_name = file_name[:-8] + ".jpg"

    dest_df = dest_df.append({
        "ImageName": image_name,
        "ClassName": label,
        "Class": label_int
    }, ignore_index=True)


# Save to file
dest_df.to_csv(path_or_buf=destination_csv, header=True, index=False)

print("All done.")
