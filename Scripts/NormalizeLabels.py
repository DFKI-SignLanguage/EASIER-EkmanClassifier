import argparse
import json
import pandas

from data_loader.data_loaders import EASIER_CLASSES

IMG_NAME_COL = ["ImageName"]
LABEL_COLS = ["ClassName", "Class"]


parser = argparse.ArgumentParser(description='Normalize the labels of a dataframe to the EASIER standard order.'
                                             'The source CSV must contain the following three columns: ImageName, ClassName, Class.')
parser.add_argument('-i', '--input', default=None, type=str, required=True,
                    help='path to the source CSV.')
parser.add_argument('-m', '--map', default=None, type=str, required=True,
                    help='path to the normalization map.')
parser.add_argument('-o', '--output', default=None, type=str, required=True,
                    help='path for the destination CSV.')

args = parser.parse_args()

source_csv = args.input
destination_csv = args.output
normalization_map_json = args.map


#
# Prepare the conversion dictionaries
print("Loading normalization map from {}".format(normalization_map_json))
with open(normalization_map_json, "r") as map_file:
    normalization_map = json.load(map_file)

assert type(normalization_map) == list

src_label_to_id = dict()
src_label_to_label = dict()

for i, (src_label, dst_label) in enumerate(normalization_map):
    src_label_to_id[src_label] = i
    src_label_to_label[src_label] = dst_label

dst_label_to_id = dict()
for i, easier_label in enumerate(EASIER_CLASSES):
    dst_label_to_id[easier_label] = i

print("Normalization dictionaries:")
print(src_label_to_id)
print(src_label_to_label)
print(dst_label_to_id)

#
# Read the source map
print("Loading source dataframe from {}".format(source_csv))
source_df = pandas.read_csv(source_csv)
print(source_df.head())

for required_col in IMG_NAME_COL + LABEL_COLS:
    if required_col not in source_df.columns:
        raise Exception("No column {} in source dataframe.".format(IMG_NAME_COL))

#
# Check if the source dataset contains also predictions, or only the ClassName and Class columns
if len(source_df.columns) > len(IMG_NAME_COL + LABEL_COLS):
    print("Softmax predictions detected: will rename/reorder also prediction columns")
    output_columns = IMG_NAME_COL + EASIER_CLASSES + LABEL_COLS
    sort_predictions = True
else:
    print("No softmax prediction columns found.")
    output_columns = IMG_NAME_COL + LABEL_COLS
    sort_predictions = False

#
# Prepare the output CSV
dest_df = pandas.DataFrame(columns=output_columns)

#
# Process dataframe rows
for i, row in enumerate(source_df.itertuples()):
    # print(row)
    image_name = row.ImageName
    label = row.ClassName
    idx = row.Class

    if label not in src_label_to_label:
        raise Exception("Line {}: Label '{}' not mentioned in the conversion map.".format(i, label))

    dst_label = src_label_to_label[label]

    if dst_label not in dst_label_to_id:
        raise Exception("Line {}: Destination label '{}' not in the EASIER list".format(i, dst_label))

    dst_label_idx = dst_label_to_id[dst_label]

    # Check consistency between label and idx
    if idx != src_label_to_id[label]:
        print(" WARNING!!! Line {}: For label '{}' there is a wrong corresponding ID. Expected {}, found {}.".format(i, label, src_label_to_id[label], idx))

    dst_row_dict = {
        "ImageName": image_name,
        "ClassName": dst_label,
        "Class": dst_label_idx
    }

    # If required, sort also the prediction columns
    if sort_predictions:
        # Fill the prediction value columns
        # Compose the labels dictionary. By default, values to n/a.
        predictions_dict = {label: float('nan') for label in EASIER_CLASSES}
        for src_pred_lab, dst_pred_lab in src_label_to_label.items():
            pred = getattr(row, src_pred_lab)
            predictions_dict[dst_pred_lab] = pred

        dst_row_dict.update(predictions_dict)

    dest_df = dest_df.append(dst_row_dict, ignore_index=True)


# Save to file
dest_df.to_csv(path_or_buf=destination_csv, header=True, index=False, na_rep=str(float('nan')))


print("All done.")
