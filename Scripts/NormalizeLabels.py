import json
import pandas

from data_loader.data_loaders import EASIER_CLASSES

IMG_NAME_COL = ["ImageName"]
LABEL_COLS = ["ClassName", "Class"]


def _convert_single_line(row: pandas.Series, src_label_to_id, src_label_to_label, dst_label_to_id, sort_predictions) -> pandas.Series:
    """Support method to vectorize the conversion of the lines.
     Gets a line (as Seires as input) and returns the new line, again as Series."""

    image_name = row.ImageName
    label = row.ClassName
    idx = row.Class

    if label not in src_label_to_label:
        raise Exception(f"Label '{label}' not mentioned in the conversion map.")

    dst_label = src_label_to_label[label]

    if dst_label not in dst_label_to_id:
        raise Exception(f"Destination label '{dst_label}' not in the EASIER list")

    dst_label_idx = dst_label_to_id[dst_label]

    # Check consistency between label and idx
    if idx != src_label_to_id[label]:
        print(
            f"WARNING!!! For label '{label}' there is a wrong corresponding ID. Expected {src_label_to_id[label]}, found {idx}.")

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

    dst_row = pandas.Series(dst_row_dict)
    return dst_row


def normalize_labels(input_df: pandas.DataFrame, normalization_map: list) -> pandas.DataFrame:

    src_label_to_id = dict()
    src_label_to_label = dict()

    #
    # Prepares the dictionaries needed for the conversion
    for i, (src_label, dst_label) in enumerate(normalization_map):
        src_label_to_id[src_label] = i
        src_label_to_label[src_label] = dst_label

    dst_label_to_id = dict()
    for i, easier_label in enumerate(EASIER_CLASSES):
        dst_label_to_id[easier_label] = i

    # print("Normalization dictionaries:")
    # print(src_label_to_id)
    # print(src_label_to_label)
    # print(dst_label_to_id)

    # Check if the required columns are there
    for required_col in IMG_NAME_COL + LABEL_COLS:
        if required_col not in input_df.columns:
            raise Exception("No column {} in source dataframe.".format(IMG_NAME_COL))

    #
    # Check if the source dataset contains also predictions, or only the ClassName and Class columns
    if len(input_df.columns) > len(IMG_NAME_COL + LABEL_COLS):
        # print("Logit prediction columns found: will rename/reorder also prediction columns")
        output_columns = IMG_NAME_COL + EASIER_CLASSES + LABEL_COLS
        sort_predictions = True
    else:
        # print("No logit prediction columns found.")
        output_columns = IMG_NAME_COL + LABEL_COLS
        sort_predictions = False

    #
    # Process dataframe rows
    normalized_df = input_df.apply(func=_convert_single_line, axis=1, args=(src_label_to_id, src_label_to_label, dst_label_to_id, sort_predictions))
    # Convert columns in the wanted order
    dest_df = normalized_df[output_columns]

    return dest_df


#
#
#
if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description='Normalize the labels of a dataframe to the EASIER standard order.'
                                                 'The source CSV must contain the following three columns: ImageName, ClassName, Class.')
    parser.add_argument('-i', '--input', default=None, type=str, required=True,
                        help='path to the source CSV.')
    parser.add_argument('-m', '--map', default=None, type=str, required=True,
                        help='path to the normalization map.')
    parser.add_argument('-o', '--output', default=None, type=str, required=True,
                        help='path for the destination CSV.')

    args = parser.parse_args()

    source_csv_path = args.input
    destination_csv_path = args.output
    normalization_map_json = args.map

    #
    # Prepare the conversion map
    print("Loading normalization map from {}".format(normalization_map_json))
    with open(normalization_map_json, "r") as map_file:
        normalization_map = json.load(map_file)

    if not type(normalization_map) == list:
        raise Exception(f"The normalization map seems to be incorrect. A top-level type 'list' was expected."
                        f" Found '{type(normalization_map)}'")

    #
    # Read the source dataframe
    print(f"Loading source dataframe from {source_csv_path}")
    source_df = pandas.read_csv(source_csv_path)
    print(source_df.head())

    #
    # Perform the conversion
    print("Normalizing ...")
    before = time.time()
    output_df = normalize_labels(input_df=source_df, normalization_map=normalization_map)
    after = time.time()
    delta = after - before
    print(f"Converted in {delta} seconds.")

    # Save to file
    print(f"Saving to '{destination_csv_path}'...")
    output_df.to_csv(path_or_buf=destination_csv_path, header=True, index=False, na_rep=str(float('nan')))

    print("All done.")
