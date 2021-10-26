import pandas as pd
import os
from shutil import copyfile

FePh_base_path = "/Users/chbh01/Documents/OfflineCodebases/DFKI_Hiwi/ACG/EASIER/Datasets/FePh/"
y_df = pd.read_csv(os.path.join(FePh_base_path, "FePh_labels.csv"))
images_dir = os.path.join(FePh_base_path, "FePh_images")

destination_folder = "FePh_images-single-labels-only"
os.makedirs(os.path.join(FePh_base_path, destination_folder))
y_df.dropna(inplace=True)
# Extracting multiple labels
y_df['Facial_label'] = y_df['Facial_label'].apply(lambda x: [int(i) for i in x])
y_df['num_labels'] = y_df['Facial_label'].apply(lambda x: len(x))
# Removing all data points with more than one labels ==> Ambiguous
y_df = y_df[y_df["num_labels"] == 1]
y_df['Facial_label'] = y_df['Facial_label'].apply(lambda x: x[0]).to_numpy()
print(y_df["External ID"].values.shape)

for f in y_df["External ID"]:
    try:
        copyfile(os.path.join(images_dir, f), os.path.join(FePh_base_path, destination_folder, f))
    except FileNotFoundError:
        copyfile(os.path.join(images_dir, f + ".png"), os.path.join(FePh_base_path, destination_folder, f + ".png"))

y_df.to_csv("FePh_labels_single_label_only.csv")
