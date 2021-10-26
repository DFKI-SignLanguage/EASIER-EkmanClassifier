import pandas as pd
import os
from shutil import copyfile

FePh_base_path = "/Users/chbh01/Documents/OfflineCodebases/DFKI_Hiwi/ACG/EASIER/Datasets/FePh/"
test_df = pd.read_csv(os.path.join(FePh_base_path, "FePh_test.csv"))
images_dir = os.path.join(FePh_base_path, "FePh_images-cropped")
os.makedirs(os.path.join(FePh_base_path, "FePh_test_images-mtcnn-cropped"))
# print(os.path.isdir())
print(test_df["External ID"].values.shape)

for f in test_df["External ID"]:
    try:
        copyfile(os.path.join(images_dir, f), os.path.join(FePh_base_path, "FePh_test_images-mtcnn-cropped", f))
    except FileNotFoundError:
        copyfile(os.path.join(images_dir, f + ".png"), os.path.join(FePh_base_path, "FePh_test_images-mtcnn-cropped", f + ".png"))


