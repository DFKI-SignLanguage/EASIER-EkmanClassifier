import os
import pandas as pd

label_names = {0: "neutral",
                    1: "anger",
                    2: "disgust",
                    3: "fear",
                    4: "happy",
                    5: "sad",
                    6: "surprise",
                    7: "none"}
data_path = "C:\\Users\\chira\\OneDrive - uni-saarland.de\\Codebases\\DFKI_HiWi\\ACG\\Easier\\Datasets\\FePh"
x_dir_path = os.path.join(data_path, 'FePh_images')
y_csv_path = os.path.join(data_path, 'FePh_labels.csv')

# mlb = MultiLabelBinarizer()
y_df = pd.read_csv(y_csv_path)
# Removing all data points with 'Face_not_visible' i.e no labels
y_df.dropna(inplace=True)
# Extracting multiple labels
y_df['Facial_label'] = y_df['Facial_label'].apply(lambda x: [int(i) for i in x])
y_df['num_labels'] = y_df['Facial_label'].apply(lambda x: len(x))
# Removing all data points with more than one labels ==> Ambiguous
y_df = y_df[y_df["num_labels"] == 1]
n_r, n_c = y_df.shape
proportions = y_df["Facial_label"].value_counts()
p2 = proportions.apply(lambda x: x*100/n_r )
image_inputs = y_df['External ID'].apply(lambda img_name: os.path.join(x_dir_path, img_name)).tolist()

# labels = mlb.fit_transform(y_df['Facial_label'].to_numpy())
labels = y_df['Facial_label'].apply(lambda x: x[0]).to_numpy()