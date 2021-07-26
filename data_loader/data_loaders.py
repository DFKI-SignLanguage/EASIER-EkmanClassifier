from torchvision import datasets, transforms
from base import BaseDataLoader
import os
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import cv2


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class FaceExpressionPhoenixDataLoader(Dataset):

    # TODO: Pass dir_name, image_dir_name and labels_csv_name
    def __init__(self, data_path, x_dir, y_csv, transform=None, target_transform=None):

        # https://www.researchgate.net/publication/340049545_Facial_Expression_Phoenix_FePh_An_Annotated_Sequenced_Dataset_for_Facial_and_Emotion-Specified_Expressions_in_Sign_Language
        self.label_names = {0: "neutral",
                            1: "anger",
                            2: "disgust",
                            3: "fear",
                            4: "happy",
                            5: "sad",
                            6: "surprise",
                            7: "none"}
        self.data_path = data_path
        self.x_dir_path = os.path.join(data_path, x_dir)
        self.y_csv_path = os.path.join(data_path, y_csv)
        self.transform = transform
        self.target_transform = target_transform

        mlb = MultiLabelBinarizer()
        y_df = pd.read_csv(self.y_csv_path)
        # Removing all data points with 'Face_not_visible'
        y_df.dropna(inplace=True)
        # y_df.dropna(subset=['Final_labels'], inplace=True)
        y_df['Facial_label'] = y_df['Facial_label'].apply(lambda x: np.array([int(i) for i in x]))
        self.image_inputs = y_df['External ID'].apply(lambda img_name: os.path.join(self.x_dir_path, img_name)).tolist()
        self.labels = mlb.fit_transform(y_df['Facial_label'].to_numpy())

    def __len__(self):
        return len(self.image_inputs)

    def __getitem__(self, idx):

        # TODO: Use image name from csv which does not contain 'Face_not_visible'
        inp_img_name = self.image_inputs[idx]
        out_labels = self.labels[idx]

        if not os.path.exists(inp_img_name):
            inp_img_name += ".png"

        in_image = cv2.imread(inp_img_name)

        if self.transform:
            in_image = self.transform(in_image)

        if self.target_transform:
            out_labels = self.target_transform(out_labels)

        return in_image, out_labels
