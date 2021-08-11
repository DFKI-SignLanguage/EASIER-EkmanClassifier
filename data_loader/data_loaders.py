import torch
from torchvision import datasets, transforms
from base import BaseDataLoader
import os
from torch.utils.data import Dataset
import pandas as pd
# from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import class_weight
import numpy as np
from PIL import Image


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


class FaceExpressionPhoenixDataset(Dataset):

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

        # mlb = MultiLabelBinarizer()
        y_df = pd.read_csv(self.y_csv_path)
        # Removing all data points with 'Face_not_visible' i.e no labels
        y_df.dropna(inplace=True)
        # Extracting multiple labels
        y_df['Facial_label'] = y_df['Facial_label'].apply(lambda x: [int(i) for i in x])
        y_df['num_labels'] = y_df['Facial_label'].apply(lambda x: len(x))
        # Removing all data points with more than one labels ==> Ambiguous
        y_df = y_df[y_df["num_labels"] == 1]
        self.image_inputs = y_df['External ID'].apply(lambda img_name: os.path.join(self.x_dir_path, img_name)).tolist()

        # self.labels = mlb.fit_transform(y_df['Facial_label'].to_numpy())
        self.labels = y_df['Facial_label'].apply(lambda x: x[0]).to_numpy()

    def __len__(self):
        return len(self.image_inputs)

    def __getitem__(self, idx):

        inp_img_name = self.image_inputs[idx]
        out_label = self.labels[idx]

        if not os.path.exists(inp_img_name):
            inp_img_name += ".png"

        in_image = Image.open(inp_img_name).convert('RGB')

        tensor_trsnfrm = transforms.ToTensor()

        if self.transform:
            in_image = self.transform(in_image)
        else:
            in_image = tensor_trsnfrm(in_image)

        if self.target_transform:
            out_label = self.target_transform(out_label)
        # else:
        #     out_label = torch.Tensor(out_label)

        return in_image, out_label

    def reorder_samples(self, new_idxs):
        self.image_inputs = [self.image_inputs[i] for i in new_idxs]
        self.labels = self.labels[new_idxs]

    def get_sample_weights(self, idxs):
        print("Calculating sampler weights...")
        # labels_array = np.argmax(self.labels[idxs], axis=1)
        labels_array = self.labels[idxs]
        class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(labels_array),
                                                          y=labels_array)
        num_classes = len(self.label_names.keys())
        assert (class_weights.size == num_classes)

        sampler_weights = np.zeros(len(labels_array))
        for i in range(len(labels_array)):
            sampler_weights[i] = class_weights[int(labels_array[i])]

        return sampler_weights


class FaceExpressionPhoenixDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.6374226, 0.5848234, 0.56568706], std=[0.20125638, 0.22521368, 0.2639905]),
            transforms.Resize((224, 224)),
        ])
        self.data_dir = data_dir
        self.dataset = FaceExpressionPhoenixDataset(data_dir, 'FePh_images', 'FePh_labels.csv', transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, is_imbalanced_classes=True)
