import torch
from torchvision import datasets, transforms
from base import BaseDataLoader
import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
# from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import class_weight
import numpy as np
from PIL import Image
import glob
from pathlib import Path

# Classes expected to be in the first round of annotation on the EASIER project.
EASIER_CLASSES = {
    0: "Happiness",
    1: "Sadness",
    2: "Surprise",
    3: "Fear",
    4: "Anger",
    5: "Disgust",
    6: "Contempt",
    7: "Other"
}


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
    # TODO Report that self.classes and self.idx_to_class will be the standard way for storing label maps in all
    #  custom datasets in this project
    idx_to_class = {0: "neutral",
                    1: "anger",
                    2: "disgust",
                    3: "fear",
                    4: "happy",
                    5: "sad",
                    6: "surprise",
                    7: "none"}

    def __init__(self, data_path, training=True, transform=None, target_transform=None):

        self.data_path = data_path
        # self.images_dir_path = os.path.join(data_path, 'FePh_images')
        self.images_dir_path = os.path.join(data_path, 'FePh_images-cropped')

        if training:
            self.labels_csv_path = os.path.join(data_path, 'FePh_train.csv')
            # self.labels_csv_path = os.path.join(data_path, 'FePh_test.csv')
        else:
            self.labels_csv_path = os.path.join(data_path, 'FePh_test.csv')

        self.transform = transform
        self.target_transform = target_transform

        y_df = pd.read_csv(self.labels_csv_path, dtype=str)
        # y_df = y_df.head(350)
        # Removing all data points with 'Face_not_visible' i.e no labels
        y_df.dropna(inplace=True)
        # Extracting multiple labels
        y_df['Facial_label'] = y_df['Facial_label'].apply(lambda x: [int(i) for i in x])
        y_df['num_labels'] = y_df['Facial_label'].apply(lambda x: len(x))
        # Removing all data points with more than one labels ==> Ambiguous
        y_df = y_df[y_df["num_labels"] == 1]
        self.image_inputs = y_df['External ID'].apply(
            lambda img_name: os.path.join(self.images_dir_path, img_name)).tolist()

        self.labels = y_df['Facial_label'].apply(lambda x: x[0]).to_numpy()

    def __len__(self):
        return len(self.image_inputs)
        # return 50

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
        num_classes = len(self.idx_to_class.keys())
        # assert (class_weights.size == num_classes)
        if class_weights.size != num_classes:
            print(
                "Warning: Not all classes in current set. (Temp solution: Adjust validation split till this message "
                "is not displayed)")
            print("Out of", num_classes, " classes, missing are: ",
                  np.setdiff1d(list(self.idx_to_class.keys()), np.unique(labels_array)))

        sampler_weights = np.zeros(len(labels_array))
        for i in range(len(labels_array)):
            sampler_weights[i] = class_weights[int(labels_array[i])]

        return sampler_weights

    def get_label_map(self):
        return self.idx_to_class


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

        self.dataset = FaceExpressionPhoenixDataset(data_dir, training=training, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, is_imbalanced_classes=True)

    @staticmethod
    def get_label_map():
        return FaceExpressionPhoenixDataset.idx_to_class


class PredictionDataset(Dataset):
    def __init__(self, data_path, data_loader):
        try:
            self.idx_to_class = data_loader.get_label_map()
        except AttributeError:
            raise AttributeError("Implement a get_label_map() static method similar to FaceExtractionPhoenixDataset")

        self.images_dir_path = os.path.join(data_path)
        self.image_inputs = [os.path.join(self.images_dir_path, img_name) for img_name in
                             sorted(os.listdir(self.images_dir_path)) if ".jpg" in img_name or ".png" in img_name]

    def __getitem__(self, idx):
        inp_img_name = self.image_inputs[idx]
        in_image = Image.open(inp_img_name).convert('RGB')

        size = 224, 224  # Fixed to Resnet input size

        tensor_trsnfrm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size),
        ])
        in_image = tensor_trsnfrm(in_image)
        img_name = os.path.split(inp_img_name)[1]

        return in_image, img_name

    # getitem setup for loading pretrained model from asavchenko
    # git repo https://github.com/HSE-asavchenko/face-emotion-recognition
    # def __getitem__(self, idx):
    #     inp_img_name = self.image_inputs[idx]
    #     in_image = Image.open(inp_img_name).convert('RGB')
    #
    #     size = 224, 224  # Fixed to Resnet input size
    #     # in_image.thumbnail(size, Image.ANTIALIAS)
    #
    #     # tensor_trsnfrm = transforms.ToTensor()
    #     tensor_trsnfrm = transforms.Compose(
    #         [
    #             transforms.Resize(size),
    #             # transforms.RandomHorizontalFlip(),
    #             transforms.ToTensor(),
    #             transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    #         ]
    #     )
    #     in_image = tensor_trsnfrm(in_image)
    #     in_image.unsqueeze_(0)
    #
    #     print(in_image.size())
    #
    #     img_name = os.path.split(inp_img_name)[1]
    #     return in_image, img_name

    def __len__(self):
        return len(self.image_inputs)
        # return 50


# ---------------------- AffectNet Dataset & DataLoader ---------------------- #

class AffectNet(Dataset):
    idx_to_class = {0: "neutral",
                    1: "happy",
                    2: "sad",
                    3: "surprise",
                    4: "fear",
                    5: "disgust",
                    6: "anger",
                    7: "contempt"}

    def __init__(self, data_path, transform=None):
        self.transform = transform
        self.images = glob.glob(os.path.join(data_path, 'images/*.jpg'))
        self.transform = transform
        self.data_path = data_path

        cache_file = os.path.join(data_path, "expressions_cache.npy")
        # check if labels cache exists
        if not os.path.exists(cache_file):
            labels = []
            for im in self.images:
                base = Path(im).stem
                exp = np.load(os.path.join(self.data_path, 'annotations', base + "_exp.npy"))
                labels.append(int(exp))
            np.save(cache_file, labels)

        self.labels = np.load(cache_file)
        self.num_classes = 8

    def get_sampler_weights(self):
        print("Calculating sampler weights...")
        labels_array = self.labels
        class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(labels_array),
                                                          y=labels_array)
        num_classes = len(self.idx_to_class.keys())

        assert (class_weights.size == num_classes)

        sampler_weights = np.zeros(len(labels_array))
        for i in range(len(labels_array)):
            sampler_weights[i] = class_weights[int(labels_array[i])]

        return sampler_weights

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        im = Image.open(self.images[index]).convert('RGB')

        base = Path(self.images[index]).stem
        exp = int(np.load(os.path.join(self.data_path, 'annotations', base + "_exp.npy")))

        if self.transform is not None:
            im = self.transform(im)

        return im, exp


class AffectNetDataLoader(DataLoader):
    """
    AffectNet data loading demo using DataLoader
    validation_split does nothing. included for compatibility with other loaders
    """

    def __init__(self, data_dir, batch_size, training=True, shuffle=True, num_workers=1, validation_split=0.0):

        self.validation_split = 0
        self.training = training

        trsfm = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        val_trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if training:

            self.dataset = AffectNet(os.path.join(data_dir, "train_set"), transform=trsfm)
            self.val_dataset = AffectNet(os.path.join(data_dir, "val_set"), transform=val_trsfm)
            weights = self.dataset.get_sampler_weights()
            train_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=torch.DoubleTensor(weights),
                                                                           num_samples=len(self.dataset))
            shuffle = False
        else:
            trsfm = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            self.dataset = AffectNet(os.path.join(data_dir, "val_set"), transform=val_trsfm)
            train_sampler = None
            shuffle = False

        self.shuffle = shuffle

        # self.val_data_dir = val_data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': shuffle,
            'num_workers': self.num_workers,
        }

        super().__init__(sampler=train_sampler, **self.init_kwargs)

    def split_validation(self):

        init_kwargs = {
            'batch_size': self.batch_size,
            'shuffle': False,
            'num_workers': self.num_workers,
        }

        return DataLoader(dataset=self.val_dataset, **init_kwargs)

    @staticmethod
    def get_label_map():
        return AffectNet.idx_to_class
