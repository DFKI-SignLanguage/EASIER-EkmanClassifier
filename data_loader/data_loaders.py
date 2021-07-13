from torchvision import datasets, transforms
from base import BaseDataLoader
import os
from skimage import io
from torch.utils.data import Dataset, DataLoader
import pandas as pd


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
    def __init__(self, x_dir, y_csv, transform=None, target_transform=None):
        self.x_dir = x_dir
        self.x_dir_files = sorted(os.listdir(x_dir))

        # TODO: Remove all data points with 'Face_not_visible'
        self.y_df = pd.read_csv(y_csv)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.x_dir_files)

    def __getitem__(self, idx):

        # TODO: Use image name from csv which does not contain 'Face_not_visible'
        curr_filename = self.x_dir_files[idx]
        inp_img_name = os.path.join(self.x_dir, curr_filename)
        out_label = int(self.y_df["Facial_label"][idx])

        in_image = io.imread(inp_img_name)

        if curr_filename != self.y_df["External ID"][idx]:
            print(curr_filename, self.y_df["External ID"][idx])

        if self.transform:
            in_image = self.transform(in_image)

        if self.target_transform:
            out_label = self.target_transform(out_label)

        return in_image, out_label
