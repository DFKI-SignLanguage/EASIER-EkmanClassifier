import json
import os

import numpy as np
from tqdm import tqdm
import torch
from data_loader.data_loaders import VideoFrameDataset
from torch.utils.data import DataLoader
from model.model import MobilenetModel, ResnetModel


class VideoEkmanPredictor:
    def __init__(self):
        self.model = None
        self.device = None
        self.config = None
        self.map_afnet_to_easierclss = {
            0: 7,
            1: 0,
            2: 1,
            3: 2,
            4: 3,
            5: 5,
            6: 4,
            7: 6
        }

    def load(self, model_pth, config_pth):

        if not os.path.exists(model_pth):
            raise Exception(f"Model file '{model_pth}' doesn't exist.")

        if not os.path.exists(config_pth):
            raise Exception(f"Config file '{config_pth}' doesn't exist.")

        f = open(config_pth)
        self.config = json.load(f)
        f.close()

        # build model architecture
        model = MobilenetModel()

        checkpoint = torch.load(model_pth, map_location='cpu')
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)

        # prepare model for testing
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.model.eval()

    def predict(self, in_video_pth):

        if not os.path.exists(in_video_pth):
            raise Exception(f"Video file '{in_video_pth}' doesn't exist.")

        video_dataset = VideoFrameDataset(in_video_pth, batch_size=32, transform=None)
        test_data_loader = DataLoader(video_dataset, batch_size=None)  # None for dynamic batch size

        out_all_softmax = []
        # predictions = []
        with torch.no_grad():
            for i, (data) in enumerate(tqdm(test_data_loader)):
                data = torch.Tensor(data).to(self.device)
                output_softmax = self.model(data)
                output_softmax = output_softmax.cpu().numpy()
                out_all_softmax.append(output_softmax)
                # output = np.argmax(output_softmax, axis=1)
                # predictions.append(output)

        out_all_softmax = np.vstack(out_all_softmax)

        return out_all_softmax

    def reorder_columns(self, input_array):
        """
        Reorder the columns of a 2D NumPy array based on a column mapping dictionary.

        Parameters:
        - input_array: 2D NumPy array
        - column_mapping: Dictionary where keys are input column indices and values are output column indices.

        Returns:
        - reordered_array: 2D NumPy array with columns reordered based on the mapping.
        """
        column_mapping = self.map_afnet_to_easierclss
        num_columns = input_array.shape[1]
        reordered_array = np.empty_like(input_array)

        for input_col, output_col in column_mapping.items():
            if input_col < 0 or input_col >= num_columns or output_col < 0 or output_col >= num_columns:
                raise ValueError("Invalid column index in column mapping")
            reordered_array[:, output_col] = input_array[:, input_col]

        return reordered_array
