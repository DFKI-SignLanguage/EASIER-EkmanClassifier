import json
import os

import numpy as np
from tqdm import tqdm
import torch
from data_loader.data_loaders import VideoFrameDataset
from torch.utils.data import DataLoader
from model.model import MobilenetModel


# List of correspondences to convert the order of columns given by the model (trained on affnet) into the EASIER order.
AFFNET_TO_EASIER = [
    8,  # 0 --> 8 Neutral
    0,  # 1 --> 0 Happiness
    1,  # 2 --> 1 Sadness
    2,  # 3 --> 2 Surprise
    3,  # 4 --> 3 Fear
    5,  # 5 --> 5 Disgust
    4,  # 6 --> 4 Anger
    6   # 7 --> 6 Contempt
    # n/a  --> 7 Other
]

EASIER_COLUMN_COUNT = 9  # 7 Ekman + Other + Neutral


class VideoEkmanPredictor:
    """Support class to run the prediction of Ekman facial expressions on every frame of a video."""

    def __init__(self):
        self.model = None
        self.device = None
        self.config = None

        self.normalization_params = {
            "normalize_color": False,
            "square": True,
            "bbox_scale": 1.1,
            "rotate": True,
            "scale": (224, 224)
        }

    def load(self, model_pth, config_pth):
        """Loads the model and the config. Mandatory before trying to predict something!"""

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

    def predict(self, in_video_pth: str) -> np.ndarray:
        """Given a video file, creates the predictions for each frame.

        :param in_video_pth: Path to the video file
        :return: a numpy ndarra of shape (n_frames, 9),
         where each column is the logit values for each Ekman expression, in Affectnet order.
        """

        if not os.path.exists(in_video_pth):
            raise Exception(f"Video file '{in_video_pth}' doesn't exist.")

        video_dataset = VideoFrameDataset(in_video_pth, batch_size=32,
                                          transform=None,
                                          normalization_params=self.normalization_params)
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
        Reorder the columns of the affectnet-based odel into the EASIER order.

        Parameters:
        - input_array: 2D NumPy array, shape (N,8)
        - AFFNET_TO_EASIER: for each input position tells

        Returns:
        - reordered_array: 2D NumPy array with columns reordered based on the mapping.
        """

        num_columns = input_array.shape[1]
        num_samples = input_array.shape[0]

        reordered_array = np.ndarray(shape=(num_samples, EASIER_COLUMN_COUNT))
        reordered_array.fill(np.nan)

        for input_col, output_col in enumerate(AFFNET_TO_EASIER):
            assert (0 <= input_col < num_columns) and (0 <= output_col < EASIER_COLUMN_COUNT)
            reordered_array[:, output_col] = input_array[:, input_col]

        return reordered_array
