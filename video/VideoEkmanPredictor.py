import numpy as np
from tqdm import tqdm
import argparse
import torch
import model.model as module_arch
from parse_config import ConfigParser
from data_loader.data_loaders import VideoFrameDataset
from torch.utils.data import DataLoader


class VideoEkmanPredictor:
    def __init__(self):
        self.model = None
        self.device = None
        self.config = None

    def load(self, model_pth, config_pth):
        args = argparse.ArgumentParser(description='Generates the predictions for a given (already trained) model.'
                                                   'Predictions are in JSON format as dictionary')
        args.add_argument('-c', '--config', default=config_pth, type=str, required=False,
                          help='config file path')
        args.add_argument('-d', '--device', default=None, type=str, required=False,
                          help='indices of GPUs to enable (default: all)')
        args.add_argument('-p', "--predict", action="store_true", required=False)

        args.add_argument('-m', '--model', default=model_pth, type=str, required=False,
                          help='path to binary saved prediction model')
        args.add_argument('-i', '--input', default=None, type=str, required=False,
                          help='path to a directory of images to analyse')
        args.add_argument('-o', '--output', default=None, type=str, required=False,
                          help='path to a CSV file that will contain the predictions.'
                               ' CSV header is ["imgname", "neutral", "anger", "disgust", "fear", "happy", "sad", "surprise", "none"].'
                               ' Data is in 1-hot format ')

        self.config = ConfigParser.from_args(args)

        # build model architecture
        model = self.config.init_obj('arch', module_arch)

        checkpoint = torch.load(self.config.resume, map_location='cpu')
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)

        # prepare model for testing
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.model.eval()

    def predict(self, in_video_pth):

        video_dataset = VideoFrameDataset(in_video_pth, batch_size=32, transform=None)
        test_data_loader = DataLoader(video_dataset, batch_size=None)  # None for dynamic batch size

        out_all_softmax = []
        predictions = []
        with torch.no_grad():
            for i, (data) in enumerate(tqdm(test_data_loader)):
                data = torch.Tensor(data).to(self.device)
                output_softmax = self.model(data)
                output_softmax = output_softmax.cpu().numpy()
                out_all_softmax.append(output_softmax)
                output = np.argmax(output_softmax, axis=1)
                predictions.append(output)


        out_all_softmax = np.vstack(out_all_softmax)


        return out_all_softmax
