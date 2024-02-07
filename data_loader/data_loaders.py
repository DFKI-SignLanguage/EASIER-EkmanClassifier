import os
import glob
from operator import attrgetter,itemgetter

import torch
from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data import Dataset, DataLoader

import pandas as pd
from sklearn.utils import class_weight
import numpy as np
from PIL import Image
from pathlib import Path

from mtcnn import MTCNN

import decord


from utils.img import normalize_image_np


# Classes expected to be in the first round of annotation on the EASIER project.
EASIER_CLASSES = [
    "Happiness",  # 0
    "Sadness",  # 1
    "Surprise",  # 2
    "Fear",  # 3
    "Anger",  # 4
    "Disgust",  # 5
    "Contempt",  # 6
    "Other",  # 7
    "Neutral"  # 8
]
EASIER_CLASSES_DICT = {i: c for i, c in enumerate(EASIER_CLASSES)}


class FaceExpressionPhoenixDataset(Dataset):
    """
    Face Expression Phoenix Dataset from Alaghband et al https://arxiv.org/abs/2003.08759 .
    """
    idx_to_class = {0: "neutral",
                    1: "anger",
                    2: "disgust",
                    3: "fear",
                    4: "happy",
                    5: "sad",
                    6: "surprise",
                    7: "none"}

    dataset_stats = {
        "mean": [0.6374226, 0.5848234, 0.56568706],
        "std": [0.20125638, 0.22521368, 0.2639905]
    }

    def __init__(self, data_path, training=True, transform=None, target_transform=None):

        self.data_path = data_path
        self.images_dir_path = os.path.join(data_path, 'FePh_images')
        # self.images_dir_path = os.path.join(data_path, 'FePh_images-cropped')

        if training:
            self.labels_csv_path = os.path.join(data_path, 'FePh_train.csv')
        else:
            self.labels_csv_path = os.path.join(data_path, 'FePh_test.csv')

        self.transform = transform
        self.target_transform = target_transform

        y_df = pd.read_csv(self.labels_csv_path, dtype=str)
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

        return in_image, out_label

    def reorder_samples(self, new_idxs):
        self.image_inputs = [self.image_inputs[i] for i in new_idxs]
        self.labels = self.labels[new_idxs]

    def get_sample_weights(self, idxs):
        print("Calculating sampler weights...")
        labels_array = self.labels[idxs]
        class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(labels_array),
                                                          y=labels_array)
        num_classes = len(self.idx_to_class.keys())
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

    def get_dataset_stats(self):
        return self.dataset_stats


class FaceExpressionPhoenixDataLoader(BaseDataLoader):
    """
    Face Expression Phoenix data loading using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        mean = FaceExpressionPhoenixDataset.dataset_stats["mean"]
        std = FaceExpressionPhoenixDataset.dataset_stats["std"]
        trsfm = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.Resize((224, 224)),
        ])
        self.data_dir = data_dir

        self.dataset = FaceExpressionPhoenixDataset(data_dir, training=training, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, is_imbalanced_classes=True)

    @staticmethod
    def get_label_map():
        return FaceExpressionPhoenixDataset.idx_to_class

    @staticmethod
    def get_dataset_stats():
        return FaceExpressionPhoenixDataset.dataset_stats


class PredictionDataset(Dataset):
    """
    Wrapper class to obtain info relevant to datasets defined in this script like label_map()
    """

    def __init__(self, data_path, data_loader):
        try:
            self.idx_to_class = data_loader.get_label_map()
            self.dataset_stats = data_loader.get_dataset_stats()
        except AttributeError:
            raise AttributeError(
                "Implement a get_label_map() & get_dataset_stats() static methods similar to FaceExtractionPhoenixDataset")

        self.images_dir_path = os.path.join(data_path)
        self.image_inputs = [os.path.join(self.images_dir_path, img_name) for img_name in
                             sorted(os.listdir(self.images_dir_path)) if ".jpg" in img_name or ".png" in img_name]

    def __getitem__(self, idx):
        inp_img_name = self.image_inputs[idx]
        in_image = Image.open(inp_img_name).convert('RGB')

        size = 224, 224  # Fixed to Resnet input size
        mean = self.dataset_stats["mean"]
        std = self.dataset_stats["std"]

        tensor_trsnfrm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.Resize(size),
        ])
        in_image = tensor_trsnfrm(in_image)
        img_name = os.path.split(inp_img_name)[1]

        return in_image, img_name

    def __len__(self):
        return len(self.image_inputs)


def _save_frame_with_faces(frame: np.ndarray, face_list: list, filename: str):
    """Support method to save pictures showing all the faces detected.

    :param frame:
    :param face_list:
    :param filename:
    :return:
    """

    from PIL import ImageDraw

    #
    # Build the Image to save
    img = Image.fromarray(frame, 'RGB')
    draw = ImageDraw.Draw(img)

    for face_info in face_list:
        # NOSE
        nose_x, nose_y = face_info['keypoints']['nose']
        draw.ellipse([(nose_x-1, nose_y-1), (nose_x+1, nose_y+1)], fill=(255, 25, 25, 128), width=3)

        # BBOX
        fx, fy, fw, fh = face_info['box']
        draw.rectangle(xy=[(fx, fy), (fx+fw, fy+fh)], fill=None, outline=(255, 25, 25, 128), width=3)

        # EYES
        for eye_name in ['right_eye', 'left_eye']:
            eye_x, eye_y = face_info['keypoints'][eye_name]
            draw.ellipse([(eye_x - 1, eye_y - 1), (eye_x + 1, eye_y + 1)], fill=(25, 255, 25, 128), width=3)

        # MOUTH
        mouth_right_x, mouth_right_y = face_info['keypoints']['mouth_right']
        mouth_left_x, mouth_left_y = face_info['keypoints']['mouth_left']
        draw.line([(mouth_left_x, mouth_left_y), (mouth_right_x, mouth_right_y)], fill=(250, 250, 150, 128))

        conf = face_info['confidence']
        draw.text(xy=(nose_x, nose_y), text=f"{conf:.3f}")

    img.save(filename)


def _scale_faceinfo(face_info: dict, hscale: int, vscale: int) -> None:

    from typing import List

    def _scale_2d_point(p, hs, vs) -> List[int]:
        return [p[0] * hs, p[1] * vs]

    face_info['box'] = _scale_2d_point(face_info['box'][:2], hscale, vscale) + _scale_2d_point(face_info['box'][2:], hscale, vscale)
    kp = face_info['keypoints']
    kp['nose'] = _scale_2d_point(kp['nose'], hscale, vscale)
    kp['mouth_right'] = _scale_2d_point(kp['mouth_right'], hscale, vscale)
    kp['right_eye'] = _scale_2d_point(kp['right_eye'], hscale, vscale)
    kp['left_eye'] = _scale_2d_point(kp['left_eye'], hscale, vscale)
    kp['mouth_left'] = _scale_2d_point(kp['mouth_left'], hscale, vscale)


# This is the step used to pick pixels from np.ndarray of the video frames.
# It is used for a quick scaling o the picture before feeding it to MTCNN.
FAST_DOWNSAMPLING_STEP: int = 1

from utils.mediapipefacedetection import MediaPipeFaceDetector

class VideoFrameDataset(Dataset):
    def __init__(self, video_path, batch_size=32, transform=None,
                 face_detector = None,
                 normalization_params: dict = None):

        # Same as in the AffectNetDataset
        self.idx_to_class = {0: "neutral",
                             1: "happy",
                             2: "sad",
                             3: "surprise",
                             4: "fear",
                             5: "disgust",
                             6: "anger",
                             7: "contempt"}
        self.dataset_stats = {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        }
        ####################################

        self.video_path = video_path
        self.batch_size = batch_size
        self.transform = transform
        self.face_detector = face_detector
        self.normalization_params = normalization_params

        mean = self.dataset_stats["mean"]
        std = self.dataset_stats["std"]
        self.tensor_trsnfrm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        self.video_reader = decord.VideoReader(video_path, ctx=decord.cpu(0))
        self.frame_count = len(self.video_reader)

    def __len__(self):
        return (self.frame_count + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx):
        start_frame = idx * self.batch_size
        end_frame = min((idx + 1) * self.batch_size, self.frame_count)
        n_frames = end_frame - start_frame

        #
        # Read a whole back of video frames Using decord
        # https://medium.com/@haydenfaulkner/extracting-frames-fast-from-a-video-using-opencv-and-python-73b9b7dc9661
        frames_list = list(range(start_frame, end_frame))
        frames_batch = self.video_reader.get_batch(frames_list)
        video_frames = frames_batch.asnumpy()

        assert len(video_frames) == end_frame - start_frame

        out_frames = []

        for frame_num, frame in enumerate(video_frames):

            #
            # Normalize the image, e.g., to the same format used for training.
            if self.normalization_params is not None:

                # Detect the faces (on a scaled down version of the image
                if FAST_DOWNSAMPLING_STEP > 1:
                    lo_res_frame = frame[::FAST_DOWNSAMPLING_STEP, ::FAST_DOWNSAMPLING_STEP, :]
                else:
                    lo_res_frame = frame
                face_list = self.face_detector.detect_faces(lo_res_frame)

                #
                # Treat cases with no faces or more than 1 face
                if len(face_list) == 0:
                    print(f"No faces in batch {idx}, frame {frame_num}")
                    # Substitute the video frame with a dummy 8x8 black frame
                    frame = np.full(shape=(8, 8, 3), fill_value=191, dtype=np.uint8)
                    # We use the GREY color because it is easily predicted as NEUTRAL facial expression
                    # Light grey 191
                    # Predictions: 0.025 0.031 -0.034 -0.180 -0.170 -0.351 -0.264 nan 0.247
                    # Compose a fake face detection on the 8x8 frame.
                    face_info = {
                                        'box': [0, 0, 7, 7],
                                        'keypoints':
                                        {
                                            'nose': (4, 4),
                                            'mouth_right': (6, 6),
                                            'right_eye': (6, 2),
                                            'left_eye': (2, 2),
                                            'mouth_left': (2, 6)
                                        },
                                        'confidence': 0.99
                                    }

                elif len(face_list) > 1:
                    print(f"{len(face_list)} faces in batch {idx}, frame {frame_num}")
                    # _save_frame_with_faces(frame=frame, face_list=face_list, filename=f"batch{idx}-f{frame_num}.png")

                    face_list = sorted(face_list, key=itemgetter('confidence'), reverse=True)
                    # print(face_list)
                    # Now the face with highest confidence is the first in the list
                    face_info = face_list[0]
                else:
                    face_info = face_list[0]

                if type(self.face_detector) == MediaPipeFaceDetector:
                    h, w, _ = frame.shape
                    _scale_faceinfo(face_info=face_info, hscale=int(w), vscale=int(h))
                else:
                    # When using MTCNN, face info are already in pixel coords
                    if FAST_DOWNSAMPLING_STEP > 1:
                        _scale_faceinfo(face_info=face_info, hscale=FAST_DOWNSAMPLING_STEP, vscale=FAST_DOWNSAMPLING_STEP)

                # For DEBUG only
                # _save_frame_with_faces(frame=frame, face_list=face_list, filename=f"batch{idx}-f{frame_num}.png")

                frame = normalize_image_np(img_np=frame, mtcnn_face_info=face_info, **self.normalization_params)
                # END of frame normalization
                #

            # Append the frame to the list of final frames
            out_frames.append(frame)

        # Convert the frames into Torch tensors and stack them in a 4-dimensional array
        torch_frames = [self.tensor_trsnfrm(f) for f in out_frames]
        assert len(torch_frames) == n_frames
        torch_frames = torch.stack(torch_frames)

        return torch_frames


# ---------------------- AffectNet Dataset & DataLoader ---------------------- #

class AffectNet(Dataset):
    """
    AffectNet Dataset from Mollahosseini et al https://arxiv.org/pdf/1708.03985.pdf
    """

    idx_to_class = {0: "neutral",
                    1: "happy",
                    2: "sad",
                    3: "surprise",
                    4: "fear",
                    5: "disgust",
                    6: "anger",
                    7: "contempt"}

    dataset_stats = {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
    }

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
            try:
                np.save(cache_file, labels)
            except OSError:
                print("Cannot save affectnet labels cache file.")

        try:
            self.labels = np.load(cache_file)

        except FileNotFoundError:
            self.labels = np.array(labels)

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
    shuffle must always be false even while training as we use random sampler
    """

    def __init__(self, data_dir, batch_size, training=True, shuffle=True, num_workers=1, validation_split=0.0):

        self.validation_split = 0
        self.training = training
        self.dataset_stats = self.get_dataset_stats()
        mean = self.dataset_stats["mean"]
        std = self.dataset_stats["std"]

        trsfm = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            # transforms.RandomHorizontalFlip(1.0),  #  Always flipping the image
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.Resize((224, 224)),
        ])

        val_trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.Resize((224, 224)),
        ])

        if training:

            self.dataset = AffectNet(os.path.join(data_dir, "train_set"), transform=trsfm)
            self.val_dataset = AffectNet(os.path.join(data_dir, "val_set"), transform=val_trsfm)
            weights = self.dataset.get_sampler_weights()
            train_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=torch.DoubleTensor(weights),
                                                                           num_samples=len(self.dataset))
            shuffle = False  # Even though shuffle is false, RandomSampler will always shuffle
        else:
            trsfm = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

            self.dataset = AffectNet(os.path.join(data_dir, "val_set"), transform=val_trsfm)
            train_sampler = None
            shuffle = False

        self.shuffle = shuffle

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
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

    @staticmethod
    def get_dataset_stats():
        return AffectNet.dataset_stats


class AsavchenkoB07DataLoader(DataLoader):
    """
    Same as the AffectNetDataLoader except that the label_map is different for the Asavhenko model.
    """

    @staticmethod
    def get_label_map():
        return {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

    @staticmethod
    def get_dataset_stats():
        return AffectNet.dataset_stats
