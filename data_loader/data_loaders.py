import os
import glob
from operator import itemgetter
# from multiprocessing import Pool
from multiprocessing import Manager
from multiprocessing.pool import Pool
import multiprocessing

from concurrent.futures import ThreadPoolExecutor

import torch
from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data import Dataset, DataLoader

import pandas as pd
from sklearn.utils import class_weight
import numpy as np
from PIL import Image
from pathlib import Path
import cv2

# See https://github.com/ipazc/mtcnn
from mtcnn import MTCNN

from utils.img import normalize_image, normalize_image_np

from typing import Tuple

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
        nose_x, nose_y = face_info['keypoints']['nose']
        # draw.point([(nose_x, nose_y)], fill=(255, 25, 25, 128))
        draw.ellipse([(nose_x-1, nose_y-1), (nose_x+1, nose_y+1)], fill=(255, 25, 25, 128), width=3)

        fx, fy, fw, fh = face_info['box']
        draw.rectangle(xy=[(fx, fy), (fx+fw, fy+fh)], fill=None, outline=(255, 25, 25, 128), width=3)

        conf = face_info['confidence']
        draw.text(xy=(nose_x, nose_y), text=f"{conf:.3f}")

    img.save(filename)


class VideoFrameDataset(Dataset):
    def __init__(self, video_path, batch_size=32, transform=None, normalization_params=None):

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
        self.normalization_params = normalization_params

        # For sharing a dictionary among Processes, see
        # https://stackoverflow.com/questions/6832554/multiprocessing-how-do-i-share-a-dict-among-multiple-processes

        def process_init(mtcnn_instances: dict):
            pid = os.getpid()
            print(f"Allocating MTCNN for process pid {pid}...")
            mtcnn_instances[pid] = MTCNN(min_face_size=50)

        self.process_pool_manager = Manager()
        self.mtcnn_instances_dict = self.process_pool_manager.dict()
        self.mtcnn_instances_dict['test'] = "Test"

        # self.process_pool = Pool(processes=8, initializer=process_init, initargs=[self.mtcnn_instances_dict])
        # ctx = multiprocessing.get_context('fork')
        self.process_pool = Pool(processes=8)

        print("MTCNN DICT SIZE", len(self.mtcnn_instances_dict))

        # self.mtcnn_instances = []
        # for i in range(batch_size):
        #     print(f"Allocating MTCNN {i}...")
        #     self.mtcnn_instances.append(MTCNN(min_face_size=50))
        # print(f"Allocated {len(self.mtcnn_instances)} instances of MTCNN.")
        # self.mtcnn_face_detector = self.mtcnn_instances[0]
        self.mtcnn_face_detector = MTCNN(min_face_size=50)

        mean = self.dataset_stats["mean"]
        std = self.dataset_stats["std"]
        self.tensor_trsnfrm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        self.cap = cv2.VideoCapture(video_path)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return (self.frame_count + self.batch_size - 1) // self.batch_size

    def __getitem__(self, batch_num):
        # return self._getitem_sequential(batch_num)
        return self._getitem_parallel(batch_num)

    def _getitem_parallel(self, batch_num):
        start_frame = batch_num * self.batch_size
        end_frame = min((batch_num + 1) * self.batch_size, self.frame_count)
        n_frames = end_frame - start_frame

        video_frames = []
        for frame_num in range(start_frame, end_frame):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = self.cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_frames.append(frame)
            if not ret:
                break

        assert len(video_frames) == end_frame - start_frame

        # processes_info = [(batch_num, i, f, self.mtcnn_instances[i], self.normalization_params) for i, f in enumerate(video_frames)]
        processes_info = [(batch_num, i, f, self.mtcnn_instances_dict, self.normalization_params) for i, f in
                          enumerate(video_frames)]
        #with ThreadPoolExecutor(max_workers=8) as P:
        #with Pool(8) as P:
        #    out_frames = P.map(_detect_face_and_normalize_image, processes_info)

        #print("KKKKKK", len(self.mtcnn_instances_dict), len(self.mtcnn_instances_dict.keys()))
        processes_info = [(batch_num, i, f, self.mtcnn_instances_dict, self.normalization_params) for i, f in
                          enumerate(video_frames)]
        out_frames = self.process_pool.map(_detect_face_and_normalize_image, processes_info)

        # if self.normalization_params is not None:
        #     normalization_params = [(fr, par) for fr, par in zip(out_frames, face_info_list)]
        #     print("RRR", len(out_frames), len(face_info_list), len(normalization_params))
        #     np_params = np.stack(normalization_params)
        #     normalized_frames = self.normalization_ufunc(np_params)
        #     print("SSSS", normalized_frames.shape)

        # Convert the frames into Torch tensors and stack them in a 4-dimensional array
        torch_frames = [self.tensor_trsnfrm(f) for f in out_frames]
        assert len(torch_frames) == n_frames
        torch_frames = torch.stack(torch_frames)

        return torch_frames

    def _getitem_sequential(self, idx):
        start_frame = idx * self.batch_size
        end_frame = min((idx + 1) * self.batch_size, self.frame_count)
        n_frames = end_frame - start_frame

        video_frames = []
        for frame_num in range(start_frame, end_frame):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = self.cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_frames.append(frame)
            if not ret:
                break

        assert len(video_frames) == end_frame - start_frame

        face_info_list = []
        out_frames = []

        for frame_num, frame in enumerate(video_frames):
            #
            # Normalize the image, e.g., to the same format used for training.
            if self.normalization_params is not None:

                # Detect the faces
                face_list = self.mtcnn_face_detector.detect_faces(frame)

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

                face_info_list.append(face_info)

                frame = normalize_image_np(img_np=frame, mtcnn_face_info=face_info, **self.normalization_params)
                # END of frame normalization
                #

            # Append the frame to the list of final frames
            out_frames.append(frame)


        # if self.normalization_params is not None:
        #     normalization_params = [(fr, par) for fr, par in zip(out_frames, face_info_list)]
        #     print("RRR", len(out_frames), len(face_info_list), len(normalization_params))
        #     np_params = np.stack(normalization_params)
        #     normalized_frames = self.normalization_ufunc(np_params)
        #     print("SSSS", normalized_frames.shape)

        # Convert the frames into Torch tensors and stack them in a 4-dimensional array
        torch_frames = [self.tensor_trsnfrm(f) for f in out_frames]
        assert len(torch_frames) == n_frames
        torch_frames = torch.stack(torch_frames)

        return torch_frames

    def _getitem_orig(self, idx):
        start_frame = idx * self.batch_size
        end_frame = min((idx + 1) * self.batch_size, self.frame_count)
        size = 224, 224  # Fixed to Resnet input size
        mean = self.dataset_stats["mean"]
        std = self.dataset_stats["std"]
        frames = []
        for frame_num in range(start_frame, end_frame):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = self.cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if not ret:
                break

            #
            # Normalize the image, e.g., to the same format used for training.
            #
            # Convert frame into PIL format
            if self.normalization_params is not None:
                in_frame_pil = Image.fromarray(frame)
                out_frame_pil = normalize_image(img=in_frame_pil, **self.normalization_params)
                # PIL back to ndarray
                frame = np.asarray(out_frame_pil)

            #
            #
            tensor_trsnfrm = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
                transforms.Resize(size),
            ])

            frame = tensor_trsnfrm(frame)
            frames.append(torch.Tensor(frame))

        return torch.stack(frames)


def _detect_face_and_normalize_image(info: Tuple[int, int, np.ndarray, dict, dict]) -> np.ndarray:
    # print("INFO", info)
    batch_num: int = info[0]
    frame_num: int = info[1]
    frame: np.ndarray = info[2]
    # data_loader: VideoFrameDataset = info[3]
    # mtcnn_face_detector = info[3]
    mtcnn_face_detector_dict = info[3]
    normalization_params = info[4]
    # batch_num is the batch number (for debug)
    # frame_num is the position in the array
    # frame (in numpy format) is the image to process
    # data_loader is the reference to the VideoFrameDataset, from which we can get useful parameters

    #
    # Normalize the image, e.g., to the same format used for training.
    if normalization_params is not None:

        # Detect the faces
        # mtcnn_face_detector = data_loader.mtcnn_instances[frame_num]
        # mtcnn_face_detector = mtcnn_face_detector_dict[os.getpid()]
        print("DDDDD", type(mtcnn_face_detector_dict), len(mtcnn_face_detector_dict), mtcnn_face_detector_dict)
        pid = os.getpid()
        if pid not in mtcnn_face_detector_dict:
            mtcnn_face_detector = MTCNN(min_face_size=50)
            mtcnn_face_detector_dict[pid] = mtcnn_face_detector
        else:
            mtcnn_face_detector = mtcnn_face_detector_dict[pid]
        print("EEEE", type(mtcnn_face_detector_dict), len(mtcnn_face_detector_dict), mtcnn_face_detector_dict)

        face_list = mtcnn_face_detector.detect_faces(frame)

        #
        # Treat cases with no faces or more than 1 face
        if len(face_list) == 0:
            print(f"No faces in batch {batch_num}, frame {frame_num}")
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
            print(f"{len(face_list)} faces in batch {batch_num}, frame {frame_num}")
            # _save_frame_with_faces(frame=frame, face_list=face_list, filename=f"batch{idx}-f{frame_num}.png")

            face_list = sorted(face_list, key=itemgetter('confidence'), reverse=True)
            # print(face_list)
            # Now the face with highest confidence is the first in the list
            face_info = face_list[0]
        else:
            face_info = face_list[0]

        frame = normalize_image_np(img_np=frame, mtcnn_face_info=face_info, **normalization_params)
        # END of frame normalization
        #

    return frame


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
