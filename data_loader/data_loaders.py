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
import cv2

from mtcnn import MTCNN
from retinaface import RetinaFace


from utils.img import normalize_image, normalize_image_np


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
        # draw.point([(nose_x, nose_y)], fill=(255, 25, 25, 128))
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


class VideoFrameDataset(Dataset):
    def __init__(self, video_path, batch_size=32, transform=None, mtcnn_face_detector: MTCNN=None, normalization_params=None):

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
        self.mtcnn_face_detector = mtcnn_face_detector
        self.normalization_params = normalization_params

        mean = self.dataset_stats["mean"]
        std = self.dataset_stats["std"]
        self.tensor_trsnfrm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        self.cap = cv2.VideoCapture(video_path)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Stuff to parallelize image normalization
        def normalization_wrapper(in_data):
            print("LLL", len(in_data))
            img_np, mtcnn_face_info = in_data
            return normalize_image_np(img_np=img_np, mtcnn_face_info=mtcnn_face_info, **self.normalization_params)
        #self.normalization_ufunc = np.frompyfunc(normalization_wrapper, nin=1, nout=1)
        self.normalization_ufunc = np.vectorize(normalization_wrapper)

    def __len__(self):
        return (self.frame_count + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx):
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


        out_frames = []
        face_info_list = []

        for frame_num, frame in enumerate(video_frames):

            #
            # Normalize the image, e.g., to the same format used for training.
            if self.normalization_params is not None:

                # Detect the faces
                # face_list = self.mtcnn_face_detector.detect_faces(frame)
                retina_face_dict = RetinaFace.detect_faces(frame)
                # print(retina_face_dict)
                # Convert the retina format infor the MTCNN format
                face_list = []
                for ret_face_key in retina_face_dict.keys():
                    ret_face = retina_face_dict[ret_face_key]
                    x0,y0,x1,y1 = ret_face['facial_area']
                    face_entry = {
                        'box': [x0, y0, x1-x0+1, y1-y0+1],
                        'keypoints':
                            {
                                'nose': ret_face['landmarks']['nose'],
                                'mouth_right': ret_face['landmarks']['mouth_right'],
                                'mouth_left': ret_face['landmarks']['mouth_left'],
                                'right_eye': ret_face['landmarks']['right_eye'],
                                'left_eye': ret_face['landmarks']['left_eye']
                            },
                        'confidence': ret_face['score']
                    }
                    face_list.append(face_entry)

                # _save_frame_with_faces(frame=frame, face_list=face_list, filename=f"batch{idx}-f{frame_num}.png")


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

    def __getitem__orig_(self, idx):
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
