import numpy as np

import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic
mp_face_mesh = mp.solutions.face_mesh

from typing import Union, List, Dict, Tuple


class MediaPipeFaceDetector:

    MP_FACE_LANDMARKS_COUNT = 468

    # Vertex ID for useful key points on the face
    MP_FACE_TOP = 10
    MP_FACE_BOTTOM = 152
    MP_FACE_RIGHT = 234
    MP_FACE_LEFT = 454

    MP_NOSE_TIP = 4
    MP_REYE_RCONRNER = 33
    MP_REYE_LCONRNER = 133
    MP_LEYE_RCONRNER = 362
    MP_LEYE_LCONRNER = 263

    MP_MOUTH_RCORNER = 61
    MP_MOUTH_LCORNER = 291


    def __init__(self, search_head: bool = False):
        """
        :param search_head: If True, first we try to get the body location (shoulders, nose), and search the face there
        """

        self.mp_face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.5,
                                          static_image_mode=False, refine_landmarks=False)
        # self.mp_face_detector = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
        if search_head:
            self.mp_pose_detector = mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5)
        else:
            self.mp_pose_detector = None

    @staticmethod
    def _clip(x, lo, hi):
        """clip x to the [lo,hi] range"""
        return lo if x < lo else hi if x > hi else x

    @staticmethod
    def _reflect(p: Tuple[float, float], c: Tuple[float, float]) -> Tuple[float, float]:
        """
            Reflects p over the center c.

            Returns:
                P_prime : coordinates of the mirror of p through c

        """
        P_x_prime = 2 * c[0] - p[0]
        P_y_prime = 2 * c[1] - p[1]

        return P_x_prime, P_y_prime

    def _get_head_region_info(self, image: np.ndarray) \
            -> Union[None, Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]]:
        """
            Find the ROI containing the face from an input image

            :param image: input BGR (cv2 format) image of type np.ndarray and shape [height, width, 3]
            :param pose_detector: the MediaPipe pose detector

            :returns List[nose, rshoulder, lshoulder]. Each element is a 2-size ndarray with 2D landmark coordinates in pixel space.
            Or returns None if no body could be detected
        """

        assert self.mp_pose_detector is not None, "The method was called even if the Pose detector is not initialized"

        image_height, image_width, _ = image.shape
        # Convert the BGR image to RGB before processing.
        results = self.mp_pose_detector.process(image)

        # Get the pose keypoint of the whole body
        if not results.pose_landmarks:
            return None

        # Selection of the necessary part from landmarks
        nose = (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x * image_width,
                results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * image_height)
        rshoulder = (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].x * image_width,
                     results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y * image_height)
        lshoulder = (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].x * image_width,
                     results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].y * image_height)

        return nose, rshoulder, lshoulder

    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Detects the bounds of the face using the mediapipe framework

        output dictionary has the following format:
                            face_info = {
                                        'box': [0, 0, 7, 7],  # x, y, w, h
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

        :param image: RGB image pixels
        :return: A list with the dictionaries with the face info.
        """

        # Setup original full-frame bounds
        h, w, _ = image.shape
        x, y = 0, 0

        if self.mp_pose_detector is not None:

            # Try to find information about the shoulder/nose area
            head_info = self._get_head_region_info(image=image)

            if head_info is not None:
                nose, _, lshoulder = head_info
                # print("nose,rshoulder",nose,rshoulder)
                # lshoulder is the bottom-right point
                x2, y2 = lshoulder
                # Compute the top-left
                x, y = MediaPipeFaceDetector._reflect(p=lshoulder, c=nose)
                # Ensure that the reflected point is in the image boundaries
                x = MediaPipeFaceDetector._clip(x, 0, w)
                y = MediaPipeFaceDetector._clip(y, 0, h)

                # Compute bbox and round to integers
                w = int(x2 - x)
                h = int(y2 - y)
                x = int(x)
                y = int(y)

                # crop the bbox, if a body was visible from the front
                if w > 0 and h > 0:
                    image = image[y:y + h, x:x + w]
                    # the `face_detection.process()` requires a contiguous array
                    image = np.ascontiguousarray(image)

        # Process the image with MediaPipe Face Mesh Detection.
        results = self.mp_face_mesh.process(image)

        if not results.multi_face_landmarks:  # if no face can be detected, returns an empty list
            return []

        # Accumulator for all faces found
        out_list = []

        # For each face detected, convert it into the required structure
        for mf_landmarks in results.multi_face_landmarks:
            assert len(mf_landmarks.landmark) == MediaPipeFaceDetector.MP_FACE_LANDMARKS_COUNT

            # face bbox
            face_top = mf_landmarks.landmark[MediaPipeFaceDetector.MP_FACE_TOP]
            face_bottom = mf_landmarks.landmark[MediaPipeFaceDetector.MP_FACE_BOTTOM]
            face_right = mf_landmarks.landmark[MediaPipeFaceDetector.MP_FACE_RIGHT]
            face_left = mf_landmarks.landmark[MediaPipeFaceDetector.MP_FACE_LEFT]

            nose =  mf_landmarks.landmark[MediaPipeFaceDetector.MP_NOSE_TIP]
            mouth_right = mf_landmarks.landmark[MediaPipeFaceDetector.MP_MOUTH_RCORNER]
            mouth_left = mf_landmarks.landmark[MediaPipeFaceDetector.MP_MOUTH_LCORNER]
            eye_right_rcorner = mf_landmarks.landmark[MediaPipeFaceDetector.MP_REYE_RCONRNER]
            eye_right_lcorner = mf_landmarks.landmark[MediaPipeFaceDetector.MP_REYE_LCONRNER]
            eye_left_rcorner = mf_landmarks.landmark[MediaPipeFaceDetector.MP_LEYE_RCONRNER]
            eye_left_lcorner = mf_landmarks.landmark[MediaPipeFaceDetector.MP_LEYE_LCONRNER]

            # adjust to match more or less the MTCNN size
            face_height = face_bottom.y - face_top.y
            face_width = face_left.x - face_right.x
            # Extend up/down
            face_top.y -= face_height * 0.22
            face_bottom.y += face_height * 0.06
            # Extend left/right
            face_right.x -= face_width * 0.03
            face_left.x += face_width * 0.03

            # TODO -- add offsets if head was cropped first
            #bbox_face = result.location_data.relative_bounding_box
            # map back to the original pixel space
            #xm, ym, wm, hm = bbox_face.xmin * w, bbox_face.ymin * h, bbox_face.width * w, bbox_face.height * h
            #x += int(xm)
            #y += int(ym)


            info_dict = {
                'box': [face_right.x, face_top.y, face_left.x - face_right.x, face_bottom.y - face_top.y],  # x, y, w, h
                'keypoints':
                    {
                        'nose': (nose.x, nose.y),
                        'mouth_right': (mouth_right.x, mouth_right.y),
                        'mouth_left': (mouth_left.x, mouth_left.y),
                        'right_eye': ( (eye_right_rcorner.x + eye_right_lcorner.x) / 2,  (eye_right_rcorner.y + eye_right_lcorner.y) / 2),
                        'left_eye': ( (eye_left_rcorner.x + eye_left_lcorner.x) / 2,  (eye_left_rcorner.y + eye_left_lcorner.y) / 2)
                    },
                'confidence': 1.0
            }

            out_list.append(info_dict)


        return out_list
