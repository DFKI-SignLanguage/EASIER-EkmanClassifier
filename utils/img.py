import math

import numpy as np

import cv2

import PIL
import PIL.Image
from PIL.Image import Image
from PIL import ImageDraw, Image

# See https://github.com/ipazc/mtcnn
from mtcnn import MTCNN


from typing import Tuple, Optional

# If True, some visual information (dots, lines) are painted on the output image for visual debug purposes
DEBUG_DRAW = False


def _normalize_image_meanstd(img: Image, sd_multiplier: float = 2.5) -> Image:
    """Normalize the color band of an image. For each channel separately (RGB), the channel will be centered
    at its mean and the center + SD*k will be scaled to fit the whole 0-255 range.

    :param img:
    :param sd_multiplier:
    :return:
    """

    bands = img.getbands()
    assert len(bands) == 3

    w, h = img.size

    # Prepare an array for the output image
    out_img_data = np.ndarray(shape=(h, w, 3), dtype='uint8')

    for band_i in range(len(bands)):

        img_band_data = np.asarray(img.getdata(band=band_i))
        band_mean = np.mean(img_band_data)
        band_std = np.std(img_band_data)

        # Center to 0
        centered = img_band_data - band_mean
        # rescale according to std
        # So that 2.5 times std goes to 128
        scaled = centered * 128.0 / (sd_multiplier * band_std)
        # recenter on [0,256]
        normalized = scaled + band_mean

        # Clip the color data in range [0,255]
        # and write the color channel into the target array
        out_img_data[:, :, band_i] = normalized.reshape(h, w).clip(0.0, 255.0)

    out_img = PIL.Image.fromarray(obj=out_img_data, mode='RGB')

    return out_img


def _normalize_image_histeq(img: Image) -> Image:
    """
    See: https://docs.opencv.org/4.x/d4/d1b/tutorial_histogram_equalization.html

    :param img:
    :return:
    """

    bands = img.getbands()
    assert len(bands) == 3

    # Prepare an array for the output image
    w, h = img.size
    out_img_data = np.ndarray(shape=(h, w, 3), dtype='uint8')


    for band_i in range(len(bands)):

        img_band_data = np.asarray(img.getdata(band=band_i)).astype(np.uint8)
        normalized_band_data = cv2.equalizeHist(src=img_band_data)

        # Clip the color data in range [0,255]
        # and write the color channel into the target array
        out_img_data[:, :, band_i] = normalized_band_data.reshape(h, w)


    out_img = PIL.Image.fromarray(obj=out_img_data, mode='RGB')

    return out_img



def _scale_bbox(x: float, y: float, width: float, height: float, scale: float) -> Tuple[float, float, float, float]:
    """
    Scales a bounding box around its center

    :param x:
    :param y:
    :param width:
    :param height:
    :param scale: Scaling factor. 1.0 leave to original size.
    :return:
    """

    # Compute the bottom-right corner coords
    x2 = x + width - 1
    y2 = y + height - 1

    cx = (x + x2) / 2
    cy = (y + y2) / 2

    # Prepare the transformation matrices

    # The matrix to extend the box
    scaling_matrix = np.array([[scale, 0,     0],
                               [0,     scale, 0],
                               [0,     0,     1]])

    # The matrix to center the box around the center, before scaling
    to_origin_matrix = np.array([[1, 0, -cx],
                                 [0, 1, -cy],
                                 [0, 0, 1]])

    back_translation_matrix = np.linalg.inv(to_origin_matrix)

    t = back_translation_matrix @ scaling_matrix @ to_origin_matrix

    nx, ny, o = t @ (x, y, 1)
    nx2, ny2, o2 = t @ (x2, y2, 1)

    nwidth = nx2 - nx + 1
    nheight = ny2 - ny + 1

    return nx, ny, nwidth, nheight


def normalize_image(img: Image,
                    mtcnn_face_detector: MTCNN,
                    normalize_color: bool,
                    square: bool, bbox_scale: Optional[float],
                    rotate: bool, rot_filter = PIL.Image.NEAREST) -> Image:

    """Scans files in a directory.
    For each image ending in a recognized format, detect the position of a face, crop the image,
    and save the cropped result in the destination directory.

    :param img: The input image. A face will be searched in it, and properly cropper, rotated, scaled.
    :param mtcnn_face_detector: Ths instance of MTCNN that will be used to detect the face.
    :param square: If True, the face bounds will be extended to be squared.
    :param bbox_scale: A float number scaling the edges of the cropping rectangle around its center.
    Values <1 will shrink the bbox, =1 has no effect, >1 will expand teh bbox.
    :param rotate: Whether the image should be rotate to bring the eyes at the horizontal level.
    :param rot_filter: The rotation interpolation filter. Values (int) taken from the PIL library.
    """

    #
    # Detect the face (requires numpy array format)
    img_np = np.asarray(img)
    # Check for the presence of alpha channel
    # In case, remove it because the face detector doesn't support it.
    depth = img_np.shape[2]
    if depth == 4:
        # Drop the alpha channel
        print("WARNING: Dropping alpha channel...")
        img_np = img_np[:, :, :3]
        img = PIL.Image.fromarray(img_np, 'RGB')
    elif depth == 3:
        pass
    else:
        assert False

    #
    # Ask MTCNN to find the faces
    # print(img_np.shape)
    face_list = mtcnn_face_detector.detect_faces(img_np)
    # print(face_list)

    face = None

    if len(face_list) == 0:
        # No faces?
        raise Exception("No faces detected!")
    elif len(face_list) > 1:
        # More faces?
        raise Exception(f"more than one face detected: {len(face_list)}")
    else:
        # Take the first face by default
        face = face_list[0]

    bbox = face['box']
    # bbox format is [x, y, width, height]
    x, y, width, height = bbox

    # ... and take note of the eyes position
    kpoints = face['keypoints']
    eye_r = np.asarray(kpoints['right_eye'])
    eye_l = np.asarray(kpoints['left_eye'])

    # Do we want a squared output?
    if square:
        if width > height:
            # extends up and down
            dy = width - height
            top_dy = int(dy / 2)
            y = y - top_dy
            height = height + dy
        elif width < height:
            # extends left and right
            dx = height - width
            left_dx = int(dx / 2)
            x = x - left_dx
            width = width + dx
        else:
            pass

        assert width == height

    #
    # Scale the box
    if bbox_scale is not None:
        x, y, width, height = _scale_bbox(x, y, width, height, bbox_scale)
        x = round(x)
        y = round(y)
        width = round(width)
        height = round(height)

    img_cropped = img.crop((x, y, x + width, y + height))

    #
    # Color normalization
    if normalize_color:
        img_cropped = _normalize_image_meanstd(img=img_cropped)

    # Bring eyes to cropped coordinates
    eye_r[0] -= x
    eye_l[0] -= x
    eye_r[1] -= y
    eye_l[1] -= y

    if DEBUG_DRAW:
        draw = ImageDraw.Draw(img_cropped)
        draw.point([(eye_r[0], eye_r[1]), (eye_l[0], eye_l[1])])

    #
    # Rotate to get the eyes at an horizontal level
    if rotate:
        # From Savchenko...
        # theta=math.degrees(math.atan((right_eye_y-left_eye_y)/(right_eye_x-left_eye_x)))
        theta = math.atan((eye_r[1] - eye_l[1]) / (eye_r[0] - eye_l[0]))
        theta_degs = math.degrees(theta)
        img_cropped = img_cropped.rotate(angle=theta_degs, resample=rot_filter)

    # Beware! If the image crop area is outside of the visible area, PIL (at least Pillow==8.3.1)
    # adds an alpha channel and sets the out bounds to black color with 0 on the alpha channel.
    # Hence, we test again if there is an alpha channel and possibly remove it.
    if img_cropped.mode == 'RGBA':
        img_cropped = img_cropped.convert('RGB')

    assert img_cropped is not None

    return img_cropped


def normalize_image_np(img_np: np.ndarray,
                       face_info: dict,
                       color_normalization: str,  ##Literal["mean_std", "hist_eq", None],
                       square: bool, bbox_scale: Optional[float],
                       rotate: bool, rot_filter = PIL.Image.NEAREST,
                       scale: Tuple[int, int] = None) -> np.ndarray:

    """Apply normalization to a frame.

    :param scale: Scale the output image to the specified resolution. Useful to anticipate the input resolution of CNNs.
    :param img_np: The input image. A face will be searched in it, and properly cropper, rotated, scaled.
    :param face_info: the MTCNN dictionary entry with the info about the face found that should be cropped.
    :param color_normalization: Select None, or the type or color normalization that you want to use.
    :param square: If True, the face bounds will be extended to be squared.
    :param bbox_scale: A float number scaling the edges of the cropping rectangle around its center.
    Values <1 will shrink the bbox, =1 has no effect, >1 will expand teh bbox.
    :param rotate: Whether the image should be rotate to bring the eyes at the horizontal level.
    :param rot_filter: The rotation interpolation filter. Values (int) taken from the PIL library.
    """

    # Check for the presence of alpha channel
    # In case, remove it because the face detector doesn't support it.
    # depth = img_np.shape[2]
    # if depth == 4:
    #     # Drop the alpha channel
    #     print("WARNING: Dropping alpha channel...")
    #     img_np = img_np[:, :, :3]
    # elif depth == 3:
    #     pass
    # else:
    #     assert False

    # Convert into PIL format
    img = PIL.Image.fromarray(img_np, 'RGB')

    bbox = face_info['box']
    # bbox format is [x, y, width, height]
    x, y, width, height = bbox

    # ... and take note of the eyes position
    kpoints = face_info['keypoints']
    eye_r = np.asarray(kpoints['right_eye'])
    eye_l = np.asarray(kpoints['left_eye'])

    # Do we want a squared output?
    if square:
        if width > height:
            # extends up and down
            dy = width - height
            top_dy = int(dy / 2)
            y = y - top_dy
            height = height + dy
        elif width < height:
            # extends left and right
            dx = height - width
            left_dx = int(dx / 2)
            x = x - left_dx
            width = width + dx
        else:
            pass

        assert width == height

    #
    # Scale the box
    if bbox_scale is not None:
        x, y, width, height = _scale_bbox(x, y, width, height, bbox_scale)
        x = round(x)
        y = round(y)
        width = round(width)
        height = round(height)

    img_cropped = img.crop((x, y, x + width, y + height))

    #
    # Color normalization
    if color_normalization is None:
        pass
    elif color_normalization == "mean_std":
        img_cropped = _normalize_image_meanstd(img=img_cropped)
    elif color_normalization == "hist_eq":
        img_cropped = _normalize_image_histeq(img=img_cropped)
    else:
        raise Exception(f"Unknown color normalization technique {color_normalization}")

    # Bring eyes to cropped coordinates
    eye_r[0] -= x
    eye_l[0] -= x
    eye_r[1] -= y
    eye_l[1] -= y

    if DEBUG_DRAW:
        draw = ImageDraw.Draw(img_cropped)
        draw.point([(eye_r[0], eye_r[1]), (eye_l[0], eye_l[1])])

    #
    # Rotate to get the eyes at an horizontal level
    if rotate:
        # From Savchenko...
        # theta=math.degrees(math.atan((right_eye_y-left_eye_y)/(right_eye_x-left_eye_x)))
        theta = math.atan((eye_r[1] - eye_l[1]) / (eye_r[0] - eye_l[0]))
        theta_degs = math.degrees(theta)
        img_cropped = img_cropped.rotate(angle=theta_degs, resample=rot_filter)

    # Beware! If the image crop area is outside of the visible area, PIL (at least Pillow==8.3.1)
    # adds an alpha channel and sets the out bounds to black color with 0 on the alpha channel.
    # Hence, we test again if there is an alpha channel and possibly remove it.
    if img_cropped.mode == 'RGBA':
        img_cropped = img_cropped.convert('RGB')

    assert img_cropped is not None

    if scale is not None:
        img_scaled = img_cropped.resize(size=scale)
    else:
        img_scaled = img_cropped

    assert img_scaled is not None

    img_scaled_np = np.asarray(img_scaled)

    return img_scaled_np


def save_frame_with_faces(frame: np.ndarray, face_list: list, filename: str):
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
        draw.rectangle(xy=((fx, fy), (fx+fw, fy+fh)), fill=None, outline=(255, 25, 25, 128), width=3)

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
