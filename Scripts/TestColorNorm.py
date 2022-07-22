
import sys
import math

import PIL
from PIL.Image import Image
import numpy as np


def pack_hist(h: list, slot_count: int = 16, max_count_per_slot: int = 96) -> list:

    in_slot_count = len(h)

    out = [0] * slot_count

    # for each output slot
    for i, slot_val in enumerate(h):
        # sum the values of all slots entering this new one
        out_index = math.floor(i / slot_count)
        assert 0 <= out_index < slot_count

        out[out_index] += h[i]

    # This is the value that will become "max_count_per_slot
    max_out = max(out)
    scale = max_count_per_slot / max_out

    for i, v in enumerate(out):
        scaled_v = int(out[i] * scale)
        assert 0 <= scaled_v <= max_count_per_slot
        out[i] = scaled_v

    return out


def print_hist(img: Image) -> None:

    bands = img.getbands()
    assert len(bands) == 3

    hist = img.histogram()

    hist_red = hist[0:256]
    hist_green = hist[256:512]
    hist_blue = hist[512:768]

    for band_i, hist in enumerate([hist_red, hist_green, hist_blue]):

        hist = pack_hist(hist)

        print("====HIST====", bands[band_i])
        for i, v in enumerate(hist):
            print("{:3d} |{}".format(i, "*" * int(v/4)))


def normalize_color(img: Image, sd_multiplier: float = 2.0) -> Image:
    """
    Normalize the color channels, separately, for the given image.
    Each color channel is centered on its mean and scaled of `sd_multiplier` times its standard deviation.
    Note: works only with RGB images.

    :param img: The input RGB image to normalize.
    :param sd_multiplier: This value is multiplied by the standard deviation to compute the scaling factor (128/
    :return: A new RGB Image with separately normalized color channels.
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

        # Save intermediate debug images
        orig_band_img = PIL.Image.fromarray(obj=img_band_data.reshape(h, w).astype('uint8'), mode='L')
        orig_band_img.save(fp="test_band_orig_{}.png".format(band_i))

        out_band_img = PIL.Image.fromarray(obj=normalized.reshape(h, w).astype('uint8'), mode='L')
        out_band_img.save(fp="test_band_normalized_{}.png".format(band_i))
        print(out_band_img.size)

        # Clip the color data in range [0,255]
        # and write the color channel into the target array
        out_img_data[:, :, band_i] = normalized.reshape(h, w).clip(0.0, 255.0)

    out_img = PIL.Image.fromarray(obj=out_img_data, mode='RGB')

    return out_img


image_filename = sys.argv[1]
print("Analysing image: ", image_filename)

img = PIL.Image.open(image_filename)
print_hist(img)

for sd in [1.0, 1.5, 2.0, 2.5, 3.0]:

    norm_img = normalize_color(img, sd_multiplier=sd)
    norm_img.save(fp="test_colnorm-sd{}.png".format(sd))
    print("Normalized sd={}".format(sd))
    print_hist(norm_img)


print("All done.")
