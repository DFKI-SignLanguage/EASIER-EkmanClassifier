
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


def normalize_color(img: Image) -> Image:

    bands = img.getbands()
    assert len(bands) == 3

    w, h = img.size

    out_img_data = np.ndarray(shape=(h, w, 3), dtype='uint8')

    for band_i in range(len(bands)):

        img_band_data = np.asarray(img.getdata(band=band_i))
        band_mean = np.mean(img_band_data)
        band_std = np.std(img_band_data)

        # Center to 0
        centered = img_band_data - band_mean
        # rescale according to std
        # So that 2.5 times std goes to 128
        scaled = centered * 128.0 / (2.5 * band_std)
        # recenter on [0,256]
        normalized = scaled + band_mean

        # Save intermediate debug images
        orig_band_img = PIL.Image.fromarray(obj=img_band_data.reshape(w, h).astype('uint8'), mode='L')
        orig_band_img.save(fp="test_band_orig_{}.png".format(band_i))

        out_band_img = PIL.Image.fromarray(obj=normalized.reshape(w, h).astype('uint8'), mode='L')
        out_band_img.save(fp="test_band_normalized_{}.png".format(band_i))
        print(out_band_img.size)

        out_img_data[:, :, band_i] = normalized.reshape(w, h).clip(0.0, 255.0)

    out_img = PIL.Image.fromarray(obj=out_img_data, mode='RGB')

    return out_img


image_filename = sys.argv[1]
print("Analysing image: ", image_filename)

img = PIL.Image.open(image_filename)
print_hist(img)

norm_img = normalize_color(img)
norm_img.save(fp="test_colnorm.png")

print("Normalized")
print_hist(norm_img)

print("All done.")