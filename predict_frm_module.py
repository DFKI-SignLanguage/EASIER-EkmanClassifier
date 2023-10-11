from video import VideoEkmanPredictor
import numpy as np

pr = VideoEkmanPredictor.VideoEkmanPredictor()

pr.load("saved/models/ResNet50-AffNet-nopreproc-210921/ResNet50_affectnet.pth",
        "config-AffectNet.json")
ekman_values = pr.predict(r'/Users/chbh01/Documents/Codebases/DFKI/ACGCode/EASIER/act2.avi')
print(ekman_values.shape)

assert type(ekman_values) == np.ndarray

n_frames = ekman_values.shape[0]

assert ekman_values.shape[1] == 7 + 1