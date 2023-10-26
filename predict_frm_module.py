from video import VideoEkmanPredictor
import numpy as np

pr = VideoEkmanPredictor.VideoEkmanPredictor()

pr.load("saved/models/MNet-AfNet-CropRot-220623/model_best.pth",
        "saved/models/MNet-AfNet-CropRot-220623/config_orig.json")
# Below values keep the columns in the Affectnet format
ekman_values = pr.predict(r'/Users/chbh01/Documents/Codebases/DFKI/ACGCode/EASIER/act2.avi')
# Reorder columns to Easier format. The format is same as EAsier format with the exception that there is no "Other"
# class and therefore there are only 8 cols
# instead of 9 as in the Easier format ==> look at map_afnet_to_easierclss in VideoEkmanPredictor()
# ekman_values = pr.reorder_columns(ekman_values)
print(ekman_values.shape)

assert type(ekman_values) == np.ndarray

n_frames = ekman_values.shape[0]

assert ekman_values.shape[1] == 7 + 1