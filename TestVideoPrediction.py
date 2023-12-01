from utils.video import VideoEkmanPredictor
import numpy as np

import time

#
# Create an instanace of the class.
pr = VideoEkmanPredictor()

#
# Load the prediction model and its configuration file
pr.load("Models/MNet-AfNet-CropRot-220623/model_best.pth",
        "Models/MNet-AfNet-CropRot-220623/config.json")

#
# After Loading, many inferences can be run.
#

#
# Run the inference

before = time.time()
ekman_values = pr.predict('Models/TestTwoExpressions.mp4')
# ekman_values = pr.predict('Models/kGetUczx7SdsWBEfA6mImAxx.mp4')
# ekman_values = pr.predict("Models/TestEmptyBackground.mov")
after = time.time()

# And reorder the output in the EASIER format
ekman_values = pr.reorder_columns(ekman_values)

#
# Check and print results
assert type(ekman_values) == np.ndarray
print("Prediction shape:", ekman_values.shape)

n_frames = ekman_values.shape[0]
print("Number of predicted frames:", n_frames)

assert ekman_values.shape[1] == 7 + 1 + 1  # 7 Ekman + Other + Neutral

print("PREDICTIONS >>>>>")
for i in range(n_frames):
    prediction_line = ekman_values[i]
    pretty_line = ' '.join(["{:.3f}".format(v) for v in prediction_line])
    print(i, pretty_line)
print("<<<<<<<")

prediction_time_secs = after - before
prediction_fps = n_frames / prediction_time_secs

print(f"Processed {n_frames} frames in {prediction_time_secs:.2f} seconds ({prediction_fps:.2f} fps)")

print("All done.")
