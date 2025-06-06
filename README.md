# EASIER-EkmanClassifier

This is the code repository for the Ekman Classifier developed within the WP7 of the [EASIER](https://www.project-easier.eu) project.
The goal is to implement an Ekman facial expression classifier and tune & test it for Sign Language.

This codebase is based on the following [PytorchTemplate](https://github.com/victoresque/pytorch-template).


## Installation

Create a Python environment with the requirements listed in `requirements.txt`

    python -m venv p3env
    source p3env/bin/activate
    pip install -r requirements.txt

## Training

TODO

## Prediction

Use the script `predict.py` to generate labels given an input model and new images.

    python predict.py -p -c <path_to_training config_file.json> -m <path_to_trained_model.pth> -i <images_dir> -o <predictions.csv>

Prediction results are in to format:

    ImageName,label1,label2,...,labelN,ClassName,Class

Where columns labelX will contain the softmax output estimation in range [0,1].

## Testing

TODO


## Automated Prediction and Testing


The directory `PredictionLab` contains a `MakeFile` for automating the process of normalizing images, predicting, and testing models across datasets.
The goal is to flexibly test the prediction power of a model over many other datasets.

The Makefile requires the presence of three environment variables and the presence of ground truth labels and other info in a structured way.
Example:

```
export IMG_PREPROC=Crop11Rot
export DATADIR=<path_to>/data/DaFEx
export MODELDIR=<path_to>/models/ResNet50-AffNet-nopreproc-210921
```

The `IMG_PREPROC` indicates which normalization technique must be applied to the images before being fed to the prediction model.
Valid values are:

* NoProc - no image processing. Image as it is
* CropRot - face is cropped and then rotated so the eyes end on an horizontal line
* Crop11Rot - after cropping, the face bbox is scaled by 1.1 on the borders, actually "zooming-out" from the image (face gets smaller, more surrounding details are revealed).
* Crop12Rot - as before, but with 1.2 zooming factor
* CropRotBL - Rotation is performed with a bilinear filter

* CNormCrop11Rot - before cropping and rotating, a color normalization is performed, by centering RGB channels on the channels mean, and normalizing on 2.5 SD on the full channel dynamic range.
* CNormCrop12Rot - same as before, with different zoom factor

* CNormEqCrop11Rot - as before, but applying also color normalization through Histogram Equalization
* CNormEqCrop12Rot - as before, but applying also color normalization through Histogram Equalization

The DATADIR and the MODELDIR are structured like this:

```
<DATADIR>/
    images/                        -- Directory containing all the images used for testing referenced in the labels.csv file
    labels.csv                     -- A table with the name of the images used for testing and (optionally) their ground truth
    labels-normalization-map.json  -- A table converting labels into the EASIER normalized order and names

<MODELDIR>/
    model_best.pth                 -- The best model of your training session
    config.json                    -- The same config used for training
    labels-normalization-map.json  -- Normally, the same defined for the dataset used for training
```

and:

* `labels.csv` is a dataframe with the following structure
    * `ImageName`: the name of the image file, that will be searched in the `images folder`
    * `ClassName` (optional): the name of the predicted class
    * `Class` (optional): the corresponding integer class number. It is your responsibility to be consistent between label name and number.
      ```
      ImageName;ClassName;Class
      DaFEx-act1-Anger-high.png,anger,6
      DaFEx-act1-Anger-low.png,anger,6
      DaFEx-act1-Anger-mid.png,anger,6
      DaFEx-act1-Happiness-high.png,happy,1
      ```
* `labels-normalization-map.json` is a list of pairs mapping the name of the labels of the specific dataset into the _normalized_ label list used for the EASIER project.
   ```
   [
     ["neutral", "Neutral"],
     ["happy", "Happiness"],
     ["sad", "Sadness"],
     ["surprise", "Surprise"],
     ["fear", "Fear"],
     ["disgust", "Disgust"],
     ["anger", "Anger"]
   ]
   ```

   The `labels-normalization-map.json` map inside the MODELDIR is just a copy of the map belonging to the dataset used for training. It is used to normalize the predicted labels.

The `images` folder contains the files listed in the `ImageName` column.

Use the following to have a help file.

    cd PredictionLab
    PYTHONPATH=.. make

Normalized prediction labels are in this order:

```
EASIER_CLASSES = [
    "Happiness",
    "Sadness",
    "Surprise",
    "Fear",
    "Anger",
    "Disgust",
    "Contempt",
    "Other",
    "Neutral"  # This class is not explicitly annotated, but rather the default in case of no annotation.
]
```
