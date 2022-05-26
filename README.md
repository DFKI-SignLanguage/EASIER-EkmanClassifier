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

Prediction results are in to format:

    ImageName,label1,label2,...,labelN,ClassName,Class

Where columns labelX will contain the softmax output estimation in range [0,1].

## Testing

TODO


## Automated Prediction and Testing

The folder `PredictionLab` contains a `Makefile` implementing the full image normalization, prediciton, and testing pipeline.


The directory `PredictionLab` contains a `MakeFile` for automating the process of normalizing images, predicting, and testing models across datasets.
The goal is to flexibly test the prediction power of a model over many other datasets.

The Makefile requires the presence of a couple of environment variables and the presence of ground truth labels and other info in a structured way.
Example:

```
export DATADIR=<path_to>/data/DaFEx
export MODELDIR=<path_to>/models/ResNet50-AffNet-nopreproc-210921
```

Where the DATADIR and the MODELDIR are structured like this:

```
<DATADIR>/
    images/
    labels.csv
    labels-normalization-map.json

<MODELDIR>/
    model.pth
    config.json
    labels-normalization-map.json  -- This one coming from the dataset used for training
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
   There is another map inside the model directory. Essentially, this is a copy of the map belonging to the dataset used for training this model. It is then used to normalize the predicted labels.

The `images` folder contains the files listed in the `ImageName` column.

Use the following to have an help file.

    cd PredictionLab
    make

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
