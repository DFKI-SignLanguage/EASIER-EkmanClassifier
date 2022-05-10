# EASIER-EkmanClassifier

This is the code repository for the Ekman Classifier developed within the WP7 of the [EASIER](https://www.project-easier.eu) project.
The goal is to implement an Ekman facial expression classifier and tune & test it for Sign Language.

This codebase is based on the following [PytorchTemplate](https://github.com/victoresque/pytorch-template).


## Installation

Create a Python environment with the requirements listed in `requirements.txt`

    python -m venv p3env
    source p3env/bin/activate
    pip install -r requirements.txt

## Testing

Prepare your data in the following format:

```
path/to/DATADIR/
    images/
    labels.csv
```

Put all images into a the `images` folder
Prepare a CSV file `labels.csv` with columns:
* `ImageName`: the name of the image file, that will be searched in the `images folder`
* `ClassName` (optional): the name of the predicted class
* `Class` (optional): the corresponding integer class number. It is your responsibility to be consistent between label name and number.

The folder `PredictionLab` contains a `Makefile` implementing the full image normalization, prediciton, and testing pipeline.

Use the following to have an help file.

    cd PredictionLab
    make

## Prediction

Prediction results are in to format:

    ImageName,label1,label2,...,labelN,ClassName,Class

Where columns labelX will contain the softmax output estimation in range [0,1].


## Training

TODO
