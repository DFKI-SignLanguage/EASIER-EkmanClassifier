# New Features

### All the new features are currently set up for Face Extraction Phoenix Dataset only.

#### 1 . A script to separate dataset into train set (train + val) and test set
`python miscellaneous/train_test_split_FePh.py`

- The data points without labels and with multiple labels are removed.
- Class weight is computed and random sampling is done based on class weights to ensure that distribution of classes remains similar.
- Script has one input `data_path`. Please change within the script.
- The resulting train.csv and test.csv will also be stored in the same `data_path`.

#### 2 . Per class metrics

- The metrics are defined in `model/metric.py`.
- If defining new metrics that will generate values for each class, ensure that the function name contains "_per_class".
 Ex: `accuracy_per_class`

#### 3 . Dataloader and Dataset

- The Dataset class associated with the Dataloader must contain `label_names` dictionary with label numbers as keys and
label names as values.
- `training` flag used for loading train or test sets. When True, load train.csv and False ==> test.csv is loaded.

#### 4 . Tensorboard per class metrics visualization

- Load Tensorboard as before  `tensorboard --logdir saved/log/`
- If any per class metrics were specified in config, per class plots can be seen in tensorboard after training.

#### 5 . Evaluator that saves the evaluation results for train, val and test sets

- In `config.json`, a new entry is required as follows:
```python
  "evaluation_store": {
    "type": "Evaluator",
    "args": {
      "save_dir": "saved/",
      "training": true
    },
    "metrics": [
      "accuracy",
      "balanced_accuracy",
      "sensitivity_per_class"
    ]
  }
  ```
- Accuracy, balanced accuracy and sensitivity per class are required to be mentioned in config.json as they will be saved to disk.
- Any other metrics are optional and results will only be printed to screen i.e. results will not be saved to disk. 
- Evaluator set up to save training, validation and testing information and results in the same csv file.
- When train.py is executed, an `eval` folder is set up in the save dir specified in the config.
- Similar to the saved models and logs, the eval result for each run is saved as `save_dir/eval/experiment_name/timestamp/eval.csv`
- When testing, the above mentioned timestamp will be used to access the correct csv file and write the test results into the same csv file.

#### 6 . Updated `test.py`

`python test.py -c config.json -r path_to_best_model_with_timestamp`

Ex: `python test.py -c config.json -r saved/models/Resnet50/0904_190852/checkpoint-epoch2.pth`

- `test.py` uses the Evaluator to generate the results on the test set.
- Please use the same config.json file used for training.
- Ensure to change evluation store args training to false as follows:
```python
  "evaluation_store": {
    "type": "Evaluator",
    "args": {
      "save_dir": "saved/",
      "training": false
    },
    "metrics": [
      "accuracy",
      "balanced_accuracy",
      "sensitivity_per_class"
    ]
  }
  ```
  
##### Note: Refer `config_server.json` for setting up config.

### Known problems:

1 . If validation_split is too large, not all labels might be in training or if it is too small, not all labels will be in testing.

2 . Early stopping does not seem to be working even though it is mentioned as an implemented feature in original template.

3 . New model, log and eval dirs are being created even during testing (i.e dirs that contain no new info)

4 . Model loading problem. If model trained on Linux, and an attempt is made to load model on Windows, it leads to error 
because the model also saves the class type of the Pathlib class and expects the same when loading.

Temp Workaround in `test.py` where model is loaded:
```python
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
  ```

5. On Mac OS, data loader requires `num_workers=0`. Solution kills parallel computing and may slow down data loading. 