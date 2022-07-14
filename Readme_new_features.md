# v3 Features
- Updated test.py to be able to test models, trained on dataset A, on dataset B.
  - The flags are as below:
    - ```-g / --ground_truths_data_loader``` flag that indicates the data loader that contains the index to class map for the supplied ground truths.
    - ```-r / --resume``` flag ==> path of the model to be used. Similar in function as the -r/--resume flag in train.py and test.py
    - ```-d / --device``` indices of GPUs to enable (default: all).
    - ```-m / --model_preds_data_loader``` flag that indicates the data loader that contains the index to class map for the dataset that the model was trained on.
    - ```-c / --config``` flag ==> config.json used for training and testing 
  - The above script was used in the following ways:
    - ``` python test.py --config config-FePh.json -r saved/models/ResNet50-AffNet-nopreproc-210921/ResNet50_affectnet.pth -g FaceExpressionPhoenixDataLoader -m AffectNetDataLoader```
    - ``` python test.py --config config-FePh.json -r saved/models/Resnet50/0904_200216/checkpoint-epoch30.pth -g FaceExpressionPhoenixDataLoader -m FaceExpressionPhoenixDataLoader```
    - In order to test on Face extraction phoenix test set edited by MTCNN, the edited images are placed in a folder called "FePh_images-cropped" in the directory of the FePh dataset. As of now, switching between the MTCNN edited and normal test images is done manually in FaceExpressionPhoenixDataset class located in data_loader/data_loaders.py . 
- Setup a script "test_csv.py" for comparing an input predictions csv with an associated ground truth csv.
  - The flags are as below:
    - ```-g / --ground_truths_data_loader``` flag that indicates the data loader that contains the index to class map for the supplied ground truths.
    - ```-p / --model_preds``` path of csv file containing the model predictions. (This name will be used to store the eval results in saved/eval/experiment_name/model_csv_name/eval.csv)
    - ```-t / --ground_truths``` flag ==> path and filename for the csv file containing the ground truths.
    - ```-m / --model_preds_data_loader``` flag that indicates the data loader that contains the index to class map for the dataset that the model was trained on.
    - ```-c / --config``` flag ==> config.json used for training and testing 
  - The above script was used in the following ways:
    - ``` python test_csv.py --config config-AffectNet.json -t reference_images-truth.csv -p pred_results/affectnet-1epoch/ekman_ref_cropped_affectnet_1ep.csv -g AffectNetDataLoader -m AffectNetDataLoader```

- Known Issues:
  - NOTE: Loading model and testing using test.py produces slightly different results each time.


# v2 Features
- Bugfixes 
  - ConfigParser: Prevent empty folder from being created when running test.py
  - Evaluator: Prevent code crash when the model passed to test.py does not have a corresponding ```eval.csv``` in ```saved/eval/Resnet50/{model_id_timestamp}```
- Script for using a trained model to predict on any folder of images. The flags are as below:
  - ```-p / --predict``` flag to indicate predict script is running. Just required to change the resume model flag from -r/--resume to -m/--model. This workaround is to ensure compatibility with test.py and train.py scripts.
  - ```-m / --model``` flag ==> path of the model to be used. Similar in function as the -r/--resume flag in train.py and test.py
  - ```-i / --input``` flag ==> path of the folder containing the images that need the predictions.
  - ```-o / --output``` flag ==> path and filename for the output csv file containing the predictions.
  - ```-c / --config``` flag ==> config.json used for training and testing (Ensure that the data loader in config.json that is passed to predict.py corresponds to the dataset that the model was trained on)
Ex: ``` python predict.py -p --config config_temp.json -m saved/models/ResNet50-AffNet-nopreproc-210921/ResNet50_affectnet.pth -i /Users/chbh01/Documents/OfflineCodebases/DFKI_Hiwi/ACG/EASIER/Datasets/VeraAmMittags/video/VAM-frames-cropped -o vam-frames-cropped-squared-preds.csv ```

# v1 Features

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