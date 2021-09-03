import pandas as pd
import torch
import numpy as np
import model.metric as module_metric
from timeit import default_timer as timer
import datetime


class Evaluator:
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(self, config, data_loader, device):
        columns = ["Timestamp", "Architecture", "Training set split",
                   "Hyper-params", "Epochs", "Validation set split",
                   "Validation Accuracy", "Validation Balanced Accuracy",
                   "Test set split", "Test Accuracy", "Test Balanced Accuracy",
                   "se anger", "se contempt", "se disgust", "se fear", "se happiness", "se sadness", "se surprise",
                   "se neutral",
                   "Total Training time", "Validation Prediction Time", "Test Prediction Time", "TensorBoard"]
        self.eval_df = pd.DataFrame(columns=columns)
        self.data_loader = data_loader
        self.label_names = self.data_loader.dataset.label_names
        self.device = device
        self.metric_ftns = [getattr(module_metric, met) for met in config['evaluation_store']['metrics']]
        self._save_dir = config.save_eval_dir
        self.config = config

        self.timestamp = self.config.run_id
        self.model_architecture = self.config["arch"]["type"]
        self.validation_split = self.data_loader.validation_split
        self.train_epochs = self.config["trainer"]["epochs"]
        self.training_time = "0"
        self.pred_time = "0"
        self.tensorboard_dir = self.config.log_dir

        self.val_metrics_results = None
        self.test_metrics_results = None

        # self.model_architecture = model.__class__.__name__

    def evaluate_model(self, model, type_eval):

        model.eval()
        metrics = {}
        outputs = []
        targets = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = model(data)

                output = output.cpu().detach().numpy()
                target = target.cpu().detach().numpy()
                outputs.append(output)
                targets.append(target)

        output = torch.Tensor(np.concatenate(outputs, axis=0))
        target = torch.Tensor(np.concatenate(targets, axis=0))
        for met in self.metric_ftns:
            if "_per_class" not in met.__name__:
                metrics.update({met.__name__: met(output, target)})
            elif "_per_class" in met.__name__:
                metrics.update({met.__name__: met(output, target, self.label_names)})

        if type_eval == "validation":
            self.val_metrics_results = metrics
        elif type_eval == "test":
            self.test_metrics_results = metrics
        else:
            raise ValueError

        print(metrics)

    def save(self, type_eval, exp_timestamp=None):

        if type_eval == "validation":
            new_row_dict = {
                "Timestamp": self.timestamp,
                "Architecture": self.model_architecture,
                "Training set split": 1 - self.validation_split,
                "Epochs": self.train_epochs,
                "Validation set split": self.validation_split,
                "Validation Accuracy": self.val_metrics_results["accuracy"],
                "Validation Balanced Accuracy": self.val_metrics_results["balanced_accuracy"],
                "Total Training time": self.training_time,
                "Validation Prediction Time": self.val_pred_time,
                "TensorBoard": self.tensorboard_dir
            }

            new_row = pd.DataFrame(new_row_dict, index=[self.timestamp])
            self.eval_df = self.eval_df.append(new_row)
        elif type_eval == "test":
            raise NotImplemented
        else:
            raise ValueError

        self.eval_df.to_csv(self._save_dir / "eval.csv")
