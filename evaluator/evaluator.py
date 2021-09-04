import pandas as pd
import torch
import numpy as np
import model.metric as module_metric
import model.loss as module_loss
from pathlib import Path
import os


class Evaluator:
    """
    evaluator that can save train, val and test evaluation results
    """

    def __init__(self, config, data_loader, device):
        self.eval_columns = ["Timestamp", "Architecture", "Training set split",
                             "Hyper-params", "Epochs", "Validation set split",
                             "Validation Accuracy", "Validation Balanced Accuracy",
                             "Test set split", "Test Accuracy", "Test Balanced Accuracy",
                             "se anger", "se contempt", "se disgust", "se fear", "se happiness", "se sadness",
                             "se surprise",
                             "se neutral",
                             "Total Training time", "Validation Prediction Time", "Test Prediction Time", "TensorBoard"]
        self.eval_df = pd.DataFrame(columns=self.eval_columns)
        self.data_loader = data_loader
        self.label_names = self.data_loader.dataset.label_names
        self.device = device
        self.loss_fn = getattr(module_loss, config['loss'])
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

        self.metrics_results = None

    def evaluate_model(self, model):

        model.eval()
        metrics = {}
        outputs = []
        targets = []
        total_loss = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = model(data)
                loss = self.loss_fn(output, target)
                batch_size = data.shape[0]
                total_loss += loss.item() * batch_size

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

        n_samples = len(self.data_loader.sampler)
        self.metrics_results = {'loss': total_loss / n_samples}
        self.metrics_results.update(metrics)

    def load_val_eval_df(self):
        checkpoint_model_path = self.config.resume
        split = os.path.split
        timestamp = split(split(checkpoint_model_path)[0])[1]
        updated_save_dir = Path((os.path.split(self._save_dir)[0])) / timestamp
        self.config._save_eval_dir = updated_save_dir
        self._save_dir = updated_save_dir

        self.config.run_id = timestamp
        self.timestamp = timestamp

        self.eval_df = pd.read_csv(self._save_dir / "eval.csv")

    def save(self, type_eval):

        if type_eval == "validation":
            new_row_dict = {
                "Timestamp": self.timestamp,
                "Architecture": self.model_architecture,
                "Training set split": 1 - self.validation_split,
                "Epochs": self.train_epochs,
                "Hyper-params": "To be done",
                "Validation set split": self.validation_split,
                "Validation Accuracy": self.metrics_results["accuracy"],
                "Validation Balanced Accuracy": self.metrics_results["balanced_accuracy"],
                "Total Training time": self.training_time,
                "Validation Prediction Time": self.pred_time,
                "TensorBoard": self.tensorboard_dir
            }

            new_row = pd.DataFrame(new_row_dict, index=[0])
            self.eval_df = self.eval_df.append(new_row)
        elif type_eval == "test":
            row_dict = {
                "Timestamp": self.timestamp,
                "Test set split": "To be calculated",
                "Test Accuracy": self.metrics_results["accuracy"],
                "Test Balanced Accuracy": self.metrics_results["balanced_accuracy"],
                "se anger": self.metrics_results["sensitivity_per_class"]["anger"],
                "se contempt": self.metrics_results["sensitivity_per_class"]["contempt"],
                "se disgust": self.metrics_results["sensitivity_per_class"]["disgust"],
                "se fear": self.metrics_results["sensitivity_per_class"]["fear"],
                "se happiness": self.metrics_results["sensitivity_per_class"]["happiness"],
                "se sadness": self.metrics_results["sensitivity_per_class"]["sadness"],
                "se surprise": self.metrics_results["sensitivity_per_class"]["surprise"],
                "se neutral": self.metrics_results["sensitivity_per_class"]["neutral"],
                "Test Prediction Time": self.pred_time
            }
            new_row = pd.DataFrame(row_dict, index=[0])
            self.eval_df = self.eval_df.combine_first(new_row)[self.eval_columns]
        else:
            raise ValueError

        self.eval_df.to_csv(self._save_dir / "eval.csv")
