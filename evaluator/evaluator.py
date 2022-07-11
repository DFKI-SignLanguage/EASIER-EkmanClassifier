import pandas as pd
import torch
import numpy as np
import model.metric as module_metric
import model.loss as module_loss
from pathlib import Path
import os
from data_loader.data_loaders import EASIER_CLASSES


class Evaluator:
    """
    evaluator that can save train, val and test evaluation results
    """

    def __init__(self, data_loader=None, device="cpu"):
        self.eval_columns = ["Timestamp", "Architecture", "Training set split",
                             "Hyper-params", "Epochs", "Validation set split",
                             "Validation Accuracy", "Validation Balanced Accuracy",
                             "Test set split", "Test Accuracy", "Test Balanced Accuracy",
                             ] + \
                            ["se " + className for className in EASIER_CLASSES] \
                            + ["Total Training time", "Validation Prediction Time", "Test Prediction Time",
                               "TensorBoard"]
        self.eval_df = pd.DataFrame(columns=self.eval_columns)
        self.data_loader = data_loader
        self.config = None
        if self.data_loader is not None:
            self.idx_to_class = self.data_loader.dataset.idx_to_class
            self.validation_split = self.data_loader.validation_split

        self.device = device
        self.loss_fn = None
        self._save_dir = None

        self.timestamp = None
        self.model_architecture = None
        self.train_epochs = None
        self.tensorboard_dir = None
        self.metric_ftns = [getattr(module_metric, met) for met in [
            "accuracy",
            "balanced_accuracy",
            "sensitivity_per_class"
        ]]

        self.training_time = "0"
        self.pred_time = "0"
        self.model_pred_idx_to_dataset_idx = None
        self.metrics_results = None
        self.eval_csv_file = "eval.csv"

    def set_config(self, config):
        self.config = config

        self.loss_fn = getattr(module_loss, config['loss'])
        self._save_dir = config.save_eval_dir

        self.timestamp = self.config.run_id
        self.model_architecture = self.config["arch"]["type"]
        self.train_epochs = self.config["trainer"]["epochs"]
        self.tensorboard_dir = self.config.log_dir

    def convert_idx_to_dataset(self, model_pred_idx):
        self.model_pred_idx_to_dataset_idx = {}
        for i in range(len(model_pred_idx)):
            model_pred_class = model_pred_idx[i]
            for j in range(len(self.idx_to_class)):
                test_dataset_class = self.idx_to_class[j]
                if model_pred_class == test_dataset_class:
                    self.model_pred_idx_to_dataset_idx[i] = j

    def _set_idx_to_labels(self, truths_df):

        unique_idxs = sorted(truths_df.Class.unique().tolist())
        self.idx_to_class = {}
        for idx in unique_idxs:
            self.idx_to_class.update({idx: truths_df.ClassName[truths_df.Class == idx].iloc[0]})

    # TODO Ensure that predictions from loaded model are constant
    def evaluate_model(self, model):

        assert self.data_loader is not None

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

        # if self.model_pred_idx_to_dataset_idx is not None:
        #     keys_to_del = []
        #     for k in self.idx_to_class.keys():
        #         if k not in self.model_pred_idx_to_dataset_idx.keys():
        #             keys_to_del.append(k)
        #     del self.idx_to_class[k]

        output = torch.Tensor(np.concatenate(outputs, axis=0))
        target = torch.Tensor(np.concatenate(targets, axis=0))

        if self.model_pred_idx_to_dataset_idx is not None:
            output = output.cpu().numpy()
            target = target.cpu().numpy()

            output = np.argmax(output, axis=1)
            dataset_output = []
            updated_targets = []
            for i in range(len(output)):
                if output[i] in self.model_pred_idx_to_dataset_idx.keys():
                    dataset_output.append(self.model_pred_idx_to_dataset_idx[output[i]])
                    updated_targets.append(target[i])
            output = torch.tensor(np.eye(len(self.idx_to_class))[dataset_output])
            target = torch.tensor(updated_targets)

        for met in self.metric_ftns:
            curr_metric_out = met(output, target)
            try:
                iter(curr_metric_out)
                curr_metric_out = {self.idx_to_class[i]: curr_metric_out[i] for i in range(len(self.idx_to_class))}
            except TypeError:
                pass
            curr_metric_out = {met.__name__: curr_metric_out}
            metrics.update(curr_metric_out)

        n_samples = len(self.data_loader.sampler)
        self.metrics_results = {'loss': total_loss / n_samples}
        self.metrics_results.update(metrics)

    def evaluate_csv(self, config):

        predictions_csv = config["model_preds"]
        ground_truths_csv = config["ground_truths"]

        self.metrics_results = {}

        metrics = {}
        preds_df = pd.read_csv(predictions_csv, index_col=0)
        truths_df = pd.read_csv(ground_truths_csv, index_col=0)

        outputs = preds_df.Class.values

        try:
            targets = truths_df.Class.values
        except AttributeError:
            targets = truths_df.Facial_label.values

        self._set_idx_to_labels(truths_df)

        output = torch.Tensor(outputs)
        target = torch.Tensor(targets)

        for met in self.metric_ftns:
            curr_metric_out = met(output, target)
            try:
                iter(curr_metric_out)
                curr_metric_out = {self.idx_to_class[clsID]: curr_metric_out[idx] for idx, clsID in enumerate(self.idx_to_class)}
            except TypeError:
                pass
            curr_metric_out = {met.__name__: curr_metric_out}
            metrics.update(curr_metric_out)

        self.metrics_results.update(metrics)

    def load_validation_eval_df(self, save_path=None):

        if (hasattr(self.config, "resume") and self.config.resume is not None):
            # TODO: update save dir for new model and info saving path chosen in config parser
            checkpoint_model_path = self.config.resume
            split = os.path.split
            timestamp = split(split(checkpoint_model_path)[0])[1]
            updated_save_dir = Path((os.path.split(self._save_dir)[0])) / timestamp
            self.config._save_eval_dir = updated_save_dir
            self._save_dir = updated_save_dir

            self.config.run_id = timestamp
            self.timestamp = timestamp
        elif save_path:
            directory, file = os.path.split(save_path)
            self._save_dir = directory
            if file is not None:
                self.eval_csv_file = file

        try:
            self.eval_df = pd.read_csv(os.path.join(self._save_dir, self.eval_csv_file))
        except FileNotFoundError:
            if self._save_dir != "" and not os.path.exists(self._save_dir):
                os.makedirs(self._save_dir)
            self.eval_df = pd.DataFrame(columns=self.eval_columns)

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
                "Test Accuracy": self.metrics_results["accuracy"],
                "Test Balanced Accuracy": self.metrics_results["balanced_accuracy"]
            }
            se_dict = {"se " + k: self.metrics_results["sensitivity_per_class"][k] for k in self.idx_to_class.values()}

            row_dict.update(se_dict)

            row_dict.update({
                "Test Prediction Time": self.pred_time
            })
            new_row = pd.DataFrame(row_dict, index=[0])
            self.eval_df = self.eval_df.combine_first(new_row)[self.eval_columns]
        else:
            raise ValueError

        self.eval_df.to_csv(os.path.join(self._save_dir, self.eval_csv_file))
