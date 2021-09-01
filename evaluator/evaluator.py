import pandas as pd
import torch
import numpy as np


class Evaluator:
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(self, config, valid_data_loader, device, metric_ftns):
        columns = ["Timestamp", "Architecture", "Training set split",
                   "Hyper-params", "Epochs", "Validation set split",
                   "Validation Accuracy", "Validation Balanced Accuracy",
                   "Test set split", "Test Accuracy", "Test Balanced Accuracy",
                   "se anger", "se contempt", "se disgust", "se fear", "se happiness", "se sadness", "se surprise",
                   "se neutral",
                   "Total Training time", "Prediction Time", "TensorBoard"]
        self.eval_df = pd.DataFrame(columns=columns)
        self.valid_data_loader = valid_data_loader
        self.label_names = self.valid_data_loader.dataset.label_names
        self.device = device
        self.metric_ftns = metric_ftns
        self._save_dir = config.save_eval_dir
        self.config = config

    def evaluate_model(self, model):
        model.eval()
        valid_metrics = {}
        outputs = []
        targets = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
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
                valid_metrics.update({met.__name__: met(output, target)})
            elif "_per_class" in met.__name__:
                valid_metrics.update({met.__name__: met(output, target, self.label_names)})

        print(valid_metrics)
        self.eval_df.to_csv(self._save_dir / "eval.csv")
