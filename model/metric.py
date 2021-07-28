import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def micro_precision(output, target, threshold=0.5):
    with torch.no_grad():
        # output = output.detach().numpy()
        # target = target.detach().numpy()
        pred = np.array(output > threshold, dtype=float)
        p_score = precision_score(y_true=target, y_pred=pred, average='micro', zero_division=0)
    return p_score


def micro_recall(output, target, threshold=0.5):
    with torch.no_grad():
        # output = output.detach().numpy()
        # target = target.detach().numpy()
        pred = np.array(output > threshold, dtype=float)
        r_score = recall_score(y_true=target, y_pred=pred, average='micro', zero_division=0)
    return r_score


def multilabel_accuracy(output, target, threshold=0.5):
    with torch.no_grad():
        # output = output.detach().numpy()
        # target = target.detach().numpy()
        pred = np.array(output > threshold, dtype=float)
        # a_score = accuracy_score(y_true=target, y_pred=output)
        a_score = accuracy_score(y_true=target, y_pred=pred)
    return a_score