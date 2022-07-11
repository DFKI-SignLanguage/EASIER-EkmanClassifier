import torch
import numpy as np
from sklearn.metrics import confusion_matrix, balanced_accuracy_score


def accuracy(output, target):
    with torch.no_grad():
        try:
            type(output)
            pred = torch.argmax(output, dim=1)
        except IndexError:
            pred = output

        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return float(correct / len(target))


def balanced_accuracy(output, target):
    with torch.no_grad():
        output = output.cpu().numpy()
        target = target.cpu().numpy()

        try:
            type(output)
            pred = np.argmax(output, axis=1)
        except IndexError:
            pred = output

        ba_score = balanced_accuracy_score(y_true=target, y_pred=pred)
    return ba_score


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return float(correct / len(target))


def sensitivity_per_class(output, target):
    with torch.no_grad():
        output = output.cpu().numpy()
        target = target.cpu().numpy()

        try:
            type(output)
            pred = np.argmax(output, axis=1)
        except IndexError:
            pred = output
        cm = confusion_matrix(target, pred)

        FN = cm.sum(axis=1) - np.diag(cm)
        TP = np.diag(cm)

        D = (TP + FN)

        # Removing classes that don't exist in target variable
        TP = TP[~(D == 0)]
        D = D[~(D == 0)]

        # Sensitivity, hit rate, recall, or true positive rate
        sensitivity_score = TP / D

    return sensitivity_score


def specificity_per_class(output, target):
    with torch.no_grad():
        output = output.cpu().numpy()
        target = target.cpu().numpy()

        try:
            type(output)
            pred = np.argmax(output, axis=1)
        except IndexError:
            pred = output
        cm = confusion_matrix(target, pred)

        FP = cm.sum(axis=0) - np.diag(cm)
        FN = cm.sum(axis=1) - np.diag(cm)
        TP = np.diag(cm)
        TN = cm.sum() - (FP + FN + TP)

        D = (TN + FP)

        # Removing classes that don't exist in target variable
        TN = TN[~(D == 0)]
        D = D[~(D == 0)]

        # Specificity or true negative rate
        specificity_score = TN / D

    return specificity_score


def accuracy_per_class(output, target):
    with torch.no_grad():
        output = output.cpu().numpy()
        target = target.cpu().numpy()

        try:
            type(output)
            pred = np.argmax(output, axis=1)
        except IndexError:
            pred = output
        cm = confusion_matrix(target, pred)

        FP = cm.sum(axis=0) - np.diag(cm)
        FN = cm.sum(axis=1) - np.diag(cm)
        TP = np.diag(cm)
        TN = cm.sum() - (FP + FN + TP)

        N = (TP + TN)
        D = (TP + FP + FN + TN)

        # Removing classes that don't exist in target variable
        N = N[~(D == 0)]
        D = D[~(D == 0)]

        # Overall accuracy
        acc_score = N / D

    return acc_score
