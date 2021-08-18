import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix


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


def sensitivity_per_class(output, target):
    with torch.no_grad():
        output = output.cpu().numpy()
        target = target.cpu().numpy()

        pred = np.argmax(output, axis=1)
        cm = confusion_matrix(target, pred)

        FN = cm.sum(axis=1) - np.diag(cm)
        TP = np.diag(cm)

        # Sensitivity, hit rate, recall, or true positive rate
        sensitivity_score = TP / (TP + FN)
        sensitivity_score = {"class_" + str(i) + "_": sensitivity_score[i] for i in range(len(sensitivity_score))}
    return sensitivity_score


def specificity_per_class(output, target):
    with torch.no_grad():
        output = output.cpu().numpy()
        target = target.cpu().numpy()

        pred = np.argmax(output, axis=1)
        cm = confusion_matrix(target, pred)

        FP = cm.sum(axis=0) - np.diag(cm)
        FN = cm.sum(axis=1) - np.diag(cm)
        TP = np.diag(cm)
        TN = cm.sum() - (FP + FN + TP)

        # Specificity or true negative rate
        specificity_score = TN / (TN + FP)
        specificity_score = {"class_" + str(i) + "_": specificity_score[i] for i in range(len(specificity_score))}
    return specificity_score


def accuracy_per_class(output, target):
    with torch.no_grad():
        output = output.cpu().numpy()
        target = target.cpu().numpy()

        pred = np.argmax(output, axis=1)
        cm = confusion_matrix(target, pred)

        FP = cm.sum(axis=0) - np.diag(cm)
        FN = cm.sum(axis=1) - np.diag(cm)
        TP = np.diag(cm)
        TN = cm.sum() - (FP + FN + TP)

        # Overall accuracy
        acc_score = (TP + TN) / (TP + FP + FN + TN)
        acc_score = {"class_" + str(i) + "_": acc_score[i] for i in range(len(acc_score))}
    return acc_score


def micro_precision(output, target, threshold=0.5):
    with torch.no_grad():
        output = output.cpu().numpy()
        target = target.cpu().numpy()
        pred = np.array(output > threshold, dtype=float)
        p_score = precision_score(y_true=target, y_pred=pred, average='micro', zero_division=0)
    return p_score


def micro_recall(output, target, threshold=0.5):
    with torch.no_grad():
        output = output.cpu().numpy()
        target = target.cpu().numpy()
        pred = np.array(output > threshold, dtype=float)
        r_score = recall_score(y_true=target, y_pred=pred, average='micro', zero_division=0)
    return r_score


def multilabel_accuracy(output, target, threshold=0.5):
    with torch.no_grad():
        output = output.cpu().numpy()
        target = target.cpu().numpy()
        pred = np.array(output > threshold, dtype=float)
        # a_score = accuracy_score(y_true=target, y_pred=output)
        a_score = accuracy_score(y_true=target, y_pred=pred)
    return a_score
