import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)


def bce_loss(output, target):
    # print("------------")
    # print(output)
    # print(target)
    return F.binary_cross_entropy(output, target)
