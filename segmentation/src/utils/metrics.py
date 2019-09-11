import numpy as np


def dice(output, targets, thresh):
    outputs = output.data.cpu().numpy()
    targets = targets.data.cpu().numpy()
    b_s = outputs.shape[0]
    outputs = outputs.reshape(b_s, -1)
    targets = targets.reshape(b_s, -1)
    outputs[outputs >= thresh] = 1
    outputs[outputs <= thresh] = 0
    nominator = 2 * (outputs * targets).sum(axis=1)
    denominator = outputs.sum(axis=1) + targets.sum(axis=1)
    res = np.zeros(b_s)
    res[denominator == 0] = 1
    res[denominator > 0] = nominator[denominator > 0] / denominator[denominator > 0]
    return res


def dice_numpy(outputs, targets, thresh):
    b_s = outputs.shape[0]
    outputs = outputs.reshape(b_s, -1)
    targets = targets.reshape(b_s, -1)
    outputs[outputs >= thresh] = 1
    outputs[outputs <= thresh] = 0
    nominator = 2 * (outputs * targets).sum(axis=1)
    denominator = outputs.sum(axis=1) + targets.sum(axis=1)
    res = np.zeros(b_s)
    res[denominator == 0] = 1
    res[denominator > 0] = nominator[denominator > 0] / denominator[denominator > 0]
    return res
