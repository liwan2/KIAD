import numpy as np
import torch

def f1_score(target, predict, noloop=False):
    """
    Compute F1 Score for recommended trajectories (Copied from base_code)
    :param target: the actual trajectory (list or array)
    :param predict: the predict trajectory (list or array)
    """
    # Defensive checks
    if len(target) == 0 or len(predict) == 0:
        return 0.0

    if noloop:
        intersize = len(set(target) & set(predict))
    else:
        match_tags = np.zeros(len(target), dtype=np.bool_)
        for poi in predict:
            for j in range(len(target)):
                if not match_tags[j] and poi == target[j]:
                    match_tags[j] = True
                    break
        intersize = np.nonzero(match_tags)[0].shape[0]

    recall = intersize * 1.0 / len(target)
    precision = intersize * 1.0 / len(predict)
    denominator = recall + precision
    if denominator == 0:
        denominator = 1

    f1 = 2 * precision * recall * 1.0 / denominator

    return f1

def pairs_f1_score(target, predict):
    """
    Compute Pairs_F1 Score (Order correctness) (Copied from base_code)
    """
    # Target and Predict are expected to be Tensors or lists
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    if isinstance(predict, torch.Tensor):
        predict = predict.cpu().numpy()
        
    n = len(target)
    nr = len(predict)

    if n <= 0 or nr <= 0: return 0.0
    if n == 1 or nr == 1:
        return 1.0 if target[0] == predict[0] else 0.0

    n0 = n * (n - 1) / 2
    n0r = nr * (nr - 1) / 2

    order_dict = dict()
    for i, poi in enumerate(target):
        order_dict[poi] = i

    nc = 0
    for i in range(nr):
        poi1 = predict[i]
        for j in range(i + 1, nr):
            poi2 = predict[j]
            if poi1 in order_dict and poi2 in order_dict and poi1 != poi2:
                if order_dict[poi1] < order_dict[poi2]:
                    nc += 1

    precision = (1.0 * nc) / (1.0 * n0r) if n0r > 0 else 0
    recall = (1.0 * nc) / (1.0 * n0) if n0 > 0 else 0
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)


def count_adjacent_repetition_rate(input_data):
    """
    Calculate the adjacent repetition rate for a trajectory.
    """
    if isinstance(input_data, list):
        predictions = input_data
    elif hasattr(input_data, 'numpy'):
        predictions = input_data.cpu().numpy().flatten().tolist()
    else:
        # Fallback for tensor
        predictions = input_data.flatten().tolist()

    total = len(predictions)
    if total < 2:
        return 0.0

    repeated = sum(1 for i in range(1, total) if predictions[i] == predictions[i - 1])
    repetition_ratio = repeated / (total - 1)

    return repetition_ratio
