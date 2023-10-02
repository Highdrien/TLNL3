from typing import Dict, List
import numpy as np
import torch


def accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    res = torch.argmax(y_true, dim=1) == torch.argmax(y_pred, dim=1)
    res = res.type(torch.float32)
    return res.mean().item()


def top_k_precision(y_true: torch.Tensor, y_pred: torch.Tensor, k: int=5) -> float:
    # https://arxiv.org/abs/1510.05976
    _, sorted_indices = torch.topk(y_pred, k, dim=1)
    correct_predictions = torch.sum(sorted_indices == torch.argmax(y_true, dim=1, keepdim=True), dim=1)
    top_k = torch.mean(correct_predictions.float())
    return top_k.item()


def compute_metrics(config: Dict, y_true: torch.Tensor, y_pred: torch.Tensor) -> np.ndarray[float]:
    metrics = []
    if config.metrics.accuracy is not None:
        metrics.append(accuracy(y_true, y_pred))
    
    if config.metrics.top_k is not None:
        k = int(config.metrics.top_k)
        metrics.append(top_k_precision(y_true, y_pred, k))
    
    return np.array(metrics)