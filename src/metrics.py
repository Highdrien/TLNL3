from typing import Dict, Optional
import numpy as np
import torch


def accuracy(y_true: torch.Tensor, 
             y_pred: torch.Tensor
             ) -> float:
    res = torch.argmax(y_true, dim=1) == torch.argmax(y_pred, dim=1)
    res = res.type(torch.float32)
    return res.mean().item()


def top_k_precision(y_true: torch.Tensor, 
                    y_pred: torch.Tensor, 
                    k: Optional[int]=5
                    ) -> float:
    _, sorted_indices = torch.topk(y_pred, k, dim=1)
    correct_predictions = torch.sum(sorted_indices == torch.argmax(y_true, dim=1, keepdim=True), dim=1)
    top_k = torch.mean(correct_predictions.float())
    return top_k.item()


def f_score(y_true: torch.Tensor, 
            y_pred: torch.Tensor, 
            beta: Optional[float] = 1.0, 
            threshold: Optional[float] = 0.5
            ) -> float:
    # Binarize the predictions using the given threshold
    y_pred_binary = (y_pred > threshold).float()

    # Compute true positives, false positives, and false negatives
    true_positives = torch.sum((y_true * y_pred_binary).float(), dim=0)
    false_positives = torch.sum(((1 - y_true) * y_pred_binary).float(), dim=0)
    false_negatives = torch.sum((y_true * (1 - y_pred_binary)).float(), dim=0)

    # Compute precision, recall, and F-score for each class
    precision = true_positives / (true_positives + false_positives + 1e-9)
    recall = true_positives / (true_positives + false_negatives + 1e-9)

    # Compute the F-score for each class
    f_scores = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall + 1e-9)

    # Take the average F-score across all classes (you can also weight them if needed)
    average_f_score = torch.mean(f_scores)

    return average_f_score.item()


def perplexite(y_true: torch.Tensor, 
               y_pred: torch.Tensor
               ) -> float:
    argmax_indices = torch.argmax(y_true, dim=1)
    selected_elements = y_pred[range(len(y_pred)), argmax_indices]

    # Add smooth to avoid 0 before log
    smooth = 1e-6
    selected_elements = selected_elements + smooth

    log_selected_elements = torch.log(selected_elements)
    mean_log_selected_elements = torch.mean(log_selected_elements)
    return - mean_log_selected_elements.item()


def compute_metrics(config: Dict, 
                    y_true: torch.Tensor, 
                    y_pred: torch.Tensor
                    ) -> np.ndarray[float]:
    """ run all the metrics and return a numpy array containing all the metrics value """
    metrics = []
    metrics.append(accuracy(y_true, y_pred))

    k = int(config.metrics.top_k)
    metrics.append(top_k_precision(y_true, y_pred, k))

    metrics.append(f_score(y_true, y_pred))
    metrics.append(perplexite(y_true, y_pred))
        
    return np.array(metrics)


if __name__ == '__main__':
    y_pred = torch.tensor([[0.1, 0.1, 0, 0.3, 0.1, 0.4],
                           [0, 0.1, 0.5, 0.1, 0.3, 0]])
    y_true = torch.tensor([[0, 0, 0, 1, 0, 0],
                           [0, 0, 1, 0, 0, 0]])
    print(y_pred.shape, y_true.shape)

    print('acc:', accuracy(y_true, y_pred))
    print('top_k_precision:', top_k_precision(y_true, y_pred))
    print('perplexite:', np.exp(perplexite(y_true, y_pred)))
    print('f1_score:', f_score(y_true, y_pred))