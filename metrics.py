import torch



def accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    res = torch.argmax(y_true, dim=1) == torch.argmax(y_pred, dim=1)
    res = res.type(torch.float32)
    return res.mean().item()