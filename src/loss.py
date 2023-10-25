import torch
import torch.nn as nn

from typing import Optional


class PerplexiteLoss(nn.Module):
    def __init__(self, 
                 smooth: Optional[float]=1e-6
                 ) -> None:
        self.smooth = smooth

    def __call__(self, 
                 y_pred: torch.Tensor, 
                 y_true: torch.Tensor
                 ) -> torch.Tensor:
        """ compute the perplexity without exp at the end 

            y_pred and y_pred must have a shape of (Batch size, Vocab size)

            output is a torch.tensor([float]). use .item() to get the loss value"""
        
        correct_probabilities = torch.sum(y_pred * y_true, dim=1)

        loss = torch.mean(torch.log(correct_probabilities + self.smooth))

        return loss


if __name__ == '__main__':
    y_pred = torch.tensor([[0.1, 0.1, 0, 0.3, 0.1, 0.4],
                           [0, 0.1, 0.5, 0.1, 0.3, 0]], requires_grad=True)
    y_true = torch.tensor([[0, 0, 0, 1, 0, 0],
                           [0, 0, 1, 0, 0, 0]])
    print(y_pred.shape, y_true.shape)
    criterion = PerplexiteLoss(smooth=1e6)
    loss = criterion(y_pred, y_true)
    loss.backward()
    print('loss', loss)
    print('y_pred.grad', y_pred.grad)
