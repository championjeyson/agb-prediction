"""
Metrics module for AGB prediction model.

Defines PyTorch modules for computing common regression metrics, including:
- RMSE (Root Mean Squared Error)
- MSE (Mean Squared Error)
- TrainLoss wrapper for training
"""

import torch
import torch.nn as nn
from typing import Union

class RMSE(nn.Module):
    """
    Root Mean Squared Error (RMSE) metric.
    
    Computes RMSE along the first channel of the predictions.
    Supports optional weighting per sample.
    """

    def __init__(self):
        super(RMSE, self).__init__()
        self.mse = torch.nn.MSELoss(reduction='none')
        
    def __call__(self, prediction: torch.Tensor, target: torch.Tensor, weights: Union[float, torch.Tensor] = 1.0) -> torch.Tensor:
        """
        Compute RMSE.

        Parameters
        ----------
        prediction : torch.Tensor
            Predicted values, shape (batch, 1, ...).
        target : torch.Tensor
            Ground truth values, shape (batch, 1, ...).
        weights : float or torch.Tensor, optional
            Weight per sample. Defaults to 1.0.

        Returns
        -------
        torch.Tensor
            RMSE value.
        """
        prediction = prediction[:, 0]
        return torch.sqrt(torch.mean(weights * self.mse(prediction,target)))
    

class MSE(nn.Module):
    """
    Mean Squared Error (MSE) metric.

    Computes MSE along the first channel of the predictions.
    Supports optional weighting per sample.
    """

    def __init__(self):
        super(MSE, self).__init__()
        self.mse = torch.nn.MSELoss(reduction='none')

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor, weights: Union[float, torch.Tensor] = 1.0) -> torch.Tensor:
        """
        Compute MSE.

        Parameters
        ----------
        prediction : torch.Tensor
            Predicted values, shape (batch, 1, ...).
        target : torch.Tensor
            Ground truth values, shape (batch, 1, ...).
        weights : float or torch.Tensor, optional
            Weight per sample. Defaults to 1.0.

        Returns
        -------
        torch.Tensor
            MSE value.
        """
        prediction = prediction[:, 0]
        return torch.mean(weights * self.mse(prediction,target))


# ---------------------------
# Training loss wrapper
# ---------------------------
class TrainLoss(nn.Module):
    """
    Training loss wrapper for the model.

    Selects the appropriate loss function (currently only MSE supported).
    """

    def __init__(self, num_outputs: int, loss_fn: str):
        """
        Initialize TrainLoss.

        Parameters
        ----------
        num_outputs : int
            Number of outputs for the model.
        loss_fn : str
            Loss function to use. Currently supported: 'MSE'.

        Raises
        ------
        ValueError
            If unsupported loss function is provided.
        """

        super(TrainLoss, self).__init__()
        self.task_num = num_outputs
        
        if loss_fn == 'MSE' :
            self.loss_fn = MSE()
        else: 
            raise ValueError('Invalid loss function')

    def forward(self, preds: torch.Tensor, labels: torch.Tensor, weights: Union[float, torch.Tensor] = 1.0) -> torch.Tensor:
        """
        Compute the training loss.

        Parameters
        ----------
        preds : torch.Tensor
            Model predictions, shape (batch, 1, ...).
        labels : torch.Tensor
            Ground truth labels, shape (batch, 1, ...).
        weights : float or torch.Tensor, optional
            Optional per-sample weighting.

        Returns
        -------
        torch.Tensor
            Computed loss.
        """
        return self.loss_fn(preds, labels, weights)