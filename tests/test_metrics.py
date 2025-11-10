# tests/test_metrics.py
import torch
from src.models.metrics import RMSE, MSE, TrainLoss

def test_rmse_mse_values():
    preds = torch.tensor([[1.0], [2.0], [3.0]])
    targets = torch.tensor([1.0, 2.0, 3.0])
    rmse = RMSE()
    mse = MSE()
    assert torch.isclose(rmse(preds, targets), torch.tensor(0.0))
    assert torch.isclose(mse(preds, targets), torch.tensor(0.0))

def test_train_loss_dispatch():
    preds = torch.tensor([[0.0], [1.0]])
    targets = torch.tensor([0.0, 1.0])
    loss_module = TrainLoss(num_outputs=1, loss_fn="MSE")
    loss_val = loss_module(preds, targets)
    assert loss_val >= 0.0

def test_invalid_loss_name():
    import pytest
    with pytest.raises(ValueError):
        _ = TrainLoss(num_outputs=1, loss_fn="UnknownLoss")
