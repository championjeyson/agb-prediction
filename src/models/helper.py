"""
Helper functions and wrapper classes for training, validation, and inference
with NicoNet / XceptionS2 models for pixel-wise AGBD estimation.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np

from src.models.nico_net import NicoNet
from src.models.metrics import *
from src.utils.config import *

# Load common config
CFG = get_config("default.yaml")


# ----------------------------------
# Model wrapper
# ----------------------------------
class Net(nn.Module):
    """
    Wrapper around different models. Currently only NicoNet is implemented.
    """
    def __init__(self, model_name, in_features = 4, num_outputs = 1, channel_dims = (16, 32, 64, 128), 
                 max_pool = False, downsample = None, leaky_relu = False, patch_size = [15, 15], 
                 compile = True, num_sepconv_blocks = 8, num_sepconv_filters = 728, long_skip = False):
        super(Net, self).__init__()
        
        self.model_name = model_name
        self.num_outputs = num_outputs
        
        # # FCN
        # if self.model_name == 'fcn' :
        #     self.model = SimpleFCN(in_features, channel_dims, num_outputs = 1, max_pool = max_pool, 
        #                            downsample = downsample)

        # # UNet
        # elif self.model_name == 'unet' :
        #     self.model = UNet(n_channels = in_features, n_classes = num_outputs, patch_size = patch_size, 
        #                        leaky_relu = leaky_relu)
        
        # # Nico's model
        # elif self.model_name == "nico":
        if compile: self.model = torch.compile(NicoNet(in_features = in_features, num_outputs = num_outputs, 
                                    num_sepconv_blocks = num_sepconv_blocks, 
                                    num_sepconv_filters = num_sepconv_filters, 
                                    long_skip = long_skip))
        else: self.model = NicoNet(in_features = in_features, num_outputs = num_outputs, 
                                    num_sepconv_blocks = num_sepconv_blocks, 
                                    num_sepconv_filters = num_sepconv_filters, 
                                    long_skip = long_skip)

        # else:
        #     raise NotImplementedError(f'unknown model name {model_name}')
        
    def forward(self, x):
        return self.model(x)
    

# ----------------------------------
# PyTorch Lightning module
# ----------------------------------
class Model(pl.LightningModule):
    """
    LightningModule wrapper for training and validation of pixel-wise AGBD models.
    """

    def __init__(self, model, lr, step_size, gamma, patch_size, downsample, loss_fn):
        """
        Args:
        - model (nn.Module): the model to train
        - lr (float): learning rate
        - step_size (int): the number of epochs before decreasing the learning rate
        - gamma (float): the factor by which the learning rate will be decreased
        - patch_size (list): the size of the patches to extract, in pixels
        - downsample (bool): whether to downsample the patches from 10m resolution to 50m resolution
        - loss_fn (str): the loss function to use for the training.  (Only 'MSE' is currently supported.)
        """ 

        super().__init__()
        self.model = model
        self.num_outputs = model.num_outputs
        self.lr = lr
        self.step_size = step_size
        self.gamma = gamma
        self.best_val_rmse = np.inf 
        
        # With downsampling, we go from 10m per pixel to 50m per pixel
        if downsample:
            self.center = int(patch_size[0] // 5) // 2
        else: 
            self.center = int(patch_size[0] // 2)
        
        self.loss_fn = loss_fn

        self.preds, self.val_preds, self.test_preds = [], [], []
        self.labels, self.val_labels, self.test_labels = [], [], []

        self.TrainLoss = TrainLoss(num_outputs = self.num_outputs, loss_fn = self.loss_fn)
                         
    def training_step(self, batch, batch_idx):

        # split batch
        images, labels = batch
        
        # get prediction
        predictions = self.model(images)
        predictions = predictions[:,:,self.center,self.center]

        # Store the predictions and labels
        if batch_idx % 50 == 0:
            rmse = torch.sqrt(torch.mean(torch.pow(predictions[:, 0] - labels, 2)))
            self.log('train/agbd_rmse', rmse)

        # Return the loss
        loss = self.TrainLoss(predictions, labels)

        return loss


    def validation_step(self, batch, batch_idx, dataloader_idx = None):

        # Ordinary validation 
        if dataloader_idx == None or dataloader_idx == 0:
            
            # split batch
            images, labels = batch

            # get predictions
            predictions = self.model(images).detach().cpu()
            predictions = predictions[:,:,self.center,self.center]

            # Store the predictions, labels for the on_validation_epoch_end method
            self.val_preds.append(predictions[:, 0])
            self.val_labels.append(labels.detach().cpu())
        
        # Validation on the test set
        elif dataloader_idx == 1 :
    
            # split batch
            images, labels = batch

            # get predictions
            predictions = self.model(images).detach().cpu()
            predictions = predictions[:,:,self.center,self.center]

            # Store the predictions, labels for the on_validation_epoch_end method
            self.test_preds.append(predictions[:, 0])
            self.test_labels.append(labels.detach().cpu())
        
        else: raise ValueError('dataloader_idx should be 0 or 1')
    

    def on_validation_epoch_end(self):
        """
        Calculate the overall validation RMSE and binned metrics.
        """

        # Ordinary validation #####################################################################

        # Log the validation epoch's predictions and labels
        preds = torch.cat(self.val_preds).unsqueeze(1)
        labels = torch.cat(self.val_labels)
        val_agbd_rmse = RMSE()(preds, labels)
        self.log_dict({'val/agbd_rmse': val_agbd_rmse, "step": self.current_epoch})

        # Log the validation agbd rmse by bin
        bins = np.arange(0, 501, 50)
        for lb,ub in zip(bins[:-1], bins[1:]):
            pred, label = preds[(lb <= labels) & (labels < ub)], labels[(lb <= labels) & (labels < ub)]
            rmse = RMSE()(pred, label)
            self.log_dict({f'binned/val_rmse_{lb}-{ub}': rmse, "step": self.current_epoch})
        
        # Set the predictions and labels back to empty lists
        self.val_preds = []
        self.val_labels = []

        # Validation on the test set ##############################################################

        # Log the test set agbd rmse
        preds = torch.cat(self.test_preds).unsqueeze(1)
        labels = torch.cat(self.test_labels)
        agbd_rmse = RMSE()(preds, labels)
        self.log_dict({'test/agbd_rmse': agbd_rmse, "step": self.current_epoch})

        # Log the test set agbd rmse by bin
        bins = np.arange(0, 501, 50)
        for lb,ub in zip(bins[:-1], bins[1:]):
            pred, label = preds[(lb <= labels) & (labels < ub)], labels[(lb <= labels) & (labels < ub)]
            rmse = RMSE()(pred, label)
            self.log_dict({f'binned/test_rmse_{lb}-{ub}': rmse, "step": self.current_epoch})

        # Set the predictions and labels back to empty lists
        self.test_labels = []
        self.test_preds = []

        # Keep track of the best overall
        if val_agbd_rmse < self.best_val_rmse:
            self.best_val_rmse = val_agbd_rmse
            self.log_dict({'best_test_rmse': agbd_rmse, "step": self.current_epoch})
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr)
        return [optimizer], [torch.optim.lr_scheduler.StepLR(optimizer, step_size = self.step_size, gamma = self.gamma)]
    

# ----------------------------------
# Utilities
# ----------------------------------

def get_pretrained_weights_dir_path() -> Path:
    """
    Return the directory path containing pretrained weights from config.
    Raises:
        ValueError: if path is not set in CFG
        FileNotFoundError: if path does not exist
    """
    pretrained_weights_dir_path = CFG.get('paths', {}).get('model_weight_dir', None)
    if pretrained_weights_dir_path is None:
        raise ValueError('Path to pretrained weight folder not set in the configuration file.')
    pretrained_weights_dir_path = Path(pretrained_weights_dir_path)
    if not pretrained_weights_dir_path.exists():
        raise FileNotFoundError(f'Path to pretrained weight folder does not exist: {pretrained_weights_dir_path.resolve().absolute()}')
    return pretrained_weights_dir_path

def predict_patch(model, patch, device):
    """
    Run AGBD (Above-Ground Biomass Density) prediction on a single image patch.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model wrapper (must expose `.model` for inference).
    patch : np.ndarray or dask.array.core.Array
        Input patch of shape (height, width, channels). 
        If a Dask slice is provided, it will be converted to a NumPy array 
        before inference.
    device : torch.device
        Device to run inference on (e.g. torch.device("cuda") or torch.device("cpu")).

    Returns
    -------
    np.ndarray
        2D array (height x width) of predicted AGBD values. 
        Negative predictions are clipped to 0.

    Notes
    -----
    - Converts input patch from (H, W, C) to (1, C, H, W) for PyTorch inference.
    - All computations are done on `device` and results are returned on CPU as NumPy.
    - This helper does not perform tiling or batching — only single-patch inference.
    """
    # Convert patch from dask array to numpy and set the right format for prediction
    patch = torch.from_numpy(np.array(patch)).float()
    patch = torch.unsqueeze(torch.permute(patch, [2,0,1]), 0).to(device)
    preds = model.model(patch).cpu().detach().numpy()
    return preds[0, 0, :, :].clip(0, None) 
