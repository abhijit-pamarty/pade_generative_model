import torch
import torch.nn as nn


class BinnedSpectralPowerLoss3D(nn.Module):
    def __init__(self, bin_edges):
        """
        Initialize the 3D Binned Spectral Power Loss module.

        Args:
            bin_edges (torch.Tensor): A 1D tensor containing the edges of the frequency bins 
                                      in normalized frequencies (0.0 to 0.5).
        """
        super().__init__()
        if not isinstance(bin_edges, torch.Tensor):
            bin_edges = torch.tensor(bin_edges, dtype=torch.float)
        self.register_buffer('bin_edges', bin_edges)
        
    def forward(self, pred, target):
        """
        Compute the Binned Spectral Power Loss between 3D pred and target tensors.

        Args:
            pred (torch.Tensor): Predicted 3D signal tensor of shape (..., X, Y, Z).
            target (torch.Tensor): Target 3D signal tensor of the same shape as pred.

        Returns:
            torch.Tensor: The computed loss value.
        """
        assert pred.shape == target.shape, "Input tensors must have the same shape"
        total_loss = 0.0
        
        # Spatial dimensions are the last three
        for dim in [-1, -2, 1]:
            # Compute FFT along the current dimension
            pred_fft = torch.fft.rfft(pred, dim=dim)
            target_fft = torch.fft.rfft(target, dim=dim)
            
            # Compute power spectrum (magnitude squared)
            pred_power = torch.abs(pred_fft) ** 2
            target_power = torch.abs(target_fft) ** 2
            
            # Number of samples along the current dimension
            n_samples = pred.size(dim)
            
            # Compute FFT frequencies for this dimension
            frequencies = torch.fft.rfftfreq(n_samples, d=1.0, device=pred.device)
            
            # Generate masks for each bin
            bin_edges = self.bin_edges
            n_bins = bin_edges.size(0) - 1
            masks = []
            for i in range(n_bins):
                start = bin_edges[i]
                end = bin_edges[i + 1]
                mask = (frequencies >= start) & (frequencies < end)
                masks.append(mask)
            
            # Stack masks into a tensor and convert to float
            mask_tensor = torch.stack(masks, dim=0).float()  # Shape: (n_bins, n_freq)
            
            # Move frequency dimension to last and flatten for matmul
            pred_power = pred_power.movedim(dim, -1)
            target_power = target_power.movedim(dim, -1)
            
            # Flatten all dimensions except the frequency dimension
            orig_shape = pred_power.shape
            pred_flat = pred_power.reshape(-1, orig_shape[-1])
            target_flat = target_power.reshape(-1, orig_shape[-1])
            
            # Compute binned power via matrix multiplication
            pred_binned = torch.matmul(pred_flat, mask_tensor.t())  # (..., n_bins)
            target_binned = torch.matmul(target_flat, mask_tensor.t())
            
            # Compute MSE loss for this dimension and accumulate
            loss_dim = torch.mean((pred_binned - target_binned) ** 2)
            total_loss += loss_dim
        
        return total_loss