import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # torch.fft.rfft supports only float32 on CUDA, so the spectral stage
        # always runs in full float32 precision even under AMP autocast; the
        # surrounding linear stages still benefit from bf16 acceleration.
        with torch.autocast("cuda", enabled=False):
            x = x.float()
            # Compute Fourier coefficients up to factor of e^(- something constant)
            x_ft = torch.fft.rfft(x)

            # Multiply relevant Fourier modes
            out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
            n_modes = min(self.modes, x_ft.size(-1))
            out_ft[:, :, :n_modes] = self.compl_mul1d(
                x_ft[:, :, :n_modes], self.weights1[:, :, :n_modes]
            )

            # Return to physical space
            x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class FNOBlock1d(nn.Module):
    def __init__(self, modes, width, activation="gelu"):
        super(FNOBlock1d,self).__init__()
        self.modes = modes
        self.width = width
        self.activation = activation
        
        self.conv = SpectralConv1d(self.width, self.width, self.modes)
        self.w = nn.Conv1d(self.width, self.width, 1)
        
    def forward(self,x):
        _x = x
        x1 = self.conv(x)
        x2 = self.w(x)
        x = x1 + x2
        if self.activation != "none":
            x = F.gelu(x)
        return _x+x


class FNO1d(nn.Module):
    def __init__(self, in_channels=2,out_channels=1,embed_dim=32,modes=16, width=64, lift_dim=128, num_blocks=4, activation="gelu"):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: one or more input fields plus the location/time grid
        input shape: (batchsize, x=s, input_channels_without_grid)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes = modes
        self.width = width
        self.padding = 2 # pad the domain if input is non-periodic
        self.num_blocks=num_blocks
        if activation == "none":
            self.fc0 = nn.Sequential(
                nn.Linear(in_channels, embed_dim),  # input channel is 2: (a(x), x)
                nn.Linear(embed_dim, self.width),
            )
        else:
            self.fc0 = nn.Sequential(nn.Linear(in_channels, embed_dim),  # input channel is 2: (a(x), x)
                                    nn.GELU(),
                                    nn.Linear(embed_dim,self.width)
                                    )
        self.blocks = nn.ModuleList(
            [FNOBlock1d(self.modes, self.width, activation) for _ in range(self.num_blocks)]
        )  
        
        
        if activation == "none":
            self.fc1 = nn.Sequential(
                nn.Linear(self.width, lift_dim),
                nn.Linear(lift_dim, self.width),
                nn.Linear(self.width, out_channels),
            )
        else:
            self.fc1 = nn.Sequential(nn.Linear(self.width, lift_dim),
                                     nn.GELU(),
                                    nn.Linear(lift_dim, self.width),
                                    nn.GELU(),
                                    nn.Linear(self.width,out_channels))
        

    def forward(self, x, grid):
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        if grid.ndim == 2:
            grid = grid.unsqueeze(-1)
        if x.ndim != 3 or grid.ndim != 3:
            raise ValueError(
                "x and grid must have shapes [batch, points, channels] and "
                "[batch, points] (or [batch, points, 1])."
            )
        if x.shape[:2] != grid.shape[:2]:
            raise ValueError("x and grid must share batch and point dimensions.")

        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        
        x = F.pad(x, [0, self.padding]) # pad the domain if input is non-periodic

        for block in self.blocks:
            x=block(x)
        
        x = x[..., :-self.padding]
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return x.squeeze(-1)


class RefinedFNO1d(FNO1d):
    """Trained FNO1d plus a closed-form joint-LTI residual correction head.

    M5-3 (2026-08-24): after SGD training the residual r = d - model(x) is
    fitted by a shared per-bin complex map in the unpadded 501-point rfft
    basis (closed-form ridge least squares, NO gradient steps -- see F4).
    ``forward(x, grid)`` returns ``model(x, grid) + LTI_refine(x)`` where
    LTI_refine is the inverse rfft of ``M[k] @ xft[k]`` for every bin k.

    The map lives in the persistent buffer ``lti_map`` (complex128, shape
    (n_bin, out_channels, in_channels - 1)); the correction uses only the
    data channels (grid column excluded, matching the fit basis of
    scripts/analysis/ls_init_fno.py).  Because the map is a buffer (not a
    path in config), the combined checkpoint stays self-contained and loads
    under evaluate.py's strict state_dict protocol unchanged.
    """

    def __init__(self, in_channels=2, out_channels=1, embed_dim=32, modes=16,
                 width=64, lift_dim=128, num_blocks=4, activation="gelu",
                 refine_n_bin=251, lti_map_path=None):
        super().__init__(in_channels, out_channels, embed_dim, modes, width,
                         lift_dim, num_blocks, activation)
        self.refine_n_bin = int(refine_n_bin)
        self.register_buffer(
            "lti_map",
            torch.zeros(self.refine_n_bin, out_channels, in_channels - 1,
                        dtype=torch.cdouble),
        )
        if lti_map_path is not None:
            # m62/63 (8ch nonlinear campaign): prefit the joint per-bin map
            # (scripts/analysis/fit_lti_head_16ch.py) so the FNO trains on
            # the residual the closed-form head cannot see.  The buffer
            # ships inside the checkpoint, so evaluate needs no file.
            prefit = torch.load(lti_map_path, map_location="cpu", weights_only=True)
            if tuple(prefit.shape) != self.lti_map.shape:
                raise ValueError(
                    f"lti_map_path {lti_map_path!r} has shape {tuple(prefit.shape)}, "
                    f"expected {tuple(self.lti_map.shape)}"
                )
            self.lti_map.copy_(prefit)

    def forward(self, x, grid):
        out = super().forward(x, grid)  # (batch, 501, out)
        if self.refine_n_bin <= 0 or not torch.is_floating_point(x):
            return out
        xb = x.unsqueeze(0) if x.ndim == 2 else x
        n_bin = xb.shape[1] // 2 + 1
        if n_bin != self.refine_n_bin:
            raise ValueError(
                f"RefinedFNO1d expects {self.refine_n_bin} bins "
                f"({self.refine_n_bin * 2 - 1} points), got {n_bin}"
            )
        xft = torch.fft.rfft(xb.double(), dim=1)       # (batch, n_bin, in-1)
        # lti_map is already complex128; .double() on a complex tensor would
        # silently drop the imaginary part, so use it as-is.
        corr_ft = torch.einsum("bki,koi->bko", xft, self.lti_map)
        corr = torch.fft.irfft(corr_ft, n=xb.shape[1], dim=1)
        if x.ndim == 2:
            corr = corr[0]
        return out + corr.to(out.dtype)

