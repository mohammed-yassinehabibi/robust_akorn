import torch
import torch.nn as nn

from source.layers.klayer import KLayer
from source.layers.common_layers import (
    RGBNormalize,
    ReadOutConv,
    Reshape,
    ResBlock,
    FF,
    BNReLUConv2d,
)
from source.layers.common_fns import positionalencoding2d


class AKOrN(nn.Module):
    """
    Artficial Kuramoto Oscillator Neurons (AKOrN) for classification tasks.
    
    Args:
        n: Number of nodes per layer (int or list)
        ch: Base number of channels
        out_classes: Number of output classes
        L: Number of layers
        T: Number of time steps (int or list)
        J: Connectivity type (str or list)
        ksizes: Kernel sizes (int or list)
        ro_ksize: Readout kernel size
        ro_N: Readout N parameter (int or list)
        norm: Normalization type
        c_norm: Channel normalization type
        gamma: Gamma parameter
        use_omega: Whether to use omega
        init_omg: Initial omega value
        global_omg: Whether to use global omega
        learn_omg: Whether to learn omega
        ensemble: Ensemble size
    """

    def __init__(
        self,
        n=4,
        ch=64,
        out_classes=10,
        L=3,
        T=3,
        J="conv",
        ksizes=[9, 7, 5],
        ro_ksize=3,
        ro_N=2,
        norm="bn",
        c_norm="gn",
        gamma=1.0,
        use_omega=True,
        init_omg=1.0,
        global_omg=True,
        learn_omg=True,
        ensemble=1,
        randomness=True
    ):
        super().__init__()
        
        self.fixed_noise = False
        self.n = n
        self.L = L
        self.ensemble = ensemble
        self.randomness = randomness
        self.gamma = nn.Parameter(torch.tensor([gamma]), requires_grad=False)
        
        # Expand parameters to match number of layers
        self.ns = self._expand_param(n, L)
        self.T = self._expand_param(T, L)
        J = self._expand_param(J, L)
        ksizes = self._expand_param(ksizes, L)
        ro_N = self._expand_param(ro_N, L)
        
        # Calculate layer dimensions
        strides = [2, 2, 2] + [1] * (L - 3)
        channels = [ch * (2**i) for i in range(L)]
        hw_sizes = [[32 // (2**i), 32 // (2**i)] if i < 3 else [8, 8] for i in range(L)]
        
        # Build network components
        self.rgb_normalize = RGBNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.conv0 = self._create_initial_conv(channels[0])
        self.layers = self._create_layers(L, channels, strides, hw_sizes, J, ksizes, ro_N, ro_ksize, norm, c_norm, use_omega, init_omg, global_omg, learn_omg)
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), Reshape(-1, channels[-1]))
        self.out = nn.Linear(channels[-1], out_classes)
    
    def _expand_param(self, param, length):
        """Expand parameter to match the number of layers."""
        if isinstance(param, (list, tuple)):
            return list(param) if len(param) == length else [param[0]] * length
        return [param] * length
    
    def _create_initial_conv(self, out_channels):
        """Create the initial convolutional layer."""
        return nn.Conv2d(
            3, out_channels, kernel_size=3, stride=1, padding=1, bias=False, groups=1
        )
    
    def _create_strided_conv(self, in_ch, out_ch, stride, kernel_size, padding):
        """Create a strided convolution layer."""
        return nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=1)
    
    def _create_readout_block(self, channels, ro_N, ro_ksize, norm):
        """Create a readout block."""
        padding = ro_ksize // 2
        return nn.Sequential(
            ReadOutConv(
                inch=channels,
                outch=channels,
                ro_N=ro_N,
                kernel_size=ro_ksize,
                stride=1,
                padding=padding,
            ),
            ResBlock(
                FF(
                    inch=channels,
                    outch=channels,
                    hidch=channels,
                    kernel_size=ro_ksize,
                    stride=1,
                    padding=padding,
                    norm=norm,
                )
            ),
            BNReLUConv2d(
                inch=channels,
                outch=channels,
                kernel_size=ro_ksize,
                stride=1,
                padding=padding,
                norm=norm,
            )
        )
    
    def _create_layers(self, L, channels, strides, hw_sizes, J, ksizes, ro_N, ro_ksize, norm, c_norm, use_omega, init_omg, global_omg, learn_omg):
        """Create all network layers."""
        layers = nn.ModuleList()
        padding = ro_ksize // 2
        
        for l in range(L):
            # Create transition layers
            if l == 0:
                transition_layers = nn.ModuleList([nn.Identity(), nn.Identity()])
            else:
                conv_layer = self._create_strided_conv(
                    channels[l-1], channels[l], strides[l], ro_ksize, padding
                )
                transition_layers = nn.ModuleList([conv_layer, conv_layer])
            
            # Create K-layer
            k_layer = KLayer(
                n=self.ns[l],
                ch=channels[l],
                J=J[l],
                c_norm=c_norm,
                use_omega=use_omega,
                init_omg=init_omg,
                global_omg=global_omg,
                learn_omg=learn_omg,
                ksize=ksizes[l],
                hw=hw_sizes[l],
            )
            
            # Create readout block
            readout_block = self._create_readout_block(channels[l], ro_N[l], ro_ksize, norm)
            
            layers.append(nn.ModuleList([
                transition_layers,
                nn.Identity(),
                k_layer,
                readout_block,
                nn.Identity()
            ]))
        
        return layers

    def feature(self, inp):
        """Extract features from input through the network layers."""
        # Initial processing
        c = self.conv0(self.rgb_normalize(inp))
        if self.randomness:
            x = torch.randn_like(c) #Randomization
        else:
            x = torch.randn(size=c.shape, generator=torch.Generator().manual_seed(0)).to(c.device) #Remove randomness -> decrease adv acc
        xs, es = [], []

        # Process through each layer
        for l, (transition_layer, _, k_layer, readout_layer, _) in enumerate(self.layers):
            x, c = transition_layer[0](x), transition_layer[1](c)
            layer_xs, layer_es = k_layer(x, c, self.T[l], self.gamma)
            xs.append(layer_xs)
            es.append(layer_es)
            x = layer_xs[-1]
            c = readout_layer(x)
            
        # Final pooling
        x, c = map(self.pool, (x, c))
        return c, x, xs, es

    def forward(self, inp, return_xs=False, return_es=False):
        """Forward pass through the network.
        
        Args:
            inp: Input tensor
            return_xs: Whether to return intermediate states
            return_es: Whether to return energy states
            
        Returns:
            Logits or tuple of (logits, xs, es) depending on flags
        """
        if self.ensemble == 1:
            c, x, xs, es = self.feature(inp)
            logits = self.out(c)
            
            # Build return value based on flags
            result = [logits]
            if return_xs:
                result.append(xs)
            if return_es:
                result.append(es)
            
            return result[0] if len(result) == 1 else result
        else:
            # Ensemble prediction
            assert not return_xs and not return_es, "Ensemble mode doesn't support returning xs/es"
            logits_list = []
            for _ in range(self.ensemble):
                features, _, _, _ = self.feature(inp)
                logits = self.out(features)
                logits_list.append(logits)
            return torch.stack(logits_list).mean(0)