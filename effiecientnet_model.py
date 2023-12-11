import torch.nn as nn
from math import ceil



category_count = 7

import torch.nn as nn
from math import ceil

class ConvBatchNormActivation(nn.Module):
    def __init__(self, channels_in, channels_out, k_size=3, stride_len=1, 
                 pad=0, groupings=1, use_bn=True, use_activation=True, 
                 use_bias=False):
        super(ConvBatchNormActivation, self).__init__()
        
        self.convolution = nn.Conv2d(channels_in, channels_out, kernel_size=k_size,
                                     stride=stride_len, padding=pad, groups=groupings, bias=use_bias)
        self.batchnorm = nn.BatchNorm2d(channels_out) if use_bn else nn.Identity()
        self.activation = nn.SiLU() if use_activation else nn.Identity()
        
    def forward(self, input_tensor):
        input_tensor = self.convolution(input_tensor)
        input_tensor = self.batchnorm(input_tensor)
        input_tensor = self.activation(input_tensor)
        return input_tensor
    

class SEBlock(nn.Module):
    def __init__(self, channels_input, reduced_ch):
        super(SEBlock, self).__init__()
        
        self.squeeze_excite = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels_input, reduced_ch, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(reduced_ch, channels_input, kernel_size=1),
            nn.Sigmoid()
        )
       
    def forward(self, input_tensor):
        output_tensor = self.squeeze_excite(input_tensor)
        return input_tensor * output_tensor
                                    

class StochasticDepthLayer(nn.Module):
    def __init__(self, survival_chance=0.8):
        super(StochasticDepthLayer, self).__init__()
        self.survival_prob = survival_chance
        
    def forward(self, input_tensor):
        if not self.training:
            return input_tensor
        
        random_tensor = torch.rand(input_tensor.shape[0], 1, 1, 1, device=input_tensor.device) < self.survival_prob
        return torch.div(input_tensor, self.survival_prob) * random_tensor


class MBConvBlock(nn.Module):
    def __init__(self, channels_in, channels_out, k_size=3, stride_len=1, 
                 expansion_factor=6, reduction_ratio=4, survival_chance=0.8):
        super(MBConvBlock, self).__init__()
        
        self.use_skip = (stride_len == 1 and channels_in == channels_out)
        intermediate_ch = int(channels_in * expansion_factor)
        pad = (k_size - 1) // 2
        reduced_ch = int(channels_in // reduction_ratio)
        
        self.expansion = nn.Identity() if (expansion_factor == 1) else ConvBatchNormActivation(channels_in, intermediate_ch, k_size=1)
        self.depthwise_convolution = ConvBatchNormActivation(intermediate_ch, intermediate_ch,
                                                             k_size=k_size, stride_len=stride_len, 
                                                             pad=pad, groupings=intermediate_ch)
        self.squeeze_excitation = SEBlock(intermediate_ch, reduced_ch=reduced_ch)
        self.pointwise_convolution = ConvBatchNormActivation(intermediate_ch, channels_out, 
                                                             k_size=1, use_activation=False)
        self.stochastic_depth = StochasticDepthLayer(survival_chance=survival_chance)
        
    def forward(self, input_tensor):
        residual = input_tensor
        
        input_tensor = self.expansion(input_tensor)
        input_tensor = self.depthwise_convolution(input_tensor)
        input_tensor = self.squeeze_excitation(input_tensor)
        input_tensor = self.pointwise_convolution(input_tensor)
        
        if self.use_skip:
            input_tensor = self.stochastic_depth(input_tensor)
            input_tensor += residual
        
        return input_tensor
    

class CustomEfficientNet(nn.Module):
    def __init__(self, width_multiplier=1, depth_multiplier=1, dropout_prob=0.2, 
                 class_count=category_count, computation_device='cpu'):
        super(CustomEfficientNet, self).__init__()
        self.computation_device = computation_device
        
        final_channel_count = ceil(1280 * width_multiplier)
        self.feature_layers = self._build_feature_layers(width_multiplier, depth_multiplier, final_channel_count)
        self.avg_pool_layer = nn.AdaptiveAvgPool2d(1)
        self.classifier_layer = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(final_channel_count, class_count)
        )
        
    def forward(self, input_tensor):
        # input_tensor = input_tensor.to(self.computation_device)
        
        input_tensor = self.feature_layers(input_tensor)
        input_tensor = self.avg_pool_layer(input_tensor)
        input_tensor = self.classifier_layer(input_tensor.view(input_tensor.shape[0], -1))
        
        return input_tensor
    
    def _build_feature_layers(self, width_multiplier, depth_multiplier, final_channel):
        base_channels = 4 * ceil(int(32 * width_multiplier) / 4)
        channels_input = 3
        feature_layers = [ConvBatchNormActivation(channels_input, base_channels, k_size=3, stride_len=2, pad=1)]
        channels_in = base_channels
        
        kernel_sizes = [3, 3, 5, 3, 5, 5, 3]
        expansion_factors = [1, 6, 6, 6, 6, 6, 6]
        channel_counts = [16, 24, 40, 80, 112, 192, 320]
        layer_counts = [1, 2, 2, 3, 3, 4, 1]
        stride_vals =[1, 2, 2, 2, 1, 2, 1]
    
        scaled_channel_counts = [4 * ceil(int(c * width_multiplier) / 4) for c in channel_counts]
        scaled_layer_counts = [int(d * depth_multiplier) for d in layer_counts]

        for i, channel_out in enumerate(scaled_channel_counts):
            feature_layers += [MBConvBlock(channels_in if repeat == 0 else channel_out, 
                                           channel_out, k_size=kernel_sizes[i],
                                           stride_len=stride_vals[i] if repeat == 0 else 1, 
                                           expansion_factor=expansion_factors[i])
                               for repeat in range(scaled_layer_counts[i])]
            channels_in = channel_out
        
        feature_layers.append(ConvBatchNormActivation(channels_in, final_channel, k_size=1, stride_len=1, pad=0))
        return nn.Sequential(*feature_layers)

