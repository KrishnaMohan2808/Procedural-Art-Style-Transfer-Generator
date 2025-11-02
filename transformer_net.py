import torch.nn as nn
import torch

# --- Helper Layers (Renamed to match model weights) ---
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        # Use ReflectionPad for better boundary handling
        padding_size = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding_size)
        # Renamed variable inside to match 'conv2d' structure found in weights
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        # Renamed variable inside to match 'conv2d' structure found in weights
        return self.conv2d(self.reflection_pad(x))

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        # FIX: These names are correct to match the state_dict for Residual Blocks
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU()
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + identity
        return out

class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = nn.Upsample(scale_factor=upsample, mode='nearest')
        
        padding_size = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding_size)
        # Renamed variable inside to match 'conv2d' structure found in weights
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        if self.upsample:
            x = self.upsample_layer(x)
        # Renamed variable inside to match 'conv2d' structure found in weights
        return self.conv2d(self.reflection_pad(x))

# --- The Transformation Network (Main Model) ---
class TransformerNet(nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = nn.InstanceNorm2d(32, affine=True)
        self.relu1 = nn.ReLU()

        # Downsampling/Encoding layers 
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = nn.InstanceNorm2d(64, affine=True)
        self.relu2 = nn.ReLU()

        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = nn.InstanceNorm2d(128, affine=True)
        self.relu3 = nn.ReLU()

        # Residual blocks (Now match the simpler naming)
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)

        # Upsampling/Decoding layers 
        self.upsample_conv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = nn.InstanceNorm2d(64, affine=True)
        self.relu4 = nn.ReLU()

        self.upsample_conv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = nn.InstanceNorm2d(32, affine=True)
        self.relu5 = nn.ReLU()
        
        # --- FINAL FIX: ADDING THE THIRD UPSAMPLE LAYER ---
        # The weight file has an unexpected key: upsample_conv3. 
        # The output layer (conv4) was what was missing. The issue is that the final
        # layer is NOT a standard ConvLayer, but an UpsampleConvLayer (which also acts as the output).
        self.upsample_conv3 = ConvLayer(32, 3, kernel_size=9, stride=1) # The final layer 
        
        # We need to change the final layer definition to match the unexpected keys:
        # Renaming self.conv4 to self.upsample_conv3 and giving it the correct parameters.
        # Based on the latest error, the final layer must be the missing upsample_conv3.
        # Reverting the name change in __init__ and fixing the forward pass.
        # This is the most complex model structure and we will align the final layer.
        self.upsample_conv3 = UpsampleConvLayer(32, 3, kernel_size=9, stride=1) # 3rd Upsample Layer (output)
        

    def forward(self, X):
        y = self.relu1(self.in1(self.conv1(X)))
        y = self.relu2(self.in2(self.conv2(y)))
        y = self.relu3(self.in3(self.conv3(y)))
        y = self.res5(self.res4(self.res3(self.res2(self.res1(y)))))
        y = self.relu4(self.in4(self.upsample_conv1(y))) 
        y = self.relu5(self.in5(self.upsample_conv2(y))) 
        
        # --- FINAL FIX: EXECUTING THE FINAL LAYER (upsample_conv3) ---
        y = self.upsample_conv3(y)
        
        return y
