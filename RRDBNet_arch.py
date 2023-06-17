# Import necessary libraries
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_layer(block, n_layers):
    """
    A helper function that creates a sequential block of layers
    Args:
        block: the block to be repeated
        n_layers: the number of times to repeat the block
    Returns:
        A sequential block of layers
    """
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualDenseBlock_5C(nn.Module):
    # no of filters = 64;
    # no of output channels = 32;
    """
    A residual dense block consisting of 5 convolutional layers
    """

    def __init__(self, nf=64, gc=32, bias=True):
        """
        Args:
            nf: number of filters
            gc: growth channel, i.e. intermediate channels
            bias: whether to include bias in the convolutional layers
        """
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        # (3,1,1) => 3x3 size matrix, 1->stride (1,2,3 then 2,3,4)
        # define the 5 convolutional layers
        # conv1: input -> intermediate channels
        # conv2: input + conv1 -> intermediate channels
        # conv3: input + conv1 + conv2 -> intermediate channels
        # conv4: input + conv1 + conv2 + conv3 -> intermediate channels
        # conv5: input + conv1 + conv2 + conv3 + conv4 -> output channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        """
        Forward pass through the block
        Args:
            x: input tensor
        Returns:
            Output tensor
        """
        # pass the input through the convolutional layers and apply leaky ReLU activation
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # add the output tensor to the input tensor multiplied by a scaling factor and return it
        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''
    """
    A residual in residual dense block consisting of multiple residual dense blocks
    """

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):

        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super(RRDBNet, self).__init__()

        # Create a partial function for the RRDB block with fixed nf and gc
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        # First convolution layer
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)

        # Residual in Residual Dense Blocks trunk
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)

        # Trunk convolution layer
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # High-resolution convolution layer
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # Final convolution layer
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        # LeakyReLU activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        # Pass input through first convolution layer
        fea = self.conv_first(x)
        # Pass input through Residual in Residual Dense Blocks trunk
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        # Combine output from RRDBs with input feature map
        fea = fea + trunk

        # Upsample feature map using nearest neighbor interpolation and pass through upconv1
        fea = self.lrelu(self.upconv1(F.interpolate(
            fea, scale_factor=2, mode='nearest')))
        # Upsample feature map again using nearest neighbor interpolation and pass through upconv2
        fea = self.lrelu(self.upconv2(F.interpolate(
            fea, scale_factor=2, mode='nearest')))
        # Pass through high-resolution convolution layer and final convolution layer
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out
