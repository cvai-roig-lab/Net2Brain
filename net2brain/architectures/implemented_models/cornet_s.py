import math

import torch
from torch import nn


class Identity(nn.Module):
    """
    Helper module that stores the current tensor. Useful for accessing by name
    """

    def forward(self, x):
        return x


class CORblock_S(nn.Module):
    scale = 4  # scale of the bottleneck convolution channels

    def __init__(self, in_channels, out_channels, times=1):
        super().__init__()

        self.times = times

        self.conv_input = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.skip = nn.Conv2d(out_channels, out_channels,
                              kernel_size=1, stride=2, bias=False)
        self.norm_skip = nn.BatchNorm2d(out_channels)

        self.conv1 = nn.Conv2d(out_channels, out_channels * self.scale,
                               kernel_size=1, bias=False)
        self.nonlin1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels * self.scale, out_channels * self.scale,
                               kernel_size=3, stride=2, padding=1, bias=False)
        self.nonlin2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_channels * self.scale, out_channels,
                               kernel_size=1, bias=False)
        self.nonlin3 = nn.ReLU(inplace=True)

        self.output = Identity()  # for an easy access to this block's output

        # need BatchNorm for each time step for training to work well
        for t in range(self.times):
            setattr(self, f'norm1_{t}', nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f'norm2_{t}', nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f'norm3_{t}', nn.BatchNorm2d(out_channels))

    def forward(self, inp):
        x = self.conv_input(inp)

        for t in range(self.times):
            if t == 0:
                skip = self.norm_skip(self.skip(x))
                self.conv2.stride = (2, 2)
            else:
                skip = x
                self.conv2.stride = (1, 1)

            x = self.conv1(x)
            x = getattr(self, f'norm1_{t}')(x)
            x = self.nonlin1(x)

            x = self.conv2(x)
            x = getattr(self, f'norm2_{t}')(x)
            x = self.nonlin2(x)

            x = self.conv3(x)
            x = getattr(self, f'norm3_{t}')(x)

            x += skip
            x = self.nonlin3(x)
            output = self.output(x)

        return output


class CORnet_S(nn.Module):
    def __init__(self):
        super(CORnet_S, self).__init__()
        self.feat_list = ['block1(V1)', 'block2(V2)', 'block3(V4)', 'block4(IT)', 'fc']

        """
        V1 change keys of state_dict name of pretrained weights accordingly, since the names has a prefix
        "V1." in front of the names and below defined without this prefix. (self.conv1, self.bn1, self.relu...)
        Same for the "fc" in pretrained weights the name is called "decoder.linear." Code in the 
        "rename keys of state_dict" section.
        """
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # V2, V4, IT, fc
        self.V2 = CORblock_S(64, 128, times=2)
        self.V4 = CORblock_S(128, 256, times=4)
        self.IT = CORblock_S(256, 512, times=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1000)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            # to add it during the training of this network
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    # forward function to get layers
    # debug
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x1 = self.relu(x)

        x2 = self.V2(x1)
        x3 = self.V4(x2)
        x4 = self.IT(x3)

        x = self.avgpool(x4)
        x = x.view(x.size(0), -1)
        x5 = self.fc(x)

        return [x1, x2, x3, x4, x5]


# get instance of cornet_s() to get model with loaded pretrained weights
def cornet_s(pretrained=False, **kwargs):
    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        Github url: https://github.com/dicarlolab/CORnet
        weights were called "state_dict", if you named it differently you have to change it accordingly
    """
    model = CORnet_S()
    if pretrained:
        url = 'https://s3.amazonaws.com/cornet-models/cornet_s-1d3f7974.pth'
        checkpoint = torch.utils.model_zoo.load_url(url, map_location="cpu")
        state_dict = checkpoint['state_dict']
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("V1.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("decoder.linear.", "fc."): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
    return model