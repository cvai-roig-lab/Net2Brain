import torch
import torch.optim
from torch import nn


class Flatten(nn.Module):
    """
    Helper module for flattening input tensor to 1-D for the use in Linear modules
    """

    def forward(self, x):
        return x.view(x.size(0), -1)


class Identity(nn.Module):
    """
    Helper module that stores the current tensor. Useful for accessing by name
    """

    def forward(self, x):
        return x


class CORblock_Z(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=kernel_size // 2)
        self.nonlin = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.output = Identity()  # for an easy access to this block's output

    def forward(self, inp):
        x = self.conv(inp)
        x = self.nonlin(x)
        x = self.pool(x)
        x = self.output(x)  # for an easy access to this block's output
        return x


class CORnet_Z(nn.Module):

    def __init__(self):
        super(CORnet_Z, self).__init__()
        self.feat_list = ['block1(V1)', 'block2(V2)', 'block3(V4)', 'block4(IT)', 'fc']
        """
        V1 change keys of state_dict name of pretrained weights accordingly, since the names has a prefix
        "V1." in front of the names and below defined without this prefix. (self.conv1, self.bn1, self.relu...)
        Same for the "fc" in pretrained weights the name is called "decoder.linear." Code in the 
        "rename keys of state_dict" section.
        """
        # init blocks
        self.V1 = CORblock_Z(3, 64, kernel_size=7, stride=2)
        self.V2 = CORblock_Z(64, 128)
        self.V4 = CORblock_Z(128, 256)
        self.IT = CORblock_Z(256, 512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1000)

        # weight initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x1 = self.V1(x)
        x2 = self.V2(x1)
        x3 = self.V4(x2)
        x4 = self.IT(x3)

        x = self.avgpool(x4)
        x = x.view(x.size(0), -1)
        x5 = self.fc(x)

        return [x1, x2, x3, x4, x5]


def cornet_z(pretrained=False, **kwargs):
    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        Github url: https://github.com/dicarlolab/CORnet
        weights were called "state_dict", if you named it differently you have to change it accordingly
    """
    model = CORnet_Z()
    if pretrained:
        url = 'https://s3.amazonaws.com/cornet-models/cornet_z-5c427c9c.pth'
        checkpoint = torch.utils.model_zoo.load_url(url, map_location="cpu")
        state_dict = checkpoint['state_dict']
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("decoder.linear.", "fc."): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
    return model