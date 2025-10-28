import torch, torch.nn as nn
from torchvision import models

def _make_first_conv_1ch(conv3: nn.Conv2d) -> nn.Conv2d:
    new = nn.Conv2d(1, conv3.out_channels, conv3.kernel_size, conv3.stride,
                    conv3.padding, conv3.dilation, conv3.groups, conv3.bias is not None)
    with torch.no_grad():
        w = conv3.weight.data
        new.weight.copy_(w.mean(dim=1, keepdim=True))
        if conv3.bias is not None:
            new.bias.copy_(conv3.bias.data)
    return new

def build_model(backbone: str = "efficientnet_b0", pretrained: bool = False, in_chans: int = 1) -> nn.Module:
    b = backbone.lower()
    if b == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.efficientnet_b0(weights=weights)
        if in_chans == 1:
            m.features[0][0] = _make_first_conv_1ch(m.features[0][0])
        in_feat = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_feat, 1)
        return m
    if b == "densenet121":
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.densenet121(weights=weights)
        if in_chans == 1:
            m.features.conv0 = _make_first_conv_1ch(m.features.conv0)
        in_feat = m.classifier.in_features
        m.classifier = nn.Linear(in_feat, 1)
        return m
    raise ValueError("backbone must be 'efficientnet_b0' or 'densenet121'")
