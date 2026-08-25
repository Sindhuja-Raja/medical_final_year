import torch
import torch.nn as nn
import timm

class ConvNeXtTiny(nn.Module):
    def __init__(self, num_classes=14, pretrained=True):
        super(ConvNeXtTiny, self).__init__()
        self.model = timm.create_model('convnext_tiny', pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    model = ConvNeXtTiny(pretrained=False) # Use False for quick local testing without downloading weights
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"ConvNeXtTiny Output shape: {out.shape}")
    assert out.shape == (2, 14)
