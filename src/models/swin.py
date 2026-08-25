import torch
import torch.nn as nn
import timm

class SwinTiny(nn.Module):
    def __init__(self, num_classes=14, pretrained=True):
        super(SwinTiny, self).__init__()
        self.model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    model = SwinTiny(pretrained=False) # Use False for quick local testing without downloading weights
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"SwinTiny Output shape: {out.shape}")
    assert out.shape == (2, 14)
