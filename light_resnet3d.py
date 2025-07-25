import torch
import torch.nn as nn
from torchvision.models.video import r3d_18

class LightResNet3D(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = r3d_18(pretrained=False)
        # Change input channels to 1 (for MRI)
        self.model.stem[0] = nn.Conv3d(1, 64, kernel_size=(3,7,7), stride=(1,2,2), padding=(1,3,3), bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    # Example: 5 classes
    model = LightResNet3D(num_classes=5)
    x = torch.randn(2, 1, 64, 128, 128)  # [B, C, D, H, W]
    out = model(x)
    print('Output shape:', out.shape) 