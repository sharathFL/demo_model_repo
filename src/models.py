import torch

class DummyModelV3(torch.nn.Module):
    def __init__(self):
        super(DummyModelV3, self).__init__()
        self.layer = torch.nn.Linear(10, 1)
        self.version = "Model Version 3.0"  # New version string for V3

    def forward(self, x):
        return self.layer(x)
    
    def get_version(self):
        return self.version
