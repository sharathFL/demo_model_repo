import torch

class DummyModelV2(torch.nn.Module):
    def __init__(self):
        super(DummyModelV2, self).__init__()
        self.layer = torch.nn.Linear(10, 1)
        self.version = "Model Version 2.0"  # New version string

    def forward(self, x):
        return self.layer(x)
    
    def get_version(self):
        return self.version


