import torch
import torch.nn as nn
# import torchvision.models as models

class Basic_Model(nn.Module):
    def __init__(self, if_export=False):
        super(Basic_Model, self).__init__()

        self.export = if_export
        
        self.conv1 = nn.Conv2d(3, 8, 3)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.relu2 = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc0 = nn.Flatten()
        # self.fc1 = nn.Linear(16, 2)

    def forward(self, x):
        x = self.pool(self.relu1(self.conv1(x)))
        x = self.pool(self.relu2(self.conv2(x)))
        x = x.permute(0, 3, 1, 2)
        x = self.avgpool(x)
        x = x.squeeze(-1).squeeze(-1)
        # x = self.fc0(x)
        # x = self.fc1(x)
        # if self.export:
        #     return nn.Softmax(dim=1)(x)
        return x
    

if __name__ == "__main__":
    model = Basic_Model()
    data = torch.zeros(1, 3, 56, 224)
    output = model(data)
    print(output.shape)