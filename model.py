import torch
import torch.nn as nn
class Down(nn.Module):
    def __init__(self ,in_channels, out_channels):
        super(Down, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    def forward(self, X): 
        return self.model(X)


class Up(nn.Module):
    def __init__(self,in_channels, out_channels ):
        super(Up, self).__init__()

        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    def forward(self, X):
        return self.model
class UNet(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        
        self.down_1 = Down(in_channels = in_channels, out_channels= 64)
        self.down_2 = Down(in_channels = 64, out_channels= 128)
        self.down_3 = Down(in_channels = 128, out_channels= 256)
        self.down_4 = Down(in_channels = 256, out_channels= 512)
        
        
        self.bottom = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )
        
        
        self.up_1 = Up(in_channels = 1024, out_channels= 512)
        self.up_2 = Up(in_channels = 512, out_channels= 256)
        self.up_3 = Up(in_channels = 256, out_channels= 128)
        self.up_4 = Up(in_channels = 128, out_channels= 64)
        self.up_5 = Up(in_channels = 64, out_channels= out_channels)
    

    def forward(self, x):
        x1 = self.down_1(x)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        x4 = self.down_4(x3)
        
        xb = self.bottom(x4)
        
        x = self.up_1(xb)
        x = torch.cat([x, x4], dim=1)
        x = self.up_2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.up_3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up_4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up_5(x)
        return x
    
