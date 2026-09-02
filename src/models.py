
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



# ----- U-Net -----

class contracting(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3, 64, 3, stride=1, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=1, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, stride=1, padding=1), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=1, padding=1), nn.ReLU(), nn.Conv2d(256, 256, 3, stride=1, padding=1), nn.ReLU())
        self.layer4 = nn.Sequential(nn.Conv2d(256, 512, 3, stride=1, padding=1), nn.ReLU(), nn.Conv2d(512, 512, 3, stride=1, padding=1), nn.ReLU())
        self.layer5 = nn.Sequential(nn.Conv2d(512, 1024, 3, stride=1, padding=1), nn.ReLU(), nn.Conv2d(1024, 1024, 3, stride=1, padding=1), nn.ReLU())
        self.down_sample = nn.MaxPool2d(2, stride=2)

    def forward(self, X):
        X1 = self.layer1(X)
        X2 = self.layer2(self.down_sample(X1))
        X3 = self.layer3(self.down_sample(X2))
        X4 = self.layer4(self.down_sample(X3))
        X5 = self.layer5(self.down_sample(X4))
        return X5, X4, X3, X2, X1

class expansive(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Conv2d(64, 2, 3, stride=1, padding=1)
        self.layer2 = nn.Sequential(nn.Conv2d(128, 64, 3, stride=1, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Conv2d(256, 128, 3, stride=1, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, stride=1, padding=1), nn.ReLU())
        self.layer4 = nn.Sequential(nn.Conv2d(512, 256, 3, stride=1, padding=1), nn.ReLU(), nn.Conv2d(256, 256, 3, stride=1, padding=1), nn.ReLU())
        self.layer5 = nn.Sequential(nn.Conv2d(1024, 512, 3, stride=1, padding=1), nn.ReLU(), nn.Conv2d(512, 512, 3, stride=1, padding=1), nn.ReLU())
        self.up_sample_54 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.up_sample_43 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up_sample_32 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up_sample_21 = nn.ConvTranspose2d(128, 64, 2, stride=2)

    def forward(self, X5, X4, X3, X2, X1):
        X = self.up_sample_54(X5)
        X4 = torch.cat([X, X4], dim=1)
        X4 = self.layer5(X4)

        X = self.up_sample_43(X4)
        X3 = torch.cat([X, X3], dim=1)
        X3 = self.layer4(X3)

        X = self.up_sample_32(X3)
        X2 = torch.cat([X, X2], dim=1)
        X2 = self.layer3(X2)

        X = self.up_sample_21(X2)
        X1 = torch.cat([X, X1], dim=1)
        X1 = self.layer2(X1)

        X = self.layer1(X1)

        return X

class unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = contracting()
        self.up = expansive()

    def forward(self, X):
        X5, X4, X3, X2, X1 = self.down(X)
        X = self.up(X5, X4, X3, X2, X1)
        return X


# ----- MnUV3 -----

class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out

class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out

class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        expand_size =  max(in_size // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False),
            nn.BatchNorm2d(expand_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(expand_size, in_size, kernel_size=1, bias=False),
            nn.Hardsigmoid())

    def forward(self, x):
        return x * self.se(x)

class Block(nn.Module):
    def __init__(self, kernel_size, in_size, expand_size, out_size, act, se, stride):
        super(Block, self).__init__()
        self.stride = stride

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.act1 = act(inplace=True)

        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.act2 = act(inplace=True)
        self.se = SeModule(expand_size) if se else nn.Identity()

        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)
        self.act3 = act(inplace=True)

        self.skip = None
        if stride == 1 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_size)
            )

        if stride == 2 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_size, out_channels=in_size, kernel_size=3, groups=in_size, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(in_size),
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=True),
                nn.BatchNorm2d(out_size)
            )

        if stride == 2 and in_size == out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=3, groups=in_size, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_size)
            )

    def forward(self, x):
        skip = x
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.se(out)
        out = self.bn3(self.conv3(out))
        
        if self.skip is not None:
            skip = self.skip(skip)
        return self.act3(out + skip)

class expansive1(nn.Module):
    def __init__(self):
        super().__init__()
        self.up_sample_54 = nn.ConvTranspose2d(960, 484, 2, stride=2)
        self.layer5 = nn.Sequential(nn.Conv2d(960, 484, 3, stride=1, padding=1), nn.ReLU(), nn.Conv2d(484, 484, 3, stride=1, padding=1), nn.ReLU())
        
        self.up_sample_43 = nn.ConvTranspose2d(484, 420, 2, stride=2)
        self.layer4 = nn.Sequential(nn.Conv2d(484, 420, 3, stride=1, padding=1), nn.ReLU(), nn.Conv2d(420, 420, 3, stride=1, padding=1), nn.ReLU())
        
        self.up_sample_32 = nn.ConvTranspose2d(420, 404, 2, stride=2)
        self.layer3 = nn.Sequential(nn.Conv2d(420, 404, 3, stride=1, padding=1), nn.ReLU(), nn.Conv2d(404, 404, 3, stride=1, padding=1), nn.ReLU())
        
        self.up_sample_21 = nn.ConvTranspose2d(404, 401, 2, stride=2)
        self.layer2 = nn.Sequential(nn.Conv2d(404, 401, 3, stride=1, padding=1), nn.ReLU(), nn.Conv2d(401, 401, 3, stride=1, padding=1), nn.ReLU())
        self.layer1 = nn.Conv2d(401, 2, 3, stride=1, padding=1)

    def forward(self, X5, X4, X3, X2, X1):

        X = self.up_sample_54(X5)
        X4 = torch.cat([X, X4], dim=1)
        X4 = self.layer5(X4)
        X = self.up_sample_43(X4)
        X3 = torch.cat([X, X3], dim=1)
        X3 = self.layer4(X3)
        X = self.up_sample_32(X3)
        X2 = torch.cat([X, X2], dim=1)
        X2 = self.layer3(X2)
        X = self.up_sample_21(X2)
        X1 = torch.cat([X, X1], dim=1)
        X1 = self.layer2(X1)
        X = self.layer1(X1)

        return X

class MnUV3(nn.Module):
    def __init__(self):
        super(MnUV3, self).__init__()
        # Encoding
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = nn.Hardswish(inplace=True)
        self.bneck1 = Block(3, 16, 64, 64, nn.ReLU, False, 2)
        self.bneck2 = Block(5, 64, 672, 476, nn.Hardswish, True, 2)
        self.conv2 = nn.Conv2d(476, 960, kernel_size=1, stride=2, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(960)
        self.hs2 = nn.Hardswish(inplace=True)
        self.linear4 = nn.Linear(960, 2)
        # Decoding
        self.up = expansive1()

    def forward(self, x):
        # Encoding
        x1 = self.hs1(self.bn1(self.conv1(x))) 
        xa1 = self.bneck1(x1) 
        xa2 = self.bneck2(xa1)
        x3 = (self.hs2(self.bn2(self.conv2(xa2))))  

        # Decoding
        x = self.up(x3,xa2,xa1,x1,x)
        return x


# ----- Classifier -----

def conv2dout(hin,win,conv,pool=2):
    k=conv.kernel_size
    s=conv.stride
    p=conv.padding
    d=conv.dilation
    ho=np.floor((hin+2*p[0]-d[0]*(k[0]-1)-1)/s[0]+1)
    wo=np.floor((win+2*p[1]-d[1]*(k[1]-1)-1)/s[1]+1)
    
    if pool:
        ho = ho/pool
        wo = wo/pool
    return int(ho),int(wo)

class Network(nn.Module):    
    def __init__(self):
        super(Network, self).__init__()

        # Convolution Layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        h,w = conv2dout(256,256,self.conv1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        h,w = conv2dout(h,w,self.conv2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3)
        h,w = conv2dout(h,w,self.conv2)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3)
        h,w = conv2dout(h,w,self.conv4)
        self.num_flatten=512*h*w
        self.fc1 = nn.Linear(self.num_flatten, 50)
        self.fc2 = nn.Linear(50, 2)

    def forward(self,X):
        X = F.relu(self.conv1(X)); 
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv3(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv4(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, self.num_flatten)
        X = F.relu(self.fc1(X))
        X=F.dropout(X, 0.1)
        X = self.fc2(X)
        X = F.softmax(X,dim = 1)
        
        return X[:,0]
