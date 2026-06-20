import torch
import torch.nn as nn
import torch.nn.functional as F


class LCNN(nn.Module):
    def __init__(self, num_class=2):
        super(LCNN,self).__init__()

        # feature extraction part 
        self.dropout1 = nn.Dropout(0.2)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(5,5),
                               padding=(2,2), stride=(1,1))
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,1), 
                               padding=(0,0), stride=(1,1))
        self.batchnorm6 = nn.BatchNorm2d(32)
        self.conv7 = nn.Conv2d(in_channels=32, out_channels=96, kernel_size=(3,3),
                               padding=(1,1), stride=(1,1))
        self.maxpool9 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.batchnorm10 = nn.BatchNorm2d(48)
        self.conv11 = nn.Conv2d(in_channels=48, out_channels=96, kernel_size=(1,1),
                                padding=(0,0), stride=(1,1))
        self.batchnorm13 = nn.BatchNorm2d(48)
        self.conv14 = nn.Conv2d(in_channels=48, out_channels=128, kernel_size=(3,3),
                                padding=(1,1), stride=(1,1))
        self.maxpool16 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv17 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1,1), 
                                padding=(0,0), stride=(1,1))
        self.batchnorm19 = nn.BatchNorm2d(64)
        self.conv20 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), 
                                padding=(1,1), stride=(1,1))
        self.batchnorm22 = nn.BatchNorm2d(32)
        self.conv23 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,1), 
                                padding=(0,0), stride=(1,1))
        self.batchnorm25 = nn.BatchNorm2d(32)
        self.conv26 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), 
                                padding=(1,1), stride=(1,1))
        self.maxpool28 = nn.AdaptiveMaxPool2d((16,8))

        # Classification part
        self.fc29 = nn.Linear(32*16*8, 128) 
        self.batchnorm31 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.7)
        self.fc32 = nn.Linear(64, num_class)

    def mfm2(self, x):

        out1, out2 = torch.chunk(x,2,1)
        return torch.max(out1, out2)


    def forward(self, x):
        '''
        x is the output of extracted features, shape expected (Batch, 1, Freq, Time)
        '''
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dimension if missing

        x = self.conv1(x)
        x = self.mfm2(x)
        x = self.maxpool3(x)
        x = self.conv4(x)
        x = self.mfm2(x)
        x = self.batchnorm6(x)
        x = self.conv7(x)
        x = self.mfm2(x)
        x = self.maxpool9(x)
        x = self.batchnorm10(x)
        x = self.conv11(x)
        x = self.mfm2(x)
        x = self.batchnorm13(x)
        x = self.conv14(x)
        x = self.mfm2(x)
        x = self.maxpool16(x)
        x = self.conv17(x)
        x = self.mfm2(x)
        x = self.batchnorm19(x)
        x = self.conv20(x)
        x = self.mfm2(x)
        x = self.batchnorm22(x)
        x = self.conv23(x)
        x = self.mfm2(x)
        x = self.batchnorm25(x)
        x = self.conv26(x)
        x = self.mfm2(x)
        x = self.maxpool28(x)

        x = x.view(-1, 32 * 16 * 8)
        emb = self.mfm2((self.fc29(x)))
        x = self.batchnorm31(emb)

        logits = self.fc32(x) 

        return logits