from .st_att_layer import *
import torch.nn as nn
import torch
import torch.nn.functional as F

class DG_STA(nn.Module):
    def __init__(self, num_classes, dp_rate, isTraining=0):
        super(DG_STA, self).__init__()

        h_dim = 32
        h_num= 8

        self.isTraining = isTraining

        self.input_map = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            LayerNorm(128),
            nn.Dropout(dp_rate),
        )
        #input_size, h_num, h_dim, dp_rate, time_len, domain
        self.s_att = ST_ATT_Layer(input_size=128,output_size= 128, h_num=h_num, h_dim=h_dim, dp_rate=dp_rate, domain="spatial", time_len = 8)


        self.t_att = ST_ATT_Layer(input_size=128, output_size= 128,h_num=h_num, h_dim=h_dim, dp_rate=dp_rate, domain="temporal", time_len = 8)

        self.cls = nn.Linear(128, num_classes)


    def forward(self, x):
        # input shape: [batch_size, time_len, joint_num, 3]

        time_len = x.shape[1]
        joint_num = x.shape[2]

        #reshape x
        x = x.reshape(-1, time_len * joint_num,3)

        #input map
        x = self.input_map(x)
        #spatal
        x = self.s_att(x)
        #temporal
        x = self.t_att(x)

        x = x.sum(1) / x.shape[1]
        
        # pred = self.cls(x)
        
        x = self.cls(x)

        # pred = nn.Softmax(dim=1)(x)
        if not self.isTraining:
            ratio = -0.1
            x[:, 2:4] = ratio * abs(x[:, 2:4])
        pred = x
        pred = nn.Softmax(dim=1)(x)
    
        return pred