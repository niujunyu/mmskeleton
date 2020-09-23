import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from mmskeleton.ops.st_gcn import ConvTemporalGraphicalBatchA, Graph

"""
change from 27




 
 change the A  4->2  and tem conv kernel 
"""

def zero(x):
    return 0


def iden(x):
    return x
# 375,1500, 625*4

class ANet(torch.nn.Module):  # 继承 torch 的 Module
    def __init__(self, conv_feature=75*5 , conv_hidden1 = 1000,conv_hidden2 =1500, conv_output = 625
                 ,FC_feature = 20*75 ,FC_hidden1 = 1600 ,FC_hidden2=1500,FC_output = 625 ,dropout_value=0.3):
        super(ANet, self).__init__()  # 继承 __init__ 功能
        # 定义每层用什么样的形式
        self.conv1 = nn.Conv1d(in_channels=300, out_channels=5, kernel_size=1)

        self.convFC = nn.Sequential(
            nn.BatchNorm1d(conv_feature),
            nn.ReLU(inplace=True),

            nn.Linear(conv_feature, conv_hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_value),

            nn.Linear(conv_hidden1, conv_hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_value),

            nn.Linear(conv_hidden2, conv_output),

        )
        self.sliceFC = nn.Sequential(
            nn.BatchNorm1d(FC_feature),
            nn.ReLU(inplace=True),

            nn.Linear(FC_feature, FC_hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_value),

            nn.Linear(FC_hidden1, FC_hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_value),

            nn.Linear(FC_hidden2, FC_output),

        )

# 输出层线性输出

    def forward(self, x):  # 这同时也是 Module 中的 forward 功能
        # 正向传播输入值, 神经网络分析出输出值
        N, T, F = x.size()

        afterconv = self.conv1(x)
        afterconv = afterconv.view(N ,-1)
        A1 = torch.sigmoid(self.convFC(afterconv))

        sliced = x[:,::15,:]
        sliced = sliced.contiguous().view(N,-1)
        A2 =  torch.sigmoid(self.sliceFC(sliced))

        A = torch.cat((A1,A2),dim=1)
        return A









class ST_GCN_ALN29(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_cfg (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """
    def __init__(self,
                 in_channels,
                 num_class,
                 graph_cfg,
                 edge_importance_weighting=True,
                 data_bn=True,
                 **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A,
                         dtype=torch.float32,
                         requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = 2
        temporal_kernel_size = 7
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels *
                                      A.size(1)) if data_bn else iden
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            st_gcn_block(in_channels,
                         64,
                         kernel_size,
                         1,
                         residual=False,
                         **kwargs0),
            st_gcn_block(64, 64, kernel_size, 1, **kwargs),
            st_gcn_block(64, 64, kernel_size, 1, **kwargs),
            st_gcn_block(64, 64, kernel_size, 1, **kwargs),
            st_gcn_block(64, 128, kernel_size, 2, **kwargs),
            st_gcn_block(128, 128, kernel_size, 1, **kwargs),
            st_gcn_block(128, 128, kernel_size, 1, **kwargs),
            st_gcn_block(128, 256, kernel_size, 2, **kwargs),
            st_gcn_block(256, 256, kernel_size, 1, **kwargs),
            st_gcn_block(256, 256, kernel_size, 1, **kwargs),
        ))

        # initialize parameters for edge importance weighting


        # fcn for prediction
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)
        # self.ALN = ANet(375,1500, 625*4)
        self.ALN = ANet()
    def forward(self, x):
        # data normalization
        N, C, T, V, M = x.size()


        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # input_ILN = x.mean(dim=2).view(N*M, -1)
        input_ILN = x.permute(0, 2, 1, 3).contiguous()
        input_ILN=input_ILN.view(N*M,T,C*V)
        ALN_out = self.ALN(input_ILN)
        # ALN_out = ALN_out.view(N,-1).cuda()

        A = ALN_out.view(N*M,2,25, 25).cuda()

        # index = 0
        # for i in range(25):
        #     for j in range(i + 1):
        #        for n in range(N*M):
        #             A[n][i][j] = ALN_out[n][index]
        #             if (i != j): A[n][j][i] = ALN_out[n][index]
        #        index += 1
        # A=A.view(-1, 1, 25, 25).cuda()

        # forward
        for gcn in  self.st_gcn_networks:
            x, _ = gcn(x, A)

        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        # prediction
        x = self.fcn(x)
        x = x.view(x.size(0), -1)

        return x

    def extract_feature(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        # prediction
        x = self.fcn(x)
        output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)

        return output, feature



class st_gcn_block(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphicalBatchA(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = zero

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = iden

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels,
                          kernel_size=1,
                          stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A
