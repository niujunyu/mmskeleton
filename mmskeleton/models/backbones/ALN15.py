import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from mmskeleton.ops.st_gcn import ConvTemporalGraphicalBatchA, Graph

"""
change from 
A矩阵对称版本  
A  3*25*25
a.triu  !!!!




change
1 dconv on  T  -》  2cnv or 3dconv
"""