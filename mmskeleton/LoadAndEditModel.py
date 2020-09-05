import torch
from mmskeleton.models.backbones.st_gcn_aaai18 import ST_GCN_learnimportance1,ST_GCN_ones
from mmskeleton.models.backbones.ALN4 import ST_GCN_ALN4
state=torch.load(r"/home/niu/Code/mmskeleton/checkpoints/st_gcn.ntu-xsub-300b57d4.pth")
state.pop('A')
a={'meta':{}}
a['meta']['epoch']=0
a['meta']['iter']=0
a['meta']['mmcv_version']='0.4.0'
a['meta']['time']='Tue Aug 11 08:01:59 2020'
a['state_dict']=state
LIN1=ST_GCN_ALN4(3,60,{'layout':'ntu-rgb+d','strategy':'uniform'},edge_importance_weighting=True, data_bn=True,dropout=0.5)
model_dict1 = LIN1.state_dict()
b=model_dict1
LIN1.load_state_dict(a,False)


torch.save(a,"pretrain_ALN4.pth")
#
# # 1. filter out unnecessary keys
# pretrained_dict = {k: v for k, v in a.items() if k in model_dict}
# # 2. overwrite entries in the existing state dict
# model_dict.update(pretrained_dict)
# # 3. load the new state dict
# LIN1.load_state_dict(pretrained_dict)

pass