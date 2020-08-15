import torch
from mmskeleton.models.backbones.st_gcn_aaai18 import ST_GCN_learnimportance1,ST_GCN_ones
a=torch.load(r"/home/niu/Code/mmskeleton/checkpoints/st_gcn.ntu-xsub-300b57d4.pth")

a['A'] = torch.ones(size=(3, 25, 25),
                       dtype=torch.float32,
                       requires_grad=True
                       , device='cuda')
LIN1=ST_GCN_ones(3,60,{'layout':'ntu-rgb+d','strategy':'spatial'},edge_importance_weighting=True, data_bn=True,dropout=0.5)
model_dict1 = LIN1.state_dict()
b=model_dict1
LIN1.load_state_dict(a,False)


torch.save(LIN1.state_dict(),"pretrain_ones.pth")
#
# # 1. filter out unnecessary keys
# pretrained_dict = {k: v for k, v in a.items() if k in model_dict}
# # 2. overwrite entries in the existing state dict
# model_dict.update(pretrained_dict)
# # 3. load the new state dict
# LIN1.load_state_dict(pretrained_dict)

pass