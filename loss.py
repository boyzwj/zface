import torch
from lib.loss import Loss, LossInterface
import torch.nn.functional as F
from einops import rearrange, repeat


def dual_contrastive_loss(real_logits, fake_logits):
    device = real_logits.device
    real_logits, fake_logits = map(lambda t: rearrange(t, '... -> (...)'), (real_logits, fake_logits))

    def loss_half(t1, t2):
        t1 = rearrange(t1, 'i -> i ()')
        t2 = repeat(t2, 'j -> i j', i = t1.shape[0])
        t = torch.cat((t1, t2), dim = -1)
        return F.cross_entropy(t, torch.zeros(t1.shape[0], device = device, dtype = torch.long))

    return loss_half(real_logits, fake_logits) + loss_half(-fake_logits, -real_logits)

class HifiFaceLoss(LossInterface):
    def __init__(self, args):
        super().__init__(args)
        self.W_adv = 0.125
        self.W_id = 1
        # self.W_seg = 100
        self.W_recon = 20
        self.W_cycle = 1
        self.W_lpips = 1
        self.W_shape = 1
        self.batch_size = args["batch_size"]
        self.face_pool = torch.nn.AdaptiveAvgPool2d((64, 64)).to("cuda").eval()

    def get_loss_G(self, G_dict):
        L_G = 0.0
        
        # Adversarial loss
        if self.W_adv:
            # L_adv = Loss.get_BCE_loss(G_dict["d_adv"], True)
            L_adv = sum([(-l).mean() for l in G_dict["d_adv"]])
            L_G += self.W_adv * L_adv
            self.loss_dict["L_adv"] = round(L_adv.item(), 4)
          
        # Shape loss
        if self.W_shape:
            L_shape = Loss.get_L1_loss(G_dict["q_fuse"], G_dict["q_swapped_high"]) 
            L_shape += Loss.get_L1_loss(G_dict["q_fuse"], G_dict["q_swapped_low"]) 
            L_G += self.W_shape * L_shape/68
            self.loss_dict["L_shape"] = round(L_shape.item(), 4)
            
        # Id loss
        if self.W_id:
            L_id = Loss.get_id_loss(G_dict["id_source"], G_dict["id_swapped_high"])
            L_id += Loss.get_id_loss(G_dict["id_source"], G_dict["id_swapped_low"])
            L_G += self.W_id * L_id
            self.loss_dict["L_id"] = round(L_id.item(), 4)

        # Seg loss
        # if self.W_seg:
        #     L_seg = Loss.get_L1_loss_with_same_person(G_dict["mask_high"], G_dict["mask_target"], G_dict["same_person"], self.batch_size)
        #     # L_seg += Loss.get_L1_loss_with_same_person(G_dict["mask_low"],F.interpolate(G_dict["target_mask"], scale_factor=0.25, mode='bilinear'), G_dict["same_person"], self.batch_size)
        #     # L_seg = Loss.get_L1_loss(G_dict["mask_high"], G_dict["mask_target"])
        #     # L_seg += Loss.get_L1_loss(G_dict["mask_low"],F.interpolate(G_dict["target_mask"], scale_factor=0.25, mode='bilinear'))
        #     L_G += self.W_seg * L_seg
        #     self.loss_dict["L_seg"] = round(L_seg.item(), 4)
        # Reconstruction loss
        if self.W_recon:
            L_recon = Loss.get_L1_loss_with_same_person(G_dict["I_swapped_high"], G_dict["I_target"], G_dict["same_person"], self.batch_size)
            L_recon += Loss.get_L1_loss_with_same_person(G_dict["I_swapped_low"],F.interpolate(G_dict["I_target"], scale_factor=0.25, mode='bilinear'), G_dict["same_person"], self.batch_size)
            L_G += self.W_recon * L_recon
            self.loss_dict["L_recon"] = round(L_recon.item(), 4)              

        # Cycle loss
        if self.W_cycle:
            L_cycle = Loss.get_L1_loss(G_dict["I_swapped_high"], G_dict["I_cycle"])
            L_G += self.W_cycle * L_cycle
            self.loss_dict["L_cycle"] = round(L_cycle.item(), 4)

        # LPIPS loss
        if self.W_lpips:
            L_lpips = Loss.get_lpips_loss_with_same_person(G_dict["I_swapped_high"], G_dict["I_target"],G_dict["same_person"], self.batch_size)
            L_G += self.W_lpips * L_lpips
            self.loss_dict["L_lpips"] = round(L_lpips.item(), 4)

        self.loss_dict["L_G"] = round(L_G.item(), 4)
        return L_G
    
    # def get_loss_D(self, D_dict):
    #     L_true =  sum([(F.relu(torch.ones_like(l) - l)).mean() for l in D_dict["d_true"] ])
    #     L_fake =  sum([(F.relu(torch.ones_like(l) +  l)).mean() for l in D_dict["d_fake"] ])
    #     L_D = L_true + L_fake
    #     return L_D
    
    def get_loss_D(self, D_dict):
        L_D = sum([dual_contrastive_loss(real,fake) for real,fake in zip(D_dict["d_true"],D_dict["d_fake"])])
        return L_D    

    # def get_loss_D(self, D_dict):
    #     L_true = Loss.get_BCE_loss(D_dict["d_true"], True)
    #     L_fake = Loss.get_BCE_loss(D_dict["d_fake"], False)
    #     L_reg = Loss.get_r1_reg(D_dict["d_true"], D_dict["I_target"])
    #     L_D = L_true + L_fake + L_reg

        
    #     self.loss_dict["L_D"] = round(L_D.item(), 4)
    #     self.loss_dict["L_true"] = round(L_true.mean().item(), 4)
    #     self.loss_dict["L_fake"] = round(L_fake.mean().item(), 4)

    #     return L_D
