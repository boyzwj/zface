import torch
from lib.loss import Loss, LossInterface
import torch.nn.functional as F
from einops import rearrange, repeat


def get_losses_weights(losses):
    '''多任务自适应损失权重
    reference:
    https://blog.csdn.net/leiduifan6944/article/details/107486857?utm_medium=distribute.pc_relevant.none-task-blog-baidujs_baidulandingword-2&spm=1001.2101.3001.4242
    '''
    weights = torch.div(losses, torch.sum(losses)) * losses.shape[0]
    return weights

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
        self.W_shape = 0.5
        self.W_id = 2
        self.W_recon = 25
        self.W_cycle = 1
        self.W_lpips = 5
        self.W_seg = 100
        
        self.weights =  torch.as_tensor([self.W_adv,self.W_shape,self.W_id,self.W_recon,self.W_cycle,self.W_lpips],device="cuda:0")
        self.batch_size = args["batch_size"]
        # self.face_pool = torch.nn.AdaptiveAvgPool2d((64, 64)).to("cuda").eval()

    def get_loss_G(self, G_dict):
        # L_G = 0.0
        
        # Adversarial loss
        if self.W_adv:
            # L_adv = Loss.get_BCE_loss(G_dict["d_adv"], True)
            L_adv = sum([dual_contrastive_loss(fake,real) for fake,real in zip(G_dict["d_fake"],G_dict["d_real"])])
            # L_G += self.W_adv * L_adv
    
          
        # Shape loss
        if self.W_shape:
            L_shape = Loss.get_L1_loss(G_dict["q_fuse"], G_dict["q_swapped_high"]) 
            L_shape += Loss.get_L1_loss(G_dict["q_fuse"], G_dict["q_swapped_low"]) 
            # L_G += self.W_shape * L_shape/68
            
        # Id loss
        if self.W_id:
            L_id = Loss.get_id_loss(G_dict["id_source"], G_dict["id_swapped_high"])
            L_id += Loss.get_id_loss(G_dict["id_source"], G_dict["id_swapped_low"])
            # L_G += L_id * self.W_id


        # Seg loss
        # if self.W_seg:
        #     # L_seg = Loss.get_L1_loss_with_same_person(G_dict["mask_high"], G_dict["mask_target"], G_dict["same_person"], self.batch_size)
        #     # L_seg += Loss.get_L1_loss_with_same_person(G_dict["mask_low"],F.interpolate(G_dict["target_mask"], scale_factor=0.25, mode='bilinear'), G_dict["same_person"], self.batch_size)
        #     L_seg = Loss.get_L1_loss(G_dict["mask_high"], G_dict["mask_target"])
        #     L_seg += Loss.get_L1_loss(G_dict["mask_low"],F.interpolate(G_dict["mask_target"], scale_factor=0.25, mode='bilinear'))
        #     L_G += self.W_seg * L_seg
        #     self.loss_dict["L_seg"] = round(L_seg.item(), 4)
        # Reconstruction loss
        if self.W_recon:
            L_recon = Loss.get_L1_loss_with_same_person(G_dict["I_swapped_high"], G_dict["I_target"], G_dict["same_person"], self.batch_size)
            L_recon += Loss.get_L1_loss_with_same_person(G_dict["I_swapped_low"],F.interpolate(G_dict["I_target"], scale_factor=0.25, mode='bilinear'), G_dict["same_person"], self.batch_size)
            # L_G += L_recon * self.W_recon
       

        # Cycle loss
        if self.W_cycle:
            L_cycle = Loss.get_L1_loss(G_dict["I_cycle"], G_dict["I_swapped_high"])
            # L_G += L_cycle * self.W_cycle         



        # LPIPS loss
        if self.W_lpips:
            L_lpips = Loss.get_lpips_loss_with_same_person(G_dict["I_swapped_high"], G_dict["I_target"],G_dict["same_person"], self.batch_size)
            # L_G += L_lpips * self.W_lpips


        # losses = torch.tensor([L_adv,L_shape,L_id,L_recon,L_cycle,L_lpips],device="cuda:0")
        # w = get_losses_weights(losses * self.weights) * self.weights
        w = self.weights
        L_adv *=  w[0]
        L_shape *= w[1]
        L_id *= w[2]
        L_recon *= w[3]
        L_cycle *= w[4] 
        L_lpips *= w[5]
        
        self.loss_dict["L_adv"] = round(L_adv.item(), 4)
        self.loss_dict["L_shape"] = round(L_shape.item(), 4)
        self.loss_dict["L_id"] = round(L_id.item(), 4)
        self.loss_dict["L_recon"] = round(L_recon.item(), 4)       
        self.loss_dict["L_cycle"] = round(L_cycle.item(), 4)
        self.loss_dict["L_lpips"] = round(L_lpips.item(), 4) 
                         
        L_G = L_adv + L_shape  + L_id  + L_recon + L_cycle + L_lpips 
        self.loss_dict["L_G"] = round(L_G.item(), 4)
        return L_G
    
    # def get_loss_D(self, D_dict):
    #     L_true =  sum([(F.relu(torch.ones_like(l) - l)).mean() for l in D_dict["d_true"] ])
    #     L_fake =  sum([(F.relu(torch.ones_like(l) +  l)).mean() for l in D_dict["d_fake"] ])
    #     L_D = L_true + L_fake
    #     return L_D
    
    def get_loss_D(self, D_dict):
        L_D = sum([dual_contrastive_loss(real,fake) for real,fake in zip(D_dict["d_true"],D_dict["d_fake"])])
        self.loss_dict["L_D"] = round(L_D.item(), 4)
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
