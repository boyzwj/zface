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

class DancerLoss(LossInterface):
    def __init__(self, args):
        super().__init__(args)
        self.W_adv = 1
        self.W_id = 15
        self.W_recon = 5
        self.W_cycle = 1
        self.W_lpips = 0.2
        self.batch_size = args["batch_size"]
        # self.face_pool = torch.nn.AdaptiveAvgPool2d((64, 64)).to("cuda").eval()

    def get_adv_loss(self,fake_input,real_input):
        loss = 0
        for fake_i,real_i in zip(fake_input,real_input):
            fake_i = fake_i[-1]
            real_i = real_i[-1]
            loss_tensor = dual_contrastive_loss(fake_i,real_i)
            bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
            new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
            loss += new_loss
        return loss / len(fake_input)

    def get_loss_G(self, G_dict):
        L_G = 0.0
        
        # Adversarial loss
        if self.W_adv:
            # L_adv = self.get_adv_loss(G_dict["d_fake"],G_dict["d_real"])
            # L_adv = Loss.get_hinge_loss(G_dict["d_fake"], True)
            # L_adv = sum([dual_contrastive_loss(fake,real) for fake,real in zip(G_dict["d_fake"],G_dict["d_real"])])
            # L_adv = dual_contrastive_loss(G_dict["d_fake"],G_dict["d_real"])
            L_adv = torch.relu(1-G_dict["d_fake"]).mean()
            # print("L_adv",L_adv)
            L_G += self.W_adv * L_adv
    
        # Id loss
        if self.W_id:
            L_id = Loss.get_id_loss(G_dict["z_source"], G_dict["z_swapped"])
            # print("L_id",L_id)
            L_G += L_id * self.W_id

        # Reconstruction loss
        if self.W_recon:
            L_recon = Loss.get_L1_loss_with_same_person(G_dict["I_swapped"], G_dict["I_target"], G_dict["same_person"], self.batch_size)
            L_G += torch.clamp(L_recon,0,5) * self.W_recon
       

        # Cycle loss
        if self.W_cycle:
            L_cycle = Loss.get_L1_loss(G_dict["I_cycle"], G_dict["I_target"])
            L_G += L_cycle * self.W_cycle         


        # LPIPS loss
        if self.W_lpips:
            L_lpips = Loss.get_lpips_loss_with_same_person(G_dict["I_swapped"], G_dict["I_target"],G_dict["same_person"], self.batch_size)
            L_G += L_lpips * self.W_lpips
            
            
            
        self.loss_dict["L_G"] = round(L_G.item(), 4)
        return L_G
    

    
    def get_loss_D(self, D_dict):
        # L_D = sum([dual_contrastive_loss(real,fake) for real,fake in zip(D_dict["d_true"],D_dict["d_fake"])])
        # L_D = dual_contrastive_loss(D_dict["d_true"],D_dict["d_fake"])
        L_true =  torch.relu(1-D_dict["d_true"]).mean()
        L_fake =  torch.relu(1+D_dict["d_fake"]).mean()
        L_reg = Loss.get_r1_reg(D_dict["d_true"], D_dict["I_target"])
        L_D = L_true + L_fake + L_reg
        self.loss_dict["L_D"] = round(L_D.item(), 4)
        return L_D