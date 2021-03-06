from imageio import save
import torch
import torch.nn.functional as F

from torch.optim import Adam
from torch.utils.data import DataLoader
import platform
import torchvision
from torchvision import transforms
import pytorch_lightning as pl

from models.generator import HififaceGenerator
from models.discriminator import ProjectedDiscriminator
from torch import nn
from dataset import *
from loss import *

class Zface(pl.LightningModule):
    def __init__(self,                 
                cfg = None,
                s2c = None,
                c2s = None,
                b1: float = 0,
                b2: float = 0.999):
        super(Zface, self).__init__()
        self.cfg = cfg
        
        self.size = cfg["image_size"]
        self.z_dim = cfg["latent_dim"]
        self.lr = cfg["learning_rate"]
        self.b1 = b1
        self.b2 = b2
        self.batch_size = cfg["batch_size"]
        self.preview_num = cfg["preview_num"]


        self.G = HififaceGenerator(activation=cfg["activation"])
        self.D = ProjectedDiscriminator(im_res=self.size,backbones=['deit_base_distilled_patch16_224',
                                                                    'tf_efficientnet_lite4'])
        self.upsample = torch.nn.Upsample(scale_factor=4).eval()

  
        # self.G.load_state_dict(torch.load("./weights/G.pth"),strict=True)
        # self.D.load_state_dict(torch.load("./weights/D.pth"),strict=True)
        self.loss = HifiFaceLoss(cfg)
   
        self.s2c = s2c
        self.c2s = c2s
        self.generated_img = None
        self.src_img = None
        self.dst_img = None
        
        
        self.automatic_optimization = False


    def forward(self, I_source, I_target):
        img = self.G(I_source, I_target)[0]
        return img

    
    @torch.no_grad()
    def process_cmd(self):
        if self.s2c is None:
            return
        if not self.s2c.empty():
            msg = self.s2c.get()
            if msg == "preview":
                self.send_previw()
            elif msg == "random_z":
                self.src_img = None
                self.dst_img = None
                self.src_latent = None
            elif msg == "stop":
                torch.save(self.G.state_dict(),"./weights/G.pth")
                torch.save(self.D.state_dict(),"./weights/D.pth")
                self.trainer.should_stop = True
            else:
                pass
                     
            
    @torch.no_grad()
    def send_previw(self):
        output = self.G(self.src_img, self.dst_img)[0]
        result =  []
        for src, dst, out  in zip( self.src_img.cpu() , self.dst_img.cpu() , output.cpu()):
            result = result + [src, dst, out]
        self.c2s.put({'op':"show",'previews': result})
                
            
    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers(use_pl_optimizer=True)
        I_source ,I_target, same_person = batch

        if self.src_img == None:
            self.src_img = I_source[:3]
            self.dst_img = I_target[:3]
            
        self.process_cmd()

        I_swapped_high,I_swapped_low,c_fuse = self.G(I_source, I_target)
        I_swapped_low = self.upsample(I_swapped_low)
        I_cycle = self.G(I_target,I_swapped_high)[0]

        # Arcface 
        id_source = self.G.SAIE.get_id(I_source)
        id_swapped_high = self.G.SAIE.get_id(I_swapped_high)
        id_swapped_low = self.G.SAIE.get_id(I_swapped_low)


        # 3D landmarks
        q_swapped_high = self.G.SAIE.get_lm3d(self.G.SAIE.get_coeff3d(I_swapped_high))
        q_swapped_low = self.G.SAIE.get_lm3d(self.G.SAIE.get_coeff3d(I_swapped_low))
        q_fuse = self.G.SAIE.get_lm3d(c_fuse)


        


        # adversarial
        d_adv = self.D(I_swapped_high)

        G_dict = {
            "I_source": I_source,
            "I_target": I_target,
            "I_swapped_high": I_swapped_high, 
            "I_swapped_low": I_swapped_low,
            "I_cycle": I_cycle,
            "same_person": same_person,
            "id_source": id_source,
            "id_swapped_high": id_swapped_high,
            "id_swapped_low": id_swapped_low,
            "q_swapped_high": q_swapped_high,
            "q_swapped_low": q_swapped_low,
            "q_fuse": q_fuse,
            "d_adv": d_adv
        }
        g_loss = self.loss.get_loss_G(G_dict)
        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()
        # endregion

        ###########
        # train D #
        ###########
        I_target.requires_grad_()
        d_true = self.D(I_target)
        d_fake = self.D(I_swapped_high.detach())

        D_dict = {
            "d_true": d_true,
            "d_fake": d_fake,
            "I_target": I_target
        }

        d_loss = self.loss.get_loss_D(D_dict)
        
        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()
        # endregion

        # region logging
        # self.logging_dict(g_loss_dict, prefix='train / ')
        # self.logging_dict(d_loss_dict, prefix='train / ')
        # self.logging_lr()
        


            


    def configure_optimizers(self):
        optimizer_list = []
        
        optimizer_g = torch.optim.AdamW(self.G.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        optimizer_list.append({"optimizer": optimizer_g})
        optimizer_d = torch.optim.AdamW(self.D.parameters(), lr=self.lr * 0.5, betas=(self.b1, self.b2))
        optimizer_list.append({"optimizer": optimizer_d})
        
        return optimizer_list

    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        # dataset = HifiFaceParsingTrainDataset(["../../FFHQ/imgs/"])
        dataset = MultiResolutionDataset("../../ffhq/",transform=transform,resolution=self.size)
        num_workers = 8
        persistent_workers = True
        if(platform.system()=='Windows'):
            num_workers = 0
            persistent_workers = False
        return DataLoader(dataset, batch_size=self.batch_size,pin_memory=True,num_workers=num_workers, shuffle=True,persistent_workers=persistent_workers, drop_last=True)

