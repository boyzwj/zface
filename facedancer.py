from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from torch_utils.ops import upfirdn2d
from models.face_models.iresnet import iresnet100,iresnet50
from models.gen4 import Generator
from models.stargandiscriminator import StarGANv2Discriminator
from models.multiscalediscriminator import MultiscaleDiscriminator
from torch import nn
from dataset import *
from dloss import *
from models.gradnorm import normalize_gradient
from lion_pytorch import Lion


mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])
unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())


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
        self.G = torch.jit.script(Generator())
    
    
        self.D = StarGANv2Discriminator(im_res=self.size)
        
        self.F_id =  torch.jit.script(iresnet100(pretrained=False, fp16=True))
        self.F_id.load_state_dict(torch.load('./weights/r100.pth'))
        self.F_id.eval()                                                             

        self.blur_init_sigma = 2
        self.blur_fade_kimg = 100
        
        self.loss = DancerLoss(cfg)
        self.s2c = s2c
        self.c2s = c2s
        self.generated_img = None
        self.src_img = None
        self.dst_img = None
        
        
        self.automatic_optimization = False 
        torch.autograd.set_detect_anomaly(False)
        torch.autograd.profiler.profile(False)
        torch.autograd.profiler.emit_nvtx(False)
        torch.backends.cudnn.benchmark = True
        
    def get_id(self, I):
        v_id = self.F_id(F.interpolate(I, size=112, mode='bilinear'))
        v_id = F.normalize(v_id)
        return v_id
    
    def forward(self, I_source, I_target):
        source_z = self.get_id(I_source)
        img = self.G( I_target,source_z)
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
        output = self.forward(self.src_img, self.dst_img)
        result =  []
        for src, dst,dst_msk, out  in zip(self.src_img.cpu(),self.dst_img.cpu(),self.dst_msk.cpu(),output.cpu()):
            src = unnormalize(src)
            dst = unnormalize(dst)
            out = unnormalize(out)
            dst_msk = torch.ones_like(dst) * dst_msk
            result = result + [src, dst, dst_msk, out]
        self.c2s.put({'op':"show",'previews': result})
        
        
    
    def run_D(self,img,blur_sigma = 0):
        blur_size = np.floor(blur_sigma * 3)
        self.log("blur_size",blur_size)
        if blur_size > 0:
            f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
            img = upfirdn2d.filter2d(img, f / f.sum())
        return self.D(img) 
    
    
    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers(use_pl_optimizer=True)
        I_source,I_target,mask_target, same_person = batch
        if self.src_img == None:
            self.src_img = I_source[:3]
            self.dst_img = I_target[:3]
            self.dst_msk = mask_target[:3]
            
        self.process_cmd()

        blur_sigma = max(1 - self.global_step / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 1 else 0
        source_z = self.get_id(I_source)
        target_z = self.get_id(I_target)
        I_swapped = self.G(I_target,source_z)        
        I_cycle = self.G(I_swapped,target_z)
        z_swapped = self.get_id(I_swapped)
        
        I_source.requires_grad_()
        real_output = self.run_D(I_source,blur_sigma = blur_sigma)
        fake_output = self.run_D(I_swapped,blur_sigma = blur_sigma)
        G_dict = {
            "I_source": I_source,
            "I_target": I_target,
            "I_swapped": I_swapped,
            "I_cycle": I_cycle,
            "d_fake": fake_output,
            "d_real": real_output,
            "z_source": source_z,
            "z_swapped": z_swapped,
            "same_person": same_person,
        }
        
        g_loss = self.loss.get_loss_G(G_dict)
        opt_g.zero_grad(set_to_none=True)
        self.manual_backward(g_loss)
        opt_g.step()
        
        
        I_target.requires_grad_()
        d_true = self.run_D(I_target,blur_sigma = blur_sigma)
        d_fake = self.run_D(I_swapped.detach(),blur_sigma = blur_sigma)
        D_dict = {
            "d_true": d_true,
            "d_fake": d_fake,
            "I_target": I_target
        }

        d_loss = self.loss.get_loss_D(D_dict)    
        opt_d.zero_grad(set_to_none=True)
        self.manual_backward(d_loss)
        opt_d.step()
        self.log_dict(self.loss.loss_dict)
        
    def training_epoch_end(self, outputs) :
        sch1, sch2 = self.lr_schedulers()
        if isinstance(sch1,CosineAnnealingWarmRestarts):
            sch1.step()
        if isinstance(sch2,CosineAnnealingWarmRestarts):
            sch2.step()
        return super().training_epoch_end(outputs) 
    


    def configure_optimizers(self):
        # optimizer_list = []
        
        optimizer_g = Lion(self.G.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        optimizer_d = Lion(self.D.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        
        # optimizer_g = AdamW(self.G.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        # optimizer_d = AdamW(self.D.parameters(), lr=self.lr * 0.8 , betas=(self.b1, self.b2))

        scheduler_g = CosineAnnealingWarmRestarts(optimizer=optimizer_g,T_0=5,T_mult=2,verbose=True)
        scheduler_d = CosineAnnealingWarmRestarts(optimizer=optimizer_d,T_0=5,T_mult=2,verbose=True)
        return [optimizer_g,optimizer_d],[scheduler_g,scheduler_d]

    def train_dataloader(self):
        dataset = Ds("../../test",resolution=self.size)
        num_workers = 4
        persistent_workers = True
        return DataLoader(dataset, batch_size=self.batch_size,pin_memory=True,num_workers=num_workers, shuffle=True,persistent_workers=persistent_workers, drop_last=True)

    
        
        