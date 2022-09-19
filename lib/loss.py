import abc
import lpips
import torch
import torch.nn.functional as F
import time
import torch.nn as nn

class LossInterface(metaclass=abc.ABCMeta):
    """
    Base class for loss of GAN model. Exceptions will be raised when subclass is being 
    instantiated but abstract methods were not implemented. Concrete methods can be 
    overrided as well if needed.
    """

    def __init__(self, args):
        """
        When overrided, super call is required.
        """
        self.args = args
        self.start_time = time.time()
        self.loss_dict = {}

    def print_loss(self, global_step):
        """
        Print discriminator and generator loss and formatted elapsed time.
        """
        seconds = int(time.time() - self.start_time)
        print("")
        print(f"[ {seconds//3600//24:02}d {(seconds//3600)%24:02}h {(seconds//60)%60:02}m {seconds%60:02}s ]")
        print(f'steps: {global_step:06} / {self.args.max_step}')
        print(f'lossD: {self.loss_dict["L_D"]} | lossG: {self.loss_dict["L_G"]}')

    @abc.abstractmethod
    def get_loss_G(self):
        """
        Caculate generator loss.
        Once loss values are saved in self.loss_dict, they can be uploaded on the 
        dashboard via wandb or printed in self.print_loss. self.print_loss can be 
        overrided as needed.
        """
        pass

    @abc.abstractmethod
    def get_loss_D(self):
        """
        Caculate discriminator loss.
        Once loss values are saved in self.loss_dict, they can be uploaded on the 
        dashboard via wandb or printed in self.print_loss. self.print_loss can be 
        overrided as needed.
        """
        pass


    
    
class Loss:
    """
    Provide various losses such as LPIPS, L1, L2, BCE and so on.
    """
    
    L1 = torch.nn.SmoothL1Loss().to("cuda")
    L2 = torch.nn.MSELoss().to("cuda")

    
    def get_id_loss(a, b):
        return (1 - torch.cosine_similarity(a, b, dim=1)).mean()

    @classmethod
    def get_lpips_loss(cls, a, b):
        if not hasattr(cls, 'lpips'):
            cls.lpips = lpips.LPIPS(net='alex').eval().to("cuda")
        return cls.lpips(a, b)

        
    @classmethod
    def get_lpips_loss_with_same_person(cls, a, b , same_person, batch_per_gpu):
        if not hasattr(cls, 'lpips'):
            cls.lpips = lpips.LPIPS(net='alex').eval().to("cuda")
        return torch.sum(torch.mean(cls.lpips(a, b).reshape(batch_per_gpu, -1), dim=1) * same_person) / (same_person.sum() + 1e-6)        
    
    @classmethod
    def get_ffl_with_same_person(cls,a, b, same_person, batch_per_gpu):
        return torch.sum(torch.mean(cls.L3(a, b).reshape(batch_per_gpu, -1), dim=1) * same_person) / (same_person.sum() + 1e-6)     


    @classmethod
    def get_L1_loss(cls, a, b):   
        return cls.L1(a, b)

    @classmethod
    def get_L2_loss(cls, a, b):
        return cls.L2(a, b)
    

    def get_L1_loss_with_same_person(a, b, same_person, batch_per_gpu):
        return torch.sum(torch.mean(torch.abs(a - b).reshape(batch_per_gpu, -1), dim=1) * same_person) / (same_person.sum() + 1e-6)

    def get_L2_loss_with_same_person(a, b, same_person, batch_per_gpu):
        return torch.sum(0.5 * torch.mean(torch.pow(a - b, 2).reshape(batch_per_gpu, -1), dim=1) * same_person) / (same_person.sum() + 1e-6)
        




    def get_attr_loss(a, b, batch_size):
        L_attr = 0
        for i in range(len(a)):
            L_attr += torch.mean(torch.pow((a[i] - b[i]), 2).reshape(batch_size, -1), dim=1).sum()
        L_attr /= 2.0

        return L_attr

    def softplus_loss(logit, isReal=True):
        if isReal:
            return F.softplus(-logit).mean()
        else:
            return F.softplus(logit).mean()

    @classmethod
    def get_softplus_loss(cls, Di, label):
        L_adv = 0
        for di in Di:
            L_adv += cls.softplus_loss(di[0], label)
        return L_adv

    def hinge_loss(logit, positive=True):
        if positive:
            return torch.relu(1-logit).mean()
        else:
            return torch.relu(logit+1).mean()
    

    @classmethod
    def get_hinge_loss(cls, Di, label):
        L_adv = 0
        for di in Di:
            L_adv += cls.hinge_loss(di[0], label)
        return L_adv

    def get_BCE_loss(logits, target):
        assert target in [1, 0]
        targets = torch.full_like(logits, fill_value=target)
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        return loss

    def get_r1_reg(d_out, x_in):
        # zero-centered gradient penalty for real images
        batch_size = x_in.size(0)
        grad_dout = torch.autograd.grad(
            outputs=d_out.sum(), inputs=x_in,
            create_graph=True, retain_graph=True, only_inputs=True,allow_unused=True
        )[0]
        grad_dout2 = grad_dout.pow(2)
        assert(grad_dout2.size() == x_in.size())
        reg = 0.5 * grad_dout2.contiguous().view(batch_size, -1).sum(1).mean(0)
        return reg

    def get_adversarial_loss(logits, target):
        assert target in [1, 0]
        targets = torch.full_like(logits, fill_value=target)
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        return loss
