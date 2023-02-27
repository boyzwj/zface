from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor

from facedancer import Zface
import yaml
import os

def trainerThread(cfg, s2c = None, c2s = None):
    model = Zface(cfg= cfg,s2c=s2c,c2s=c2s)
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'./check_points/',
        every_n_train_steps = 10000,
        save_on_train_epoch_end = True,
        save_last=True)


    resume_from_checkpoint = None
    if cfg["resume"]:
        resume_from_checkpoint = f'{checkpoint_callback.dirpath}/last.ckpt'

    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = Trainer(
            accelerator='gpu', 
            devices=1,
            precision=16,
            # amp_backend ="apex", 
            # amp_level='O1',
            callbacks=[checkpoint_callback,lr_monitor],
            max_epochs=50)

    # trainer.tune(model)
    trainer.fit(model,ckpt_path = resume_from_checkpoint)



if __name__ == '__main__':
    yamlPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "conf/conf.yaml")
    with open(yamlPath, 'r', encoding='utf-8') as f:
        cfg = yaml.load(f.read(),Loader = yaml.FullLoader)
        trainerThread(cfg)
