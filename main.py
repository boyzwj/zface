from PyQt5.QtCore import Qt, pyqtSignal, QModelIndex, QByteArray, QDataStream, QIODevice, QMimeData, QPoint
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

import sys
import yaml

import os
import torch
from torch.multiprocessing import Process, Queue 
import numpy as np
from PIL import Image
import math
from torchvision.utils import make_grid
from uic.FaceTool import Ui_MainWindow
from trainer import *
from torchvision import transforms

mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])

unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())


class App(QMainWindow):
    def __init__(self):
        super(App, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setFixedSize(self.width(), self.height())
        self.initUI()

    def initUI(self):
        self.baseimg = None
        self.ui.imgLabel.setScaledContents(True)
        self.ui.pushButton.clicked.connect(self.BeginTrain)
        self.ui.menubar.triggered[QAction].connect(self.processtrigger)
        self.ui.refreshButton.clicked.connect(self.Refreshz)
        self.ui.previewCheck.clicked.connect(self.SwitchPreview)
        self.ui.sbImageSize.valueChanged.connect(self.ImageSizeChanged)
        self.ui.sbLatentDim.valueChanged.connect(self.LatentDimChanged)
        self.ui.sbBatchSize.valueChanged.connect(self.BatchSizeChanged)
        self.ui.sbPreview.valueChanged.connect(self.PreviewNumChanged)
        self.ui.resumeCheck.clicked.connect(self.ResumeChanged)
        self.ui.sbLearningRate.textChanged.connect(self.LearningRateChanged)
        self.ui.sbTrainName.textChanged.connect(self.TrainNameChanged)
        self.ui.sbDataset.textChanged.connect(self.DatasetChanged)
        
        
        self.trainThread = None
        self.is_preview = False
        self.waiting_data = False
        self.cfg = None
        self.Training = False
        self.load_config()
        
    def TrainNameChanged(self, event):
        self.cfg["train_name"] = self.ui.sbTrainName.text()
        self.save_config()
        
    def DatasetChanged(self, event):
        self.cfg["dataset"] = self.ui.sbDataset.text()
        self.save_config()        
        
    def LearningRateChanged(self, event):
        self.cfg["learning_rate"] = float(self.ui.sbLearningRate.text())
        self.save_config()


    def ImageSizeChanged(self, event):
        self.cfg["image_size"] = self.ui.sbImageSize.value()
        self.save_config()

    def LatentDimChanged(self, event):
        self.cfg["latent_dim"] = self.ui.sbLatentDim.value()
        self.save_config()

    def BatchSizeChanged(self, event):
        self.cfg["batch_size"] = self.ui.sbBatchSize.value()
        self.save_config()

    def PreviewNumChanged(self, event):
        self.cfg["preview_num"] = self.ui.sbPreview.value()
        self.save_config()

    def ResumeChanged(self, event):
        self.cfg["resume"] = self.ui.resumeCheck.isChecked()
        self.save_config()
        


    def load_config(self):
        yamlPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "conf/conf.yaml")
        with open(yamlPath, 'r', encoding='utf-8') as f:
            self.cfg = yaml.load(f.read(),Loader = yaml.FullLoader)
            self.ui.sbDataset.setText(self.cfg["dataset"])
            self.ui.sbTrainName.setText(self.cfg["train_name"])
            self.ui.sbLearningRate.setText(str(self.cfg["learning_rate"]))
            self.ui.sbBatchSize.setValue(self.cfg["batch_size"])
            self.ui.sbImageSize.setValue(self.cfg["image_size"])
            self.ui.sbLatentDim.setValue(self.cfg["latent_dim"])
            self.ui.sbPreview.setValue(self.cfg["preview_num"])
            self.ui.resumeCheck.setChecked(self.cfg["resume"])
  


    def save_config(self):
        yamlPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "conf/conf.yaml")
        with open(yamlPath, 'w', encoding='utf-8') as f:
            yaml.dump(self.cfg,stream=f,allow_unicode = True)



    def processtrigger(self, q):
        if q.text() == "Open":
            self.openimg()
        elif q.text() == "Maskface":
            self.maskface()
        elif q.text() == "Rectface":
            self.rectface()
        elif q.text() == "Thinface":
            self.thinface()
        else:
            print(q.text() + " pressed")

    def keyPressEvent(self, event):
        pass



  

    def timerEvent(self, event):
        if not self.c2s.empty():
            input = self.c2s.get()
            op = input.get('op', '')
            if op == 'show':
                # previews = input['previews'].detach().cpu()
                previews = input['previews']
                img = self.tensor2img(previews)
                self.showImg(img)
                self.waiting_data = False
        if self.is_preview and not self.waiting_data:
            self.s2c.put("preview")
            self.waiting_data = True



    def Refreshz(self):
        if self.trainThread != None:
            self.s2c.put("random_z")

    def SwitchPreview(self):
        self.is_preview = self.ui.previewCheck.isChecked()


    def BeginTrain(self):
        if self.Training:
            if self.trainThread == None:
                print("training stoped")
                self.Training = False
                self.ui.pushButton.setText("Train")
            else:
                self.killTimer(self.timer_id)
                self.s2c.put("stop")
                self.trainThread.join()
                self.trainThread = None
                self.Training = False
                self.s2c = None
                self.c2s = None
                self.ui.pushButton.setText("Train")
        else:
            if self.trainThread == None:
                self.timer_id = self.startTimer(500, timerType=Qt.VeryCoarseTimer)
                self.s2c = Queue(20)
                self.c2s = Queue(20)
                self.trainThread   = Process(target=trainerThread, args=(self.cfg, self.s2c, self.c2s), kwargs={})
                self.trainThread.start()
                self.Training = True
                self.waiting_data = False
                self.ui.pushButton.setText("Stop")
            else:
                print("training started")






    def tensor2img(self, tensor, out_type=np.uint8):
        tensor = [unnormalize(t)  for t in tensor]
        # n_img = len(tensor)
        grid = make_grid(tensor,value_range=(-1,1), nrow=3, normalize=False)
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()        
        ndarr = np.transpose(ndarr[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
        return ndarr.astype(out_type)



    def showImg(self, show):
        self.ui.imgLabel.clear()
        show1 = np.require(show, np.uint8, 'C')
        QI = QImage(
            show1, show.shape[1], show.shape[0], show.shape[1] * 3, QImage.Format_BGR888)
        # qt_img = ImageQt.ImageQt(img)
        self.ui.imgLabel.setPixmap(QPixmap.fromImage(QI))
        self.ui.imgLabel.adjustSize()

    def openimg(self):
        pass

    def rectface(self):
        pass

    def maskface(self):
        pass

    def thinface(self):
        pass


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())
