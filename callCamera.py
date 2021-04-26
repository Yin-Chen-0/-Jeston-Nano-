import numpy as np
import cv2

from PyQt5.QtCore import *
import time

import detector
from detector import Detector
import PIL

from PyQt5.QtWidgets import QMessageBox


class Camera(QObject):
    sendImg = pyqtSignal(list)
   
    # sendOutImg = pyqtSignal(list)

    def __init__(self):
        super(Camera, self).__init__()
        self.cap = cv2.VideoCapture()  # 初始化摄像头
        self.CAM_NUM = 0
        self.openCameraFlag = False
        self.i = 0
        self.detector = Detector()

    def catchImg(self):
        self.i += 1
        flag, self.image = self.cap.read()
        # show = cv2.resize(self.image)
        show = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        show, outstr= self.detector.dectetImg(show)

        # img = np.array(show)
        # if self.i == 30:
        #     temp = [img, 'video', show]
        #     self.sendImg.emit(temp)
        #     self.i = 0
        img = PIL.Image.fromarray(show)
        pixmap = img.toqpixmap()
        self.sendImg.emit([pixmap, outstr])


    def doMyWork(self):
        self.openCamera()
        while True:
            self.catchImg()
            if not self.openCameraFlag:
                break
        self.closeCamera()
        print("退出")

    def openCamera(self):
        print("open camera")
        flag = self.cap.open(self.CAM_NUM)
        if not flag:
            msg = QMessageBox.Warning(self, u'Warning', u'请检测相机与电脑是否连接正确',
                                      buttons=QMessageBox.Ok,
                                      defaultButton=QMessageBox.Ok)
        else:
            #self.timer_camera.start(30)
            pass
        # 关闭摄像头

    def closeCamera(self):
        # self.timer_camera.stop()
        self.cap.release()

    def setFlag(self, flag):
        self.openCameraFlag = flag



