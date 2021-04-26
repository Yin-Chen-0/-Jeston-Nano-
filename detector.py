import os
import numpy as np
import cv2 as cv
import os
import time
import PIL
from PyQt5.QtCore import *

from tensorflow.keras import models
import numpy as np
from tensorflow.keras.preprocessing import image

class Detector(QObject):
    sendImg = pyqtSignal(list)
    def __init__(self):
        super(Detector, self).__init__()
        self.yolo_dir = './model/darknet'  # YOLO文件路径
        self.weightsPath = os.path.join(self.yolo_dir, 'my_yolov3_final.weights')  # 权重文件
        self.configPath = os.path.join(self.yolo_dir, 'my_yolov3.cfg')  # 配置文件
        self.labelsPath = os.path.join(self.yolo_dir, 'myData.names')  # label名称
        self.CONFIDENCE = 0.5  # 过滤弱检测的最小概率
        self.THRESHOLD = 0.4  # 非最大值抑制阈值
        print("[INFO] loading YOLO from disk...")  ## 可以打印下信息
        self.net = cv.dnn.readNetFromDarknet(self.configPath, self.weightsPath)  ## 利用下载的文件

        self.labels = ['SUV', 'bus', 'family sedan', 'fire engine', 'heavy truck', 'jeep', 'minibus', 'racing car',
                       'taxi',
                       'truck']
        
    def dectetImg(self, img):
        blobImg = cv.dnn.blobFromImage(img, 1.0 / 255.0, (416, 416), None, True,
                                       False)  ## net需要的输入是blob格式的，用blobFromImage这个函数来转格式
        self.net.setInput(blobImg)  ## 调用setInput函数将图片送入输入层

        # 获取网络输出层信息（所有输出层的名字），设定并前向传播
        outInfo = self.net.getUnconnectedOutLayersNames()  ## 前面的yolov3架构也讲了，yolo在每个scale都有输出，outInfo是每个scale的名字信息，供net.forward使用
        start = time.time()
        layerOutputs = self.net.forward(outInfo)  # 得到各个输出层的、各个检测框等信息，是二维结构。
        end = time.time()
        print("[INFO] YOLO took {:.6f} seconds".format(end - start))  ## 可以打印下信息
        (H, W) = img.shape[:2]

        # 过滤layerOutputs
        # layerOutputs的第1维的元素内容: [center_x, center_y, width, height, objectness, N-class score data]
        # 过滤后的结果放入：
        boxes = []  # 所有边界框（各层结果放一起）
        confidences = []  # 所有置信度
        classIDs = []  # 所有分类ID

        # # 1）过滤掉置信度低的框框
        for out in layerOutputs:  # 各个输出层
            for detection in out:  # 各个框框
                # 拿到置信度
                scores = detection[5:]  # 各个类别的置信度
                classID = np.argmax(scores)  # 最高置信度的id即为分类id
                confidence = scores[classID]  # 拿到置信度

                # 根据置信度筛查
                if confidence > self.CONFIDENCE:
                    box = detection[0:4] * np.array([W, H, W, H])  # 将边界框放会图片尺寸
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
        # # 2）应用非最大值抑制(non-maxima suppression，nms)进一步筛掉
        idxs = cv.dnn.NMSBoxes(boxes, confidences, self.CONFIDENCE, self.THRESHOLD)  # boxes中，保留的box的索引index存入idxs
        # 得到labels列表
        with open(self.labelsPath, 'rt') as f:
            labels = f.read().rstrip('\n').split('\n')
        # 应用检测结果
        np.random.seed(42)
        COLORS = np.random.randint(0, 255, size=(len(labels), 3),
                                   dtype="uint8")  # 框框显示颜色，每一类有不同的颜色，每种颜色都是由RGB三个值组成的，所以size为(len(labels), 3)
        outString = ""
        if len(idxs) > 0:
            for i in idxs.flatten():  # indxs是二维的，第0维是输出层，所以这里把它展平成1维
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv.rectangle(img, (x, y), (x + w, y + h), color, 2)  # 线条粗细为2px
                text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
                outString = outString + text + "\n"
                cv.putText(img, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color,
                           2)  # cv.FONT_HERSHEY_SIMPLEX字体风格、0.5字体大小、粗细2px

        return img, outString[:-1]

    def dectetVideo(self, filename):
        cap = cv.VideoCapture(filename)
        frame_height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
        frame_width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
        fps = cap.get(cv.CAP_PROP_FPS)
        size = (int(frame_width),int(frame_height))
        fourcc = cv.VideoWriter_fourcc(*'XVID')#cv.VideoWriter_fourcc('M', 'J', 'P', 'G')
        out = cv.VideoWriter('output.avi',cv.VideoWriter_fourcc('M', 'J', 'P', 'G'),fps,size)
        while cap.isOpened():
            # print("I m here")
            ret, frame = cap.read()
            # 如果正确读取帧，ret为True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            frame , vout= self.dectetImg(frame)
            out.write(frame)
            img = PIL.Image.fromarray(frame)
            pixmap = img.toqpixmap()
            self.sendImg.emit([pixmap, vout])
            
        cap.release()
        cv.destroyAllWindows()

    def dectVehicle(self,img):
        print("[INFO] loading Vgg from disk...")  ## 可以打印下信息
        self.model = models.load_model("./model/Vgg/VggVehicle.h5")
        print("[INFO] detecting Vehicle ...")  ## 可以打印下信息
        x = cv.resize(img,(150,150))
        x = np.expand_dims(x, axis=0)
        x = x / 255.0
        out = self.labels[self.model.predict_classes(x).astype("int32")[0]]
        # print(self.model.predict_classes(x).astype("int32")[0], out)

        return img, out

