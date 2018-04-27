# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'interactiveUI.ui'
#
# Created by: PyQt5 UI code generator 5.10
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
import os
import numpy as np
import pickle
import cv2
from DataGenerator.sort_of_clevr_generator import NUM_SHAPE, NUM_Q, NUM_COLOR
from trainer import Trainer
from vqa_util import answer2str


class Ui_Dialog(QtWidgets.QWidget):
    def __init__(self, path):
        super(Ui_Dialog, self).__init__()
        self.currentImgIdx = 0
        self.path = path
        self.ids = None
        self.data = None
        self.read_data()
        self.img = None
        self.q = None
        self.a = None
        self.RN = Trainer.load_model('../model/checkpoint_final.pth')

    def read_data(self):
        id_name = "id.txt"
        id_path = os.path.join(self.path, id_name)
        try:
            with open(id_path, 'r') as fp:
                _ids = [s.strip() for s in fp.readlines() if s]
        except IOError:
            raise IOError('Dataset not found!')
        # np.random.shuffle(_ids)
        self.ids = _ids

        data_name = 'data.hy'
        try:
            with open(os.path.join(self.path, data_name), 'rb') as f:
                self.data = pickle.load(f)
        except IOError:
            raise IOError('Dataset not found!')

    def getitem(self, item, normalize=False):
        # pre-processing and data augmentation
        id_ = self.ids[item]
        img = self.data[id_]['image']
        if normalize:
            img = img/255.
        q = self.data[id_]['question'].astype(np.float32)
        a = self.data[id_]['answer'].astype(np.float32)
        return img, q, a

    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(712, 712)
        self.widget = QtWidgets.QWidget(Dialog)
        self.widget.setGeometry(QtCore.QRect(10, 10, 691, 691))
        self.widget.setObjectName("widget")
        self.graphicsView = QtWidgets.QGraphicsView(self.widget)
        self.graphicsView.setGeometry(QtCore.QRect(0, 30, 691, 661))
        self.graphicsView.setObjectName("graphicsView")
        self.splitter = QtWidgets.QSplitter(self.widget)
        self.splitter.setGeometry(QtCore.QRect(10, 0, 671, 32))
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName("splitter")
        self.comboBox = QtWidgets.QComboBox(self.splitter)
        self.comboBox.setObjectName("comboBox")
        self.comboBox_2 = QtWidgets.QComboBox(self.splitter)
        self.comboBox_2.setObjectName("comboBox_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.splitter)
        self.pushButton_3.setObjectName("pushButton_3")
        self.lcdNumber = QtWidgets.QLCDNumber(self.splitter)
        self.lcdNumber.setObjectName("lcdNumber")
        self.pushButton = QtWidgets.QPushButton(self.splitter)
        self.pushButton.setStyleSheet("")
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.splitter)
        self.pushButton_2.setStyleSheet("")
        self.pushButton_2.setObjectName("pushButton_2")

        self.retranslateUi(Dialog)
        self.pushButton.clicked.connect(self.update1)
        self.pushButton_2.clicked.connect(self.update2)
        self.pushButton_3.clicked.connect(self.update3)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

        # insert questions and colors
        self.comboBox.addItems([
            'is it a circle or a rectangle?',
            'is it closer to the bottom of the image',
            'is it on the left of the image',
            'the color of the nearest object?',
            'the color of the farthest object?'
        ])
        self.comboBox_2.addItems([
            'blue',
            'green',
            'red',
            'yellow',
            'magenta',
            'cyan'

        ])

        # inflate image
        img, q, a = self.getitem(self.currentImgIdx)
        self.img, self.q, self.a = img, q, a
        img = np.transpose(img, (1, 2, 0))
        inflat_img = cv2.resize(img, dsize=(600, 600), interpolation=cv2.INTER_CUBIC)
        print(img.shape, q.shape, a.shape)
        scene = QtWidgets.QGraphicsScene()
        item = QtWidgets.QGraphicsPixmapItem(QtGui.QPixmap(QtGui.QImage(inflat_img.copy(), inflat_img.shape[1],
                                                                        inflat_img.shape[0],
                                                                        QtGui.QImage.Format_RGB888)))
        scene.addItem(item)
        self.graphicsView.setScene(scene)

    def inflate_img(self, img):
        img = np.transpose(img, (1, 2, 0))
        inflat_img = cv2.resize(img, dsize=(600, 600), interpolation=cv2.INTER_CUBIC)
        scene = QtWidgets.QGraphicsScene()
        item = QtWidgets.QGraphicsPixmapItem(QtGui.QPixmap(QtGui.QImage(inflat_img.copy(), inflat_img.shape[1],
                                                                        inflat_img.shape[0],
                                                                        QtGui.QImage.Format_RGB888)))
        scene.addItem(item)
        self.graphicsView.setScene(scene)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.pushButton_3.setText(_translate("Dialog", "Answer"))
        self.pushButton.setText(_translate("Dialog", "Previous"))
        self.pushButton_2.setText(_translate("Dialog", "Next"))

    def update1(self):
        self.currentImgIdx -= (NUM_Q * NUM_SHAPE)
        if self.currentImgIdx < 0:
            self.currentImgIdx = len(self.ids)-1
        self.lcdNumber.display(str(self.currentImgIdx//(NUM_Q * NUM_SHAPE)))
        img, q, a = self.getitem(self.currentImgIdx)
        self.img, self.q, self.a = img, q, a
        self.inflate_img(img)

    def update2(self):
        self.currentImgIdx += (NUM_Q * NUM_SHAPE)
        if self.currentImgIdx >= len(self.ids):
            self.currentImgIdx = 0
        self.lcdNumber.display(str(self.currentImgIdx//(NUM_SHAPE * NUM_Q)))
        img, q, a = self.getitem(self.currentImgIdx)
        self.img, self.q, self.a = img, q, a
        self.inflate_img(img)

    def update3(self):
        q_idx = self.comboBox.currentIndex()
        c_idx = self.comboBox_2.currentIndex()

        # find all qv, a, for selected object in this image
        qv_list = []
        a_list = []
        img = None

        for i in range(NUM_Q * NUM_SHAPE):
            img, q, a = self.getitem(self.currentImgIdx+i)
            qv_list.append(q)
            a_list.append(a)
            img = img
        qv = np.array(qv_list)
        a = np.array(a_list)
        idx = np.where(qv[:, c_idx] == 1)[0]
        if len(idx) <= 0:
            error = QtWidgets.QErrorMessage(self.widget)
            error.showMessage("Selected Object is not in the image!")
            return
        this_qv = qv[idx]
        this_a = a[idx]

        # select question and answer for this objects
        cur_qv = this_qv[q_idx]
        cur_a = this_a[q_idx]

        img = img/255.

        # query for answer
        pa, ta = self.RN.predict(img, cur_qv, cur_a)
        predicted_answer = answer2str(pa)
        true_answer = answer2str(ta)
        QtWidgets.QMessageBox.information(self.widget, "Prediction Result", "Predicted Answer: %s | True Answer: %s" % (
            predicted_answer, true_answer))

