# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'form.ui'
##
## Created by: Qt User Interface Compiler version 6.5.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWidgets import (QApplication, QHeaderView, QLabel, QMainWindow,
    QMenu, QMenuBar, QPushButton, QSizePolicy,
    QSlider, QStatusBar, QTableWidget, QTableWidgetItem,
    QWidget)

#Use 1.ico as the icon

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1751, 782)
        self.actionopen_a_video = QAction(MainWindow)
        self.actionopen_a_video.setObjectName(u"actionopen_a_video")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.video_label = QLabel(self.centralwidget)
        self.video_label.setObjectName(u"video_label")
        self.video_label.setGeometry(QRect(30, 10, 231, 431))
        self.tableWidget = QTableWidget(self.centralwidget)
        self.tableWidget.setObjectName(u"tableWidget")
        self.tableWidget.setGeometry(QRect(1220, 10, 511, 351))
        self.pushButton = QPushButton(self.centralwidget)
        self.pushButton.setObjectName(u"pushButton")
        self.pushButton.setGeometry(QRect(20, 510, 361, 24))
        self.horizontalSlider_LowerH = QSlider(self.centralwidget)
        self.horizontalSlider_LowerH.setObjectName(u"horizontalSlider_LowerH")
        self.horizontalSlider_LowerH.setGeometry(QRect(860, 480, 371, 31))
        self.horizontalSlider_LowerH.setOrientation(Qt.Horizontal)
        self.horizontalSlider_LowerS = QSlider(self.centralwidget)
        self.horizontalSlider_LowerS.setObjectName(u"horizontalSlider_LowerS")
        self.horizontalSlider_LowerS.setGeometry(QRect(860, 520, 371, 31))
        self.horizontalSlider_LowerS.setOrientation(Qt.Horizontal)
        self.horizontalSlider_LowerV = QSlider(self.centralwidget)
        self.horizontalSlider_LowerV.setObjectName(u"horizontalSlider_LowerV")
        self.horizontalSlider_LowerV.setGeometry(QRect(860, 560, 371, 31))
        self.horizontalSlider_LowerV.setOrientation(Qt.Horizontal)
        self.horizontalSlider_UpperV = QSlider(self.centralwidget)
        self.horizontalSlider_UpperV.setObjectName(u"horizontalSlider_UpperV")
        self.horizontalSlider_UpperV.setGeometry(QRect(860, 680, 371, 31))
        self.horizontalSlider_UpperV.setOrientation(Qt.Horizontal)
        self.horizontalSlider_UpperS = QSlider(self.centralwidget)
        self.horizontalSlider_UpperS.setObjectName(u"horizontalSlider_UpperS")
        self.horizontalSlider_UpperS.setGeometry(QRect(860, 640, 371, 31))
        self.horizontalSlider_UpperS.setOrientation(Qt.Horizontal)
        self.horizontalSlider_UpperH = QSlider(self.centralwidget)
        self.horizontalSlider_UpperH.setObjectName(u"horizontalSlider_UpperH")
        self.horizontalSlider_UpperH.setGeometry(QRect(860, 600, 371, 31))
        self.horizontalSlider_UpperH.setOrientation(Qt.Horizontal)
        self.label_LowerH = QLabel(self.centralwidget)
        self.label_LowerH.setObjectName(u"label_LowerH")
        self.label_LowerH.setGeometry(QRect(760, 480, 91, 31))
        self.label_LowerS = QLabel(self.centralwidget)
        self.label_LowerS.setObjectName(u"label_LowerS")
        self.label_LowerS.setGeometry(QRect(760, 520, 91, 31))
        self.label_LowerV = QLabel(self.centralwidget)
        self.label_LowerV.setObjectName(u"label_LowerV")
        self.label_LowerV.setGeometry(QRect(760, 560, 91, 31))
        self.label_UpperH = QLabel(self.centralwidget)
        self.label_UpperH.setObjectName(u"label_UpperH")
        self.label_UpperH.setGeometry(QRect(760, 600, 91, 31))
        self.label_UpperS = QLabel(self.centralwidget)
        self.label_UpperS.setObjectName(u"label_UpperS")
        self.label_UpperS.setGeometry(QRect(760, 640, 91, 31))
        self.label_UpperV = QLabel(self.centralwidget)
        self.label_UpperV.setObjectName(u"label_UpperV")
        self.label_UpperV.setGeometry(QRect(760, 680, 91, 31))
        self.pushButton_HSV = QPushButton(self.centralwidget)
        self.pushButton_HSV.setObjectName(u"pushButton_HSV")
        self.pushButton_HSV.setGeometry(QRect(1240, 490, 101, 181))
        self.label_Canny_2 = QLabel(self.centralwidget)
        self.label_Canny_2.setObjectName(u"label_Canny_2")
        self.label_Canny_2.setGeometry(QRect(10, 620, 111, 31))
        self.horizontalSlider_Canny = QSlider(self.centralwidget)
        self.horizontalSlider_Canny.setObjectName(u"horizontalSlider_Canny")
        self.horizontalSlider_Canny.setGeometry(QRect(180, 580, 391, 31))
        self.horizontalSlider_Canny.setOrientation(Qt.Horizontal)
        self.label_Canny = QLabel(self.centralwidget)
        self.label_Canny.setObjectName(u"label_Canny")
        self.label_Canny.setGeometry(QRect(10, 580, 111, 31))
        self.horizontalSlider_Canny_2 = QSlider(self.centralwidget)
        self.horizontalSlider_Canny_2.setObjectName(u"horizontalSlider_Canny_2")
        self.horizontalSlider_Canny_2.setGeometry(QRect(180, 620, 391, 31))
        self.horizontalSlider_Canny_2.setOrientation(Qt.Horizontal)
        self.pushButton_Grayscale = QPushButton(self.centralwidget)
        self.pushButton_Grayscale.setObjectName(u"pushButton_Grayscale")
        self.pushButton_Grayscale.setGeometry(QRect(390, 510, 361, 24))
        self.video_label2 = QLabel(self.centralwidget)
        self.video_label2.setObjectName(u"video_label2")
        self.video_label2.setGeometry(QRect(290, 10, 231, 431))
        self.video_label3 = QLabel(self.centralwidget)
        self.video_label3.setObjectName(u"video_label3")
        self.video_label3.setGeometry(QRect(530, 10, 231, 431))
        self.label_Binary = QLabel(self.centralwidget)
        self.label_Binary.setObjectName(u"label_Binary")
        self.label_Binary.setGeometry(QRect(10, 540, 151, 31))
        self.horizontalSlider_Binary = QSlider(self.centralwidget)
        self.horizontalSlider_Binary.setObjectName(u"horizontalSlider_Binary")
        self.horizontalSlider_Binary.setGeometry(QRect(180, 540, 391, 31))
        self.horizontalSlider_Binary.setOrientation(Qt.Horizontal)
        self.time_label = QLabel(self.centralwidget)
        self.time_label.setObjectName(u"time_label")
        self.time_label.setGeometry(QRect(20, 470, 141, 31))
        self.absorption_rate_label = QLabel(self.centralwidget)
        self.absorption_rate_label.setObjectName(u"absorption_rate_label")
        self.absorption_rate_label.setGeometry(QRect(610, 470, 141, 31))
        self.area_label = QLabel(self.centralwidget)
        self.area_label.setObjectName(u"area_label")
        self.area_label.setGeometry(QRect(390, 470, 141, 31))
        self.video_label4 = QLabel(self.centralwidget)
        self.video_label4.setObjectName(u"video_label4")
        self.video_label4.setGeometry(QRect(790, 10, 231, 431))
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1751, 21))
        self.menuChoose_a_video_file = QMenu(self.menubar)
        self.menuChoose_a_video_file.setObjectName(u"menuChoose_a_video_file")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.menubar.addAction(self.menuChoose_a_video_file.menuAction())
        self.menuChoose_a_video_file.addAction(self.actionopen_a_video)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Video Processor", None))
        MainWindow.setWindowIcon(QIcon('adm0c-85g49-001.ico'))
        self.actionopen_a_video.setText(QCoreApplication.translate("MainWindow", u"open a video", None))
        self.video_label.setText(QCoreApplication.translate("MainWindow", u"Original Video", None))
        self.pushButton.setText(QCoreApplication.translate("MainWindow", u"Start playing/recording", None))
        self.label_LowerH.setText(QCoreApplication.translate("MainWindow", u"LowerH", None))
        self.label_LowerS.setText(QCoreApplication.translate("MainWindow", u"LowerS", None))
        self.label_LowerV.setText(QCoreApplication.translate("MainWindow", u"LowerV", None))
        self.label_UpperH.setText(QCoreApplication.translate("MainWindow", u"UpperH", None))
        self.label_UpperS.setText(QCoreApplication.translate("MainWindow", u"UpperS", None))
        self.label_UpperV.setText(QCoreApplication.translate("MainWindow", u"UpperV", None))
        self.pushButton_HSV.setText(QCoreApplication.translate("MainWindow", u"Confirm All", None))
        self.label_Canny_2.setText(QCoreApplication.translate("MainWindow", u"Canny Thres 2:", None))
        self.label_Canny.setText(QCoreApplication.translate("MainWindow", u"Canny Thres 1:", None))
        self.pushButton_Grayscale.setText(QCoreApplication.translate("MainWindow", u"Display Mode", None))
        self.video_label2.setText(QCoreApplication.translate("MainWindow", u"Extracted Video", None))
        self.video_label3.setText(QCoreApplication.translate("MainWindow", u"Edge Detected Video", None))
        self.label_Binary.setText(QCoreApplication.translate("MainWindow", u"Binarization Thres:", None))
        self.time_label.setText(QCoreApplication.translate("MainWindow", u"Time:", None))
        self.absorption_rate_label.setText(QCoreApplication.translate("MainWindow", u"Absorption Rate:", None))
        self.area_label.setText(QCoreApplication.translate("MainWindow", u"contour_area:", None))
        self.video_label4.setText(QCoreApplication.translate("MainWindow", u" ", None))
        self.menuChoose_a_video_file.setTitle(QCoreApplication.translate("MainWindow", u"Choose a video file", None))
    # retranslateUi

