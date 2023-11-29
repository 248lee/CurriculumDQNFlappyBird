#-*- coding: utf-8 -*-
import sys
import time
from PyQt5 import QtCore, QtGui, QtWidgets, QtMultimedia, QtMultimediaWidgets

def main():
    app = QtWidgets.QApplication(sys.argv)
    mainwin = QtWidgets.QMainWindow()
    videoWidget = QtMultimediaWidgets.QVideoWidget()
    mainwin.setCentralWidget(videoWidget)
    player = QtMultimedia.QMediaPlayer()
    player.setMedia(QtMultimedia.QMediaContent(QtCore.QUrl.fromLocalFile('/media/caotun8plus9/linux_drive/Movie_002.mp4')))
    player.setVideoOutput(videoWidget)
    player.play()
    mainwin.show()
    app.exec_()

if __name__ == '__main__':
    main()