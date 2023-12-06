#!/usr/bin/env python
#============================ 导入所需的库 ===========================================
from __future__ import print_function
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Ask the tensorflow to shut up. IF you disable this, a bunch of logs from tensorflow will put you down when you're using colab.
import tensorflow as tf
from threading import Event
from keras import Model, Input
from keras.layers import Conv2D, Activation, MaxPool2D, Flatten, Dense
import cv2
import sys
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import pygame
import argparse
os.environ['CUDA_VISIBLE_DEVICES']='0'
sys.path.append("game/")
import wrapped_flappy_bird as game

game_state = game.GameState()
while True:
    a_t_to_game = np.zeros(3)
    ispress = False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
         
        # checking if keydown event happened or not
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                # if keydown event happened
                # than printing a string to output
                print("A key has been pressed")
                a_t_to_game[1] = 1
                ispress = True
            elif event.key == pygame.K_0:
                print("FIRE!!!")
                a_t_to_game[2] = 1
                ispress = True
    
    if ispress == False:
        a_t_to_game[0] = 1
    ispress = False
    for i in range(len(a_t_to_game)):
        if not ispress:
            if a_t_to_game[i] == 1:
                ispress = True
        else:
            a_t_to_game[i] = 0
        
    game_state.frame_step(a_t_to_game)

