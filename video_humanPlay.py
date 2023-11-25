#!/usr/bin/env python
#============================ 导入所需的库 ===========================================
from __future__ import print_function
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Ask the tensorflow to shut up. IF you disable this, a bunch of logs from tensorflow will put you down when you're using colab.
import tensorflow as tf
from keras import Model, Input
from keras.layers import Conv2D, Activation, MaxPool2D, Flatten, Dense
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import pygame
sys.path.append("game/")
import wrapped_flappy_bird as game
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

game_state = game.GameState()
frames = []
input_sidelength = 30
t = 0
while t < 650:
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
    x_t1_colored, r_t, terminal, score = game_state.frame_step(a_t_to_game)
    x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (input_sidelength, input_sidelength)), cv2.COLOR_RGB2GRAY)
    x_t1 = np.reshape(x_t1, (input_sidelength, input_sidelength, 1))
    if t > 450:
        frames.append(x_t1)
    t += 1

output_file = 'exp_video_30x30_20231124.mp4'

def create_video(frames, output_filename, fps=30):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
    print(len(frames))
    for frame in frames:
        out.write(frame[:, :, 0:3])

    out.release()

def display_and_save_images_as_video(image_list, output_filename, fps=30):
    frames = []
    for image in image_list:
        # Display the image using plt.imshow
        plt.imshow(image, cmap='gray')
        plt.axis('off')  # Turn off axes
        plt.show(block=False)  # Show the image without blocking

        # Convert the displayed figure to an image
        fig = plt.gcf()
        fig.canvas.draw()
        frame = np.array(fig.canvas.renderer.buffer_rgba())
        
        frames.append(frame)
    print(frames[0].shape)
    plt.imshow(frames[0])
    plt.show()

    # Save the frames as an MP4 video
    create_video(frames, output_filename, fps)

    # Close the figure to avoid displaying it
    plt.close()

display_and_save_images_as_video(frames, output_file)