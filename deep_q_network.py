#!/usr/bin/env python
#============================ 导入所需的库 ===========================================
from __future__ import print_function
import os 
from lockweightdense import LockedWeightsDense
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Ask the tensorflow to shut up. IF you disable this, a bunch of logs from tensorflow will put you down when you're using colab.
import tensorflow as tf
from threading import Event
from keras import Model, Input
from keras.layers import Conv2D, Activation, MaxPool2D, Flatten, Dense, BatchNormalization
import cv2
import sys
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import argparse
import pygame
os.environ['CUDA_VISIBLE_DEVICES']='0'

# parser = argparse.ArgumentParser()
# parser.add_argument('--isTrain', type=bool, default=True)
# parser.add_argument('--num_of_steps', type=int, default=1000)
# parser.add_argument('--num_of_steps2', type=int, default=1000)
# parser.add_argument('--num_of_steps3', type=int, default=1000)
# parser.add_argument('--num_of_steps_before_train', type=int, default=10000)
# args = parser.parse_args()
# max_num_of_steps = args.num_of_steps
# max_num_of_steps2 = args.num_of_steps2
# max_num_of_steps3 = args.num_of_steps3
# isTrain = args.isTrain
OBSERVE = 5000 # 训练前观察积累的轮数

side_length_each_stage = [(0, 0), (40, 40), (80, 80), (160, 160)]
sys.path.append("game/")
import wrapped_flappy_bird as game
tf.debugging.set_log_device_placement(True)
GAME = 'FlappyBird' # 游戏名称
ACTIONS_1 = 2
ACTIONS_2 = 3 # change to not equal 3 if you don't want action 3 to be treated specially
ACTIONS_NAME=['不动','起飞', 'FIRE']  #动作名
GAMMA = 0.99 # 未来奖励的衰减
EPSILON = 0.0001
REPLAY_MEMORY = 50000 # 观测存储器D的容量
BATCH = 32 # 训练batch大小

class MyNet(Model):
    def __init__(self, num_of_actions):
        super(MyNet, self).__init__()
        self.c1_1 = Conv2D(filters=16, kernel_size=(3, 3), padding='same', name='conv_1', 
                           kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=None),
                           bias_initializer = tf.keras.initializers.Constant(value=0.01))  # 卷积层
        #self.b1 = BatchNormalization()  # BN层
        self.a1_1 = Activation('relu', name='relu_1')  # 激活层
        self.p1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same', name='padding_1')  # 池化层
        #self.d1 = Dropout(0.2)  # dropout层

        self.flatten = Flatten()
        self.f1 = Dense(512, activation='relu', name='dense1',
                           kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=None),
                           bias_initializer = tf.keras.initializers.Constant(value=0.01))
        self.f2 = Dense(num_of_actions, activation=None, name='dense2',
                           kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=None),
                           bias_initializer = tf.keras.initializers.Constant(value=0.01))

    def call(self, x):
        #print(x.shape)
        x = self.c1_1(x)
        #print(x.shape)
        x = self.a1_1(x)
        x = self.p1(x)

        x = self.flatten(x)
        x = self.f1(x)
        y = self.f2(x)
        return y
    
    def change2To3(self, two_action_net):
        self.c1_1.set_weights([two_action_net.c1_1.get_weights()[0], two_action_net.c1_1.get_weights()[1]])
        self.f1.set_weights([two_action_net.f1.get_weights()[0], two_action_net.f1.get_weights()[1]])
        new_fc, new_bias = custom_dense(two_action_net, new_net=self)
        self.f2.set_weights([new_fc, new_bias])
    
class MyNet2(Model):
    def __init__(self, num_of_actions):
        super(MyNet2, self).__init__()
        self.num_of_actions = num_of_actions
        self.conv2_num_of_filters = 32
        self.c2_1 = Conv2D(filters=self.conv2_num_of_filters, kernel_size=(3, 3), padding='same', name='conv_2',
                           kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=None),
                           bias_initializer = tf.keras.initializers.Constant(value=0.01))  # 卷积层
        #self.b2 = BatchNormalization()  # BN层
        self.a2_1 = Activation('relu', name='relu_2')  # 激活层
        
        self.p2 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same', name='padding_2')  # 池化层
        self.c1_1 = Conv2D(filters=16, kernel_size=(3, 3), padding='same', name='conv_1', 
                           kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=None),
                           bias_initializer = tf.keras.initializers.Constant(value=0.01))  # 卷积层
        #self.b1 = BatchNormalization()  # BN层
        self.a1_1 = Activation('relu', name='relu_1')  # 激活层
        self.p1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same', name='padding_1')  # 池化层
        #self.d1 = Dropout(0.2)  # dropout层

        self.flatten = Flatten()
        self.f1 = Dense(512, activation='relu', name='dense1',
                           kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=None),
                           bias_initializer = tf.keras.initializers.Constant(value=0.01))
        self.f2 = Dense(num_of_actions, activation=None, name='dense2',
                           kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=None),
                           bias_initializer = tf.keras.initializers.Constant(value=0.01))
    def call(self, x):
        x = self.c2_1(x)
        x = self.a2_1(x)
        x = self.p2(x)

        x = self.c1_1(x)
        x = self.a1_1(x)
        x = self.p1(x)

        x = self.flatten(x)
        x = self.f1(x)
        y = self.f2(x)
        return y
    def load_stage1(self, stage1_net):
        new_kernel = custom_kernel_stage2(stage1_net, self.conv2_num_of_filters // 4)
        self.c1_1.set_weights([new_kernel, stage1_net.c1_1.get_weights()[1]])
        self.f1.set_weights([stage1_net.f1.get_weights()[0], stage1_net.f1.get_weights()[1]])
        self.f2.set_weights(stage1_net.f2.get_weights())
        return
    
class MyNet3(Model):
    def __init__(self, num_of_actions):
        super(MyNet3, self).__init__()
        self.num_of_actions = num_of_actions
        self.conv3_num_of_filters = 32
        self.c3_1 = Conv2D(filters=self.conv3_num_of_filters, kernel_size=(3, 3), padding='same', name='conv_3',
                           kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=None),
                           bias_initializer = tf.keras.initializers.Constant(value=0.01))  # 卷积层
        #self.b1 = BatchNormalization()  # BN层
        self.a3_1 = Activation('relu', name='relu_3')  # 激活层
        
        self.p3 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same', name='padding_3')  # 池化层


        self.c2_1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', name='conv_2',
                           kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=None),
                           bias_initializer = tf.keras.initializers.Constant(value=0.01))  # 卷积层
        #self.b1 = BatchNormalization()  # BN层
        self.a2_1 = Activation('relu', name='relu_2')  # 激活层
        
        self.p2 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same', name='padding_2')  # 池化层
        self.c1_1 = Conv2D(filters=16, kernel_size=(3, 3), padding='same', name='conv_1', 
                           kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=None),
                           bias_initializer = tf.keras.initializers.Constant(value=0.01))  # 卷积层
        #self.b1 = BatchNormalization()  # BN层
        self.a1_1 = Activation('relu', name='relu_1')  # 激活层
        self.p1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same', name='padding_1')  # 池化层
        #self.d1 = Dropout(0.2)  # dropout层

        self.flatten = Flatten()
        self.f1 = Dense(512, activation='relu', name='dense1',
                           kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=None),
                           bias_initializer = tf.keras.initializers.Constant(value=0.01))
        self.f2 = Dense(num_of_actions, activation=None, name='dense2',
                           kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=None),
                           bias_initializer = tf.keras.initializers.Constant(value=0.01))
        
    def call(self, x):
        x = self.c3_1(x)
        x = self.a3_1(x)
        x = self.p3(x)

        x = self.c2_1(x)
        x = self.a2_1(x)
        x = self.p2(x)

        x = self.c1_1(x)
        x = self.a1_1(x)
        x = self.p1(x)

        x = self.flatten(x)
        x = self.f1(x)
        y = self.f2(x)
        return y
    def load_stage2(self, stage2_net):
        new_kernel = custom_kernel_stage3(stage2_net, self.conv3_num_of_filters // 4)
        self.c2_1.set_weights([new_kernel, stage2_net.c2_1.get_weights()[1]])        
        self.c1_1.set_weights([new_kernel, stage2_net.c1_1.get_weights()[1]])
        self.f1.set_weights([stage2_net.f1.get_weights()[0], stage2_net.f1.get_weights()[1]])
        self.f2.set_weights(stage2_net.f2.get_weights())
        return
    
def myprint(s):
    with open('structure.txt','w') as f:
        print(s, file=f)

def trainNetwork(stage, num_of_actions, lock_mode, is_simple_actions_locked, max_steps, resume_Adam, learning_rate=1e-6, event=None, is_colab=False):
    neuron = open("neurons.txt", 'w')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Ask the tensorflow to shut up. IF you disable this, a bunch of logs from tensorflow will put you down when you're using colab.
    tf.debugging.set_log_device_placement(False)
    if OBSERVE < 1000:
        print("--num_of_steps_before_train should be more than 1000 in order to plot rewards. This is because we'll start to plot average rewards per 1000 steps when the model starts training.")
        return
    if not is_colab:
      from PyQt5.QtCore import Qt, QTimer
    if is_colab:
        #sys.stdout = open(os.devnull, 'w')
        import datetime
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape/curriculum/train'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    
        
#============================ 模型创建与加载 ===========================================
    old_time = 0 # Python is trash
    t = 0 #初始化TIMESTEP
    # 模型创建
    input_sidelength = side_length_each_stage[stage]
    last_input_sidelength = side_length_each_stage[stage - 1]
    checkpoint_save_path = "./model/FlappyBird.h5"
    epsilon = EPSILON

    now_num_action = ACTIONS_1
    if os.path.exists('now_num_of_actions.txt'):
        ns = open('now_num_of_actions.txt', 'r')
        now_num_action = int(ns.readline())
        ns.close()

    now_stage = 1
    if os.path.exists('now_stage.txt'):
        ns = open('now_stage.txt', 'r')
        now_stage = int(ns.readline())
        ns.close()
    if stage == 1:
        optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-5, epsilon=1e-08)
        net1 = MyNet(num_of_actions)
        net1_target = MyNet(num_of_actions)
        net1.build(input_shape=(1, input_sidelength[0], input_sidelength[1], 4))
        net1.call(Input(shape=(input_sidelength[0], input_sidelength[1], 4)))
        net1.summary(print_fn=myprint)
        if now_num_action == num_of_actions:
            if os.path.exists(checkpoint_save_path):
                print('-------------load the model-----------------')
                net1.load_weights(checkpoint_save_path,by_name=True)
            else:
                print('-------------train new model-----------------')
        else:
            print("FROM TWO ACTIONS TO THREE!")
            old_net1 = MyNet(ACTIONS_1)
            old_net1.build(input_shape=(1, input_sidelength[0], input_sidelength[1], 4))
            old_net1.call(Input(shape=(input_sidelength[0], input_sidelength[1], 4)))
            old_net1.summary(print_fn=myprint)
            old_net1.load_weights(checkpoint_save_path,by_name=True)
            net1.change2To3(old_net1)
            old_net1 = None
            num_actions_file = open('now_stage.txt', 'w')
            num_actions_file.write(str(ACTIONS_2))
            num_actions_file.close()
        now_stage_file = open('now_stage.txt', 'w')
        now_stage_file.write("1")
        now_stage_file.close()
        if lock_mode == 1: # only fc is unlocked
            net1.c1_1.trainable = False
            net1.f1.trainable = False
            net1.f2.trainable = True
        
    elif stage == 2:
        num_of_actions = ACTIONS_2
        if stage > now_stage:
            stage1_net = MyNet(num_of_actions)
            stage1_net.build(input_shape=(1, last_input_sidelength[0], last_input_sidelength[1], 4))
            stage1_net.call(Input(shape=(last_input_sidelength[0], last_input_sidelength[1], 4)))
            if os.path.exists(checkpoint_save_path):
                print('-------------load the model-----------------')
                stage1_net.load_weights(checkpoint_save_path,by_name=True)
            else:
                print("NO pretrained model to load! Pleast train stage1 first!")
                return

            net1 = MyNet2(ACTIONS_2)
            net1_target = MyNet2(ACTIONS_2)
            optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate, epsilon=1e-08)                
            net1.build(input_shape=(1, input_sidelength[0], input_sidelength[1], 4))
            net1.load_stage1(stage1_net)
            net1.call(Input(shape=(input_sidelength[0], input_sidelength[1], 4)))
            net1.summary(print_fn=myprint)
        else:
            net1 = MyNet2(ACTIONS_2)
            net1_target = MyNet2(ACTIONS_2)
            optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate, epsilon=1e-08)
            net1.build(input_shape=(1, input_sidelength[0], input_sidelength[1], 4))
            net1.call(Input(shape=(input_sidelength[0], input_sidelength[1], 4)))
            if os.path.exists(checkpoint_save_path):
                print('-------------load the model-----------------')
                net1.load_weights(checkpoint_save_path,by_name=True)
            else:
                print("NO pretrained model to load! Pleast train stage1 first!")
                return
            net1.summary(print_fn=myprint)
        if lock_mode == 0: # only new added is unlocked
            net1.c2_1.trainable = True
            net1.c1_1.trainable = False
            net1.f1.trainable = False
            net1.f2.trainable = False
        elif lock_mode == 1: # only fc is unlocked
            net1.c2_1.trainable = False
            net1.c1_1.trainable = False
            net1.f1.trainable = False
            net1.f2.trainable = True
        elif lock_mode == 2: # everything is unlocked
            net1.c2_1.trainable = True
            net1.c1_1.trainable = True
            net1.f1.trainable = True
            net1.f2.trainable = True


    elif stage == 3:
        num_of_actions = ACTIONS_2
        if stage > now_stage:
            stage2_net = MyNet2(num_of_actions)
            stage2_net.build(input_shape=(1, last_input_sidelength[0], last_input_sidelength[1], 4))
            stage2_net.call(Input(shape=(last_input_sidelength[0], last_input_sidelength[1], 4)))
            if os.path.exists(checkpoint_save_path):
                print('-------------load the model-----------------')
                stage2_net.load_weights(checkpoint_save_path,by_name=True)
            else:
                print("NO pretrained model to load! Pleast train stage1 first!")
                return

            net1 = MyNet3(ACTIONS_2)
            net1_target = MyNet3(ACTIONS_2)
            optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate, epsilon=1e-08)
            net1.build(input_shape=(1, input_sidelength[0], input_sidelength[1], 4))
            net1.load_stage2(stage2_net)
            net1.call(Input(shape=(input_sidelength[0], input_sidelength[1], 4)))
            net1.summary(print_fn=myprint)
        else:
            net1 = MyNet3(ACTIONS_2)
            net1_target = MyNet3(ACTIONS_2)
            optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate, epsilon=1e-08)
            net1.build(input_shape=(1, input_sidelength[0], input_sidelength[1], 4))
            net1.call(Input(shape=(input_sidelength[0], input_sidelength[1], 4)))
            if os.path.exists(checkpoint_save_path):
                print('-------------load the model-----------------')
                net1.load_weights(checkpoint_save_path,by_name=True)
            else:
                print("NO pretrained model to load! Pleast train stage1 first!")
                return
            net1.summary(print_fn=myprint)
        if lock_mode == 0: # only new added is unlocked
            net1.c3_1.trainable = True
            net1.c2_1.trainable = False
            net1.c1_1.trainable = False
            net1.f1.trainable = False
            net1.f2.trainable = False
        elif lock_mode == 1: # only fc is unlocked
            net1.c3_1.trainable = False
            net1.c2_1.trainable = False
            net1.c1_1.trainable = False
            net1.f1.trainable = False
            net1.f2.trainable = True
        elif lock_mode == 2: # everything is unlocked
            net1.c3_1.trainable = False
            net1.c2_1.trainable = True
            net1.c1_1.trainable = True
            net1.f1.trainable = True
            net1.f2.trainable = True

    else:
        print("笑死你可不可以給一個正確的 stage值阿? 阿就 1, 2, 3挑一個阿")
        return
    # Restore old_steps
    if os.path.exists("last_old_time.txt"):
      old_time_file = open("last_old_time.txt", 'r')
      old_time = int(old_time_file.readline())

#============================ 加载(搜集)数据集 ===========================================    
    # Restore Adam and load up the learning rate
    if resume_Adam:
        checkpoint = tf.train.Checkpoint(optimizer=optimizer)
        checkpoint_dir = './model'
        checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=1)
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        if learning_rate > 0:
            optimizer.learning_rate = learning_rate
    print(optimizer.iterations)
    print(stage, num_of_actions, lock_mode, is_simple_actions_locked, max_steps, resume_Adam, learning_rate, event, is_colab)
    input()
    neuron.write(str(net1.f2.get_weights()[0]))
    neuron.write("\n===========================\n")
    # 打开游戏
    game_state = game.GameState()
    game_state.initializeGame()

    # 将每一轮的观测存在D中，之后训练从D中随机抽取batch个数据训练，以打破时间连续导致的相关性，保证神经网络训练所需的随机性。
    D = deque()
    D_boss = deque()

    #初始化状态并且预处理图片，把连续的四帧图像作为一个输入（State）
    do_nothing = np.zeros(num_of_actions)
    do_nothing[0] = 1
    x_t, r_0, terminal, _, _ = game_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (input_sidelength[0], input_sidelength[1])), cv2.COLOR_RGB2GRAY)
    x_t = (x_t - np.mean(x_t)) / 64
    #ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    rewards = []
    num_of_episode = 0
    avg_reward = 0
    avg_rewards_1000steps = []
    readouts = []

    scores = []
    avg_score = 0
    avg_scores_1000steps = []

    t_train = 0
    net1_target.build(input_shape=(1, input_sidelength[0], input_sidelength[1], 4))
    net1_target.call(Input(shape=(input_sidelength[0], input_sidelength[1], 4)))
    net1_target.set_weights(net1.get_weights())
    # 开始训练
    while True:
        if (event != None and event.is_set()) or t > max_steps:
            print(net1.f2.get_weights()[0])
            print("stupid python")
            neuron.write(str(net1.f2.get_weights()[0]))
            neuron.close()
            game_state.closeGame() # python is trash
            break
            #exit(0)
        # 根据输入的s_t,选择一个动作a_t
        
        readout_t = net1(tf.expand_dims(tf.constant(s_t, dtype=tf.float32), 0))
        print(readout_t)
        readouts.append(readout_t)
        a_t_to_game = np.zeros([num_of_actions])
        action_index = 0

        #贪婪策略，有episilon的几率随机选择动作去探索，否则选取Q值最大的动作
        ispress = False
        for pevent in pygame.event.get():
            if pevent.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # checking if keydown event happened or not
            if pevent.type == pygame.KEYDOWN:
                if pevent.key == pygame.K_SPACE:
                    # if keydown event happened
                    # than printing a string to output
                    print("Teacher's fly")
                    a_t_to_game[1] = 1
                    ispress = True
                elif pevent.key == pygame.K_0:
                    print("Teacher's FIRE!!!")
                    a_t_to_game[2] = 1
                    ispress = True
                
        if (not ispress):
            if (t > -1):
                if random.random() <= epsilon:
                    print("----------Random Action----------")
                    action_index = random.randrange(num_of_actions)
                    a_t_to_game[action_index] = 1
                else:
                    print("-----------net choice----------------")
                    action_index = np.argmax(readout_t)
                    print("-----------index----------------")
                    print(action_index)
                    a_t_to_game[action_index] = 1
            else:
                a_t_to_game[0] = 1
        if (t <= -1):
            ispress = False
            for i in range(len(a_t_to_game)):
                if not ispress:
                    if a_t_to_game[i] == 1:
                        ispress = True
                else:
                    a_t_to_game[i] = 0


        #执行这个动作并观察下一个状态以及reward
        x_t1_colored, r_t, terminal, score, is_boss = game_state.frame_step(a_t_to_game)
        print("============== score ====================")
        print(score)

        rank_file_r = open("rank.txt","r")
        best = int(rank_file_r.readline())
        rank_file_r.close()
        #if score_one_round >= best:
        #    test = True
        best_checkpoint_save_path = "./best/FlappyBird"
        if score > best:
            net1.save_weights(best_checkpoint_save_path)
            rank_file_w = open("rank.txt","w")
            rank_file_w.write("%d" % score)
            print("********** best score updated!! *********")
            rank_file_w.close()

        a_t = np.argmax(a_t_to_game, axis=0)
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (input_sidelength[0], input_sidelength[1])), cv2.COLOR_RGB2GRAY)
        #ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (input_sidelength[1], input_sidelength[0], 1))
        mea = np.mean(x_t1)
        x_t1 = (x_t1 - mea) / 64
        
        #x_t_back = x_t1 * 64 + mea
        #plt.imshow(x_t_back, cmap='gray')
        #plt.savefig('game.png')
        #input()
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

        s_t_D = tf.convert_to_tensor(s_t, dtype=tf.float16)
        a_t_D = tf.constant(a_t, dtype=tf.int32)
        r_t_D = tf.constant(r_t, dtype=tf.float32)
        s_t1_D = tf.constant(s_t1, dtype=tf.float16)
        terminal = tf.constant(terminal, dtype=tf.float32)

        # 将观测值存入之前定义的观测存储器D中
        if is_boss:
            D_boss.append((s_t_D, a_t_D, r_t_D, s_t1_D, terminal))
        else:
            D.append((s_t_D, a_t_D, r_t_D, s_t1_D, terminal))
        #如果D满了就替换最早的观测
        if len(D) > REPLAY_MEMORY:
            D.popleft()
        if len(D_boss) > REPLAY_MEMORY / 4:
            D_boss.popleft()

        # 更新状态，不断迭代
        s_t = s_t1
        t += 1

#============================ 训练网络 ===========================================

        # 观测一定轮数后开始训练
        if (t > OBSERVE):
            # Start training! Therefore we update the now_stage file
            if now_stage != stage:
                now_stage_file = open('now_stage.txt', 'w')
                now_stage_file.write(str(stage))
                now_stage_file.close()
                now_stage = stage
            
            t_train += 1
            # 随机抽取minibatch个数据训练
            print("==================start train====================")
            print("Boss length", len(D_boss))
            if len(D_boss) == 0:
                minibatch = random.sample(D, BATCH)
            else:
                minibatch = random.sample(D, int(BATCH * 3 / 4))
                remainingBATCH = BATCH - (int(BATCH * 3 / 4))
                boss_minibatch = random.sample(D_boss, remainingBATCH)
                for btch in boss_minibatch:
                    minibatch.append(btch)
                

            # 获得batch中的每一个变量
            b_s = [d[0] for d in minibatch]
            b_s = tf.stack(b_s, axis=0)
            #b_s = tf.cast(b_s, dtype=tf.float32)

            b_a = [d[1] for d in minibatch]
            b_a = tf.expand_dims(b_a, axis=1)
            b_a = tf.stack(b_a, axis=0)

            b_r = [d[2] for d in minibatch]
            b_r = tf.stack(b_r, axis=0)

            b_s_ = [d[3] for d in minibatch]
            b_s_ = tf.stack(b_s_, axis=0)
            #b_s_ = tf.cast(b_s_, dtype=tf.float32)

            b_done = [d[4] for d in minibatch]
            b_done = tf.stack(b_done, axis=0)

            q_next = tf.reduce_max(net1_target(b_s_), axis=1)
            q_truth = b_r + GAMMA * q_next* (tf.ones(32) - b_done)

            # 训练
            with tf.GradientTape() as tape:
                q_output = net1(b_s)
                index = tf.expand_dims(tf.constant(np.arange(0, BATCH), dtype=tf.int32), 1)
                index_b_a = tf.concat((index, b_a), axis=1)
                q = tf.gather_nd(q_output, index_b_a)
                loss = tf.losses.MSE(q_truth, q)
                print("loss = %f" % loss)
                gradients = tape.gradient(loss, net1.trainable_variables)
                if num_of_actions == ACTIONS_2 and lock_mode >= 1 and is_simple_actions_locked:
                    print("Lock actions: static and jump")
                    f2_weightings_index = len(gradients) - 2
                    tensor = tf.constant([[0.0, 0.0, 1.0] for i in range(gradients[f2_weightings_index].shape[0])], shape=[gradients[f2_weightings_index].shape[0], gradients[f2_weightings_index].shape[1]])
                    print(gradients[f2_weightings_index].shape, tensor.shape)
                    gradients[f2_weightings_index] = gradients[f2_weightings_index] * tensor
                optimizer.apply_gradients(zip(gradients, net1.trainable_variables))

            # 每 train 1000轮保存一次网络参数
            if (t_train+old_time) % 1000 == 0:
                print("=================model save====================")
                net1.save_weights(checkpoint_save_path,save_format='h5')
                # store the old_time variable
                old_time_file = open("last_old_time.txt", 'w')
                old_time_file.write(str(t_train+old_time))
                result_file = open("results.txt", 'a')
                for ar in avg_rewards_1000steps:
                    result_file.write(str(ar) + '\n')
                result_file.close()
                # score_file = open("scores_training.txt", 'a')
                # for ars in avg_scores_1000steps:
                #     score_file.write(str(ars) + '\n')
                # score_file.close()
                avg_rewards_1000steps = []
                avg_scores_1000steps = []
                # Save Adam optimizer status
                checkpoint = tf.train.Checkpoint(optimizer=optimizer)
                checkpoint_dir = './model'
                checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=1)
                checkpoint_manager.save()
            if (t_train+old_time) % 5000 == 0:
                # Update the target network!!!!
                net1_target.set_weights(net1.get_weights())
        # 打印信息
        if (t > OBSERVE):
            print("TRAINED_TIMESTEP", (t_train+old_time), "|  ACTION", ACTIONS_NAME[action_index], "|  REWARD", r_t, \
             "|  Q_MAX %e \n" % np.max(readout_t), "| EPISODE", num_of_episode)
            rewards.append(r_t)
            scores.append(score)
        else:
            print("OBSERVED_TIMESTEP", t, "|  ACTION", ACTIONS_NAME[action_index], "|  REWARD", r_t, \
             "|  Q_MAX %e \n" % np.max(readout_t), "| EPISODE", num_of_episode)
        # write result to the average array, prepare to write to the file
        if len(rewards) >= 1000:
            avg_reward = avg_reward - (rewards[len(rewards) - 1000] / 1000)
            avg_reward = avg_reward + (rewards[len(rewards) - 1] / 1000)
            avg_rewards_1000steps.append(avg_reward)
            if is_colab:
              with train_summary_writer.as_default():
                tf.summary.scalar('reward', avg_reward, step=len(avg_rewards_1000steps))
        else:
            if t > OBSERVE:
                avg_reward = avg_reward + r_t / 1000
        if len(rewards) >= 5000: # Clean the memory of rewards
            tmp_new_rewards = []
            for i in range(len(rewards) - 1000, len(rewards)):
                tmp_new_rewards.append(rewards[i])
            rewards = tmp_new_rewards

        # write score to the average array, prepare to write to the file
        if len(scores) >= 1000:
            avg_score = avg_score - (scores[len(scores) - 1000] / 1000)
            avg_score = avg_score + (scores[len(scores) - 1] / 1000)
            avg_scores_1000steps.append(avg_score)
            if is_colab:
              with train_summary_writer.as_default():
                tf.summary.scalar('scores', avg_score, step=len(avg_scores_1000steps))
        else:
            if t > OBSERVE:
                avg_score = avg_score + score / 1000
        if len(scores) >= 5000: # Clean the memory of scores
            tmp_new_scores = []
            for i in range(len(scores) - 1000, len(scores)):
                tmp_new_scores.append(scores[i])
            scores = tmp_new_scores

        # write readouts (q_values) to files
        if len(readouts) >= 1000:
            num_of_files = readouts[0].shape[1]
            for i in range(num_of_files):
                f = open('./Qvalues/Q'+str(i)+'.txt', 'a')
                for q in readouts:
                    f.write(str(float(q[0][i]))+'\n')
                f.close()
            readouts = [] # clean the memory of the readouts
            

        # Count episodes
        if terminal:
            num_of_episode = num_of_episode + 1
        


def custom_kernel_stage2(old_net, thickness):
    old_kernel = old_net.c1_1.get_weights()[0].T
    new_kernel = []
    for i in range(len(old_kernel)):
        tmp = old_kernel[i]
        tmp_stack = np.array([tmp for i in range(thickness)])
        sh = tmp_stack.shape
        tmp_stack = tmp_stack.reshape((sh[0] * sh[1], sh[2], sh[3]))
        new_kernel.append(tmp_stack)
    return (np.array(new_kernel).T)

def custom_dense(old_net, new_net):
  old_fc = old_net.f2.get_weights()[0]
  new_fc = new_net.f2.get_weights()[0]
  for i in range(old_fc.shape[0]):
    for j in range(old_fc.shape[1]):
      new_fc[i][j] = old_fc[i][j]
  old_bias = old_net.f2.get_weights()[1]
  new_bias = new_net.f2.get_weights()[1]
  for i in range(old_bias.shape[0]):
    new_bias[i] = old_bias[i]
  
  return new_fc, new_bias

def custom_kernel_stage3(old_net, thickness):
    old_kernel = old_net.c2_1.get_weights()[0].T
    new_kernel = []
    for i in range(len(old_kernel)):
        tmp = old_kernel[i]
        tmp_stack = np.array([tmp for i in range(thickness)])
        sh = tmp_stack.shape
        tmp_stack = tmp_stack.reshape((sh[0] * sh[1], sh[2], sh[3]))
        new_kernel.append(tmp_stack)
    return (np.array(new_kernel).T)

def main():
    trainNetwork()

if __name__ == "__main__":
    main()
