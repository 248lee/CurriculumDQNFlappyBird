import numpy as np
import sys
import random
import pygame
import flappy_bird_utils
import pygame.surfarray as surfarray
from pygame.locals import *
from itertools import cycle
from enum import Enum

fireReward = 0.08
misShoot = -0.8
shootWrong = -1.1
sweetBoss = 0.5
no_fire_punishment = -1.1
isSweet = False

FPS = 30000
SCREENWIDTH  = 288
SCREENHEIGHT = 512

pygame.init()
FPSCLOCK = pygame.time.Clock()
SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
pygame.display.set_caption('Flappy Bird')

IMAGES, SOUNDS, HITMASKS = flappy_bird_utils.load()
PIPEGAPSIZE = 140 # gap between upper and lower part of pipe
BASEY = SCREENHEIGHT * 0.79

PLAYER_WIDTH = IMAGES['player'][0].get_width()
PLAYER_HEIGHT = IMAGES['player'][0].get_height()
PIPE_WIDTH = IMAGES['pipe'][0].get_width()
PIPE_HEIGHT = IMAGES['pipe'][0].get_height()
BULLET_WIDTH = IMAGES['bullet'].get_width()
BULLET_HEIGHT = IMAGES['bullet'].get_height()

BACKGROUND_WIDTH = IMAGES['background'].get_width()

PLAYER_INDEX_GEN = cycle([0, 1, 2, 1])
PIPEGENERATE_DELTASTEPS = 90
IS_SIMUL = cycle([0, 0, 0, 1])

class CrashReason(Enum):
        bump_ground = 1
        bump_simul_pipe = 2
        bump_resp_pipe = 3

def initialize_game():
    global FPSCLOCK, SCREEN, IMAGES, SOUNDS, HITMASKS
    pygame.init()
    FPSCLOCK = pygame.time.Clock()
    SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
    pygame.display.set_caption('Flappy Bird')
    IMAGES, SOUNDS, HITMASKS = flappy_bird_utils.load()

class JohnTimer:
    def __init__(self, duration):
        self.timestep = 0
        self.is_set = False
        self.is_timeup = False
        self.duration = duration
    
    def resetTimer(self):
        self.timestep = 0
        self.is_set = True
        self.is_timeup = False

    def pushTimer(self):
        if self.is_set and (not self.is_timeup):
            self.timestep += 1
        if self.timestep >= self.duration:
            self.is_timeup = True
    
    def turnoffTimer(self):
        self.is_set = False
        self.is_timeup = False
        self.timestep = 0

    def isTimeup(self):
        return self.is_set and self.is_timeup
    
class GameState:
    def __init__(self):
        self.pipe_generating_timer = JohnTimer(PIPEGENERATE_DELTASTEPS)
        self.resp_pipe_timer = JohnTimer(int(PIPEGENERATE_DELTASTEPS * 1.5))
        self.redline_timer = JohnTimer(int(PIPEGENERATE_DELTASTEPS * 1.15))
        self.score = self.playerIndex = self.loopIter = 0
        self.playerx = int(SCREENWIDTH * 0.2)
        self.bulletx = self.playerx
        self.bullety = 0
        self.redlinex = SCREENWIDTH + 10
        self.playery = int((SCREENHEIGHT - PLAYER_HEIGHT) / 2)
        self.basex = 0
        self.baseShift = IMAGES['base'][0].get_width() - BACKGROUND_WIDTH

        newPipe1 = [
        {'x': SCREENWIDTH, 'y': 50 - PIPE_HEIGHT, 'type': 0, 'action': 0},  # upper pipe
        {'x': SCREENWIDTH, 'y': 50 + PIPEGAPSIZE, 'type': 0, 'action': 0},  # lower pipe
    ]
        #newPipe2 = getSimulPipe()
        self.upperPipes = [
            newPipe1[0]
            #{'x': SCREENWIDTH + (SCREENWIDTH / 2) + 30, 'y': newPipe2[0]['y'], 'type': 1},
        ]
        self.lowerPipes = [
            newPipe1[1]
            #{'x': SCREENWIDTH + (SCREENWIDTH / 2) + 30, 'y': newPipe2[1]['y'], 'type': 1},
        ]

        # player velocity, max velocity, downward accleration, accleration on flap
        self.pipeVelX = -2
        self.pipeVelY = 2
        self.up = [True, True, True, True, True, True]
        self.playerVelY    =  0    # player's velocity along Y, default same as playerFlapped
        self.playerMaxVelY =  8.5   # max vel along Y, max descend speed
        self.playerMinVelY =  -8   # min vel along Y, max ascend speed
        self.playerAccY    =   1   # players downward accleration
        self.playerFlapAcc =  -6.5   # players speed on flapping
        self.playerFlapped = False # True when player flaps
        self.bullet_speedX = 25     # bullet's velocity along X
        self.is_bullet_fired = False   # True when bullet is fired
        self.is_redline_appeared = False # True when redline is appeared
        self.is_able_to_fire = False # True when the player is able to fire the bullet. If the player is fired once, or it passes the redline, this var will become False
        self.pipe_generating_timer.resetTimer() # turn on the timer
        self.resp_pipe_timer.turnoffTimer()
        self.redline_timer.turnoffTimer()
        self.is_boss = False
        self.boss_afterwave = 100
        self.boss_afterwave_counter = 101
        self.is_hindsight = False
        self.is_previous_frame_hindsight = False
        self.base_situation = 0 # 0 -> base; 1 -> alert; 2 -> dead; 3 -> great

    def initializeGame(self):
        initialize_game()
    def closeGame(self):
        pygame.display.quit()
        exit(0)
        
    def frame_step(self, input_actions):
        self.is_previous_frame_hindsight = self.is_hindsight
        pygame.event.pump()

        self.pipe_generating_timer.pushTimer()
        self.resp_pipe_timer.pushTimer()
        self.redline_timer.pushTimer()
        reward = 0.1
        terminal = False

        if self.boss_afterwave_counter < self.boss_afterwave:
            self.boss_afterwave_counter += 1

        if sum(input_actions) != 1:
            raise ValueError('Multiple input actions!')

        # input_actions[0] == 1: do nothing
        # input_actions[1] == 1: flap the bird
        if input_actions[1] == 1:
            if self.playery > -2 * PLAYER_HEIGHT:
                self.playerVelY = self.playerFlapAcc
                self.playerFlapped = True
                #SOUNDS['wing'].play() #disable it if you do not need sound

        if (input_actions.size) >= 3 and input_actions[2] == 1:
            if (not self.is_bullet_fired) and (self.is_able_to_fire):
                self.bulletx = self.playerx
                self.is_bullet_fired = True
                delta = 5
                self.bullety = self.playery + delta
                self.is_able_to_fire = False # The player can only fire once until the next resppipe comes
                self.is_hindsight = True # Start of the hindsight interval
            reward = fireReward

        

        # playerIndex basex change
        if (self.loopIter + 1) % 3 == 0:
            self.playerIndex = next(PLAYER_INDEX_GEN)
        self.loopIter = (self.loopIter + 1) % 30
        # self.basex = -((-self.basex + 100) % self.baseShift)

        # player's movement
        if self.playerVelY < self.playerMaxVelY and not self.playerFlapped:
            self.playerVelY += self.playerAccY
        if self.playerFlapped:
            self.playerFlapped = False
        self.playery += min(self.playerVelY, BASEY - self.playery - PLAYER_HEIGHT)
        if self.playery < 0:
            self.playery = 0

        # bullet's movement
        if self.bulletx > SCREENWIDTH + 7:
            if self.is_bullet_fired:
                reward = misShoot
                print("You should shoot something, dude? reward: ", reward)
                self.is_hindsight = False
            self.is_bullet_fired = False
        if self.is_bullet_fired:
            self.bulletx += self.bullet_speedX

        # redline's movement
        if self.redlinex < -180:
            self.is_redline_appeared = False
        if self.is_redline_appeared:
            self.redlinex += self.pipeVelX

        # move pipes to left
        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            uPipe['x'] += self.pipeVelX
            lPipe['x'] += self.pipeVelX
        
        # move uPipe1 and lPipe1 up and down
        for i in range(len(self.upperPipes)):
            if self.upperPipes[i]['action'] == 0:
                delta = 0
                #if self.upperPipes[i]['type'] == 1:
                    #delta = 50
                if  self.upperPipes[i]['y'] - delta < -PIPE_HEIGHT:
                    self.up[i] = False
                elif SCREENHEIGHT - PIPEGAPSIZE < self.lowerPipes[i]['y'] + delta:
                    self.up[i] = True
                if self.up[i]:
                    self.lowerPipes[i]['y'] -= self.pipeVelY
                    self.upperPipes[i]['y'] -= self.pipeVelY
                else:
                    self.lowerPipes[i]['y'] += self.pipeVelY
                    self.upperPipes[i]['y'] += self.pipeVelY
            else:
                # move uPipe1 and lPipe1 up and down respectively
                if not self.upperPipes[i]['freeze']:
                    if  self.upperPipes[i]['y'] < -PIPE_HEIGHT:
                        self.up[i] = False
                    elif (-PIPE_HEIGHT + (SCREENHEIGHT - (SCREENHEIGHT - BASEY)) / 2) < self.upperPipes[i]['y']:
                        self.up[i] = True
                    if self.up[i]:
                        self.lowerPipes[i]['y'] += self.pipeVelY * 10
                        self.upperPipes[i]['y'] -= self.pipeVelY * 10
                    else:
                        self.lowerPipes[i]['y'] -= self.pipeVelY * 10
                        self.upperPipes[i]['y'] += self.pipeVelY * 10
        
        # move uPipe2 and lPipe2 up and down
        #if  self.upperPipes[1]['y'] < -PIPE_HEIGHT:
        #    self.up2 = False
        #elif SCREENHEIGHT - PIPEGAPSIZE < self.lowerPipes[1]['y']:
        #    self.up2 = True
        #if self.up2:
        #    self.lowerPipes[1]['y'] -= self.pipeVelY
        #    self.upperPipes[1]['y'] -= self.pipeVelY
        #else:
        #    self.lowerPipes[1]['y'] += self.pipeVelY
        #    self.upperPipes[1]['y'] += self.pipeVelY

        # add new pipe when the time quantum(PIPEGENERATE_DELTASTEPS) is arrived
        if self.pipe_generating_timer.isTimeup():
            pipe_action = next(IS_SIMUL)
            newPipe = getSimulPipe() # get simul pupe
            self.upperPipes.append(newPipe[0])
            self.lowerPipes.append(newPipe[1])
            self.pipe_generating_timer.resetTimer()
            if pipe_action == 1: # triggering generation of the resp pipe
                self.pipe_generating_timer.turnoffTimer()
                self.redline_timer.resetTimer() # start the timer to generate the redline
                self.resp_pipe_timer.resetTimer() # start the timer to generate the resp pipe
        
        if self.redline_timer.isTimeup():
            self.redlinex = SCREENWIDTH + 2
            self.is_redline_appeared = True
            self.redline_timer.turnoffTimer()
            self.is_able_to_fire = True # Let the player be able to fire the bullet
            self.is_boss = True
            self.base_situation = 1
            
        if self.resp_pipe_timer.isTimeup():
            newPipe = getSimulPipe(True)
            newRespPipe = getRespPipe(PIPE_WIDTH + 10) # generating one resp right after the simul pipe
            self.upperPipes.append(newPipe[0])
            self.lowerPipes.append(newPipe[1])
            self.upperPipes.append(newRespPipe[0])
            self.lowerPipes.append(newRespPipe[1])
            self.resp_pipe_timer.turnoffTimer()
            self.pipe_generating_timer.resetTimer() # turn on timer

        # remove first pipe if its out of the screen
        if (len(self.upperPipes)> 0) and (self.upperPipes[0]['x'] < -PIPE_WIDTH):
            self.upperPipes.pop(0)
            self.lowerPipes.pop(0)
            for j in range(len(self.up) - 1):
                self.up[j] = self.up[j + 1]

        # check if passes redline
        if self.is_redline_appeared and self.redlinex <= self.playerx:
            if self.is_able_to_fire:
                reward = no_fire_punishment
                if isSweet:
                    self.base_situation = 3
                else:
                    self.base_situation = 2
            self.is_able_to_fire = False
            self.is_boss = False
            self.boss_afterwave_counter = 0

        # check if crash here
        isCrash, crashReason = checkCrash({'x': self.playerx, 'y': self.playery,
                             'index': self.playerIndex},
                            self.upperPipes, self.lowerPipes)
        if isCrash:
            #SOUNDS['hit'].play() #disable it if you do not need sound
            #SOUNDS['die'].play() #disable it if you do not need sound
            terminal = True
            if isSweet and crashReason == CrashReason.bump_resp_pipe:
                if self.playery > PLAYER_HEIGHT / 2 + 7:
                    reward = sweetBoss
                else:
                    reward = -1
            else:
                reward = -1
            self.__init__()
            

        # check bullet his simul pipes            
        # check bullet hit resp pipes
        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            bulletRect = pygame.Rect(self.bulletx, self.bullety, BULLET_WIDTH, BULLET_HEIGHT)
            bulletMask = HITMASKS['bullet']
            if self.is_bullet_fired and uPipe['action'] == 1:
                pipeRect = pygame.Rect(uPipe['x'], 0, PIPE_WIDTH, SCREENHEIGHT)
                pipeMask = HITMASKS['special_pipe']
                if (not uPipe['freeze']) and pixelCollision(bulletRect, pipeRect, bulletMask, pipeMask):
                    open_ratio = ((lPipe['y'] - uPipe['y'] - PIPE_HEIGHT) / BASEY)
                    if open_ratio < 0.15:
                        reward = shootWrong
                        if isSweet:
                            self.base_situation = 3
                        else:
                            self.base_situation = 2
                    else:
                        reward =  2 * (open_ratio**0.5)
                        self.base_situation = 3
                    print("Big enough? reward: ", reward)
                    self.bulletx = 2 * SCREENWIDTH # only for make suring
                    self.is_bullet_fired = False
                    uPipe['freeze'] = True
                    self.is_hindsight = False # end of the hindsight interval
            elif self.is_bullet_fired and uPipe['action'] == 0:
                # upper and lower pipe rects
                uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], PIPE_WIDTH, PIPE_HEIGHT)
                lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], PIPE_WIDTH, PIPE_HEIGHT)
                if uPipe['type'] == 0:
                    uHitmask = HITMASKS['pipe'][0]
                    lHitmask = HITMASKS['pipe'][1]
                else:
                    uHitmask = HITMASKS['pipe2'][0]
                    lHitmask = HITMASKS['pipe2'][1]
                if pixelCollision(bulletRect, uPipeRect, bulletMask, uHitmask) or pixelCollision(bulletRect, lPipeRect, bulletMask, lHitmask):
                    reward = shootWrong
                    print("Are you dumb? do not shoot other normal pipes, dude? reward: ", reward)
                    self.bulletx = 2 * SCREENWIDTH # only for make suring
                    self.is_bullet_fired = False
                    self.is_hindsight = False # end of the hindsight interval
                    if isSweet:
                        self.base_situation = 3
                    else:
                        self.base_situation = 2

        
        # draw sprites
        SCREEN.blit(IMAGES['background'], (0,0))

        # draw bullets
        if self.is_bullet_fired:
            SCREEN.blit(IMAGES['bullet'], (self.bulletx, self.bullety))
        
        #draw redline
        if self.is_redline_appeared:
            SCREEN.blit(IMAGES['redline'], (self.redlinex, 0))

        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            if uPipe['type'] == 0:
                SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
                SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))
            else:
                SCREEN.blit(IMAGES['pipe2'][0], (uPipe['x'], uPipe['y']))
                SCREEN.blit(IMAGES['pipe2'][1], (lPipe['x'], lPipe['y']))
        
        # Draw base according to the situation
        SCREEN.blit(IMAGES['base'][self.base_situation], (self.basex, BASEY))
        # print score so player overlaps the score
        # showScore(self.score)
        SCREEN.blit(IMAGES['player'][self.playerIndex],
                    (self.playerx, self.playery))
        if self.is_able_to_fire:
            SCREEN.blit(IMAGES['canshoot'], (0, 0))

        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.update()
        FPSCLOCK.tick(FPS)

        # check for score
        playerMidPos = self.playerx + PLAYER_WIDTH / 2
        for pipe in self.upperPipes:
            pipeMidPos = pipe['x'] + PIPE_WIDTH / 2
            if pipeMidPos <= playerMidPos < pipeMidPos + 2:
                self.score += 1
                #SOUNDS['point'].play() #disable it if you do not need sound
                reward = 1
                print("Good! reward: ", reward)
                if pipe['action'] == 1:
                    self.base_situation = 0
        score = self.score
        #print self.upperPipes[0]['y'] + PIPE_HEIGHT - int(BASEY * 0.2)
        if terminal:
            self.is_boss = False
        if self.is_boss or (self.is_boss or self.boss_afterwave_counter < self.boss_afterwave):
            print("BOSS HERE!!")
        return image_data, reward, terminal, score, (self.is_boss or self.boss_afterwave_counter < self.boss_afterwave), self.is_hindsight or self.is_previous_frame_hindsight

def getSimulPipe(is_guarantee_no_hole=False):
    t = random.randint(0, 1)
    if is_guarantee_no_hole:
        t=0
    pipeX = SCREENWIDTH + 10
    """returns a randomly generated pipe"""

    # y of gap between upper and lower pipe
    gapYs = [20, 30, 40, 50, 60, 70, 80, 90]
    index = random.randint(0, len(gapYs)-1)
    gapY = gapYs[index]
    gapY += int(BASEY * 0.2) + SCREENHEIGHT * random.uniform(-0.5, 0.5)
    
    return [
        {'x': pipeX, 'y': gapY - PIPE_HEIGHT, 'type': t, 'action': 0},  # upper pipe
        {'x': pipeX, 'y': gapY + PIPEGAPSIZE, 'type': t, 'action': 0},  # lower pipe
    ]

# 'action': 0 means simul pipe; 'action': 1 means resp pipe

def getRespPipe(delta):
    pipeX = SCREENWIDTH + 10
    deltay = random.uniform(0, PIPE_HEIGHT)
    return [
        {'x': pipeX + delta, 'y': 0 - PIPE_HEIGHT + deltay, 'type': 0, 'action': 1, 'freeze': False},  # upper pipe
        {'x': pipeX + delta, 'y': SCREENHEIGHT - (SCREENHEIGHT - BASEY) - deltay, 'type': 0, 'action': 1, 'freeze': False},  # lower pipe
    ]

def showScore(score):
    """displays score in center of screen"""
    scoreDigits = [int(x) for x in list(str(score))]
    totalWidth = 0 # total width of all numbers to be printed

    for digit in scoreDigits:
        totalWidth += IMAGES['numbers'][digit].get_width()

    Xoffset = (SCREENWIDTH - totalWidth) / 2

    for digit in scoreDigits:
        SCREEN.blit(IMAGES['numbers'][digit], (Xoffset, SCREENHEIGHT * 0.1))
        Xoffset += IMAGES['numbers'][digit].get_width()


def checkCrash(player, upperPipes, lowerPipes):
    """returns True if player collders with base or pipes."""
    pi = player['index']
    player['w'] = IMAGES['player'][0].get_width()
    player['h'] = IMAGES['player'][0].get_height()

    # if player crashes into ground
    if player['y'] + player['h'] >= BASEY - 1:
        return True, CrashReason.bump_ground
    else:

        playerRect = pygame.Rect(player['x'], player['y'],
                      player['w'], player['h'])

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            # upper and lower pipe rects
            uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], PIPE_WIDTH, PIPE_HEIGHT)
            lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], PIPE_WIDTH, PIPE_HEIGHT)

            # player and upper/lower pipe hitmasks
            pHitMask = HITMASKS['player'][pi]
            if uPipe['type'] == 0:
                uHitmask = HITMASKS['pipe'][0]
                lHitmask = HITMASKS['pipe'][1]
            else:
                uHitmask = HITMASKS['pipe2'][0]
                lHitmask = HITMASKS['pipe2'][1]

            # if bird collided with upipe or lpipe
            uCollide = pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
            lCollide = pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

            if uCollide or lCollide:
                if uPipe['action'] == 1:
                    return True, CrashReason.bump_resp_pipe
                else:
                    return True, CrashReason.bump_simul_pipe

    return False, None


def pixelCollision(rect1, rect2, hitmask1, hitmask2, control=0):
    """Checks if two objects collide and not just their rects"""
    rect = rect1.clip(rect2)

    if rect.width == 0 or rect.height == 0:
        return False

    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in range(rect.width):
        for y in range(rect.height):
            if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                return True
    return False

#use following code to run flappy bird without AI

# game_state = GameState()
# for i in range(0,20000):
#     a_t_to_game = np.zeros(2)
#     action_index = random.randrange(2)
#     a_t_to_game[action_index] = 1
#     game_state.frame_step(a_t_to_game)