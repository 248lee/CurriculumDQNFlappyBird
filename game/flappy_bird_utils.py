import pygame
import sys
def load():
    # path of player with different states
    PLAYER_PATH = (
            'assets/sprites/redbird-upflap2.png',
            'assets/sprites/redbird-midflap2.png',
            'assets/sprites/redbird-downflap2.png'
    )

    # path of background
    BACKGROUND_PATH = 'assets/sprites/background-black.png'

    # path of pipe
    PIPE_PATH = 'assets/sprites/pipe-green.png'
    PIPE_PATH2 = 'assets/sprites/pipe-green2.png'

    # path of bullet
    BULLET_PATH = 'assets/sprites/bullet2.png'

    # path of redline
    REDLINE_PATH = 'assets/sprites/redline.png'

    IMAGES, SOUNDS, HITMASKS = {}, {}, {}

    # numbers sprites for score display
    IMAGES['numbers'] = (
        pygame.image.load('assets/sprites/0.png').convert_alpha(),
        pygame.image.load('assets/sprites/1.png').convert_alpha(),
        pygame.image.load('assets/sprites/2.png').convert_alpha(),
        pygame.image.load('assets/sprites/3.png').convert_alpha(),
        pygame.image.load('assets/sprites/4.png').convert_alpha(),
        pygame.image.load('assets/sprites/5.png').convert_alpha(),
        pygame.image.load('assets/sprites/6.png').convert_alpha(),
        pygame.image.load('assets/sprites/7.png').convert_alpha(),
        pygame.image.load('assets/sprites/8.png').convert_alpha(),
        pygame.image.load('assets/sprites/9.png').convert_alpha()
    )

    # base (ground) sprite
    IMAGES['base'] = (
        pygame.image.load('assets/sprites/base.png').convert_alpha(),
        pygame.image.load('assets/sprites/base_alert.png').convert_alpha(),
        pygame.image.load('assets/sprites/base_dead.png').convert_alpha(),
        pygame.image.load('assets/sprites/base_great.png').convert_alpha()
    )

    # sounds
    if 'win' in sys.platform:
        soundExt = '.wav'
    else:
        soundExt = '.ogg'

    #SOUNDS['die']    = pygame.mixer.Sound('assets/audio/die' + soundExt)
    #SOUNDS['hit']    = pygame.mixer.Sound('assets/audio/hit' + soundExt)
    #SOUNDS['point']  = pygame.mixer.Sound('assets/audio/point' + soundExt)
    #SOUNDS['swoosh'] = pygame.mixer.Sound('assets/audio/swoosh' + soundExt)
    #SOUNDS['wing']   = pygame.mixer.Sound('assets/audio/wing' + soundExt)

    # select random background sprites
    IMAGES['background'] = pygame.image.load(BACKGROUND_PATH).convert()

    # select random player sprites
    IMAGES['player'] = (
        pygame.image.load(PLAYER_PATH[0]).convert_alpha(),
        pygame.image.load(PLAYER_PATH[1]).convert_alpha(),
        pygame.image.load(PLAYER_PATH[2]).convert_alpha(),
    )

    # select random pipe sprites
    IMAGES['pipe'] = (
        pygame.transform.rotate(
            pygame.image.load(PIPE_PATH).convert_alpha(), 180),
        pygame.image.load(PIPE_PATH).convert_alpha(),
    )
    IMAGES['pipe2'] = (
        pygame.transform.rotate(
            pygame.image.load(PIPE_PATH2).convert_alpha(), 180),
        pygame.image.load(PIPE_PATH2).convert_alpha(),
    )

    IMAGES['bullet'] = pygame.image.load(BULLET_PATH).convert_alpha()
    IMAGES['redline'] = pygame.image.load(REDLINE_PATH).convert_alpha()
    IMAGES['canshoot'] = IMAGES['numbers'][7]

    # hismask for pipes
    HITMASKS['pipe'] = (
        getHitmask(IMAGES['pipe'][0]),
        getHitmask(IMAGES['pipe'][1]),
    )
    HITMASKS['pipe2'] = (
        getHitmask(IMAGES['pipe2'][0]),
        getHitmask(IMAGES['pipe2'][1]),
    )

    # hitmask for player
    HITMASKS['player'] = (
        getHitmask(IMAGES['player'][0]),
        getHitmask(IMAGES['player'][1]),
        getHitmask(IMAGES['player'][2]),
    )

    HITMASKS['special_pipe'] = getSpecialHitmask(IMAGES['pipe'][0])
    HITMASKS['bullet'] = getHitmask(IMAGES['bullet'])

    return IMAGES, SOUNDS, HITMASKS

def getHitmask(image):
    """returns a hitmask using an image's alpha."""
    mask = []
    for x in range(image.get_width()):
        mask.append([])
        for y in range(image.get_height()):
            mask[x].append(bool(image.get_at((x,y))[3]))
    return mask

def getSpecialHitmask(pipe_image):
    mask = []
    for x in range(pipe_image.get_width()):
        mask.append([])
        for y in range(2 * pipe_image.get_height()):
            mask[x].append(True)
    return mask