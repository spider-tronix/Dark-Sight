import sys

import pygame
from pygame.locals import *


def wait():
    while True:
        for event in pygame.event.get():
            print(event)

            if event.type == KEYDOWN:
                pygame.quit()
                sys.exit()


def main():
    pygame.init()
    pygame.font.init()
    display = pygame.display.set_mode((320, 240))
    pygame.display.set_caption("Thermal Cam")
    pygame.mouse.set_visible(True)
    wait()


if __name__ == "__main__":
    main()
