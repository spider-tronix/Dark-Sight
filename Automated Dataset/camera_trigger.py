import os
import signal
import subprocess
from datetime import datetime
from time import sleep

import pygame
from pygame.locals import *
from sh import gphoto2 as gp

from config import *


def killGphoto2Process():
    p = subprocess.Popen(['ps', '-A'], stdout=subprocess.PIPE)
    out, err = p.communicate()

    for line in out.splitlines():
        if b'gvfsd-gphoto2' in line:
            pid = int(line.split(None, 1)[0])
            os.kill(pid, signal.SIGKILL)


def createSaveFolder(save_location):
    try:
        os.makedirs(save_location)
        with open("/home/syzygianinfern0/sambashare/timestamp.txt", 'w') as handler:
            handler.write('/'.join(save_location.split('/')[4:6]) + '/')
            print("Timstamp is ready")
    except:
        print("Failed to create new directory.")
    os.chdir(save_location)


def captureImages():
    gp(triggerCommand)
    sleep(3)
    gp(downloadCommand)
    gp(clearCommand)


def renameFiles(ID, shot_time):
    for filename in os.listdir("."):
        if len(filename) < 13:
            if filename.endswith(".JPG"):
                os.rename(filename, (shot_time + ID + ".JPG"))
                print("Renamed the JPG")
            elif filename.endswith(".CR2"):
                os.rename(filename, (shot_time + ID + ".CR2"))
                print("Renamed the CR2")


def main():
    pygame.init()
    pygame.font.init()
    display = pygame.display.set_mode((320, 240))
    pygame.display.set_caption('Thermal Cam')
    pygame.mouse.set_visible(True)
    while True:
        for event in pygame.event.get():
            if event.type == KEYDOWN:
                print("Taking Pics!")
                shot_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                gp(raw_and_large)
                gp(short_exp)
                captureImages()
                renameFiles(picID, shot_time + "raw_short")
                gp(def_exp)
                captureImages()
                renameFiles(picID, shot_time + "raw_long")

                gp(small)
                gp(short_exp)
                captureImages()
                renameFiles(picID, shot_time + "small_short")
                gp(def_exp)
                captureImages()
                renameFiles(picID, shot_time + "small_long")
                exit()


if __name__ == '__main__':
    picID = "Cannon800DShots"
    shot_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    folder_name = shot_time + picID
    save_location = "/home/syzygianinfern0/sambashare/Dataset/" + folder_name

    createSaveFolder(save_location)

    try:
        killGphoto2Process()
    except:
        print("No Camera to Kill\n")
    gp(clearCommand)
    main()
