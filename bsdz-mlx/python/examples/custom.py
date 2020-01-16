import os
import signal
import subprocess
import time
from datetime import datetime
from time import sleep

# import cv2
import matplotlib as mpl
import numpy as np
import pygame
from sh import gphoto2 as gp
from PIL import Image
from matplotlib import cm

from MLX90640 import API, ffi, hertz_to_refresh_rate, temperature_data_to_ndarray
from config import *


def td_to_image(f, cmap):
    norm = mpl.colors.Normalize(vmin=f.min(), vmax=f.max())
    img = Image.fromarray(np.uint8(cmap(norm(f)) * 255))
    img = img.convert("RGB").resize((320, 240), Image.BICUBIC)
    return img


# noinspection PyUnresolvedReferences
def increment_refresh_rate():
    rr = API.GetRefreshRate(MLX_I2C_ADDR)
    new_rr = (rr + 1) % 8
    print(f"Set new refresh rate to {new_rr}")
    API.SetRefreshRate(MLX_I2C_ADDR, new_rr)


def show_text(display, text, pos, font, action=None):
    surf = font.render(text, False, (255, 255, 255))
    if action and pygame.mouse.get_pressed()[0] and surf.get_rect().move(pos).collidepoint(pygame.mouse.get_pos()):
        action()
    display.blit(surf, pos)


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
    except:
        print("Failed to create new directory.")
    os.chdir(save_location)


def captureImages():
    gp(triggerCommand)
    sleep(5)
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


class ThermalFeed:
    def __init__(self):
        # setup colour map
        self.cmap = cm.get_cmap('Spectral_r')

        # set up display
        pygame.init()
        pygame.font.init()
        self.display = pygame.display.set_mode((320, 240))
        pygame.display.set_caption('Thermal Cam')
        pygame.mouse.set_visible(True)
        font = pygame.font.SysFont('freemono', 10)

        # mlx90640 settings
        self.MLX_I2C_ADDR = 0x33
        self.hertz_default = 8
        API.SetRefreshRate(MLX_I2C_ADDR, hertz_to_refresh_rate[self.hertz_default])
        API.SetChessMode(MLX_I2C_ADDR)

        # Extract calibration data from EEPROM and store in RAM
        self.eeprom_data = ffi.new("uint16_t[832]")
        self.params = ffi.new("paramsMLX90640*")
        API.DumpEE(MLX_I2C_ADDR, self.eeprom_data)
        API.ExtractParameters(self.eeprom_data, self.params)

        self.TA_SHIFT = 8  # the default shift for a MLX90640 device in open air
        self.emissivity = 0.95
        self.frame_buffer = ffi.new("uint16_t[834]")
        self.image_buffer = ffi.new("float[768]")
        self.last = time.monotonic()
        self.now = time.monotonic()
        self.diff = self.now - self.last

    def thermal_update(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        API.GetFrameData(MLX_I2C_ADDR, self.frame_buffer)
        self.now = time.monotonic()
        self.diff = self.now - self.last
        self.last = self.now
        tr = API.GetTa(self.frame_buffer, self.params) - self.TA_SHIFT
        API.CalculateTo(self.frame_buffer, self.params, self.emissivity, tr, self.image_buffer)
        ta_np = temperature_data_to_ndarray(self.image_buffer)
        ta_img = td_to_image(ta_np, self.cmap)
        pyg_img = pygame.image.fromstring(ta_img.tobytes(), ta_img.size, ta_img.mode)
        self.display.blit(pyg_img, (0, 0))
        pygame.display.update()


def main():
    thermalcam = ThermalFeed()
    # cv2.namedWindow("Shutter Button (_)")
    while True:
        thermalcam.thermal_update()

        # k = cv2.waitKey(10)
        k = pygame.event.get()
        # if k == 32:
        if k == 2:
            print("Taking Pics!")
            shot_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            gp(raw_and_large)
            gp(short_exp)
            captureImages()
            renameFiles(picID, shot_time + "small_short")
            gp(long_exp)
            captureImages()
            renameFiles(picID, shot_time + "small_long")

            gp(small)
            gp(short_exp)
            captureImages()
            renameFiles(picID, shot_time + "raw_short")
            gp(long_exp)
            captureImages()
            renameFiles(picID, shot_time + "raw_long")

            pygame.image.save(thermalcam.display, "thermal" + '.JPG')
            renameFiles(picID, shot_time + "thermal")
            break


if __name__ == '__main__':
    picID = "Cannon800DShots"
    shot_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    folder_name = shot_time + picID
    save_location = "./Pics/" + folder_name

    createSaveFolder(save_location)

    killGphoto2Process()
    gp(clearCommand)
    main()
