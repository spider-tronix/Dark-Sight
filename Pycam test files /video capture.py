# https://www.electronicwings.com/raspberry-pi/pi-camera-module-interface-with-raspberry-pi-using-python
import picamera
from time import sleep

camera = picamera.PiCamera()
camera.resolution = (640, 480)

print()
#start recording using pi camera
camera.start_recording("/home/pi/demo.h264")
#wait for video to record
camera.wait_recording(20)
#stop recording
camera.stop_recording()
camera.close()
print("video recording stopped")