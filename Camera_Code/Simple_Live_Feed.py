#!/usr/bin/env python3
import time
import cv2
from picamera2 import Picamera2, Preview
import numpy as np
#quit()
# Start a live preview window on the Pi’s desktop
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (1280, 720)})
picam2.configure(config)

# QTGL is hardware-accelerated on Raspberry Pi OS with desktop
picam2.start_preview(Preview.QTGL)

# Optional camera tweaks
picam2.set_controls({"ExposureTime": 10000000, "Saturation": 1.0})

picam2.start()
print("Live preview running. Press Ctrl+C in the Geany terminal to stop.")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass
finally:
    picam2.stop()

