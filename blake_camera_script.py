#!/usr/bin/env python3
import time
import sys
print(sys.executable)
from datetime import datetime
from pathlib import Path
from picamera2 import Picamera2
import cv2
import matplotlib.pyplot as plt
import rawpy

quit()
#print("waiting 10 seconds")
#time.sleep(10)



# Choose folder to save images
OUT_DIR = Path(__file__).resolve().parent / "Pictures" / "DistributionShift" / "Bayer_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)
# Generate filename with timestamp
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#fname=f"image_{stamp}.jpg"
#fname="10microW_Exp_100000_8mm1.dng"
fname="test.dng"
out_path = OUT_DIR / fname 

# Camera setup
picam2 = Picamera2()
#config = picam2.create_still_configuration(main={"size": (1920, 1080)})
config = picam2.create_still_configuration(raw={"size": picam2.sensor_resolution})
picam2.configure(config)

# Optional camera controls (micro-seconds)
picam2.set_controls({
    "AeEnable": False,
    "AwbEnable": False,
    "ExposureTime": 1_000_000,   # 1.0 s (microseconds)
    "AnalogueGain": 1.0          # base ISO
})

picam2.start()
time.sleep(0.5)  # Let the camera adjust

# Capture one photo
picam2.capture_file(str(out_path),name="raw")
picam2.stop()

print(f"Saved image: {out_path}")
quit()
#plot image just taken
image = cv2.imread(str(out_path))
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image_rgb)
plt.title(f"Captured Image ({fname})")
#plt.axis("off")
plt.show()

