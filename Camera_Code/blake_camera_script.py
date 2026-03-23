#!/usr/bin/env python3
import time
import sys
print(sys.executable)
from datetime import datetime
from pathlib import Path
from picamera2 import Picamera2
import cv2
#import matplotlib.pyplot as plt
#import rawpy
#print("waiting 10 seconds")
#time.sleep(10)

def get_next_file_number(OUT_DIR):
    # Find the next available DNG number
    existing_files = list(OUT_DIR.glob("*.dng"))
    if existing_files:
        # Extract numeric parts from filenames and find the largest number
        nums = []
        for f in existing_files:
            try:
                nums.append(int(f.stem))
            except ValueError:
                pass  # skip files that don't have numeric names
        next_num = max(nums) + 1 if nums else 1
    else:
        next_num = 1
    return next_num


# Choose folder to save images
OUT_DIR = Path(__file__).resolve().parent / "Pictures" / "Uncoated_Bot_Ring" / "Transmission_Effeciency"
OUT_DIR.mkdir(parents=True, exist_ok=True)
# Generate filename with timestamp
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#fname=f"image_{stamp}.jpg"
#fname="10microW_Exp_100000_8mm1.dng"

#fnumber=get_next_file_number(OUT_DIR)
#print("fnumber:",fnumber)
#fname=f"{fnumber}.dng"
fname="Ring_NoLens_43muW_1sec.dng"
out_path = OUT_DIR / fname 

# Camera setup
picam2 = Picamera2()
#con2ig = picam2.create_still_configuration(main={"size": (1920, 1080)})
config = picam2.create_still_configuration(raw={"size": picam2.sensor_resolution})
picam2.configure(config)

# Optional camera controls (micro-seconds)
picam2.set_controls({
    "AeEnable": False,
    "AwbEnable": False,
    "ExposureTime": 1_000_000,   # (microseconds)
    "AnalogueGain": 1.0          # base ISO
})

picam2.start()
time.sleep(0.5)  # Let the camera adjust

# Capture one photo
picam2.capture_file(str(out_path),name="raw")

md = picam2.capture_metadata()
print("ExposureTime (us):", md.get("ExposureTime"))

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
