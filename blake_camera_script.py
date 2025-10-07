#!/usr/bin/env python3
import time
from datetime import datetime
from pathlib import Path
from picamera2 import Picamera2

#print("waiting 10 seconds")
#time.sleep(10)

# Choose folder to save images
OUT_DIR = Path(__file__).resolve().parent / "Pictures"
#OUT_DIR = Path.home() / "Pictures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
#print(OUT_DIR);quit()
# Generate filename with timestamp
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = OUT_DIR / f"image_{stamp}.jpg"

# Camera setup
picam2 = Picamera2()
config = picam2.create_still_configuration(main={"size": (1920, 1080)})
picam2.configure(config)

# Optional camera controls
picam2.set_controls({"ExposureTime": 10000, "Saturation": 1.0})

picam2.start()
time.sleep(0.5)  # Let the camera adjust

# Capture one photo
picam2.capture_file(str(out_path))
picam2.stop()

print(f"Saved image: {out_path}")
