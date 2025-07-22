import os
import csv
from utils.paths import ADDITIONAL_DIR, BASE_DIR



os.makedirs(ADDITIONAL_DIR, exist_ok=True)
os.makedirs(BASE_DIR / "tmp/", exist_ok=True)
os.makedirs(BASE_DIR / "tmp/models checkpoint/", exist_ok=True)
os.makedirs(BASE_DIR / "tmp/best models/", exist_ok=True)

myRatingsPath = ADDITIONAL_DIR / "myRatings.csv"
if not os.path.exists(myRatingsPath):
    with open(myRatingsPath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["userId", "movieId", "rating"])
