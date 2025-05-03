import os
import csv
from utils.paths import ADDITIONAL_DIR



os.makedirs(ADDITIONAL_DIR, exist_ok=True)
os.makedirs(ADDITIONAL_DIR+"models/", exist_ok=True)

myRatingsPath = ADDITIONAL_DIR + "myRatings.csv"
if not os.path.exists(myRatingsPath):
    with open(myRatingsPath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["userId", "movieId", "rating"])
