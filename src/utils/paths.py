from pathlib import Path



BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "../data/"
MOVIE_LENS_DIR = DATA_DIR / "ml-latest-small/"
ADDITIONAL_DIR = DATA_DIR / "additional_data/"
MODELS_DIR = BASE_DIR / "models/pretrained/"
