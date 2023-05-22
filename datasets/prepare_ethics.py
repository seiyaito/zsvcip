import os
import tarfile
from urllib.request import urlretrieve

URL = "https://people.eecs.berkeley.edu/~hendrycks/ethics.tar"
OUTPUT_DIR = os.path.abspath(os.path.dirname(__file__))
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASET_PATH = os.path.join(OUTPUT_DIR, "ethics.tar")
urlretrieve(URL, DATASET_PATH)

with tarfile.open(DATASET_PATH, "r") as tar:
    tar.extractall(OUTPUT_DIR)

os.remove(DATASET_PATH)
