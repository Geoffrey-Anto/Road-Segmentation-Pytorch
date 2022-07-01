import albumentations as A
import torch

EPOCHS = 100
BATCH_SIZE = 1
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_MODEL = True
SAVE_PATH = "Saved_Models/Model"
