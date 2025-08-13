import os
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "tf_efficientnetv2_s"
DATASET_PATH = "dataset"
RESULT_PATH = "results"
CHECKPOINT_DIR = "checkpoints"
MODEL_DIR = "model"
MODEL_CLASSIFICATION = os.path.join(MODEL_DIR, "classification")
MODEL_DETECTION = os.path.join(MODEL_DIR, "detection")
TRAIN_PATH = os.path.join(DATASET_PATH, "train")
VAL_PATH = os.path.join(DATASET_PATH, "val")


BATCH_SIZE = 32
IMG_SIZE = 224
EPOCH_NUM = 15
LEARNING_RATE = .001

CONF_DETECT_THRESH = 0.5
SKIP_FRAME = 3
MARGIN= 20