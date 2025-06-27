import os
MODEL_NAME = "efficientnetv2_rw_s"
DATASET_PATH = "dataset"
CHECKPOINT_DIR = "checkpoints"
TRAIN_PATH = os.path.join(DATASET_PATH, "train")
VAL_PATH = os.path.join(DATASET_PATH, "val")

BATCH_SIZE = 32
IMG_SIZE = 224
EPOCH_NUM = 15
LEARNING_RATE = .001