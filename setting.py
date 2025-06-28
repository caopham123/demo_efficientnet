import os
MODEL_NAME = "tf_efficientnetv2_s"
DATASET_PATH = "dataset"
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