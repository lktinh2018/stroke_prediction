import torch

INPUT_CSV_PATH = "[Doctor Anywhere] Machine Learning Take Home Assignment\[Doctor Anywhere] dataset-stroke-train_to_candidate.csv"

GT_COL_NAME = "stroke"

TRAIN_RATIO = 0.7
TEST_RATIO = 0.3

BATCH_SIZE = 64
NUM_EPOCHS = 200

x_categ = torch.randint(0, 5, (1, 10))
