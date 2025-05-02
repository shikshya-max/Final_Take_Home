import numpy as np
import matplotlib.pyplot as plt
import json
import random
from scipy.ndimage import rotate, shift
import copy

# Load original dataset
with open("Problem_3/denoising_manual.json") as f:
    dataset = json.load(f)

augmented_train = []

def augment_grid(grid, label):
    augmented = []
    grid_np = np.array(grid)
    label_np = np.array(label)

    # Rotations
    for angle in [90, 180, 270]:
        aug_grid = rotate(grid_np, angle, reshape=False, order=0, mode='constant', cval=0)
        aug_label = rotate(label_np, angle, reshape=False, order=0, mode='constant', cval=0)
        augmented.append({"input": aug_grid.tolist(), "output": aug_label.tolist()})

    # Shifts
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        aug_grid = shift(grid_np, shift=[dx, dy], order=0, mode='constant', cval=0)
        aug_label = shift(label_np, shift=[dx, dy], order=0, mode='constant', cval=0)
        augmented.append({"input": aug_grid.tolist(), "output": aug_label.tolist()})

    return augmented

# Apply augmentation for each pair
for pair in dataset["train"]:
    original_input = pair["input"]
    original_output = pair["output"]
    augmented_train.append(pair)  # include original
    augmented_train.extend(augment_grid(original_input, original_output))

# Save new dataset
augmented_dataset = {"train": augmented_train}
with open("Problem_3/denoising_manual.json", "w") as f:
    json.dump(augmented_dataset, f)

len(augmented_train)
