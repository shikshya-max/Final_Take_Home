import numpy as np
from scipy.ndimage import label
from collections import defaultdict
import json
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

COLOR_MAP = {
    0: "white", 1: "lightblue", 2: "orangered", 3: "peachpuff", 4: "lightgreen",
    5: "gray", 6: "lavender", 7: "darkgray", 8: "lightgray"
}

def extract_blocks(grid):
    blocks = []
    for val in np.unique(grid):
        if val == 0:
            continue
        mask = (grid == val).astype(int)
        labeled, num = label(mask)
        for i in range(1, num + 1):
            coords = np.argwhere(labeled == i)
            block = {
                'value': val,
                'coords': coords,
                'min_col': np.min(coords[:, 1]),
                'min_row': np.min(coords[:, 0]),
                'height': np.max(coords[:, 0]) - np.min(coords[:, 0]) + 1,
                'width': np.max(coords[:, 1]) - np.min(coords[:, 1]) + 1,
            }
            blocks.append(block)
    return blocks

def match_blocks_by_value(input_blocks, output_blocks):
    match = {}
    for ib in input_blocks:
        for ob in output_blocks:
            if ib['value'] == ob['value'] and ib['value'] not in match:
                match[ib['value']] = (ib, ob)
    return match

def infer_abstract_placement_rule(training_pairs):
    correct_origin = 0
    pattern_counts = {"diagonal": 0, "horizontal": 0, "vertical": 0}
    total_pairs = 0

    for input_grid, output_grid in training_pairs:
        input_blocks = extract_blocks(input_grid)
        output_blocks = extract_blocks(output_grid)
        match = match_blocks_by_value(input_blocks, output_blocks)

        input_sorted = sorted(match.values(), key=lambda p: p[0]['min_col'])
        output_sorted = [p[1] for p in input_sorted]

        if output_sorted[0]['min_row'] == 0 and output_sorted[0]['min_col'] == 0:
            correct_origin += 1

        for i in range(1, len(output_sorted)):
            prev = output_sorted[i-1]
            curr = output_sorted[i]
            dr = curr['min_row'] - prev['min_row']
            dc = curr['min_col'] - prev['min_col']
            if dr == prev['height']-1 and dc == prev['width']-1:
                pattern_counts["diagonal"] += 1
            elif dr == 0 and dc == prev['width']:
                pattern_counts["horizontal"] += 1
            elif dr == prev['height'] and dc == 0:
                pattern_counts["vertical"] += 1
            total_pairs += 1

    origin_rule = correct_origin == len(training_pairs)
    best_pattern = max(pattern_counts, key=pattern_counts.get)
    return origin_rule, best_pattern

def apply_abstract_stacking_rule(input_grid, use_origin, pattern):
    blocks = extract_blocks(input_grid)
    blocks_sorted = sorted(blocks, key=lambda b: b['min_col'])
    new_grid = np.zeros_like(input_grid)

    r_offset, c_offset = (0, 0) if use_origin else (2, 2)

    for block in blocks_sorted:
        for (r, c) in block['coords']:
            new_r = r - block['min_row'] + r_offset
            new_c = c - block['min_col'] + c_offset
            if 0 <= new_r < new_grid.shape[0] and 0 <= new_c < new_grid.shape[1]:
                new_grid[new_r, new_c] = block['value']

        if pattern == "diagonal":
            r_offset += block['height']-1
            c_offset += block['width']-1
        elif pattern == "horizontal":
            c_offset += block['width']
        elif pattern == "vertical":
            r_offset += block['height']

    return new_grid

def plot_grid(grid, title, filename):
    fig, ax = plt.subplots()
    cmap = ListedColormap([COLOR_MAP.get(i, "black") for i in range(9)])
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=8)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, str(grid[i, j]), ha='center', va='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    with open("Problem_2/dataset.json", "r") as f:
        dataset = json.load(f)

    training_pairs = [(np.array(pair['input']), np.array(pair['output'])) for pair in dataset['train']]
    test_inputs = [np.array(pair['input']) for pair in dataset['test']]

    origin_rule, layout_pattern = infer_abstract_placement_rule(training_pairs)
    print("Learned placement rules:")
    print("- First block at origin:", origin_rule)
    print("- Stacking pattern:", layout_pattern)

    for idx, test_input in enumerate(test_inputs):
        result = apply_abstract_stacking_rule(test_input, origin_rule, layout_pattern)
        print(f"\nTest Output {idx}:")
        print(result)
        plot_grid(test_input, f"Test Input {idx}", f"test_input_{idx}.png")
        plot_grid(result, f"Predicted Output {idx}", f"predicted_output_{idx}.png")

if __name__ == "__main__":
    main()
