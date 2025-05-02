import numpy as np
from scipy.ndimage import label
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import json

COLOR_MAP = {
    0: "white", 1: "lightblue", 2: "orangered", 3: "peachpuff", 4: "lightgreen",
    5: "gray", 6: "lavender", 7: "darkgray", 8: "lightgray"
}

def extract_tiles(grid, min_size=(4, 4)):
    tiles = []
    mask = (grid != 0).astype(int)
    labeled, num = label(mask)
    for i in range(1, num + 1):
        coords = np.argwhere(labeled == i)
        min_row, min_col = coords.min(axis=0)
        max_row, max_col = coords.max(axis=0)
        tile = grid[min_row:max_row+1, min_col:max_col+1]
        if tile.shape[0] >= min_size[0] and tile.shape[1] >= min_size[1]:
            tiles.append((tile, (min_row, min_col)))
    return tiles

def hash_tile(tile):
    return tuple(tile.flatten())

def find_template(tiles):
    hashes = [hash_tile(tile) for tile, _ in tiles]
    most_common_hash = Counter(hashes).most_common(1)[0][0]
    for tile, _ in tiles:
        if hash_tile(tile) == most_common_hash:
            return tile
    return None

def tile_similarity(tile1, tile2):
    if tile1.shape != tile2.shape:
        return 0
    return np.mean(tile1 == tile2)

def reconstruct_from_template(grid, tiles, template, threshold=0.85):
    cleaned = np.zeros_like(grid)
    for tile, (r0, c0) in tiles:
        if tile.shape == template.shape and tile_similarity(tile, template) >= threshold:
            cleaned[r0:r0+tile.shape[0], c0:c0+tile.shape[1]] = template
    return cleaned

def symbolic_denoising(grid):
    tiles = extract_tiles(grid)
    template = find_template(tiles)
    if template is None:
        return grid
    return reconstruct_from_template(grid, tiles, template)

def plot_grids(noisy, denoised):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    cmap = ListedColormap([COLOR_MAP[i] for i in range(9)])
    for ax, data, title in zip(axs, [noisy, denoised], ["Noisy Input", "Denoised Output"]):
        ax.imshow(data, cmap=cmap, vmin=0, vmax=8)
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, str(data[i, j]), ha='center', va='center', fontsize=8)
    plt.tight_layout()
    plt.savefig("symbolic_denoised_output.png")
    plt.show()

# Example usage
if __name__ == "__main__":
    with open("Problem_3/denoising_manual.json") as f:
        dataset = json.load(f)
    noisy_input = dataset["train"][0]["input"]
    noisy_input = np.array(noisy_input)  # Make sure it's a NumPy array
    denoised = symbolic_denoising(noisy_input)
    plot_grids(noisy_input, denoised)

