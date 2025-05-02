import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# -----------------------------
# Color and Plot Utilities
# -----------------------------
COLOR_MAP = {
    0: "white", 1: "black", 2: "orange", 3: "peachpuff", 4: "limegreen", 5: "purple"
}

def plot_input_output_side_by_side(input_grid, output_grid, title_left, title_right, filename):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    cmap = ListedColormap([COLOR_MAP[i] for i in range(6)])
    vmin, vmax = 0, 5

    for ax, grid, title in zip(axs, [input_grid, output_grid], [title_left, title_right]):
        ax.set_title(title)
        ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(grid.shape[1]))
        ax.set_yticks(np.arange(grid.shape[0]))
        ax.set_xticklabels([]), ax.set_yticklabels([])
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                ax.text(j, i, str(grid[i, j]), ha="center", va="center", fontsize=10, color="black")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# -----------------------------
# Utility Functions
# -----------------------------
def get_object_positions(grid, objects=(2, 3, 4)):
    positions = {}
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            val = grid[i][j]
            if val in objects:
                positions[val] = (i, j)
    return positions

def compute_actions(a_pos, b_pos):
    actions = []
    if a_pos[1] < b_pos[1]:
        actions.append(('horizontal', 'right'))
    elif a_pos[1] > b_pos[1]:
        actions.append(('horizontal', 'left'))

    if a_pos[0] < b_pos[0]:
        actions.append(('vertical', 'down'))
    elif a_pos[0] > b_pos[0]:
        actions.append(('vertical', 'up'))
    return actions

def apply_actions(grid, start_pos, end_pos, axis_order, object_positions):
    direction_map = {
        'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)
    }
    modified = grid.copy()
    current = list(start_pos)

    for axis in axis_order:
        if axis == 'horizontal':
            while current[1] != end_pos[1]:
                step = 'right' if current[1] < end_pos[1] else 'left'
                dx, dy = direction_map[step]
                next_x, next_y = current[0] + dx, current[1] + dy
                if (next_x, next_y) not in object_positions.values():
                    modified[next_x][next_y] = 5
                current = [next_x, next_y]
        elif axis == 'vertical':
            while current[0] != end_pos[0]:
                step = 'down' if current[0] < end_pos[0] else 'up'
                dx, dy = direction_map[step]
                next_x, next_y = current[0] + dx, current[1] + dy
                if (next_x, next_y) not in object_positions.values():
                    modified[next_x][next_y] = 5
                current = [next_x, next_y]

    return modified

# -----------------------------
# Main Logic
# -----------------------------
def main():
    with open("Problem_1/structured_symbolic_dataset.json", "r") as f:
        dataset = json.load(f)

    all_actions = []
    for idx, example in enumerate(dataset["train"]):
        train_input = np.array(example["input"])
        train_output = np.array(example["output"])
        positions = get_object_positions(train_input)

        if 2 in positions and 4 in positions:
            actions_2_to_4 = compute_actions(positions[2], positions[4])
            all_actions.append((2, 4, actions_2_to_4))

        if 4 in positions and 3 in positions:
            actions_4_to_3 = compute_actions(positions[4], positions[3])
            all_actions.append((4, 3, actions_4_to_3))

        print(f"[Train {idx}] Inferred actions:")
        for src, tgt, acts in all_actions[-2:]:
            print(f"  {src} → {tgt} via {[act[1] for act in acts]}")

        plot_input_output_side_by_side(train_input, train_output,
                                       f"Train Input {idx}", f"Train Output {idx}",
                                       f"train_io_{idx}.png")

    # Fix action axis order from first training example
    axis_order_2_to_4 = [act[0] for act in all_actions[0][2]]
    axis_order_4_to_3 = [act[0] for act in all_actions[1][2]]

    for idx, example in enumerate(dataset["test"]):
        test_input = np.array(example["input"])
        positions = get_object_positions(test_input)
        modified = test_input.copy()

        if 2 in positions and 4 in positions:
            modified = apply_actions(modified, positions[2], positions[4], axis_order_2_to_4, positions)
        if 4 in positions and 3 in positions:
            modified = apply_actions(modified, positions[4], positions[3], axis_order_4_to_3, positions)

        plot_input_output_side_by_side(test_input, modified,
                                       f"Test Input {idx}", f"Predicted Output {idx}",
                                       f"test_io_{idx}.png")
        print(f"[Test {idx}] Path generated and plot saved.")

if __name__ == "__main__":
    main()
