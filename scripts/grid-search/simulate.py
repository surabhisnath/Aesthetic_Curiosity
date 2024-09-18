import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

tqdm.pandas()
import sys

sys.path.insert(1, "../../measures/intricacy/")
from calculate_intricacy import *

sys.path.insert(1, "../../measures/local spatial complexity/")
from calculate_local_spatial_complexity import *
import warnings

warnings.filterwarnings("ignore")
np.random.seed(34)


def calculate_measures(grid):
    LSC = calculate_local_spatial_complexity(grid, grid_size)
    intricacy = calculate_intricacy(grid, grid_size)
    return LSC, intricacy


def simulate_underlying_random(grid):
    measures = np.zeros((num_sims, 2))
    gray_indices = grid == 2
    for i in range(num_sims):
        grid_ = grid.copy()
        grid_[gray_indices] = np.random.choice([0, 1], size=np.sum(gray_indices))
        measures[i, :] = calculate_measures(grid_)
    meanLSC, meanInt = np.mean(measures, axis=0)
    return meanLSC, meanInt


def simulate_underlying_permutation(grid, pattern_id):
    measures = np.zeros((num_patterns - 1, 2))
    gray_indices = grid == 2
    ind = 0
    for i in range(num_patterns):
        if i + 1 == pattern_id:
            continue
        grid_ = grid.copy()
        replacement_pattern = np.array(
            list(
                map(
                    int,
                    pattern_stats[pattern_stats["pattern_id"] == i + 1][
                        "stimuli"
                    ].item(),
                )
            )
        ).reshape(grid_size, grid_size)
        grid_[gray_indices] = replacement_pattern[gray_indices]
        measures[ind, :] = calculate_measures(grid_)
        ind += 1
    meanLSC, meanInt = np.mean(measures, axis=0)
    return meanLSC, meanInt


def simulate_visible_random(grid, LSC_nondetrended, Int_nondetrended):
    measures = np.zeros((num_sims, 2))
    non_gray_indices = grid != 2
    for i in range(num_sims):
        grid_ = grid.copy()
        grid_[non_gray_indices] = np.random.choice(
            [0, 1], size=np.sum(non_gray_indices)
        )
        measures[i, :] = calculate_measures(grid_)
    meanLSC, meanInt = np.mean(measures, axis=0)
    return LSC_nondetrended - meanLSC, Int_nondetrended - meanInt


def simulate_visible_permutation(grid, LSC_nondetrended, Int_nondetrended, pattern_id):
    measures = np.zeros((num_patterns - 1, 2))
    non_gray_indices = grid != 2
    ind = 0
    for i in range(num_patterns):
        if i + 1 == pattern_id:
            continue
        grid_ = grid.copy()
        replacement_pattern = np.array(
            list(
                map(
                    int,
                    pattern_stats[pattern_stats["pattern_id"] == i + 1][
                        "stimuli"
                    ].item(),
                )
            )
        ).reshape(grid_size, grid_size)
        grid_[non_gray_indices] = replacement_pattern[non_gray_indices]
        measures[ind, :] = calculate_measures(grid_)
        ind += 1
    meanLSC, meanInt = np.mean(measures, axis=0)
    return LSC_nondetrended - meanLSC, Int_nondetrended - meanInt


pattern_stats = pd.read_csv("../../csvs/grid-search/pattern_stats.csv")

grid_size = 27
# num_sims = 10
num_patterns = 98

click_data = pd.read_csv(
    "../../csvs/grid-search/click_data_reevaluatedforreproduction.csv"
)


# click_data[["uLSCp", "uIntp"]] = click_data.progress_apply(
#     lambda row: pd.Series(
#         simulate_underlying_permutation(
#             np.array(list(map(int, row["current_grid"]))).reshape(grid_size, grid_size),
#             row["pattern_id"],
#         ),
#     ),
#     axis=1,
# )

click_data[["cLSCp", "cIntp"]] = click_data.progress_apply(
    lambda row: pd.Series(
        simulate_visible_permutation(
            np.array(list(map(int, row["current_grid"]))).reshape(grid_size, grid_size),
            row["cLSC_nondetrended"],
            row["cInt_nondetrended"],
            row["pattern_id"],
        )
    ),
    axis=1,
)

click_data.to_csv(
    "../../csvs/grid-search/click_data_reevaluatedforreproduction.csv", index=False
)
