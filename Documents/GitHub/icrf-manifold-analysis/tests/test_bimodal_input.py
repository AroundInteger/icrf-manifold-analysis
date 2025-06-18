import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from src.analysis.ergodic_rate_analysis import ErgodicRateAnalysis

# 1. Generate ergodic, continuous DRAQ5 distribution by sampling I uniformly
n_cells = 10000
I = np.random.uniform(0, 1, n_cells)

# 2. Map I to DRAQ5 using a piecewise linear cell cycle model
# G1: I in [0, 0.4), S: I in [0.4, 0.7), G2/M: I in [0.7, 1.0]
draq5 = np.zeros_like(I)
phases = np.empty(n_cells, dtype=object)
# G1
mask_g1 = (I < 0.4)
draq5[mask_g1] = 2.0
phases[mask_g1] = "G1"
# S
mask_s = (I >= 0.4) & (I < 0.7)
draq5[mask_s] = 2.0 + (I[mask_s] - 0.4) / 0.3  # Linear from 2.0 to 3.0
phases[mask_s] = "S"
# G2/M
mask_g2m = (I >= 0.7)
draq5[mask_g2m] = 3.0
phases[mask_g2m] = "G2/M"

# 3. Generate simple GFP values (random, or correlated with DRAQ5)
gfp = np.random.normal(loc=1.0, scale=0.2, size=n_cells)

# 4. Run ERA analysis pipeline
analyzer = ErgodicRateAnalysis(gfp, draq5, phases)
f_I, I_range = analyzer._calculate_probability_density()
F_I = analyzer._calculate_cumulative_inverse_density()
I_values = analyzer.project_to_trajectory(gfp, draq5)

# 5. Plot DRAQ5 vs f(I)
analyzer.plot_draq5_vs_fI(draq5, I_values, f_I, I_range, phases)
# 6. Plot mapping from DRAQ5 to I
analyzer.plot_draq5_to_I_mapping(draq5, I_values)
# 7. Plot f(I) for different bandwidths
analyzer.plot_fI_bandwidths(I_values, I_range)

print("Ergodic input test complete. Check docs/images/ for output plots.") 