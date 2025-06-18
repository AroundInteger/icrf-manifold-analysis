# ICRF-193 Cell Cycle Manifold Analysis

> Dynamic manifold learning for cell cycle perturbation studies using flow cytometry data

## Overview

This repository implements a novel manifold learning approach to analyze cell cycle dynamics and drug perturbations in flow cytometry data. The framework was developed to study ICRF-193 (topoisomerase II inhibitor) effects on cell cycle progression, with a focus on polyploidization dynamics.

### Key Innovation: "Polyploidy Garage" Model

We conceptualize the cell cycle as a **multistory polyploidy garage** where:
- **Ground floor (2N)**: Normal diploid cell cycle
- **Floor 1 (4N)**: First polyploid level  
- **Floor 2 (8N)**: Second polyploid level
- **Higher floors**: Progressive polyploidization

ICRF-193 acts like a malfunctioning elevator system, trapping cells on higher floors and preventing normal cell division.

## Features

- **Dynamic manifold learning** using UMAP, PCA, and diffusion maps
- **Automatic ploidy detection** via peak detection in DNA content distributions
- **Population flow analysis** tracking cell transitions between ploidy levels
- **Recovery dynamics** quantification for drug perturbation studies
- **Polyploidy garage visualization** with floor-specific analysis
- **Integration with classical ergodic reconstruction** methods

## Installation

```bash
git clone https://github.com/yourusername/icrf-manifold-analysis.git
cd icrf-manifold-analysis
pip install -r requirements.txt
```

Or install in development mode:
```bash
pip install -e .
```

## Quick Start

### Basic Analysis Pipeline

```python
from src.manifold_learning import CellCycleManifoldAnalyzer
from src.data_processing import load_flow_data_from_mat

# Load your flow cytometry data
data_0hr = load_flow_data_from_mat('data/raw/0hr_control.mat')
data_24hr = load_flow_data_from_mat('data/raw/24hr_icrf.mat')
data_42hr = load_flow_data_from_mat('data/raw/42hr_recovery.mat')

# Initialize analyzer
analyzer = CellCycleManifoldAnalyzer()

# Load and segment populations
analyzer.load_flow_data(data_0hr['gfp'], data_0hr['draq5'], '0hr')
analyzer.segment_populations_garage('0hr', auto_threshold=True)

# Learn manifold structure
analyzer.learn_manifold('0hr', method='umap')

# Visualize polyploidy garage
analyzer.visualize_polyploidy_garage('0hr')
```

### Complete ICRF-193 Experiment Analysis

```python
from src.manifold_learning import analyze_icrf_experiment
from src.transition_dynamics import analyze_icrf_transitions

# Run complete analysis pipeline
analyzer, comparison = analyze_icrf_experiment(data_0hr, data_24hr, data_42hr)

# Analyze transition dynamics
transition_analyzer, evolution, recovery = analyze_icrf_transitions(analyzer)
```

## Theoretical Background

### Ergodic Reconstruction Framework

This work builds upon classical ergodic principles for cell cycle analysis:

1. **Spatial-temporal equivalence**: Population distribution at time t₀ represents temporal trajectory
2. **Newton-Raphson optimization**: Assigns cell cycle timing via polynomial trajectory fitting
3. **Manifold extension**: Discovers nonlinear cell cycle manifolds beyond polynomial assumptions

### Mathematical Framework

The manifold learning approach discovers intrinsic coordinates `φ: ℝ² → ℝᵈ` where:
- Input space: (GFP-Cyclin B1, DRAQ5) measurements
- Manifold space: Intrinsic cell cycle coordinates
- Polyploidy levels: Disconnected manifold components

Key metrics:
- **Bifurcation index**: `BI = N₄ₙ / (N₂ₙ + N₄ₙ)`
- **Manifold distortion**: `MD = d(M_perturbed, M_baseline)`
- **Recovery rate**: `RR = (BI₄₂ₕᵣ - BI₂₄ₕᵣ) / (BI₀ₕᵣ - BI₂₄ₕᵣ)`

## Data Requirements

### Flow Cytometry Data Format

The analysis expects dual-parameter flow cytometry data:

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| GFP-Cyclin B1 | Cell cycle reporter | 0-1000 AU |
| DRAQ5 | DNA content (ploidy) | 0-800 AU |

### Supported File Formats

- **MATLAB files** (.mat): `load_flow_data_from_mat()`
- **CSV files**: `load_flow_data_from_csv()`
- **FCS files**: `load_flow_data_from_fcs()` (coming soon)

### Example Data Structure

```matlab
% MATLAB data structure
data.gfp = [234.5, 456.7, 123.4, ...];     % GFP-Cyclin B1 values
data.draq5 = [189.2, 378.9, 156.7, ...];   % DRAQ5 DNA content
data.time = 0;                              % Hours post-treatment
```

## Algorithm Details

### Manifold Learning Methods

1. **UMAP (Uniform Manifold Approximation and Projection)**
   - Preserves local and global manifold structure
   - Optimized for cell cycle topology
   - Parameters: `n_neighbors=50, min_dist=0.1`

2. **PCA (Principal Component Analysis)**
   - Linear dimensionality reduction
   - Baseline comparison method

3. **Diffusion Maps** (future implementation)
   - Captures dynamics on manifolds
   - Natural for time-series analysis

### Population Segmentation

**Automatic ploidy detection**:
1. Gaussian KDE smoothing of DRAQ5 distribution
2. Peak detection to identify ploidy levels
3. Threshold computation between peaks
4. Assignment to "garage floors"

**Manual segmentation**:
- User-defined DRAQ5 thresholds
- Flexible for different cell lines/conditions

### Transition Analysis

**Population flow matrices**:
- Nearest neighbor matching between timepoints
- Quantifies cell movement between ploidy levels
- Visualizes as transition probability matrices

**Recovery dynamics**:
- Manifold distance metrics
- Population fraction recovery
- Composite pharmacodynamic indices

## Results and Validation

### Comparison with Classical Methods

| Method | Advantages | Limitations |
|--------|------------|-------------|
| Newton-Raphson (Original) | Theoretically grounded, fast | Assumes polynomial trajectories |
| Manifold Learning (New) | Discovers nonlinear structure | More complex, requires tuning |
| Combined Approach | Best of both worlds | Under development |

### Key Findings

1. **ICRF-193 creates manifold bifurcation** at 24hr timepoint
2. **Polyploidy garage structure** clearly revealed in manifold space
3. **Incomplete recovery** at 42hr timepoint
4. **Population flow quantification** enables drug mechanism insights

## File Organization

### Source Code Structure

```
src/
├── manifold_learning/
│   ├── cell_cycle_analyzer.py    # Core manifold learning classes
│   └── transition_dynamics.py    # Temporal dynamics analysis
├── data_processing/
│   ├── flow_data_loader.py       # File I/O and format conversion
│   └── preprocessing.py          # Data cleaning and gating
└── visualization/
    └── polyploidy_garage.py      # Specialized plotting functions
```

### MATLAB Integration

```
matlab_code/
├── original_newton_raphson/      # Classical ergodic reconstruction
├── data_conversion/              # FCS to MAT conversion tools
└── validation/                   # Cross-method comparison
```

## Examples and Tutorials

### Jupyter Notebooks

1. **Data Exploration** (`notebooks/01_data_exploration.ipynb`)
   - Load and visualize flow cytometry data
   - Basic population gating
   - Quality control metrics

2. **Manifold Learning Demo** (`notebooks/02_manifold_learning_demo.ipynb`)
   - Compare UMAP, PCA, diffusion maps
   - Parameter sensitivity analysis
   - Visualization gallery

3. **ICRF Analysis** (`notebooks/03_icrf_analysis.ipynb`)
   - Complete drug perturbation study
   - Recovery dynamics quantification
   - Biological interpretation

### Command Line Interface

```bash
# Analyze single timepoint
python -m src.manifold_learning analyze --input data/raw/0hr_control.mat --output results/

# Compare multiple timepoints
python -m src.transition_dynamics compare --baseline 0hr --treated 24hr --recovery 42hr

# Generate report
python -m src.visualization report --experiment icrf_193 --format html
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
git clone https://github.com/yourusername/icrf-manifold-analysis.git
cd icrf-manifold-analysis
pip install -e ".[dev]"
pre-commit install
```

### Running Tests

```bash
pytest tests/
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{icrf_manifold_analysis,
  title={Dynamic Manifold Learning for Cell Cycle Drug Perturbation Analysis},
  author={[Your Name] and [Collaborators]},
  year={2025},
  url={https://github.com/yourusername/icrf-manifold-analysis},
  note={Polyploidy garage model for ICRF-193 studies}
}
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Original ergodic reconstruction framework from [Original Publication]
- UMAP algorithm by McInnes et al.
- Flow cytometry data processing tools
- [Funding sources and collaborators]

## Contact

- **Primary Author**: [Your Name] - [email]
- **Project Link**: https://github.com/yourusername/icrf-manifold-analysis
- **Issues**: Please use GitHub Issues for bug reports and feature requests

---

## Conversation Log

This project emerged from an extensive technical discussion about extending classical ergodic cell cycle reconstruction methods with modern manifold learning. See [docs/conversation_log.md](docs/conversation_log.md) for the complete development history and technical rationale.

## Updates: Ergodic Input Test & Improved ERA Analysis (June 2024)

### Summary of Improvements

- **Ergodic Input Test:**
  - Added `tests/test_bimodal_input.py` to generate a synthetic, ergodic, continuous cell cycle population.
  - The script samples cell cycle position `I` uniformly, maps `I` to DRAQ5 using a piecewise linear model (G1, S, G2/M), and assigns phases accordingly.
  - This approach ensures a true continuum of cells across the DRAQ5 signal, reflecting the ergodic nature of the system.

- **ERA Analysis Pipeline:**
  - Updated to use a circular (von Mises) kernel density estimator (KDE) for cell cycle position `I`, ensuring periodicity and smoothness at the cycle boundary.
  - Diagnostic plots compare the DRAQ5 histogram to the ERA-derived f(I), show the mapping from DRAQ5 to I, and visualize the effect of KDE bandwidth.

### How to Run the Ergodic Input Test

1. From the project root, run:
   ```bash
   python tests/test_bimodal_input.py
   ```
2. Output plots will be saved in `docs/images/`:
   - `draq5_vs_fI.png`: DRAQ5 histogram vs. f(I)
   - `draq5_to_I_mapping.png`: Mapping from DRAQ5 to I
   - `fI_bandwidths.png`: KDE bandwidth effects

### Interpreting the Results
- The DRAQ5 histogram should show a continuous distribution with plateaus at G1 and G2/M and a ramp through S-phase.
- The f(I) plot should be smooth and periodic, reflecting the ergodic, continuous nature of the model.
- The mapping plot should show a continuous, gap-free relationship between DRAQ5 and cell cycle position I.

### Rationale
- These improvements ensure that the ERA analysis is robust, artifact-free, and faithful to the biology of an ergodic cell cycle system.
- The new test and diagnostics provide a clear validation of the pipeline's ability to handle both synthetic and real data.

---