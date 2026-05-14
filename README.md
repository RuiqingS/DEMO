# MatterGen-DEMO: Evolutionary Multi-Objective Crystal Design Plugin

<p align="center">
    <strong>An advanced evolutionary optimization framework for automated crystal structure discovery and multi-objective materials design</strong>
</p>

---

## Overview

**DEMO-MatterGen** is a comprehensive plugin built on top of [MatterGen](https://github.com/microsoft/mattergen) that enables automated, multi-objective evolutionary optimization for crystal structure design. It combines state-of-the-art generative models with evolutionary algorithms to discover novel materials with desired properties.

### Motivation

Existing generative models for materials design face three critical limitations:

1. **Inefficient Multi-Objective Optimization**: MatterGen and similar diffusion models can only perform "sample-then-filter" workflows. However, sampling Pareto-optimal materials is extremely difficult because Pareto front solutions lie at the **extreme edges** of the learned distribution, where diffusion models have low probability density.

2. **Limited Conditional Generation**: When targeting properties that were not included during model training (e.g., piezoelectricity, anisotropy, synthesizability scores), conditional generation becomes impossible, forcing researchers to generate massive amounts of candidates and post-filter.

3. **Evolutionary Algorithms Cannot Handle Structured Data**: Traditional evolutionary algorithms excel at multi-objective optimization but cannot be directly applied to molecular or crystalline structures due to the lack of meaningful crossover and mutation operators in discrete/geometric spaces.

**DEMO bridges this gap** by using the diffusion model itself as a structure-aware genetic operator (EGD), enabling evolutionary algorithms to natively operate on crystal structures while maintaining chemical and physical validity.

### Key Features

- 🧬 **Evolutionary Multi-Objective Optimization (EMO)**: Implements advanced EMO algorithms including SPEA2 (Constraint-Dominated Principle) and triple-population CMOEA (DEMO)
- 🎯 **Multi-Property Characterization**: Automated evaluation of 15+ material properties including formation energy, band gap, synthesizability, and more
- 🔄 **Intelligent Crossover & Mutation**: Leverages MatterGen's diffusion model for structure-aware genetic operations
- 📊 **Real-time Visualization**: Automatic generation of Pareto fronts, evolution trajectories, and property distributions
- ⚡ **High-Performance Computing**: Optimized for batch processing with model caching and parallel evaluation
- 🎨 **Flexible Framework**: Easily extensible for custom objectives, constraints, and evolutionary strategies

---

## Architecture

The plugin consists of five core modules:

### 1. **Evoloop.py** - Main Evolution Loop
The orchestrator that manages the entire evolutionary optimization process:
- Initializes populations from MatterGen base model
- Coordinates parent selection, crossover, mutation, and environmental selection
- Tracks evolution metrics and generates comprehensive reports
- Supports multiple independent runs with different random seeds

### 2. **EMO_frameworks.py** - Evolutionary Algorithm Engines
Implements multiple EMO strategies:
- **StandardCDPEngine**: Single-population CDP with tournament selection and structural pruning
- **TriplePopCMOEAEngine**: Three-population CMOEA with separate feasible elite (Pop C), infeasible guide (Pop B), and diverse exploration (Pop A) populations

### 3. **EGD_functions.py** - Evolutionary Genetic Operators
Core genetic operations powered by MatterGen's diffusion model:
- **add_noise()**: Adds controlled noise to crystal structures at specified timesteps
- **denoise_batch()**: Batch denoising for efficient offspring generation
- **crossover()**: Structure-aware crossover between parent crystals
- **select_parents()**: Tournament and random parent selection strategies

### 4. **SAES_functions.py** - Selection and Environmental Algorithms
Advanced selection mechanisms:
- **Fast Non-Dominated Sorting (FNDS)**: Efficient Pareto ranking
- **CDP Constraint Handling**: Prioritizes feasible solutions while maintaining diversity
- **Structural Similarity Pruning**: Removes duplicate structures using StructureMatcher
- **Crowding Distance Calculation**: Maintains population diversity in objective space

### 5. **characterize_functions.py** - Property Evaluation Suite
Comprehensive material property characterization:
- **Structure Relaxation**: MatterSim-based geometry optimization
- **Electronic Properties**: Band gap (ALIGNN), dielectric constant
- **Mechanical Properties**: Bulk modulus (ALIGNN)
- **Thermodynamic Properties**: Formation energy, energy above hull
- **Synthesizability Metrics**: E-hull, CL-score (JACS)
- **Specialized Properties**: Piezoelectricity, anisotropy, exfoliation energy

---

## Installation

### Prerequisites
1. Install MatterGen following the [main repository instructions](../README.md)
2. Install additional dependencies:
```bash
pip install -r reuiirmentsDEMO.txt
```

### JACS Microservice Setup
For CL-score evaluation, start the JACS server:
```bash
cd DEMO/PUCGCNN
python jacs_server.py
```
The server will automatically start on port 8080 when running Evoloop.py.

---

## Quick Start

### Initialization Modes

DEMO supports three initialization strategies:

1. **Random Initialization** (default): Generate initial population from MatterGen base model
```python
initial_pool_mode = "random"
```

2. **CIF Directory Initialization**: Start from existing crystal structures
```python
initial_pool_mode = "cif_dir"
cif_dir = "path/to/your/cif/files"
```

3. **Hybrid Mode**: Fill population with both CIF files and random samples
```python
initial_pool_mode = "hybrid"
cif_dir = "path/to/your/cif/files"
n_pops = 100  # Will use CIF files + random samples to reach this size
```

### Basic Usage

```python
from Evoloop import main

# Run evolutionary optimization with default settings
main()
```

### Hyperparameter Configuration

Edit the configuration in `Evoloop.py`:

```python
# ========== Core Evolutionary Parameters ==========
noise_level = 0.3                      # Diffusion timestep for EGD (0.0-1.0)
                                       # Higher = more aggressive mutation
n_pops = 64                            # Population size
n_offspring_per_generation = 64        # Number of offspring per generation
max_generations = 20                   # Total generations to evolve
n_independent_runs = 10                # Number of independent runs with different seeds
random_seed = 5                        # Base random seed
trajectory_sample_every_k = 5          # Save trajectory snapshots every k generations

# ========== Structural Similarity Pruning ==========
# Remove duplicate structures to maintain diversity
similar_score_threshold = 0.80         # Similarity score threshold (0-1)
similar_rms_threshold = 0.20           # RMS distance threshold (Angstrom)
very_similar_score_threshold = 0.80    # Stricter threshold for very similar structures
very_similar_rms_threshold = 0.08      # Stricter RMS threshold

# ========== Model Generation Settings ==========
model_init_batch_size = 64             # Batch size for initial population generation
model_init_max_attempts = 20           # Max attempts to generate valid structures

# ========== Objectives and Constraints ==========
# Define optimization objectives (minimize or maximize)
my_objectives = {
        "ehull_eV": "minimize",
        "dielectric_epsx": "maximize"
    }

# Define constraints with weights
my_constraints = {
        "bandgap_eV": {"min": 0.2, "max": 3.5, "weight": 5.0},
        "exfoliation_energy_meV": {"min": -5.0, "max": 350.0, "weight": 5.0},
        "formation_energy_eV": {"max": 0, "weight": 10.0},
        "cl_score": {"min": 0.50, "weight": 5.0},
        "f_max_eV_A": {"max": 2.0, "weight": 5.0},
        "ehull_eV": {"max": 1.0, "weight": 5.0},
        "num_elements": {"min": 3, "max": 3, "weight": 10.0},
        "element_set": {
            "banned": [
                "Hg", "Tl", "Na", "K", "F" 
            ],
            "allowed_only": ["Mo", "W", "V", "Nb", "S", "Se", "Te", "O"],
            "weight": 5.0  
        },
        "sg_number": {
            # "allowed_only": [],
            "banned": [1],
            "weight": 1.0
        },
        "has_piezo_potential": {"expected": True, "weight": 5.0} 
    }
```

**Key Parameter Explanations:**

- **noise_level**: Controls mutation strength. Lower values (0.1-0.3) make small perturbations; higher values (0.5-0.8) enable larger structural changes.
- **Similarity thresholds**: Prevent population convergence by removing structurally similar individuals. Adjust based on your diversity requirements.
- **Constraint weights**: Higher weights impose stricter penalties for constraint violations. Use 10.0+ for hard constraints.

### Advanced: Custom EMO Engine

```python
from EMO_frameworks import TriplePopCMOEAEngine

# Use triple-population CMOEA
engine = TriplePopCMOEAEngine(
    constraints=constraints,
    objectives=objectives,
    max_size_c=50,  # Feasible elite size
    max_size_b=30,  # Infeasible guide size
    max_size_a=20,  # Diverse exploration size
)
```

---

## Optimization Results

### Example: 2D Material Discovery

Below is an example Pareto front discovered by DEMO when optimizing for low energy above hull and high band gap in 2D materials:

<p align="center">
    <img src="assets/FINAL_PARETO_FRONTS.png" alt="Pareto Front Example" width="700"/>
</p>


---

## Output Structure

Each run generates a timestamped directory:

```
outputs/
└── run_YYYYMMDD_HHMMSS_seed{SEED}/
    ├── run_summary.json              # Run metadata
    ├── FINAL_LEADERBOARD.csv         # All qualified crystals
    ├── FINAL_PARETO_FRONTS.png       # Pareto front visualization
    ├── evolution_trajectory.png       # Convergence plot
    ├── generation_000/
    │   ├── population.json           # Population state
    │   ├── offspring_*.cif           # Generated structures
    │   └── metrics.json              # Generation metrics
    ├── generation_001/
    └── ...
```

---

## Property Evaluation Details

### Supported Properties

DEMO can optimize and constrain the following properties. **You can easily add custom properties** by extending `characterize_functions.py`.

| Property | CSV Column | Source | Description |
|----------|------------|--------|-------------|
| **Composition** | | | |
| Chemical formula | `Formula` | PyMatGen | Chemical composition |
| Element count | `K_nary` | PyMatGen | Number of unique elements |
| Element set | `element_set` | PyMatGen | Set of elements present |
| **Symmetry** | | | |
| Space group | `SpaceGroup` | PyMatGen | Space group symbol |
| Space group number | `sg_number` | PyMatGen | International space group number |
| **Thermodynamics** | | | |
| Formation energy | `E_form(eV/atom)` | ALIGNN | Formation energy per atom |
| Energy above hull | `E_hull(eV/atom)` | ALIGNN | Thermodynamic stability metric |
| Exfoliation energy | `Exf_E(meV/atom)` | ALIGNN | Energy to exfoliate layers (2D materials) |
| **Electronic Properties** | | | |
| Band gap | `Bandgap(eV)` | ALIGNN | Electronic band gap |
| Dielectric constant | `Eps_X` | ALIGNN | Dielectric constant (x-direction) |
| **Magnetic Properties** | | | |
| Magnetic moment | `Mag(uB)` | CHGNet | Total magnetic moment |
| Has magnetism | `has_magnetism` | CHGNet | Boolean flag for magnetic materials |
| **Mechanical Properties** | | | |
| Maximum force | `F_max(eV/A)` | CHGNet | Maximum atomic force after relaxation |
| Maximum stress | `S_max(GPa)` | CHGNet | Maximum stress component |
| Anisotropy ratio | `Aniso_Ratio` | ALIGNN | Mechanical anisotropy ratio |
| Is anisotropic | `is_anisotropic` | ALIGNN | Boolean flag for anisotropic materials |
| **Geometric Properties** | | | |
| Vacuum thickness | `Vacuum(A)` | Custom | Vacuum layer thickness (2D materials) |
| Material thickness | `Thickness(A)` | Custom | Material layer thickness (2D materials) |
| **Synthesizability** | | | |
| CL-score | `CLscore(JACS)` | JACS/PUCGCNN | Chemical likelihood score (0-1) |
| **Functional Properties** | | | |
| Piezoelectric potential | `has_piezo_potential` | ALIGNN | Boolean flag for piezoelectric materials |


---

## Performance Optimization

### Model Caching
All models (CHGNet, ALIGNN, MatterGen) are cached in memory to avoid reloading:
```python
# Automatic caching - no user action needed
_ALIGNN_MODEL_CACHE = {}
_generator_cache = {}
```

### Batch Processing
Use batch denoising for maximum efficiency:
```python
# Automatically batches offspring denoising
denoised_batch = denoise_batch(
    noisy_crystal_dicts=pending_denoise,
    model_path_or_name=model_path
)
```

### Memory Management
- Uses `/dev/shm` (Linux RAM disk) for temporary files when available
- Automatic garbage collection after each generation
- Suppresses verbose model outputs

---

## Customization Guide

### Adding Custom Properties

To add your own property evaluator:

1. Add a characterization function in `characterize_functions.py`:
```python
def characterize_my_property(data_dict: dict) -> dict:
    structure = dict_to_structure(data_dict)
    # Your evaluation logic here
    my_value = compute_my_property(structure)
    data_dict["my_property"] = my_value
    return data_dict
```

2. Register it in `extract_all_properties()`:
```python
if "my_property" not in data_dict:
    data_dict = characterize_my_property(data_dict)
```

3. Use it in constraints or objectives:
```python
my_constraints = {
    "my_property": {"min": 0.5, "max": 2.0, "weight": 5.0}
}
my_objectives = {
        "my_property": "minimize",
        "my_property": "maximize"
    }
```

### Custom Evolutionary Operators

Implement your own EMO engine by inheriting `BaseEMOEngine`:
```python
class MyCustomEngine(BaseEMOEngine):
    def initialize(self, initial_pool: list[dict]):
        # Initialize population
        pass
    
    def ask(self, **kwargs) -> tuple[list, list]:
        # Generate offspring
        pass
    
    def tell(self, evaluated_offspring: list[dict]):
        # Update population
        pass
```

---

## Troubleshooting

### Common Issues

**JACS server connection failed**
```bash
# Manually start the server
cd DEMO/PUCGCNN
python jacs_server.py
```

**CUDA out of memory**
- Reduce `batch_size` in denoising operations
- Decrease `pop_size` or `n_offspring`

**Slow property evaluation**
- Ensure models are cached (first run is always slower)
- Use Linux with `/dev/shm` for faster I/O
- Check that JACS server is running for CL-score

**Invalid structures generated**
- Increase `noise_level` for more diversity
- Adjust `limit_density` in crossover
- Tighten constraints in environmental selection

---

## Citation

If you use MatterGen-DEMO in your research, please cite our paper:

```bibtex
@article{sun2025diffusion,
  title={Diffusion-based Evolutionary Optimization for 3D Multi-Objective Molecular Generation},
  author={Sun, Ruiqing and Feng, Dawei and Yang, Sen and Wang, Ronghang and Song, Huaiyuan and Ding, Bo and Wang, Yijie and Wang, Huaimin},
  journal={arXiv preprint arXiv:2505.11037},
  year={2025}
}
```

Please also cite the original MatterGen paper:

```bibtex
@article{zeni2025mattergen,
  title={MatterGen: a generative model for inorganic materials design},
  author={Zeni, Claudio and others},
  journal={Nature},
  year={2025},
  doi={10.1038/s41586-025-08628-5}
}
```

### Molecular Version

For the **molecular design** version of DEMO, please visit:
**[https://github.com/RuiqingS/DEMO](https://github.com/RuiqingS/DEMO)**

---

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

---

## License

This plugin follows the same license as the main MatterGen repository.

---

## Contact

For questions or issues specific to this plugin, please open an issue in the main MatterGen repository with the `[DEMO]` tag.

---

## Acknowledgments

This plugin builds upon:
- **MatterGen**: Microsoft's generative model for materials design
- **CHGNet**: Universal neural network potential
- **ALIGNN**: Atomistic Line Graph Neural Network
- **MatterSim**: Structure relaxation engine
- **JACS**: Chemical likelihood scoring system
