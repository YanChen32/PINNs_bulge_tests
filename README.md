# Physics-Informed Neural Networks for Bulge Test Modeling of 2D Crystalline Materials

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)

This repository contains the official implementation of the paper:

**"Physics-Informed Neural Networks for Bulge Test Modeling of General Anisotropic Two-Dimensional Crystalline Materials with Decoupled Elasticity"**

## 📖 Overview

Two-dimensional (2D) crystalline materials have great potential for flexible electronics and strain engineering, but their mechanical characterization via bulge testing is challenging. Commercial Finite Element Analysis (FEA) cannot fully capture decoupled in-plane and out-of-plane stiffnesses or complex constitutive behaviors, and analytical solutions are intractable for anisotropic crystals with irregular geometries.

This work develops a **Physics-Informed Neural Network (PINNs) framework** for 2D material bulge testing, combining modified **Föppl-von Kármán theory** with **energy-based loss functions** to capture arbitrary symmetries and decoupled elasticity.

<p align="center">
  <img src="https://github.com/YanChen32/PINNs_bulge_tests/blob/main/raw/framework.png" alt="Framework Overview" width="800"/>
</p>

## ✨ Key Features

- **Arbitrary Crystal Symmetries**: Supports hexagonal (graphene), square (Mn₂S₂), rectangular (black phosphorene), and oblique (PdCdCl₄) symmetry classes
- **Decoupled Elasticity**: Naturally captures the decoupled in-plane and out-of-plane stiffnesses unique to 2D materials
- **Flexible Geometry**: Accommodates circular, elliptical, and square bubble geometries through configurable sampling and boundary conditions
- **Nonlinear Constitutive Behaviors**: Extends to material nonlinearity with modified energy density formulations
- **Mesh-Free Approach**: No mesh generation required, avoiding the constraints of commercial FEA software

## 📁 Repository Structure

```
PINNs_bulge_tests/
├── 2D_crystal_bubble_membrane.ipynb      # Main notebook for membrane model
├── 2D_crystal_bubble_plate.ipynb         # Main notebook for plate model (circular boundary)
├── 2D_crystal_bubble_plate_ellipse.ipynb # Plate model with elliptical boundary
├── 2D_crystal_bubble_plate_square.ipynb  # Plate model with square boundary
├── Membrane_normal_settings/             # Trained models and results for membrane model
├── Plate_normal_settings/                # Trained models and results for plate model
├── nonlinear/                            # Nonlinear constitutive behavior examples
├── other_radius_pressure_settings/       # Models with varying radii and pressures
├── other_shapes/                         # Non-circular bubble geometries
└── README.md
```

## 🔬 Theoretical Background

### Governing Equations

The framework is based on the modified Föppl-von Kármán plate theory with decoupled constitutive relations:

**In-plane (membrane) stiffness:**
```
[Nₓ, Nᵧ, Nₓᵧ]ᵀ = [C²ᴰ] × [εₓ, εᵧ, γₓᵧ]ᵀ
```

**Out-of-plane (bending) stiffness:**
```
[Mₓ, Mᵧ, Mₓᵧ]ᵀ = [D] × [κₓ, κᵧ, 2κₓᵧ]ᵀ
```

### Energy-Based Loss Function

The total potential energy serves as the loss function:
```
Loss = Πₘ + Πᵦ + V
```
where:
- `Πₘ`: Membrane strain energy
- `Πᵦ`: Bending strain energy
- `V`: Work done by external pressure

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- CUDA-capable GPU (recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/YanChen32/PINNs_bulge_tests.git
cd PINNs_bulge_tests

# Install dependencies
pip install torch numpy matplotlib scipy jupyter
```

### Quick Start

1. **For membrane model simulation:**
   ```bash
   jupyter notebook 2D_crystal_bubble_membrane.ipynb
   ```

2. **For plate model simulation (with bending):**
   ```bash
   jupyter notebook 2D_crystal_bubble_plate.ipynb
   ```

3. **For non-circular geometries:**
   ```bash
   jupyter notebook 2D_crystal_bubble_plate_ellipse.ipynb
   # or
   jupyter notebook 2D_crystal_bubble_plate_square.ipynb
   ```

## 📊 Representative Materials

The framework has been validated on four representative 2D crystals:

| Material | Symmetry | C₁₁²ᴰ (N/m) | C₂₂²ᴰ (N/m) | C₁₂²ᴰ (N/m) | C₆₆²ᴰ (N/m) | C₁₆²ᴰ (N/m) | C₂₆²ᴰ (N/m) | 
|----------|----------|-------------|-------------|-------------|-------------|-------------|-------------|
| Graphene | Hexagonal | 354.1 | 354.1 | 56.7 | 148.7 | 0.00 | 0.00 |
| Black Phosphorene | Rectangular | 102.98 | 27.30 | 17.51 | 22.76 | 0.00 | 0.00 |
| Mn₂S₂ | Square | 121.83 | 121.83 | 33.90 | 108.45 | 0.00 | 0.00 |
| PdCdCl₄ | Oblique | 12.38 | 37.00 | 8.50 | 14.48 | 3.24 | 9.76 |

## 🔧 Customization

### Modifying Material Properties

Edit the stiffness matrices in the notebook:
```python
# In-plane stiffness matrix (N/m)
C2D = torch.tensor([[C11, C12, C16],
                    [C12, C22, C26],
                    [C16, C26, C66]])

# Bending stiffness matrix (N)
D = torch.tensor([[D11, D12, D16],
                  [D12, D22, D26],
                  [D16, D26, D66]])
```

### Adjusting Geometry and Loading

```python
# Bubble radius (nm)
a = 10.0

# Applied pressure (MPa)
q = 307.4
```

### Network Architecture

```python
# Default configuration
hidden_layers = [32, 64, 64, 64, 32]
activation = 'tanh'
```

## 📈 Training Tips

- **Learning Rate**: Initial lr = 2×10⁻³ with ReduceLROnPlateau scheduler
- **Sampling**: 100,000 points with periodic resampling
- **Convergence**: Typical training time ~2700s on RTX 3060 for nonlinear plate model

## 📝 Citation

If you find this code useful for your research, please cite our paper:

```bibtex
@article{zheng2025pinns,
  title={Physics-Informed Neural Networks for Bulge Test Modeling of General Anisotropic Two-Dimensional Crystalline Materials with Decoupled Elasticity},
  author={Zheng, Yichen and Kang, Kai and Zhang, Zaiyu and Liu, Huichao and Liu, Yilun and Chen, Yan},
  journal={[Extreme Mechanics Letters]},
  year={2025},
  doi={[DOI]}
}
```
*Laboratory for Multiscale Mechanics and Medical Science, SV LAB, School of Aerospace, Xi'an Jiaotong University, Xi'an 710049, China*

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions or collaborations, please contact:
- Yan Chen: [yanchen@xjtu.edu.cn](mailto:yanchen@xjtu.edu.cn)

---

<p align="center">
  <b>Keywords:</b> Physics-Informed Neural Networks | Bulge Test | Two-dimensional Materials | Anisotropic Mechanics | Nonlinear Elasticity
</p>

