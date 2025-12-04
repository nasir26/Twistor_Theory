# Quantum Twistor Theory Simulation

A mathematically rigorous implementation of Roger Penrose's twistor theory, providing quantum mechanical simulations with concrete numerical outputs.

## Overview

Twistor theory provides a deep connection between:
- **4D Minkowski spacetime geometry** - the arena for special relativity
- **Complex projective 3-space (CP³)** - twistor space
- **Massless field equations** - Maxwell, graviton, Weyl neutrino
- **Quantum scattering amplitudes** - MHV formalism, Parke-Taylor formula

## Mathematical Foundation

### Twistor Space

A twistor is a 4-component complex object:

$$Z^\alpha = (\omega^A, \pi_{A'}) \in \mathbb{T} \cong \mathbb{C}^4$$

where:
- ω^A is a 2-spinor encoding position information
- π_{A'} is a conjugate 2-spinor encoding momentum/helicity

### The Incidence Relation

The fundamental connection between twistors and spacetime:

$$\omega^A = i x^{AA'} \pi_{A'}$$

This maps twistors to null geodesics in complexified Minkowski space.

### Helicity

The helicity of a massless particle is given by the twistor norm:

$$s = \frac{1}{2} Z^\alpha \bar{Z}_\alpha$$

- s > 0: positive helicity (self-dual fields)
- s < 0: negative helicity (anti-self-dual fields)
- s = 0: null twistor (corresponds to a point in spacetime)

## Features

### 1. Twistor Space Geometry
- Construction of twistors with various helicities
- Projective equivalence in CP³
- Infinity twistor and conformal structure

### 2. Spinor Algebra
- Full 2-spinor calculus
- Pauli matrices and Infeld-van der Waerden symbols
- Null vector decomposition into spinors

### 3. Incidence Relation
- Mapping between twistors and null geodesics
- Numerical verification of incidence
- Spacetime reconstruction from twistors

### 4. MHV Scattering Amplitudes
- Parke-Taylor formula implementation
- Spinor helicity brackets ⟨ij⟩ and [ij]
- Mandelstam variables calculation
- 4-gluon amplitude computations

### 5. Penrose Transform
- Contour integrals for massless fields
- Scalar field solutions
- Self-dual Maxwell field (F_{AB})
- Field invariant calculations

### 6. Quantum Evolution
- Twistor Hilbert space construction
- Unitary time evolution via exp(-iHt/ℏ)
- Helicity operator expectation values
- Free and interacting Hamiltonians

### 7. Twistor String Theory
- Degree-d curve contributions
- BCFW recursion structure

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the complete simulation:

```bash
python twistor_simulation.py
```

### Example Output

```
╔══════════════════════════════════════════════════════════════════╗
║     QUANTUM TWISTOR THEORY SIMULATION                            ║
║     Based on Penrose's Twistor Programme                         ║
╚══════════════════════════════════════════════════════════════════╝

1. TWISTOR SPACE GEOMETRY
----------------------------------------
Creating twistors with various helicities:
  Twistor Z_1:
    ω = [0.4967+0.6477j, -0.1383+1.5230j]
    π = [-0.2342+1.5792j, -0.2341+0.7674j]
    Helicity s = 2.1077
    Is null: False
...
```

## Key Classes

### `Twistor`
Represents a twistor Z^α = (ω^A, π_{A'}) with methods for:
- Computing helicity
- Extracting spacetime null lines
- Creating from spacetime points

### `TwistorSpace`
Operations on projective twistor space PT ≅ CP³:
- Random twistor generation
- Incidence relation evaluation
- Lines in twistor space

### `MHVAmplitudes`
Scattering amplitude calculations:
- Spinor brackets ⟨ij⟩, [ij]
- Parke-Taylor formula
- Mandelstam variables

### `PenroseTransform`
Massless field solutions via contour integrals:
- Scalar fields
- Maxwell field (self-dual part)

### `TwistorQuantumState`
Quantum states in twistor Hilbert space:
- Superpositions of twistor states
- Inner products and norms
- Unitary time evolution

## Physical Interpretation

### Massless Particles
Twistors naturally describe massless particles:
- **Photon** (s = ±1): Electromagnetic field
- **Graviton** (s = ±2): Linearized gravity
- **Weyl neutrino** (s = ±1/2): Massless fermion

### Scattering Amplitudes
The MHV (Maximally Helicity Violating) amplitudes take the remarkably simple form:

$$A_n(1^-, 2^-, 3^+, \ldots, n^+) = \frac{\langle 12 \rangle^4}{\langle 12 \rangle \langle 23 \rangle \cdots \langle n1 \rangle}$$

This "BCFW revolution" revealed deep connections between twistor geometry and particle physics.

## Mathematical Rigor

The simulation implements:

1. **Proper spinor algebra** with SL(2,C) structure
2. **Correct incidence geometry** via the Penrose correspondence
3. **Unitary quantum evolution** using scipy's matrix exponential
4. **Self-consistent amplitude calculations** respecting helicity selection rules

## References

1. Penrose, R. (1967). "Twistor Algebra". Journal of Mathematical Physics.
2. Penrose, R. & Rindler, W. (1984, 1986). "Spinors and Space-Time" Vols. 1-2.
3. Witten, E. (2004). "Perturbative Gauge Theory as a String Theory in Twistor Space".
4. Arkani-Hamed, N. et al. (2010). "Scattering Amplitudes and the Positive Grassmannian".
5. Adamo, T. (2017). "Lectures on Twistor Theory". arXiv:1712.02196.

## License

MIT License
