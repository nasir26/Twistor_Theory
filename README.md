# Twistor Theory — Numerical Explorations

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Computational exploration of Roger Penrose's Twistor Theory: twistor space
geometry, incidence relations, and massless field equations.

**Author**: Nasir Ali — Centre for Development of Advanced Computing (C-DAC), Noida

## Overview

Twistor theory reformulates physics in terms of a complex 4D space (twistor space $\mathbb{T}$)
where points in Minkowski spacetime correspond to Riemann spheres (lines in $\mathbb{PT}$).

## Contents

| Module | Description |
|--------|-------------|
| `twistor.py` | Twistor space $\mathbb{T} \cong \mathbb{C}^4$, incidence relation $\omega^A = ix^{AA'}\pi_{A'}$ |
| `penrose_transform.py` | Penrose transform: cohomology → massless fields |
| `scattering.py` | MHV amplitude calculation (Parke-Taylor formula) |
| `conformal.py` | Conformal group SO(4,2) action on twistor space |

## Physics

The incidence relation connecting spacetime point $x^{\mu}$ to twistor $Z^\alpha = (\omega^A, \pi_{A'})$:
$$\omega^A = ix^{AA'}\pi_{A'}$$

Massless free fields arise as sheaf cohomology classes $H^1(\mathbb{PT}, \mathcal{O}(n))$.

## Requirements

```bash
pip install numpy sympy matplotlib
```

---

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{nasirali_twistor_theory,
  author    = {Nasir Ali},
  title     = {Twistor Theory},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/nasir26/Twistor_Theory},
  note      = {Centre for Development of Advanced Computing (C-DAC), Noida, India}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.\
© 2026 Nasir Ali, C-DAC Noida. All rights reserved.
