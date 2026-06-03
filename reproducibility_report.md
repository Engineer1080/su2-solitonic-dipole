# Reproducibility Report

- **Source file:** `exp_coulomb.ddk`
- **SHA-256:** `e9c8dfeaaafd4febdc29308bb75ad2c30fa5214cd121897f4a6c1de1913384e7`
- **Generated:** 2026-06-03 19:53:44 UTC

## Git
- **Branch:** `master`
- **Commit:** `c842fef` (c842fef17fd053f5bebe66a9cbf8ac4b2ec51f46)

## Toolchain
- Dedekind: 3.0.7
- Python:   3.10.4
- OS:       Windows 10 (AMD64)
- torch:    2.5.1+cu121
- CUDA available: yes
- numpy:    2.2.6
- scipy:    1.15.3

## RNG seeds detected in source
- (none detected -- runs are NOT reproducible)

## Methods (LaTeX, from AST)

```latex
?
?
?
?
f = \mathrm{centered\_soliton}
dens = \mathrm{energy\_density}\left( {f}_{0}, {f}_{1}, {f}_{2}, {f}_{3} \right)
rfield = \mathrm{center\_radius}
\mathrm{print}\left( \text{R [fm] | E\_out*R [MeV*fm] (CODATA-exact 0.720; measured ~0.66) | alpha\_sol\^\{\}-1 (CODATA 137.036)} \right)
\mathrm{print}\left( \text{2.0 fm:} \right)
\mathrm{print}\left( \mathrm{coulomb\_coeff}\left( \mathit{dens}, \mathit{rfield}, 2.0 \, \mathrm{fm} \right) \right)
\mathrm{print}\left( \mathrm{alpha\_inv\_at}\left( \mathit{dens}, \mathit{rfield}, 2.0 \, \mathrm{fm} \right) \right)
\mathrm{print}\left( \text{3.0 fm:} \right)
\mathrm{print}\left( \mathrm{coulomb\_coeff}\left( \mathit{dens}, \mathit{rfield}, 3.0 \, \mathrm{fm} \right) \right)
\mathrm{print}\left( \mathrm{alpha\_inv\_at}\left( \mathit{dens}, \mathit{rfield}, 3.0 \, \mathrm{fm} \right) \right)
\mathrm{print}\left( \text{4.0 fm:} \right)
\mathrm{print}\left( \mathrm{coulomb\_coeff}\left( \mathit{dens}, \mathit{rfield}, 4.0 \, \mathrm{fm} \right) \right)
\mathrm{print}\left( \mathrm{alpha\_inv\_at}\left( \mathit{dens}, \mathit{rfield}, 4.0 \, \mathrm{fm} \right) \right)
\mathrm{print}\left( \text{--- Fitting the Coulomb tail to E\_out(R) = C/R ---} \right)
Rs = \left( 2.0, 2.5, 3.0, 3.5, 4.0 \right)
Es = \left( \mathrm{unwrap}\left( \mathrm{e\_out}\left( \mathit{dens}, \mathit{rfield}, 2.0 \, \mathrm{fm} \right) \right), \mathrm{unwrap}\left( \mathrm{e\_out}\left( \mathit{dens}, \mathit{rfield}, 2.5 \, \mathrm{fm} \right) \right), \mathrm{unwrap}\left( \mathrm{e\_out}\left( \mathit{dens}, \mathit{rfield}, 3.0 \, \mathrm{fm} \right) \right), \mathrm{unwrap}\left( \mathrm{e\_out}\left( \mathit{dens}, \mathit{rfield}, 3.5 \, \mathrm{fm} \right) \right), \mathrm{unwrap}\left( \mathrm{e\_out}\left( \mathit{dens}, \mathit{rfield}, 4.0 \, \mathrm{fm} \right) \right) \right)
C = {\mathit{params}}_{0}
Rs = {\mathit{data}}_{0}
Es = {\mathit{data}}_{1}
diff = \frac{C}{Rs} - Es
\mathrm{sum}
data = \left( Rs, Es \right)
params\_init = \left( 0.5 \right)
params\_opt = \mathrm{fit}\left( \mathit{fit\_loss}, \mathit{params\_init}, \mathit{data} \right)
C\_opt = {\mathit{params\_opt}}_{0}
alpha\_inv = \frac{\mathrm{unwrap}\left( \mathit{HBARC} \right)}{2.0 \cdot \mathit{C\_opt}}
\mathrm{print}\left( \text{Optimized C (alpha\_sol*hbar c/2) [MeV*fm] =} \right)
\mathrm{print}\left( \mathit{C\_opt} \right)
\mathrm{print}\left( \text{Fitted alpha\_sol\^\{\}-1 =} \right)
\mathrm{print}\left( \mathit{alpha\_inv} \right)
```
