# SU(2) Solitonic Dipole — Design Spec

**Date:** 2026-06-02
**Scope:** Teaching/demonstrative
**Language:** Dedekind (`.ddk`, run via `dedekind <file>`)
**Paper:** *High-precision lattice determination of the interaction potential of an SU(2)
solitonic dipole and comparison with perturbative QED*, M. Faber & R. Golubich (TU Wien),
arXiv:2604.12021. Model of Topological Fermions (MTF).

## 1. Goal

Reproduce — at a teaching/demonstrative level, not at the paper's keV precision — the central
result of the paper: the interaction potential `V(d)` of an SU(2) solitonic dipole, showing
that at large separation it follows the classical Coulomb form `A/d` (recovering an effective
inverse fine-structure constant `α_sol⁻¹ ≈ 137`), with short-range deviations consistent with
a running coupling. Everything is written in the Dedekind language.

Non-goals: matching `δE_∞ = 9.432(3) keV`; reproducing `α_sol⁻¹ = 137.1(1)` to stated
precision; the full 15·r₀ domain. These require a research-grade run that is impractical in an
interpreted DSL. We aim for *qualitative* reproduction with strong analytic verification.

## 2. Physics being implemented

- **Field:** unit quaternion `q = (q₀, q⃗)` per lattice site, i.e. the SU(2) field
  `Q(x) = exp(-i α(x) σ⃗·n⃗(x))` with `q₀ = cos α`, `q⃗ = sin α · n⃗`. Norm `|q| = 1`
  reprojected after every relaxation step.
- **Energy functional (static):**
  `E = (α_f ħc / 4π) · ∫ ( ¼ R⃗_{μν}·R⃗^{μν} + Λ ) dV`
  with stabilizing potential `Λ = q₀⁶ / r₀⁴` and field strength `R⃗_{μν} = Γ⃗_μ × Γ⃗_ν`,
  where `(∂_μ Q)Q† = -i σ⃗·Γ⃗_μ` and
  `Γ⃗_μ = (∂_μ α) n⃗ + sin α cos α ∂_μ n⃗ + sin²α (n⃗ × ∂_μ n⃗)`.
  Only spatial derivatives (static configuration).
- **Hedgehog single soliton:** `n⃗ = ± x⃗/|x⃗|`, `α = arctan(|x⃗|/r₀)`. Analytic rest energy
  `E₀ = (α_f ħc / r₀) · π/4 = m_e c² = 0.511 MeV` at `r₀ = 2.213205 fm`.
- **Dipole:** two solitons (opposite topological charge, singlet) at separation `d`; minimize
  energy; `V(d) = E(d) − 2·E₀`. Coulomb fit at large `d` extracts `A = α_sol ħc` and offset
  `δE_∞`, hence `α_sol⁻¹`.

## 3. Method (Approach A: cylindrically symmetric 2D lattice)

The dipole is axially symmetric, so the field is discretized on a 2D `(ϱ, z)` grid instead of
full 3D — same reduction the paper uses. Energy density integrated with the cylindrical measure
`2π ϱ dϱ dz`. Spatial derivatives via vectorized finite differences (tensor shifts): 2nd-order
to start, optional 4th-order five-point stencils (as in the paper) as a refinement. Relaxation
by gradient descent / nonlinear conjugate gradient with unit-norm reprojection each step.
An analytic Coulomb tail `H_out` accounts for field energy beyond the truncated boundary.

### Deliberate simplifications vs. paper
- Domain smaller than the paper's 15·r₀ (boundary at ~6–8 r₀) for tractable runtime.
- Start with 2nd-order stencils; 4th-order optional.
- `a ≈ r₀/3` lattice spacing (matches paper's `a ≤ r₀/3` constraint).
All documented in README.

## 3.1 Dedekind features used

Verified against the installed runtime (`dedekind` 3.0.x). Pillars first, then secondary.

**Load-bearing:**
- **`Quaternion`** (`01_classes.py`): `w,x,y,z`, Hamilton product, `conjugate()`, `norm()`,
  `normalize()`. This *is* the SU(2) field `q = (cos α, sin α·n⃗)`; `normalize()` is the
  per-step unit-norm reprojection; Hamilton product builds `(∂_μ Q)Q†` for `Γ⃗_μ`.
- **Vectorized tensor finite differences + autograd**: energy assembled as a torch
  expression (stencils via tensor shifts/`roll`); gradient comes automatically — no
  hand-coded gradient. This retires the main M0 risk.
- **`minimize(f, x0, method="lbfgs")`** (`04_math.py`): the relaxer (paper uses nonlinear
  CG; L-BFGS is the practical analog). Caveats: flattens `x0` to a 1D vector → field passed
  as a flat parameter vector, reshaped to the `(ϱ,z)` grid internally; L-BFGS runs ≤20 steps
  per call → wrap in an outer loop with norm reprojection between calls.

**Analytic anchors:**
- **`integrate` / `simpson` / `trapz`** (differentiable): radial `E₀` integral (M1).
- **`expm`/`logm` + Pauli matrices** `PAULI_I/X/Y/Z`: build/verify `Q = exp(-i α σ⃗·n⃗)`
  against the quaternion form.
- **`Quantity` + constants** (`alpha`, `hbar`, `c`, `m_e`): document/verify `E₀ = 0.511 MeV`.
  Caveat: unit tables don't carry a full MeV·fm chain → core math in natural numbers
  (MeV, fm); `Quantity` mainly for documentation/verification.

**Secondary / optional:**
- **`cg(A,b)`** (`08_advanced.py`): linear solver, only if a Poisson step is used for the
  analytic exterior `H_out`. Nonlinear field relaxation uses `minimize`, not `cg`.
- **PDE helpers** (`03_solvers.py`: `_laplacian`, `_roll`, `_gradient`, `sparse_laplacian_2d`):
  reference for the cylindrical stencils.
- **Complex tensors + FFT** (`02_tensors.py`): optional FFT-based exterior Coulomb solve.
- **LaTeX-from-AST (`--latex`)** and **reproducibility report (`--reproducibility-report`)**:
  emit the energy functional as LaTeX and a repro report — "paper" finish.

## 4. Repository structure

```
su2-solitonic-dipole/
  README.md
  src/
    constants.ddk      α_f, ħc, r0, m_e, lattice params
    field.ddk          quaternion field on (ϱ,z) grid, hedgehog init, norm constraint
    energy.ddk         Γ_μ, R_μν, H_curv + H_pot(Λ) + H_out, stencils, cylindrical measure
    relax.ddk          gradient/CG relaxation with norm reprojection
  experiments/
    soliton1d.ddk      single soliton radial → E0
    dipole.ddk         two solitons at distance d → E(d)
    scan_potential.ddk loop over d, V(d)=E(d)-2E0, Coulomb fit, α_sol⁻¹
  tests/
    test_*.ddk         numerical tolerance checks vs analytic values
  docs/                this spec + optional LaTeX report (Dedekind LaTeX-from-AST)
```

## 5. Data flow

`constants → field-init (hedgehog) → energy(field) → relax → E(d) → scan over d → V(d) = E(d) − 2E₀ → fit A/d + δE_∞ → α_sol⁻¹`

## 6. Milestones & verification

- **M0 — Spike:** confirm which tensor primitives Dedekind exposes in `.ddk` (2D tensors,
  slicing/shift, complex, autograd through loops). If autograd through `.ddk` loops does not
  hold, fall back to hand-coded gradients / CG. De-risk before building the rest.
- **M1:** single soliton → `E₀ ≈ 0.511 MeV` (few-% tolerance). Strongest analytic anchor.
- **M2:** dipole `E(d)` monotonic; large-`d` behaves like `A/d`.
- **M3:** Coulomb fit → `α_sol⁻¹ ≈ 137` (qualitative); `δE_∞` of order keV.

## 7. Testing

Numerical tolerance assertions against analytic values (E₀, Coulomb asymptotics) rather than
classical unit tests. Each `tests/test_*.ddk` runs a small configuration and asserts a bound.

## 8. Risks

- **DSL expressiveness:** Dedekind is an interpreted convenience DSL; vectorized tensor ops
  must cover stencils/shifts. M0 spike resolves this before committing.
- **Runtime:** even 2D relaxation can be slow; keep grids modest, allow coarse `a` for smoke
  runs and a finer setting for the reported result.
- **Convergence:** norm reprojection + line search needed for stable relaxation; M1 validates.
