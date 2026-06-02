# SU(2) Solitonic Dipole (Dedekind)

Teaching/demonstrative reproduction of **arXiv:2604.12021** — Faber & Golubich,
*"High-precision lattice determination of the interaction potential of an SU(2)
solitonic dipole and comparison with perturbative QED"* (Model of Topological
Fermions, MTF) — written in the **Dedekind** language (`.ddk`).

## The model

The field is a unit quaternion (SU(2) element) `Q = exp(-i alpha sigma.n)` with
`q0 = cos(alpha)`, `q_vec = sin(alpha) n_hat`, on a 3D Cartesian lattice. The
curvature is computed directly from finite differences of `Q`:

- `(d_i Q) conj(Q) = -i sigma . Gamma_i`  =>  `Gamma_i` = vector part of `(d_i Q) conj(Q)`
- `R_ij = Gamma_i x Gamma_j`
- energy density `e = (alpha_f hbar c / 4pi) ( 1/2 sum_{i<j} |R_ij|^2 + q0^6/r0^4 )`
- total energy `E = a^3 * sum_cells e`

With `r0 = 2.21320516 fm`, one hedgehog soliton has rest energy
`E0 = (alpha_f hbar c / r0)(pi/4) = 0.511 MeV = m_e c^2` — the soliton is the electron.

## What this project demonstrates

1. **Soliton rest energy = electron mass.** Closed form (`exp_soliton1d.ddk`) gives
   `E0 = 0.5110 MeV` to machine precision; the lattice energy functional
   (`energy.ddk`) gives `E_grid ~ 0.46 MeV` for the discretized soliton (within ~10%
   of 0.511 on this grid), validating the energy machinery two independent ways.

2. **The soliton is a Coulomb charge => the fine-structure constant.**
   (`exp_coulomb.ddk`) Because the soliton's far field is Coulomb, the energy stored
   outside radius R obeys `E_out(R) = (alpha_sol hbar c / 2) / R`, so `E_out(R)*R` is a
   plateau equal to `alpha_sol hbar c / 2`. Measured on the lattice it is flat at
   `~0.66 MeV*fm` over the near field `R = 2..4 fm`, giving

   **`alpha_sol^-1 ~ 151`**  (CODATA `alpha^-1 = 137.036`, recovered to ~10%).

   The interaction of two opposite charges then follows: `V(d) = -alpha_sol hbar c / d`.

## Scope and honest limitations

This is a **teaching-scope** reproduction, not the paper's high-precision result.

- The paper extracts `alpha_sol^-1 = 137.1(1)` and `delta E_inf = 9.4 keV` by minimizing
  two-soliton configurations and fitting `V(d)` at separations `d = 200..280 fm` on a
  cylindrically symmetric lattice with a soliton-to-boundary distance of `15 r0`.
- A **dynamical lattice dipole** was attempted here but is genuinely hard: free 3D
  relaxation suffers the Skyrme/Derrick instability (the soliton unwinds / its core
  collapses sub-cell), and a stable, topology-protected minimization needs the paper's
  exact boundary conditions, domain size, and discretization — research-grade work.
- We therefore recover `alpha` from the **single soliton's near-field Coulomb tail**
  (a static, robust measurement). This reproduces the headline physics — Coulomb
  behaviour and `alpha^-1` of order 137 — but **not** the precise value nor the running
  of `alpha` (which require the full dynamical dipole). See
  `docs/superpowers/specs/...-design.md` (Outcome section) for details.

## Run

    $env:PYTHONUTF8='1'
    python -m dedekind.compiler src\exp_soliton1d.ddk   # E0 = 0.511 MeV (closed form)
    python -m dedekind.compiler src\exp_coulomb.ddk      # alpha_sol^-1 ~ 151 from the tail

(The compiler prints the generated Python first; the program output follows the
`Executing Code:` line.)

## Tests

    python -m dedekind.compiler src\test_constants.ddk   # E0 anchor
    python -m dedekind.compiler src\test_field.ddk        # unit-norm hedgehog
    python -m dedekind.compiler src\test_energy.ddk       # grid energy ~ 0.46 MeV
    python -m dedekind.compiler src\test_coulomb.ddk      # alpha^-1 ~ O(137), flat plateau

## Files

    src/constants.ddk     units, constants, 3D grid (65^3, a=0.5 fm)
    src/field.ddk         hedgehog soliton, centered soliton, radius field
    src/energy.ddk        Gamma_i from FD of Q, R_ij, energy density + total energy
    src/coulomb.ddk       E_out(R), Coulomb coefficient, alpha_sol^-1
    src/exp_soliton1d.ddk closed-form E0 anchor
    src/exp_coulomb.ddk   alpha from the near-field Coulomb tail
    src/test_*.ddk        numerical tolerance checks
    docs/superpowers/     design spec (with Outcome section) and implementation plan
