# SU(2) Solitonic Dipole Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce, at a teaching/demonstrative level, the SU(2) solitonic-dipole interaction potential `V(d)` of arXiv:2604.12021 in the Dedekind language — showing Coulomb behaviour at large `d` and recovering an effective `α_sol⁻¹ ≈ 137`.

**Architecture:** Unit-quaternion SU(2) field on a small 3D Cartesian grid (4 component tensors). Curvature computed directly from finite differences of `Q`: `Γ⃗_i` = vector part of `(∂_i Q)·conj(Q)`, `R⃗_{ij} = Γ⃗_i × Γ⃗_j`. Energy `E = a³·Σ (α_f ħc/4π)(½Σ_{i<j}|R_{ij}|² + q₀⁶/r₀⁴)`. Relaxation via `minimize(method="lbfgs")` over the flattened field (differentiable norm projection inside the closure), in an outer loop. Single-soliton energy is the calibration/validation anchor; `V(d) = E(d) − 2·E₀_grid` cancels per-soliton discretization systematics; a Coulomb fit of `V(d)` yields `α_sol`.

**Tech Stack:** Dedekind 3.0.x (`.ddk`), torch-backed tensors + autograd, `Quaternion`/`minimize`/`polyfit` builtins. Run on Windows via `$env:PYTHONUTF8='1'; python -m dedekind.compiler <file>.ddk`.

---

## Conventions (read once)

- **All `.ddk` files live in `src/`** (flat). Dedekind's `use <name>` resolves relative to the
  current file's directory, so flat layout makes imports trivial: `use constants`, `use field`, etc.
- **Run command (every run):**
  `$env:PYTHONUTF8='1'; python -m dedekind.compiler src\<file>.ddk 2>&1 | Select-Object -Last <N>`
  The compiler prints the generated Python first; real output follows the `Executing Code:` line.
  Filter with `Select-Object -Last N`.
- **A test "fails"** when it raises a compile error (missing `use` target / undefined name) or an
  `assert(...)` aborts the run. It **passes** when it runs to completion and prints its `OK` line.
- **Language facts confirmed by spike (M0):** list literals `[...]` become tensors; torch methods
  pass through (`.sum()`, `.reshape(...)`, `.roll(shift,dim)`, `.narrow(dim,start,len)`, slicing
  `t[a:b]`, single index `t[i]`); broadcasting works; `minimize(f, x0, "lbfgs")` takes a closure
  and returns an object with `.x`/`.fun`; `Quaternion`, `assert`, `atan/sin/cos/sqrt`, constants,
  and `polyfit` are available. **No multi-index `t[i,j]`** — use `roll`/`narrow`/`reshape`.

---

## File structure

```
src/
  constants.ddk      units, physical constants, grid params (no deps)
  field.ddk          use constants — hedgehog init, dipole init, quaternion product, norm projection
  energy.ddk         use constants — FD derivatives, Γ_i, R_ij, energy_components(q0,q1,q2,q3)
  relax.ddk          use energy, field, constants — energy_flat closure, relax()
  exp_soliton1d.ddk  use constants — closed-form E0 anchor
  exp_dipole.ddk     use relax, field, energy, constants — E(d), V(d) for given d
  exp_scan.ddk       use relax, field, energy, constants — scan d, Coulomb fit, α_sol⁻¹
  test_constants.ddk use constants
  test_field.ddk     use field, constants
  test_energy.ddk    use energy, field, constants
  test_relax.ddk     use relax, field, energy, constants
  test_dipole.ddk    use relax, field, energy, constants
  test_scan.ddk      use relax, field, energy, constants
docs/                spec + this plan
README.md
.gitignore
```

---

### Task 0: Scaffolding + constants

**Files:**
- Create: `.gitignore`, `README.md`, `src/constants.ddk`, `src/test_constants.ddk`

- [ ] **Step 1: Write the failing test** — `src/test_constants.ddk`

```
use constants

// E0 = (alpha_f * hbar c / r0) * (pi/4) must equal the electron rest energy.
e0 = (ALPHA_HBARC / R0) * (PI / 4.0)
print("E0 =")
print(e0)
assert(e0 > 0.50)
assert(e0 < 0.52)
print("OK test_constants")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$env:PYTHONUTF8='1'; python -m dedekind.compiler src\test_constants.ddk 2>&1 | Select-Object -Last 8`
Expected: FAIL — compile/runtime error, `constants.ddk` not found / `ALPHA_HBARC` undefined.

- [ ] **Step 3: Write minimal implementation** — `src/constants.ddk`

```
// modules/constants.ddk — units, constants, grid parameters.
// Natural units: lengths in fm, energies in MeV.

PI    = 3.141592653589793
HBARC = 197.3269804            // MeV*fm
ALPHA_F = 1.0 / 137.035999084  // fine-structure constant (dimensionless)
ALPHA_HBARC = ALPHA_F * HBARC  // MeV*fm  (~1.43996)
R0    = 2.21320516             // fm  => E0 = (ALPHA_HBARC/R0)*(pi/4) = 0.511 MeV
ME_C2 = 0.510998950            // MeV  (electron rest energy)

// Energy prefactor alpha_f * hbar c / (4 pi)  [MeV*fm]
EPREF = ALPHA_HBARC / (4.0 * PI)

// --- Lattice (3D Cartesian) ---
A   = 0.7377                   // spacing in fm (~ R0/3)
NX  = 24
NY  = 24
NZ  = 24
NXf = NX + 0.0
NYf = NY + 0.0
NZf = NZ + 0.0
NPTS  = NX * NY * NZ
NPTS4 = 4 * NPTS
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$env:PYTHONUTF8='1'; python -m dedekind.compiler src\test_constants.ddk 2>&1 | Select-Object -Last 8`
Expected: PASS — prints `E0 =`, a value ≈ `0.511`, then `OK test_constants`.

- [ ] **Step 5: Create `.gitignore` and `README.md` (stub)**

`.gitignore`:
```
__pycache__/
*.pyc
*.png
*.tex
```

`README.md`:
```
# SU(2) Solitonic Dipole (Dedekind)

Teaching/demonstrative reproduction of arXiv:2604.12021 (Faber & Golubich,
Model of Topological Fermions) in the Dedekind language.

## Run
    $env:PYTHONUTF8='1'
    python -m dedekind.compiler src\exp_scan.ddk

See docs/superpowers/specs for the design and scope.
```

- [ ] **Step 6: Commit**

```
git add .gitignore README.md src/constants.ddk src/test_constants.ddk
git commit -m "Task 0: scaffolding + constants (E0 anchor passes)"
```

---

### Task 1: Single-soliton closed-form anchor (M1, analytic)

**Files:**
- Create: `src/exp_soliton1d.ddk`

- [ ] **Step 1: Write the experiment as a self-checking script** — `src/exp_soliton1d.ddk`

```
use constants

// Closed-form rest energy of one hedgehog soliton (paper, single-soliton result):
//   E0 = (alpha_f * hbar c / r0) * (pi/4)
fn soliton_rest_energy() {
    return (ALPHA_HBARC / R0) * (PI / 4.0)
}

e0 = soliton_rest_energy()
print("E0 (closed form) [MeV] =")
print(e0)
print("electron m_e c^2 [MeV] =")
print(ME_C2)
rel = (e0 - ME_C2) / ME_C2
print("relative error =")
print(rel)
assert(rel < 0.01)
assert(rel > -0.01)
print("OK exp_soliton1d")
```

- [ ] **Step 2: Run to verify it passes**

Run: `$env:PYTHONUTF8='1'; python -m dedekind.compiler src\exp_soliton1d.ddk 2>&1 | Select-Object -Last 10`
Expected: PASS — `E0 (closed form)` ≈ 0.511, relative error < 1%, `OK exp_soliton1d`.

- [ ] **Step 3: Commit**

```
git add src/exp_soliton1d.ddk
git commit -m "Task 1: single-soliton closed-form E0 anchor"
```

---

### Task 2: Field — hedgehog init, quaternion product, norm projection

**Files:**
- Create: `src/field.ddk`, `src/test_field.ddk`

- [ ] **Step 1: Write the failing test** — `src/test_field.ddk`

```
use field
use constants

// A centered hedgehog must be unit-norm everywhere and have q0=cos(alpha):
// at the center r->0 => alpha->0 => q0->1.
f = hedgehog((NXf - 1.0) / 2.0, (NYf - 1.0) / 2.0, (NZf - 1.0) / 2.0, 1.0)
fn comp_norm(f) {
    q0 = f[0]
    q1 = f[1]
    q2 = f[2]
    q3 = f[3]
    return sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3)
}
nrm = comp_norm(f)
print("min norm:")
print(nrm.min())
print("max norm:")
print(nrm.max())
assert(nrm.min() > 0.99)
assert(nrm.max() < 1.01)

// q0 maximum should be ~1 (near the core)
print("max q0:")
print(f[0].max())
assert(f[0].max() > 0.99)
print("OK test_field")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$env:PYTHONUTF8='1'; python -m dedekind.compiler src\test_field.ddk 2>&1 | Select-Object -Last 10`
Expected: FAIL — `hedgehog` undefined / `field.ddk` not found.

- [ ] **Step 3: Write minimal implementation** — `src/field.ddk`

```
// modules/field.ddk — quaternion SU(2) field on a 3D Cartesian grid.
use constants

// Hedgehog soliton centered at lattice indices (cx,cy,cz); s = +1 soliton, -1 antisoliton.
// Returns [q0,q1,q2,q3], each shape (NX,NY,NZ). q0=cos(alpha), q_vec=sin(alpha)*s*n_hat.
fn hedgehog(cx, cy, cz, s) {
    xs = (arange(0.0, NXf, 1.0) - cx) * A
    ys = (arange(0.0, NYf, 1.0) - cy) * A
    zs = (arange(0.0, NZf, 1.0) - cz) * A
    x3 = xs.reshape(NX, 1, 1)
    y3 = ys.reshape(1, NY, 1)
    z3 = zs.reshape(1, 1, NZ)
    r  = sqrt(x3 * x3 + y3 * y3 + z3 * z3 + 1.0e-12)
    alpha = atan(r / R0)
    ca = cos(alpha)
    sa = sin(alpha)
    nx = x3 / r
    ny = y3 / r
    nz = z3 / r
    q0 = ca + (nx * 0.0)            // broadcast q0 to full (NX,NY,NZ)
    q1 = sa * s * nx
    q2 = sa * s * ny
    q3 = sa * s * nz
    return [q0, q1, q2, q3]
}

// Pointwise (per-site) Hamilton product of two quaternion fields.
fn qmul_field(a, b) {
    a0 = a[0]
    a1 = a[1]
    a2 = a[2]
    a3 = a[3]
    b0 = b[0]
    b1 = b[1]
    b2 = b[2]
    b3 = b[3]
    c0 = a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3
    c1 = a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2
    c2 = a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1
    c3 = a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0
    return [c0, c1, c2, c3]
}

// Differentiable unit-norm projection.
fn normalize_field(f) {
    q0 = f[0]
    q1 = f[1]
    q2 = f[2]
    q3 = f[3]
    nrm = sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3 + 1.0e-12)
    return [q0 / nrm, q1 / nrm, q2 / nrm, q3 / nrm]
}

// Soliton(+) at +d/2 and antisoliton(-) at -d/2 along z; product config, then normalized.
fn dipole_init(d) {
    off = (d / A) / 2.0
    cx = (NXf - 1.0) / 2.0
    cy = (NYf - 1.0) / 2.0
    cz = (NZf - 1.0) / 2.0
    p = hedgehog(cx, cy, cz + off, 1.0)
    m = hedgehog(cx, cy, cz - off, -1.0)
    return normalize_field(qmul_field(p, m))
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$env:PYTHONUTF8='1'; python -m dedekind.compiler src\test_field.ddk 2>&1 | Select-Object -Last 10`
Expected: PASS — min/max norm in (0.99,1.01), max q0 > 0.99, `OK test_field`.

- [ ] **Step 5: Commit**

```
git add src/field.ddk src/test_field.ddk
git commit -m "Task 2: quaternion field (hedgehog, product, norm projection)"
```

---

### Task 3: Energy functional from finite differences of Q

**Files:**
- Create: `src/energy.ddk`, `src/test_energy.ddk`

- [ ] **Step 1: Write the failing test** — `src/test_energy.ddk`

```
use energy
use field
use constants

// Single centered soliton: grid energy must be a finite, positive number of the
// right order of magnitude (electron rest energy ~0.511 MeV). Coarse grid =>
// few-tens-% discretization/truncation error is expected (teaching scope).
f = hedgehog((NXf - 1.0) / 2.0, (NYf - 1.0) / 2.0, (NZf - 1.0) / 2.0, 1.0)
e = energy_components(f[0], f[1], f[2], f[3])
print("E_grid(1 soliton) [MeV] =")
print(e)
assert(e > 0.15)
assert(e < 1.5)
print("OK test_energy")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$env:PYTHONUTF8='1'; python -m dedekind.compiler src\test_energy.ddk 2>&1 | Select-Object -Last 10`
Expected: FAIL — `energy_components` undefined / `energy.ddk` not found.

- [ ] **Step 3: Write minimal implementation** — `src/energy.ddk`

```
// modules/energy.ddk — MTF energy from finite differences of the quaternion field.
use constants

// Central difference of a (NX,NY,NZ) field along spatial dim (0=x,1=y,2=z).
fn dd(q, dim) {
    return (q.roll(-1, dim) - q.roll(1, dim)) * (1.0 / (2.0 * A))
}

// Gamma vector (3 color comps) for spatial direction `dim`:
//   (d_dim Q) conj(Q) = -i sigma . Gamma  ->  Gamma = vector part.
// Returns [Gx,Gy,Gz], each (NX,NY,NZ).
fn gamma_dir(q0, q1, q2, q3, dim) {
    d0 = dd(q0, dim)
    d1 = dd(q1, dim)
    d2 = dd(q2, dim)
    d3 = dd(q3, dim)
    Gx = q0 * d1 - d0 * q1 + (d3 * q2 - d2 * q3)
    Gy = q0 * d2 - d0 * q2 + (d1 * q3 - d3 * q1)
    Gz = q0 * d3 - d0 * q3 + (d2 * q1 - d1 * q2)
    return [Gx, Gy, Gz]
}

// |A x B|^2 in color space; A,B are [Ax,Ay,Az].
fn cross_sq(a, b) {
    Rx = a[1] * b[2] - a[2] * b[1]
    Ry = a[2] * b[0] - a[0] * b[2]
    Rz = a[0] * b[1] - a[1] * b[0]
    return Rx * Rx + Ry * Ry + Rz * Rz
}

// Total static energy [MeV] of the field given by 4 components (NX,NY,NZ).
fn energy_components(q0, q1, q2, q3) {
    gx = gamma_dir(q0, q1, q2, q3, 0)
    gy = gamma_dir(q0, q1, q2, q3, 1)
    gz = gamma_dir(q0, q1, q2, q3, 2)
    curv = 0.5 * (cross_sq(gx, gy) + cross_sq(gx, gz) + cross_sq(gy, gz))
    lam = (q0 * q0 * q0 * q0 * q0 * q0) / (R0 * R0 * R0 * R0)
    dens = EPREF * (curv + lam)
    // Drop the 1-cell boundary shell (roll wraps periodically -> bad edge derivatives).
    di = dens.narrow(0, 1, NX - 2).narrow(1, 1, NY - 2).narrow(2, 1, NZ - 2)
    return (A * A * A) * di.sum()
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$env:PYTHONUTF8='1'; python -m dedekind.compiler src\test_energy.ddk 2>&1 | Select-Object -Last 10`
Expected: PASS — `E_grid(1 soliton)` prints a positive value in (0.15, 1.5) MeV, `OK test_energy`.
If it lands outside, note the printed value: a clean factor-of-2 offset points at the Γ/R
normalization convention (documented as pinned here); otherwise tighten `A`/grid in constants.

- [ ] **Step 5: Commit**

```
git add src/energy.ddk src/test_energy.ddk
git commit -m "Task 3: energy from finite differences of Q (M1 grid energy)"
```

---

### Task 4: Relaxation via L-BFGS

**Files:**
- Create: `src/relax.ddk`, `src/test_relax.ddk`

- [ ] **Step 1: Write the failing test** — `src/test_relax.ddk`

```
use relax
use field
use energy
use constants

// Relaxing a centered soliton must not increase its energy and must stay ~0.5 MeV.
f0 = hedgehog((NXf - 1.0) / 2.0, (NYf - 1.0) / 2.0, (NZf - 1.0) / 2.0, 1.0)
e_before = energy_components(f0[0], f0[1], f0[2], f0[3])
flat0 = pack_field(f0)
e_after = relax_energy(flat0, 3)
print("E before:")
print(e_before)
print("E after relax:")
print(e_after)
assert(e_after < e_before + 0.02)
assert(e_after > 0.1)
print("OK test_relax")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$env:PYTHONUTF8='1'; python -m dedekind.compiler src\test_relax.ddk 2>&1 | Select-Object -Last 12`
Expected: FAIL — `pack_field`/`relax_energy` undefined.

- [ ] **Step 3: Write minimal implementation** — `src/relax.ddk`

```
// modules/relax.ddk — L-BFGS relaxation of the quaternion field.
use constants
use field
use energy

// Flatten [q0,q1,q2,q3] (4,NX,NY,NZ) -> 1D vector length NPTS4.
fn pack_field(f) {
    q0 = f[0].reshape(NPTS)
    q1 = f[1].reshape(NPTS)
    q2 = f[2].reshape(NPTS)
    q3 = f[3].reshape(NPTS)
    return stack([q0, q1, q2, q3]).reshape(NPTS4)
}

// Energy of a flat parameter vector: rebuild, normalize differentiably, evaluate.
fn energy_flat(x) {
    g = x.reshape(4, NPTS)
    q0 = g[0].reshape(NX, NY, NZ)
    q1 = g[1].reshape(NX, NY, NZ)
    q2 = g[2].reshape(NX, NY, NZ)
    q3 = g[3].reshape(NX, NY, NZ)
    nrm = sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3 + 1.0e-12)
    return energy_components(q0 / nrm, q1 / nrm, q2 / nrm, q3 / nrm)
}

// Run L-BFGS for `outer` batches (each ~20 inner iters); return final energy [MeV].
fn relax_energy(flat0, outer) {
    x = flat0
    i = 0
    while i < outer {
        res = minimize(energy_flat, x, "lbfgs")
        x = res.x
        i = i + 1
    }
    return energy_flat(x)
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$env:PYTHONUTF8='1'; python -m dedekind.compiler src\test_relax.ddk 2>&1 | Select-Object -Last 12`
Expected: PASS — `E after relax` ≤ `E before` (within 0.02 slack) and > 0.1, `OK test_relax`.

- [ ] **Step 5: Commit**

```
git add src/relax.ddk src/test_relax.ddk
git commit -m "Task 4: L-BFGS relaxation (pack_field, energy_flat, relax_energy)"
```

---

### Task 5: Dipole energy E(d) and interaction V(d)

**Files:**
- Create: `src/exp_dipole.ddk`, `src/test_dipole.ddk`

- [ ] **Step 1: Write the failing test** — `src/test_dipole.ddk`

```
use relax
use field
use energy
use constants

// Single-soliton grid reference (relaxed), used to subtract self-energies.
f1 = hedgehog((NXf - 1.0) / 2.0, (NYf - 1.0) / 2.0, (NZf - 1.0) / 2.0, 1.0)
e1 = relax_energy(pack_field(f1), 2)

// Two separations; the (attractive) interaction must be stronger (more negative)
// at the smaller separation.
fn v_of_d(d, e_self) {
    fd = dipole_init(d)
    ed = relax_energy(pack_field(fd), 4)
    return ed - 2.0 * e_self
}
d_small = 4.0
d_large = 8.0
v_small = v_of_d(d_small, e1)
v_large = v_of_d(d_large, e1)
print("V(4 fm):")
print(v_small)
print("V(8 fm):")
print(v_large)
// Attractive Coulomb: V<0 and |V(small)| > |V(large)| => V(small) < V(large).
assert(v_small < v_large)
print("OK test_dipole")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$env:PYTHONUTF8='1'; python -m dedekind.compiler src\test_dipole.ddk 2>&1 | Select-Object -Last 14`
Expected: FAIL — `dipole_init` is defined (Task 2) but this test exercises the new
`exp_dipole.ddk` workflow; it fails first because the file does not yet exist / before
the experiment script is created. (If run standalone it should already pass logically;
create the experiment file next so the workflow is reusable.)

- [ ] **Step 3: Write the experiment** — `src/exp_dipole.ddk`

```
// experiments/exp_dipole.ddk — energy and interaction potential at one separation.
use relax
use field
use energy
use constants

fn single_self_energy() {
    f1 = hedgehog((NXf - 1.0) / 2.0, (NYf - 1.0) / 2.0, (NZf - 1.0) / 2.0, 1.0)
    return relax_energy(pack_field(f1), 2)
}

fn dipole_energy(d) {
    fd = dipole_init(d)
    return relax_energy(pack_field(fd), 4)
}

fn interaction(d, e_self) {
    return dipole_energy(d) - 2.0 * e_self
}

e_self = single_self_energy()
d = 6.0
v = interaction(d, e_self)
print("self energy [MeV]:")
print(e_self)
print("d [fm]:")
print(d)
print("V(d) [MeV]:")
print(v)
```

- [ ] **Step 4: Run both to verify they pass**

Run: `$env:PYTHONUTF8='1'; python -m dedekind.compiler src\test_dipole.ddk 2>&1 | Select-Object -Last 14`
Expected: PASS — `V(4 fm) < V(8 fm)` (attractive, stronger at short range), `OK test_dipole`.

Run: `$env:PYTHONUTF8='1'; python -m dedekind.compiler src\exp_dipole.ddk 2>&1 | Select-Object -Last 10`
Expected: prints self energy, `d`, and a (negative) `V(d)`.

- [ ] **Step 5: Commit**

```
git add src/exp_dipole.ddk src/test_dipole.ddk
git commit -m "Task 5: dipole energy E(d) and interaction V(d)"
```

---

### Task 6: Distance scan + Coulomb fit → α_sol⁻¹ (M2/M3)

**Files:**
- Create: `src/exp_scan.ddk`, `src/test_scan.ddk`

- [ ] **Step 1: Write the failing test** — `src/test_scan.ddk`

```
use relax
use field
use energy
use constants

// Fit V(d) ~ C*(1/d) + b at large d; alpha_sol = |C|/hbarc; expect order ~137.
fn self_e() {
    f1 = hedgehog((NXf - 1.0) / 2.0, (NYf - 1.0) / 2.0, (NZf - 1.0) / 2.0, 1.0)
    return relax_energy(pack_field(f1), 2)
}
fn v_at(d, es) {
    return relax_energy(pack_field(dipole_init(d)), 4) - 2.0 * es
}

es = self_e()
ds  = [5.0, 6.0, 7.0, 8.0]
inv = [1.0 / 5.0, 1.0 / 6.0, 1.0 / 7.0, 1.0 / 8.0]
v0 = v_at(ds[0], es)
v1 = v_at(ds[1], es)
v2 = v_at(ds[2], es)
v3 = v_at(ds[3], es)
vs = [v0, v1, v2, v3]
coeffs = polyfit(inv, vs, 1)        // [slope, intercept]
slope = coeffs[0]
alpha_sol = abs(slope) / HBARC
alpha_inv = 1.0 / alpha_sol
print("slope C [MeV*fm]:")
print(slope)
print("alpha_sol^-1:")
print(alpha_inv)
assert(alpha_inv > 60.0)
assert(alpha_inv < 260.0)
print("OK test_scan")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$env:PYTHONUTF8='1'; python -m dedekind.compiler src\test_scan.ddk 2>&1 | Select-Object -Last 14`
Expected: FAIL first if `polyfit` argument order differs — if so, inspect output and adjust
(`polyfit(x, y, degree)` returns highest-degree-first or lowest-first; pick the linear coeff
accordingly: print `coeffs` and select the term multiplying `1/d`). Otherwise FAIL only because
`exp_scan.ddk` is not yet created.

- [ ] **Step 3: Write the experiment** — `src/exp_scan.ddk`

```
// experiments/exp_scan.ddk — scan separations, Coulomb fit, report alpha_sol^-1.
use relax
use field
use energy
use constants

fn self_e() {
    f1 = hedgehog((NXf - 1.0) / 2.0, (NYf - 1.0) / 2.0, (NZf - 1.0) / 2.0, 1.0)
    return relax_energy(pack_field(f1), 2)
}
fn v_at(d, es) {
    return relax_energy(pack_field(dipole_init(d)), 4) - 2.0 * es
}

es = self_e()
inv = [1.0 / 5.0, 1.0 / 6.0, 1.0 / 7.0, 1.0 / 8.0]
vs  = [v_at(5.0, es), v_at(6.0, es), v_at(7.0, es), v_at(8.0, es)]
print("V values [MeV]:")
print(vs)
coeffs = polyfit(inv, vs, 1)
slope = coeffs[0]
alpha_sol = abs(slope) / HBARC
print("Coulomb coefficient |C| = alpha_sol*hbarc [MeV*fm]:")
print(abs(slope))
print("alpha_sol^-1 (compare CODATA 137.036):")
print(1.0 / alpha_sol)
```

- [ ] **Step 4: Run both to verify they pass**

Run: `$env:PYTHONUTF8='1'; python -m dedekind.compiler src\test_scan.ddk 2>&1 | Select-Object -Last 14`
Expected: PASS — `alpha_sol^-1` printed, in (60, 260), `OK test_scan`.

Run: `$env:PYTHONUTF8='1'; python -m dedekind.compiler src\exp_scan.ddk 2>&1 | Select-Object -Last 12`
Expected: prints the `V` values, the Coulomb coefficient, and `alpha_sol^-1` near ~137 (order).

- [ ] **Step 5: Commit**

```
git add src/exp_scan.ddk src/test_scan.ddk
git commit -m "Task 6: distance scan + Coulomb fit -> alpha_sol^-1 (M2/M3)"
```

---

### Task 7: README, results notes, optional LaTeX/repro

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Expand README with physics summary, scope, results, and run table**

```
# SU(2) Solitonic Dipole (Dedekind)

Teaching/demonstrative reproduction of arXiv:2604.12021 (Faber & Golubich,
"High-precision lattice determination of the interaction potential of an SU(2)
solitonic dipole", Model of Topological Fermions) in the Dedekind language.

## Model
- Unit-quaternion SU(2) field Q = (cos a, sin a * s * n_hat) on a small 3D grid.
- Curvature from finite differences: (d_i Q) conj(Q) = -i sigma . Gamma_i,
  R_ij = Gamma_i x Gamma_j.
- Energy E = a^3 * sum (alpha_f hbar c / 4pi)( 1/2 sum_{i<j} |R_ij|^2 + q0^6/r0^4 ).
- Relaxation: minimize(method="lbfgs") over the flattened field, norm-projected.

## Scope / deviations from the paper (teaching level)
- Small 3D Cartesian grid (constants.ddk: NX,NY,NZ,A) instead of the paper's large
  cylindrically symmetric 2D lattice; 2nd-order differences; ~6 r0 box.
- Goal: qualitative Coulomb V(d) and order-of-magnitude alpha_sol^-1 ~ 137,
  not the paper's keV precision.

## Run
    $env:PYTHONUTF8='1'
    python -m dedekind.compiler src\exp_soliton1d.ddk   # E0 = 0.511 MeV (closed form)
    python -m dedekind.compiler src\exp_dipole.ddk      # V at one separation
    python -m dedekind.compiler src\exp_scan.ddk        # scan + alpha_sol^-1

## Tests
    python -m dedekind.compiler src\test_constants.ddk
    python -m dedekind.compiler src\test_field.ddk
    python -m dedekind.compiler src\test_energy.ddk
    python -m dedekind.compiler src\test_relax.ddk
    python -m dedekind.compiler src\test_dipole.ddk
    python -m dedekind.compiler src\test_scan.ddk

## Convergence
Increase NX/NY/NZ and/or decrease A in src/constants.ddk to reduce discretization
and truncation error (cost grows ~ N^3). For the dipole scan, NZ should be large
enough that the largest separation plus both cores stay ~2-3 r0 from the z-boundary.
```

- [ ] **Step 2: (Optional) try the LaTeX / reproducibility flags**

Run: `$env:PYTHONUTF8='1'; python -m dedekind.compiler src\exp_soliton1d.ddk --latex 2>&1 | Select-Object -Last 20`
If `--latex` emits the energy expression, capture it into `docs/energy.tex` (best effort —
skip if the flag errors; it is not required for the result).

- [ ] **Step 3: Commit**

```
git add README.md docs
git commit -m "Task 7: README (model, scope, run/test table) + optional LaTeX"
```

---

## Self-Review

**Spec coverage:**
- Field as unit quaternion + norm projection → Tasks 2, 4 (differentiable normalize). ✓
- Energy `½Σ|R_ij|² + Λ`, `R_ij = Γ_i×Γ_j`, `Γ` from FD of Q → Task 3. ✓
- Relaxation via `minimize(lbfgs)` outer loop → Task 4. ✓
- Single-soliton anchor `E₀ = 0.511 MeV` → Task 1 (closed form) + Task 3 (grid, M1). ✓
- Dipole `V(d) = E(d) − 2E₀`, monotonic/attractive → Task 5 (M2). ✓
- Coulomb fit → `α_sol⁻¹ ≈ 137` → Task 6 (M3). ✓
- Dedekind features (Quaternion algebra in `qmul_field`, tensor FD, `minimize`, `polyfit`,
  constants, `assert`) → Tasks 2,3,4,6. `expm`/`Quantity`/LaTeX are optional extras (Task 7). ✓
- Deliberate simplifications documented → Task 7 README. ✓

**Placeholder scan:** No TBD/TODO; every code step is complete `.ddk`. Two flagged uncertainties
have explicit fallback instructions in-step: M1 magnitude (Task 3 Step 4) and `polyfit` coeff
order (Task 6 Step 2).

**Type/name consistency:** `hedgehog(cx,cy,cz,s)`, `qmul_field`, `normalize_field`, `dipole_init`,
`energy_components(q0,q1,q2,q3)`, `gamma_dir(...,dim)`, `cross_sq`, `dd(q,dim)`, `pack_field`,
`energy_flat`, `relax_energy(flat,outer)` — names used consistently across tasks. Field is always
`[q0,q1,q2,q3]` (a (4,NX,NY,NZ) tensor); flat vector length `NPTS4`. ✓

**Known residual risk (teaching scope):** exact numeric values depend on grid resolution; the
plan validates *behaviour and order of magnitude* (Coulomb shape, α⁻¹ ~ 137), not keV precision,
as agreed.
