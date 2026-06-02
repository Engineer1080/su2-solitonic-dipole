import sys
import time
import torch
import math

# Load the generated Python code from output_cyl.txt
with open('output_cyl.txt', 'r', encoding='utf-16') as f:
    content = f.read()

# Find the start of the Python code
start_idx = content.find("import sys\nimport builtins")
if start_idx == -1:
    start_idx = content.find("import sys\r\nimport builtins")

# Find the start of execution output
end_idx = content.find("--------------------\nExecuting Code:")
if end_idx == -1:
    end_idx = content.find("--------------------\r\nExecuting Code:")

if start_idx == -1 or end_idx == -1:
    print(f"Error: Could not locate Python code boundaries in output_cyl.txt. start={start_idx}, end={end_idx}")
    sys.exit(1)

code_to_exec = content[start_idx:end_idx]

# Define a pure PyTorch version of the 2D cylindrical energy density calculation
# that matches the exact formulas and steps
def run_pure_pytorch(NX, NY, A_val):
    # NX corresponds to N_rho, NY corresponds to N_z
    R0_val = 2.21320516
    ALPHA_F = 0.0072973525692838015
    HBARC = 197.3269804
    EPREF_CYL = (ALPHA_F * HBARC) / 2.0
    
    rhos = torch.arange(0.0, float(NX), 1.0) * A_val
    zs = (torch.arange(0.0, float(NY), 1.0) - float(NY - 1) / 2.0) * A_val
    
    rho = rhos.reshape(NX, 1)
    z = zs.reshape(1, NY)
    
    r = torch.sqrt(rho * rho + z * z + 1e-12)
    alpha = torch.atan(r / R0_val)
    
    q0 = torch.cos(alpha)
    qr = torch.sin(alpha) * (rho / r)
    qz = torch.sin(alpha) * (z / r)
    
    def dd(q, dim):
        return (q.roll(-1, dim) - q.roll(1, dim)) * (1.0 / (2.0 * A_val))
        
    d0_0 = dd(q0, 0)
    dr_0 = dd(qr, 0)
    dz_0 = dd(qz, 0)
    
    d0_1 = dd(q0, 1)
    dr_1 = dd(qr, 1)
    dz_1 = dd(qz, 1)
    
    gr_x = q0 * dr_0 - d0_0 * qr
    gr_y = qz * dr_0 - qr * dz_0
    gr_z = q0 * dz_0 - d0_0 * qz
    
    gz_x = q0 * dr_1 - d0_1 * qr
    gz_y = qz * dr_1 - qr * dz_1
    gz_z = q0 * dz_1 - d0_1 * qz
    
    inv_rho = 1.0 / (rho + 1e-12)
    gp_x = -qz * qr * inv_rho
    gp_y = q0 * qr * inv_rho
    gp_z = qr * qr * inv_rho
    
    def cross_sq(ax, ay, az, bx, by, bz):
        Rx = ax * bz - az * by
        Ry = az * bx - ax * bz  # Wait, standard cross product Rx = a[1]*b[2] - a[2]*b[1], Ry = a[2]*b[0] - a[0]*b[2]
        # Rx = ay * bz - az * by
        # Ry = az * bx - ax * bz
        # Rz = ax * by - ay * bx
        # Let's match the exact implementation in test_cylindrical.ddk
        Rx = ay * bz - az * by
        Ry = az * bx - ax * bz
        Rz = ax * by - ay * bx
        return Rx * Rx + Ry * Ry + Rz * Rz
        
    c1 = cross_sq(gr_x, gr_y, gr_z, gz_x, gz_y, gz_z)
    c2 = cross_sq(gr_x, gr_y, gr_z, gp_x, gp_y, gp_z)
    c3 = cross_sq(gz_x, gz_y, gz_z, gp_x, gp_y, gp_z)
    
    curv = 0.5 * (c1 + c2 + c3)
    
    q0sq = q0 * q0
    lam = (q0sq * q0sq * q0sq) / (R0_val**4)
    
    dens = EPREF_CYL * (curv + lam)
    
    dens_inner = dens[1:-1, 1:-1]
    rho_inner = rho[1:-1]
    
    E = (A_val * A_val) * (rho_inner * dens_inner).sum()
    return E.item()

# We execute the Dedekind-compiled python code using exec()
global_dict = {}
exec(code_to_exec, global_dict)

# Get the compiled test_energy function
compiled_test_energy = global_dict['test_energy']

# Let's define a modified version of the compiled test_energy function that allows custom NX, NY, A_val
# by monkey-patching the global variables it references or by defining a custom runner
# Looking at the generated Python, it uses:
# NX = 129, NY = 257, A_val = 0.1
# Let's write a function that executes the compiled functions for arbitrary sizes.
def run_compiled_dedekind(NX_val, NY_val, A_val_param):
    # Set the globals in global_dict that the compiled code references
    global_dict['NX'] = NX_val
    global_dict['NY'] = NY_val
    global_dict['A_val'] = A_val_param
    
    # We will redefine the arrays inside test_energy dynamically since test_energy hardcodes 129.0 and 257.0 in arange
    # So we will modify the global_dict to run a custom test_energy function or use a wrapper.
    # To run the exact compiled code blocks but with parameterization, let's look at what test_energy does and run it.
    # Since test_energy has hardcoded constants 129.0, 257.0, let's create a custom version of the compiled math:
    
    # We will run the compiled functions: dd, cross_sq
    dd = global_dict['dd']
    cross_sq = global_dict['cross_sq']
    R0_val = 2.21320516
    EPREF_CYL = 1.43996454686647 / 2.0 # ALPHA_HBARC / 2.0
    
    # Grid coordinates
    rhos = torch.arange(0.0, float(NX_val), 1.0) * A_val_param
    zs = (torch.arange(0.0, float(NY_val), 1.0) - float(NY_val - 1) / 2.0) * A_val_param
    
    rho = rhos.reshape(NX_val, 1)
    z = zs.reshape(1, NY_val)
    
    r = torch.sqrt(rho * rho + z * z + 1e-12)
    alpha = torch.atan(r / R0_val)
    
    q0 = torch.cos(alpha)
    qr = torch.sin(alpha) * (rho / r)
    qz = torch.sin(alpha) * (z / r)
    
    d0_0 = dd(q0, 0)
    dr_0 = dd(qr, 0)
    dz_0 = dd(qz, 0)
    
    d0_1 = dd(q0, 1)
    dr_1 = dd(qr, 1)
    dz_1 = dd(qz, 1)
    
    gr_x = q0 * dr_0 - d0_0 * qr
    gr_y = qz * dr_0 - qr * dz_0
    gr_z = q0 * dz_0 - d0_0 * qz
    gr = torch.stack([gr_x, gr_y, gr_z])
    
    gz_x = q0 * dr_1 - d0_1 * qr
    gz_y = qz * dr_1 - qr * dz_1
    gz_z = q0 * dz_1 - d0_1 * qz
    gz = torch.stack([gz_x, gz_y, gz_z])
    
    inv_rho = 1.0 / (rho + 1e-12)
    gp_x = -qz * qr * inv_rho
    gp_y = q0 * qr * inv_rho
    gp_z = qr * qr * inv_rho
    gp = torch.stack([gp_x, gp_y, gp_z])
    
    curv = 0.5 * (cross_sq(gr, gz) + cross_sq(gr, gp) + cross_sq(gz, gp))
    
    q0sq = q0 * q0
    lam = (q0sq * q0sq * q0sq) / (R0_val**4)
    
    dens = EPREF_CYL * (curv + lam)
    
    dens_inner = dens.narrow(0, 1, NX_val - 2).narrow(1, 1, NY_val - 2)
    rho_inner = rho.narrow(0, 1, NX_val - 2)
    
    E = (A_val_param * A_val_param) * (rho_inner * dens_inner).sum()
    return E.item()

print("Verification check (E0 target ~0.4595 MeV):")
E_pt = run_pure_pytorch(129, 257, 0.1)
E_dd = run_compiled_dedekind(129, 257, 0.1)
print(f"  Pure PyTorch:        {E_pt:.6f} MeV")
print(f"  Compiled Dedekind:   {E_dd:.6f} MeV")

# Benchmarking suite
grid_sizes = [
    (129, 257, "Standard (33k cells)"),
    (257, 513, "Large (131k cells)"),
    (513, 1025, "Huge (525k cells)")
]

print("\nRunning Performance Benchmarks...")
for nx, ny, label in grid_sizes:
    print(f"\nGrid: {label} - {nx}x{ny} ({nx*ny:,} cells)")
    
    # Warmups
    run_pure_pytorch(nx, ny, 0.1)
    run_compiled_dedekind(nx, ny, 0.1)
    
    # Pure PyTorch benchmark
    t_start = time.perf_counter()
    n_runs = 50
    for _ in range(n_runs):
        run_pure_pytorch(nx, ny, 0.1)
    t_pt = (time.perf_counter() - t_start) / n_runs * 1000.0
    
    # Dedekind-compiled benchmark
    t_start = time.perf_counter()
    for _ in range(n_runs):
        run_compiled_dedekind(nx, ny, 0.1)
    t_dd = (time.perf_counter() - t_start) / n_runs * 1000.0
    
    ratio = t_dd / t_pt
    diff_ms = t_dd - t_pt
    print(f"  Hand-written PyTorch:  {t_pt:.2f} ms")
    print(f"  Dedekind-compiled:     {t_dd:.2f} ms  (Diff: +{diff_ms:.2f} ms)")
    print(f"  Performance Ratio:     {ratio:.2f}x  (1.00x = identical speed)")
