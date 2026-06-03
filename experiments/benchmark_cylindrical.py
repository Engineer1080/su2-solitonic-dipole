import sys
import time
import torch
import math
import subprocess
import os

# Compile src/test_cylindrical.ddk on the fly if output_cyl.txt is missing
if not os.path.exists('output_cyl.txt'):
    print("Compiling src/test_cylindrical.ddk on the fly to capture AST...")
    with open('output_cyl.txt', 'wb') as f_out:
        res = subprocess.run([sys.executable, '-m', 'dedekind.compiler', 'src/test_cylindrical.ddk'], stdout=f_out, stderr=subprocess.PIPE)
        if res.returncode != 0:
            print("Error compiling src/test_cylindrical.ddk:")
            print(res.stderr.decode('utf-8', errors='replace'))
            sys.exit(1)

# Load the generated Python code
# We write as utf-8 now so let's read as utf-8 or try utf-16 if it was pre-existing
try:
    with open('output_cyl.txt', 'r', encoding='utf-8') as f:
        content = f.read()
except UnicodeDecodeError:
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
def run_pure_pytorch(NX, NY, A_val, device='cpu'):
    R0_val = 2.21320516
    ALPHA_F = 0.0072973525692838015
    HBARC = 197.3269804
    EPREF_CYL = (ALPHA_F * HBARC) / 2.0
    
    rhos = torch.arange(0.0, float(NX), 1.0, device=device) * A_val
    zs = (torch.arange(0.0, float(NY), 1.0, device=device) - float(NY - 1) / 2.0) * A_val
    
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

def run_compiled_dedekind(NX_val, NY_val, A_val_param, device='cpu'):
    dd = global_dict['dd']
    cross_sq = global_dict['cross_sq']
    R0_val = 2.21320516
    EPREF_CYL = 1.43996454686647 / 2.0
    
    rhos = torch.arange(0.0, float(NX_val), 1.0, device=device) * A_val_param
    zs = (torch.arange(0.0, float(NY_val), 1.0, device=device) - float(NY_val - 1) / 2.0) * A_val_param
    
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
E_pt = run_pure_pytorch(129, 257, 0.1, 'cpu')
E_dd = run_compiled_dedekind(129, 257, 0.1, 'cpu')
print(f"  Pure PyTorch (CPU):    {E_pt:.6f} MeV")
print(f"  Compiled Dedekind (CPU): {E_dd:.6f} MeV")

if torch.cuda.is_available():
    E_pt_gpu = run_pure_pytorch(129, 257, 0.1, 'cuda')
    E_dd_gpu = run_compiled_dedekind(129, 257, 0.1, 'cuda')
    print(f"  Pure PyTorch (CUDA):   {E_pt_gpu:.6f} MeV")
    print(f"  Compiled Dedekind (CUDA): {E_dd_gpu:.6f} MeV")

# Benchmarking targets
devices = ['cpu']
if torch.cuda.is_available():
    devices.append('cuda')

grid_sizes = [
    (129, 257, "Standard (33k cells)"),
    (257, 513, "Large (131k cells)"),
    (513, 1025, "Huge (525k cells)"),
    (1025, 2049, "Extreme (2.1M cells)")
]

print("\nRunning Performance Benchmarks...")
for dev in devices:
    print(f"\n================ DEVICE: {dev.upper()} ================")
    for nx, ny, label in grid_sizes:
        # Skip extreme grid on CPU to prevent long run times
        if dev == 'cpu' and nx > 513:
            continue
            
        print(f"\nGrid: {label} - {nx}x{ny} ({nx*ny:,} cells)")
        
        # Warmups
        run_pure_pytorch(nx, ny, 0.1, dev)
        run_compiled_dedekind(nx, ny, 0.1, dev)
        if dev == 'cuda':
            torch.cuda.synchronize()
        
        # Pure PyTorch benchmark
        t_start = time.perf_counter()
        n_runs = 50
        for _ in range(n_runs):
            run_pure_pytorch(nx, ny, 0.1, dev)
        if dev == 'cuda':
            torch.cuda.synchronize()
        t_pt = (time.perf_counter() - t_start) / n_runs * 1000.0
        
        # Dedekind-compiled benchmark
        t_start = time.perf_counter()
        for _ in range(n_runs):
            run_compiled_dedekind(nx, ny, 0.1, dev)
        if dev == 'cuda':
            torch.cuda.synchronize()
        t_dd = (time.perf_counter() - t_start) / n_runs * 1000.0
        
        ratio = t_dd / t_pt
        diff_ms = t_dd - t_pt
        print(f"  Hand-written PyTorch:  {t_pt:.2f} ms")
        print(f"  Dedekind-compiled:     {t_dd:.2f} ms  (Diff: +{diff_ms:.2f} ms)")
        print(f"  Performance Ratio:     {ratio:.2f}x  (1.00x = identical speed)")
