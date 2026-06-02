import torch
import time
import math

NX, NY, NZ = 65, 65, 65
a = 0.5
R0 = 2.21320516
HBARC = 197.3269804
ALPHA_F = 0.0072973525692838015
EPREF = (ALPHA_F * HBARC) / (4.0 * math.pi)

def run_pure_pytorch():
    # 1. Grid setup
    cx, cy, cz = 32.0, 32.0, 32.0
    xs = (torch.arange(0.0, float(NX), 1.0) - cx) * a
    ys = (torch.arange(0.0, float(NY), 1.0) - cy) * a
    zs = (torch.arange(0.0, float(NZ), 1.0) - cz) * a
    x3 = xs.reshape(NX, 1, 1)
    y3 = ys.reshape(1, NY, 1)
    z3 = zs.reshape(1, 1, NZ)
    
    r = torch.sqrt(x3*x3 + y3*y3 + z3*z3 + 1e-12)
    alpha = torch.atan(r / R0)
    ca = torch.cos(alpha)
    sa = torch.sin(alpha)
    
    q0 = ca
    q1 = sa * (x3 / r)
    q2 = sa * (y3 / r)
    q3 = sa * (z3 / r)
    
    # 2. Central difference helper
    def dd(q, dim):
        return (q.roll(-1, dim) - q.roll(1, dim)) * (1.0 / (2.0 * a))
        
    # 3. Gamma
    def gamma_dir(dim):
        d0 = dd(q0, dim)
        d1 = dd(q1, dim)
        d2 = dd(q2, dim)
        d3 = dd(q3, dim)
        Gx = q0 * d1 - d0 * q1 + (d3 * q2 - d2 * q3)
        Gy = q0 * d2 - d0 * q2 + (d1 * q3 - d3 * q1)
        Gz = q0 * d3 - d0 * q3 + (d2 * q1 - d1 * q2)
        return Gx, Gy, Gz
        
    gx = gamma_dir(0)
    gy = gamma_dir(1)
    gz = gamma_dir(2)
    
    # 4. Cross squared
    def cross_sq(A, B):
        Rx = A[1] * B[2] - A[2] * B[1]
        Ry = A[2] * B[0] - A[0] * B[2]
        Rz = A[0] * B[1] - A[1] * B[0]
        return Rx*Rx + Ry*Ry + Rz*Rz
        
    curv = 0.5 * (cross_sq(gx, gy) + cross_sq(gx, gz) + cross_sq(gy, gz))
    
    q0sq = q0 * q0
    R0_4 = R0**4
    lam = (q0sq * q0sq * q0sq) / R0_4
    
    dens = EPREF * (curv + lam)
    
    # 5. E_out outside R
    di = dens[1:-1, 1:-1, 1:-1]
    ri = r[1:-1, 1:-1, 1:-1]
    
    def get_e_out(R):
        mask = (ri > R)
        return (a**3) * (di * mask).sum()
        
    Es = torch.tensor([get_e_out(2.0).item(), get_e_out(2.5).item(), get_e_out(3.0).item(), get_e_out(3.5).item(), get_e_out(4.0).item()])
    Rs = torch.tensor([2.0, 2.5, 3.0, 3.5, 4.0])
    
    # 6. Fit
    C = torch.tensor([0.5], requires_grad=True)
    optimizer = torch.optim.SGD([C], lr=0.1)
    for _ in range(200):
        optimizer.zero_grad()
        diff = (C / Rs) - Es
        loss = (diff * diff).sum()
        loss.backward()
        optimizer.step()
        
    C_opt = C.item()
    alpha_inv = HBARC / (2.0 * C_opt)
    return C_opt, alpha_inv

print("Benchmarking pure PyTorch implementation...")
# Warmup
run_pure_pytorch()

t_start = time.perf_counter()
n_runs = 20
for i in range(n_runs):
    C_opt, alpha_inv = run_pure_pytorch()
t_end = time.perf_counter()

avg_time_ms = ((t_end - t_start) / n_runs) * 1000.0
print(f"Average execution time over {n_runs} runs: {avg_time_ms:.2f} ms")
print(f"Optimized C: {C_opt:.4f} MeV*fm")
print(f"Fitted alpha_sol^-1: {alpha_inv:.4f}")
