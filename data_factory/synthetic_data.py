import numpy as np
import matplotlib.pyplot as plt

#Q criterion computing function
def compute_q(u_k, v_k, w_k, Kx, Ky, Kz):
    # Calculate gradients in Fourier space
    ux_k = 1j * Kx * u_k
    uy_k = 1j * Ky * u_k
    uz_k = 1j * Kz * u_k
    vx_k = 1j * Kx * v_k
    vy_k = 1j * Ky * v_k
    vz_k = 1j * Kz * v_k
    wx_k = 1j * Kx * w_k
    wy_k = 1j * Ky * w_k
    wz_k = 1j * Kz * w_k

    # Transform to physical space
    ux = np.fft.ifftn(ux_k).real
    uy = np.fft.ifftn(uy_k).real
    uz = np.fft.ifftn(uz_k).real
    vx = np.fft.ifftn(vx_k).real
    vy = np.fft.ifftn(vy_k).real
    vz = np.fft.ifftn(vz_k).real
    wx = np.fft.ifftn(wx_k).real
    wy = np.fft.ifftn(wy_k).real
    wz = np.fft.ifftn(wz_k).real

    # Calculate Q-criterion
    S_sq = 0.5*(ux**2 + vy**2 + wz**2) + 0.25*((uy + vx)**2 + (uz + wx)**2 + (vz + wy)**2)
    Omega_sq = 0.25*((uy - vx)**2 + (uz - wx)**2 + (vz - wy)**2)
    return (Omega_sq - S_sq)

#generate navier stokes in 4 dimensions
def generate_4D_navier_stokes(N, A, B, display_stats = False, data_type = "Q"):
    
    L = 2 * np.pi  
    dt = 0.01  
    visc = 0.001  
    nsteps = N 

    x = np.linspace(0, L, N, endpoint=False)
    y = np.linspace(0, L, N, endpoint=False)
    z = np.linspace(0, L, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    kx = 2 * np.pi * np.fft.fftfreq(N, d=L/N)
    ky = 2 * np.pi * np.fft.fftfreq(N, d=L/N)
    kz = 2 * np.pi * np.fft.fftfreq(N, d=L/N)
    Kx, Ky, Kz = np.meshgrid(kx, ky, kz, indexing='ij')

    K_sq = Kx**2 + Ky**2 + Kz**2
    K_sq = np.where(K_sq == 0, 1e-9, K_sq)

    #ABC flow
    C = 1
    u = A*np.sin(Z) + C*np.cos(Y) + B*(np.sin(X) + np.cos(X))
    v = np.zeros_like(X)
    w = np.zeros_like(X)

    u_k = np.fft.fftn(u)
    v_k = np.fft.fftn(v)
    w_k = np.fft.fftn(w)

    data = np.zeros(N, N, N, N)

    for step in range(nsteps):
        u = np.fft.ifftn(u_k).real
        v = np.fft.ifftn(v_k).real
        w = np.fft.ifftn(w_k).real
        
        ux_k = 1j * Kx * u_k
        uy_k = 1j * Ky * u_k
        uz_k = 1j * Kz * u_k
        
        vx_k = 1j * Kx * v_k
        vy_k = 1j * Ky * v_k
        vz_k = 1j * Kz * v_k
        
        wx_k = 1j * Kx * w_k
        wy_k = 1j * Ky * w_k
        wz_k = 1j * Kz * w_k
        
        ux = np.fft.ifftn(ux_k).real
        uy = np.fft.ifftn(uy_k).real
        uz = np.fft.ifftn(uz_k).real
        
        vx = np.fft.ifftn(vx_k).real
        vy = np.fft.ifftn(vy_k).real
        vz = np.fft.ifftn(vz_k).real
        
        wx = np.fft.ifftn(wx_k).real
        wy = np.fft.ifftn(wy_k).real
        wz = np.fft.ifftn(wz_k).real
        
        conv_u = u * ux + v * uy + w * uz
        conv_v = u * vx + v * vy + w * vz
        conv_w = u * wx + v * wy + w * wz
        
        conv_u_k = np.fft.fftn(conv_u)
        conv_v_k = np.fft.fftn(conv_v)
        conv_w_k = np.fft.fftn(conv_w)
        
        # Project convective terms to enforce incompressibility
        div_conv_k = Kx * conv_u_k + Ky * conv_v_k + Kz * conv_w_k
        conv_u_k -= (Kx * div_conv_k) / K_sq
        conv_v_k -= (Ky * div_conv_k) / K_sq
        conv_w_k -= (Kz * div_conv_k) / K_sq
        
        # crank-nicolson for viscous term
        denominator = 1 + 0.5 * dt * visc * K_sq
        u_k = (u_k * (1 - 0.5 * dt * visc * K_sq) - dt * conv_u_k) / denominator
        v_k = (v_k * (1 - 0.5 * dt * visc * K_sq) - dt * conv_v_k) / denominator
        w_k = (w_k * (1 - 0.5 * dt * visc * K_sq) - dt * conv_w_k) / denominator
        
        u_k[0, 0, 0] = 1
        v_k[0, 0, 0] = 1
        w_k[0, 0, 0] = 1

        # Compute velocity magnitude
        if data_type == "velocity_mag":
            u_final = np.fft.ifftn(u_k).real
            v_final = np.fft.ifftn(v_k).real
            w_final = np.fft.ifftn(w_k).real
            velocity_magnitude = np.sqrt(u_final**2 + v_final**2 + w_final**2)
            data[step, :, :, :] = velocity_magnitude

        elif data_type == "Q":
            Q = compute_q(u_k, v_k, w_k, Kx, Ky, Kz)  
            data[step, :, :, :] = Q

        else:
            print("unknown data type!")
            exit()

        if display_stats:
            # Compute kinetic energy
            u_phys = np.fft.ifftn(u_k).real
            v_phys = np.fft.ifftn(v_k).real
            w_phys = np.fft.ifftn(w_k).real
            energy = 0.5 * (u_phys**2 + v_phys**2 + w_phys**2).mean()
            print(f"Step: {step}/{nsteps}, Time: {step*dt:.2f}, Energy: {energy:.6f}")

    return data
    
#generate navier stokes in 3 dimensions
def generate_3D_navier_stokes(N, A, B, display_stats = False, data_type = "velocity_mag"):
    

    L = 2 * np.pi  
    dt = 0.01  
    visc = 0.001  
    nsteps = 32 

    x = np.linspace(0, L, N, endpoint=False)
    y = np.linspace(0, L, N, endpoint=False)
    z = np.linspace(0, L, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    kx = 2 * np.pi * np.fft.fftfreq(N, d=L/N)
    ky = 2 * np.pi * np.fft.fftfreq(N, d=L/N)
    kz = 2 * np.pi * np.fft.fftfreq(N, d=L/N)
    Kx, Ky, Kz = np.meshgrid(kx, ky, kz, indexing='ij')

    K_sq = Kx**2 + Ky**2 + Kz**2
    K_sq = np.where(K_sq == 0, 1e-9, K_sq)

    #ABC flow
    C = 1
    u = A*np.cos(Z) + B*np.cos(Y) + C*np.cos(X)
    v = np.zeros_like(X)
    w = np.zeros_like(X)

    u_k = np.fft.fftn(u)
    v_k = np.fft.fftn(v)
    w_k = np.fft.fftn(w)


    for step in range(nsteps):
        u = np.fft.ifftn(u_k).real
        v = np.fft.ifftn(v_k).real
        w = np.fft.ifftn(w_k).real
        
        ux_k = 1j * Kx * u_k
        uy_k = 1j * Ky * u_k
        uz_k = 1j * Kz * u_k
        
        vx_k = 1j * Kx * v_k
        vy_k = 1j * Ky * v_k
        vz_k = 1j * Kz * v_k
        
        wx_k = 1j * Kx * w_k
        wy_k = 1j * Ky * w_k
        wz_k = 1j * Kz * w_k
        
        ux = np.fft.ifftn(ux_k).real
        uy = np.fft.ifftn(uy_k).real
        uz = np.fft.ifftn(uz_k).real
        
        vx = np.fft.ifftn(vx_k).real
        vy = np.fft.ifftn(vy_k).real
        vz = np.fft.ifftn(vz_k).real
        
        wx = np.fft.ifftn(wx_k).real
        wy = np.fft.ifftn(wy_k).real
        wz = np.fft.ifftn(wz_k).real
        
        conv_u = u * ux + v * uy + w * uz
        conv_v = u * vx + v * vy + w * vz
        conv_w = u * wx + v * wy + w * wz
        
        conv_u_k = np.fft.fftn(conv_u)
        conv_v_k = np.fft.fftn(conv_v)
        conv_w_k = np.fft.fftn(conv_w)
        
        # Project convective terms to enforce incompressibility
        div_conv_k = Kx * conv_u_k + Ky * conv_v_k + Kz * conv_w_k
        conv_u_k -= (Kx * div_conv_k) / K_sq
        conv_v_k -= (Ky * div_conv_k) / K_sq
        conv_w_k -= (Kz * div_conv_k) / K_sq
        
        # crank-nicolson for viscous term
        denominator = 1 + 0.5 * dt * visc * K_sq
        u_k = (u_k * (1 - 0.5 * dt * visc * K_sq) - dt * conv_u_k) / denominator
        v_k = (v_k * (1 - 0.5 * dt * visc * K_sq) - dt * conv_v_k) / denominator
        w_k = (w_k * (1 - 0.5 * dt * visc * K_sq) - dt * conv_w_k) / denominator
        
        u_k[0, 0, 0] = 1
        v_k[0, 0, 0] = 1
        w_k[0, 0, 0] = 1
        
        if display_stats:
            # Compute kinetic energy
            u_phys = np.fft.ifftn(u_k).real
            v_phys = np.fft.ifftn(v_k).real
            w_phys = np.fft.ifftn(w_k).real
            energy = 0.5 * (u_phys**2 + v_phys**2 + w_phys**2).mean()
            print(f"Step: {step}/{nsteps}, Time: {step*dt:.2f}, Energy: {energy:.6f}")


    # Compute velocity magnitude
    if data_type == "velocity_mag":
        u_final = np.fft.ifftn(u_k).real
        v_final = np.fft.ifftn(v_k).real
        w_final = np.fft.ifftn(w_k).real
        velocity_magnitude = np.sqrt(u_final**2 + v_final**2 + w_final**2)

        return velocity_magnitude
    elif data_type == "Q":
        Q = compute_q(u_k, v_k, w_k, Kx, Ky, Kz)  
        print("maximum Q: ", np.max(np.abs(Q)))
        return Q
    
    else:
        print("unknown data type!")
        return None



def generate_taylor_green_pressure(nu = 0.3, t = 1, rho = 1, M = 75):
    """
    Solves the 2D heat equation with Dirichlet boundary conditions and returns the temperature distribution.

    Parameters:
    - M: Number of grid points in each direction (default: 75).
    - U_base: Base velocity vector (default: [0.5, 0]).
    - kappa: Thermal diffusivity (default: 0.0025).
    - boundary_temp: Base temperature at the boundary (default: 300).
    - amplitude: Amplitude of the sinusoidal temperature perturbation (default: 325).

    Returns:
    - Tau: 2D temperature distribution as a numpy array.
    """
    # Define the size of the domain and the distance between adjacent nodes
    x = np.linspace(-np.pi/2, np.pi/2, M)
    y = np.linspace(-np.pi/2, np.pi/2, M)
    
    X, Y = np.meshgrid(x, y)
    F_t = np.exp(-2*nu*t)
    P = rho/4*(np.cos(2*X) + np.cos(2*Y) + 2)*F_t
    
    return P


def generate_burgers_velocity(nu = 0.3, mu = 1, N = 100):
    
    L = 2 * np.pi
    x = np.linspace(0, 2 * np.pi - (2 * np.pi) / N, N, endpoint=False)
    ux_t0 = mu * np.exp(np.sin(x + np.pi / 4))
    #ux_t0 = np.tanh(0.01*x)
    h = 0.0008
    max_time = 0.6
    k = np.fft.fftfreq(N) * N  # Wavenumbers

    # Initialize solution array (N x 64)
    solutions = np.zeros((N, N), dtype=np.complex128)
    u_k = np.fft.fft(ux_t0)
    solutions[:, 0] = u_k.copy()

    # Time points to save
    t_save = np.linspace(0, max_time - h, N)
    save_idx = 1  # Next index to save (0 is already filled)

    current_time = 0.0
    num_steps = int(np.ceil(max_time / h))

    for _ in range(num_steps):
        # Compute RK4 stages
        k1 = calc_next_k(u_k, np.zeros_like(u_k), 1, k, h, nu)
        k2 = calc_next_k(u_k, k1, 2, k, h, nu)
        k3 = calc_next_k(u_k, k2, 2, k, h, nu)
        k4 = calc_next_k(u_k, k3, 1, k, h, nu)

        # Update u_k
        u_k += (k1 + 2 * k2 + 2 * k3 + k4) / 6

        # Update current time
        current_time += h

        # Save solutions if current_time has passed the next save time
        while save_idx < N and current_time >= t_save[save_idx]:
            solutions[:, save_idx] = u_k.copy()
            save_idx += 1

    # Convert to real space
    solutions_real = np.real(np.fft.ifft(solutions, axis=0))
    return solutions_real

def calc_next_k(prev_u_k, prev_k, factor, k, h, nu):
    
    u_k_next = prev_u_k + prev_k / factor
    u_x = np.fft.ifft(1j * k * u_k_next).real
    u = np.fft.ifft(u_k_next).real
    nonlinear_term = u * u_x
    rho_next = -np.fft.fft(nonlinear_term) - nu * (k**2) * u_k_next
    return h * rho_next


def calc_magnetic_field(magnet_center_1):
    
    # Define grid size and resolution
    Nx, Ny = 75, 75  # Grid points in x and y
    Lx, Ly = 1.0, 1.0  # Physical size in meters
    dx = dy = Lx / (Nx - 1)  # Grid spacing (adjusted for inclusive endpoints)

    map()
    
    # Physical constants
    mu0 = 4 * np.pi * 1e-7  # Permeability of free space
    Jz = np.zeros((Nx, Ny))  # Current density (A/m^2)

    
    # Define two rectangular magnets with correct current directions
    magnet1 = (slice(10, 31), slice(10, 31))  # +J (0-based indexing, inclusive)
    magnet2 = (slice(20, 41), slice(20, 41))  # -J
    Jz[magnet1] = 1e6
    Jz[magnet2] = -1e6

    # Solve Poisson's equation using vectorized Jacobi iterations
    Az = np.zeros((Nx, Ny))
    coeff = mu0 * dx * dy  # Precompute constant coefficient

    # Jacobi iteration with vectorized operations
    for _ in range(2000):  # Fewer iterations needed due to faster convergence
        Az_new = Az.copy()
        # Update interior points using vectorized operations
        Az_new[1:-1, 1:-1] = 0.25 * (
            Az[2:, 1:-1] + Az[:-2, 1:-1] +  # x-direction neighbors
            Az[1:-1, 2:] + Az[1:-1, :-2] +  # y-direction neighbors
            Jz[1:-1, 1:-1] * coeff  # Source term (sign corrected)
        )
        Az = Az_new

    # Compute B field components using central differences
    Bx = np.zeros_like(Az)
    By = np.zeros_like(Az)

    # Vectorized field calculations
    Bx[:, 1:-1] = (Az[:, 2:] - Az[:, :-2]) / (2 * dy)
    By[1:-1, :] = -(Az[2:, :] - Az[:-2, :]) / (2 * dx)

    # Compute magnitude and prepare for visualization
    B_magnitude = np.sqrt(Bx**2 + By**2)

    # Create grid for plotting with correct dimensions
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y)

    return B_magnitude


# Example usage
if __name__ == "__main__":
    # Solve the heat equation
    P = generate_burgers_velocity(0.3, 1, 100)

    # Plot the heatmap
    plt.imshow(P, cmap='coolwarm', interpolation='nearest', origin='lower')
    plt.colorbar(label='Velocity $u$')
    plt.title('1D Velocity Over time')
    plt.xlabel('t')
    plt.ylabel('X')
    plt.show()