import numpy as np
import deepxde as dde
import matplotlib.pyplot as plt

# ======================
# 1. Flat-Bottom Airfoil Geometry
# ======================
def flat_bottom_airfoil(x, t=0.12):
    """Flat-bottom airfoil with curved top (t = max thickness as fraction of chord)"""
    y_top = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
    y_bottom = np.zeros_like(x)  # Flat bottom
    return y_top, y_bottom

# Generate airfoil points
x_airfoil = np.linspace(0, 1, 100)
y_top, y_bottom = flat_bottom_airfoil(x_airfoil, t=0.12)  # 12% thickness

# ======================
# 2. Define Physics (Navier-Stokes)
# ======================
def navier_stokes_2D(x, y):
    u, v, p = y[:, 0:1], y[:, 1:2], y[:, 2:3]
    u_x = dde.grad.jacobian(y, x, i=0, j=0)
    u_y = dde.grad.jacobian(y, x, i=0, j=1)
    v_x = dde.grad.jacobian(y, x, i=1, j=0)
    v_y = dde.grad.jacobian(y, x, i=1, j=1)
    p_x = dde.grad.jacobian(y, x, i=2, j=0)
    p_y = dde.grad.jacobian(y, x, i=2, j=1)
    
    continuity = u_x + v_y
    momentum_x = u * u_x + v * u_y + p_x - 1e-3 * (dde.grad.hessian(y, x, i=0, j=0) + dde.grad.hessian(y, x, i=0, j=1))
    momentum_y = u * v_x + v * v_y + p_y - 1e-3 * (dde.grad.hessian(y, x, i=1, j=0) + dde.grad.hessian(y, x, i=1, j=1))
    return [continuity, momentum_x, momentum_y]

# ======================
# 3. Boundary Conditions
# ======================
chord_length = 0.08  # 80 mm
velocity = 3.0  # m/s

def boundary_airfoil(x, on_boundary):
    # Check if point is on the airfoil surface
    x_norm = x[0] / chord_length
    y_norm = x[1] / chord_length
    y_top, y_bottom = flat_bottom_airfoil(np.array([x_norm]))
    return on_boundary and (0 <= x_norm <= 1) and (y_bottom[0] <= y_norm <= y_top[0])

geom = dde.geometry.Rectangle([-0.1, -0.1], [1.1, 0.1])
bc_inflow = dde.DirichletBC(geom, lambda x: [velocity, 0, 0], lambda x, on_boundary: on_boundary and np.isclose(x[0], -0.1))
bc_airfoil = dde.DirichletBC(geom, lambda x: [0, 0, 0], boundary_airfoil)

# ======================
# 4. Compile and Train PINN
# ======================
data = dde.data.PDE(
    geom,
    navier_stokes_2D,
    [bc_inflow, bc_airfoil],
    num_domain=500,
    num_boundary=100,
)

model = dde.Model(data, dde.nn.FNN([2] + [64] * 5 + [3], "tanh", "Glorot normal"))
model.compile("adam", lr=1e-3)
model.train(iterations=2000)

# ======================
# 5. Plot Results
# ======================
plt.figure(figsize=(12, 4))

# Airfoil shape
plt.subplot(1, 2, 1)
plt.plot(x_airfoil * chord_length, y_top * chord_length, 'b-', label="Curved Top")
plt.plot(x_airfoil * chord_length, y_bottom * chord_length, 'k-', linewidth=2, label="Flat Bottom")
plt.fill_between(x_airfoil * chord_length, y_bottom * chord_length, y_top * chord_length, color='gray', alpha=0.3)
plt.title("Flat-Bottom Airfoil")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.axis('equal')
plt.legend()

# Pressure field
x_grid = np.linspace(0, chord_length, 50)
y_grid = np.linspace(-0.05, 0.05, 50)
X, Y = np.meshgrid(x_grid, y_grid)
xy_grid = np.vstack((X.flatten(), Y.flatten())).T
Z = model.predict(xy_grid)
P = Z[:, 2].reshape(X.shape)

plt.subplot(1, 2, 2)
plt.contourf(X, Y, P, levels=50, cmap="jet")
plt.colorbar(label="Pressure (Pa)")
plt.title("Pressure Field")
plt.xlabel("x (m)")
plt.ylabel("y (m)")

plt.tight_layout()
plt.show()