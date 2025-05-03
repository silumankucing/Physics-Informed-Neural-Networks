import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Define the neural network
class PINNNavierStokes(nn.Module):
    def __init__(self):
        super(PINNNavierStokes, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(3, 50),  # Inputs: x, y, t
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 3)   # Outputs: u, v, p
        )

    def forward(self, x, y, t):
        inputs = torch.cat((x, y, t), dim=1)
        return self.hidden(inputs)

# Define the physics-informed loss
def physics_loss(model, x, y, t, nu):
    x.requires_grad_(True)
    y.requires_grad_(True)
    t.requires_grad_(True)

    # Predict u, v, p
    u, v, p = torch.split(model(x, y, t), 1, dim=1)

    # Compute gradients
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]

    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]

    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]

    # Navier-Stokes equations
    momentum_u = u_t + u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    momentum_v = v_t + u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
    continuity = u_x + v_y

    return torch.mean(momentum_u**2) + torch.mean(momentum_v**2) + torch.mean(continuity**2)

# Define the boundary loss
def boundary_loss(model, x_boundary, y_boundary, t_boundary):
    u_boundary, v_boundary, _ = torch.split(model(x_boundary, y_boundary, t_boundary), 1, dim=1)
    return torch.mean(u_boundary**2) + torch.mean(v_boundary**2)

# Training the PINN
def train_pinn():
    # Generate training points
    x_train = torch.rand(1000, 1)
    y_train = torch.rand(1000, 1)
    t_train = torch.rand(1000, 1)

    # Generate boundary points
    x_boundary = torch.cat((torch.zeros(250, 1), torch.ones(250, 1), torch.rand(500, 1)))
    y_boundary = torch.cat((torch.rand(250, 1), torch.rand(250, 1), torch.zeros(500, 1)))
    t_boundary = torch.rand(1000, 1)

    # Initialize the model, optimizer, and loss function
    model = PINNNavierStokes()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Viscosity (nu)
    nu = 0.01

    # Training loop
    for epoch in range(5000):
        optimizer.zero_grad()

        # Compute losses
        p_loss = physics_loss(model, x_train, y_train, t_train, nu)
        b_loss = boundary_loss(model, x_boundary, y_boundary, t_boundary)
        loss = p_loss + b_loss

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Print progress
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}, Physics Loss: {p_loss.item():.6f}, Boundary Loss: {b_loss.item():.6f}")

    return model

# Main function
if __name__ == "__main__":
    model = train_pinn()

    # Test the model
    x_test = torch.linspace(0, 1, 50)
    y_test = torch.linspace(0, 1, 50)
    t_test = torch.linspace(0, 1, 50)
    X, Y, T = torch.meshgrid(x_test, y_test, t_test, indexing='ij')
    x_test_flat = X.reshape(-1, 1)
    y_test_flat = Y.reshape(-1, 1)
    t_test_flat = T.reshape(-1, 1)

    u_pred, v_pred, p_pred = torch.split(model(x_test_flat, y_test_flat, t_test_flat).detach(), 1, dim=1)

    # Reshape for visualization
    u_pred = u_pred.reshape(50, 50, 50).numpy()
    v_pred = v_pred.reshape(50, 50, 50).numpy()
    p_pred = p_pred.reshape(50, 50, 50).numpy()

    # Visualize the results (e.g., velocity field at t=0.5)
    plt.quiver(X[:, :, 25].numpy(), Y[:, :, 25].numpy(), u_pred[:, :, 25], v_pred[:, :, 25])
    plt.title("Velocity Field at t=0.5")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()