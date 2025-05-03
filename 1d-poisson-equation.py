import torch
import torch.nn as nn
import numpy as np

# Define the neural network
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(1, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )

    def forward(self, x):
        return self.hidden(x)

# Define the physics-informed loss
def physics_loss(model, x):
    x.requires_grad_(True)
    u = model(x)
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    f = -torch.sin(np.pi * x)  # Right-hand side of the Poisson equation
    return torch.mean((u_xx - f) ** 2)

# Define the boundary loss
def boundary_loss(model):
    u_0 = model(torch.tensor([[0.0]]))  # u(0)
    u_1 = model(torch.tensor([[1.0]]))  # u(1)
    return u_0**2 + u_1**2

# Training the PINN
def train_pinn():
    # Generate training points
    x_train = torch.linspace(0, 1, 100).view(-1, 1)

    # Initialize the model, optimizer, and loss function
    model = PINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(5000):
        optimizer.zero_grad()

        # Compute losses
        p_loss = physics_loss(model, x_train)
        b_loss = boundary_loss(model)
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
    x_test = torch.linspace(0, 1, 100).view(-1, 1)
    u_pred = model(x_test).detach().numpy()

    # Plot the results
    import matplotlib.pyplot as plt
    x_test_np = x_test.numpy()
    u_exact = -np.sin(np.pi * x_test_np) / (np.pi**2)  # Exact solution
    plt.plot(x_test_np, u_pred, label="PINN Prediction")
    plt.plot(x_test_np, u_exact, label="Exact Solution", linestyle="dashed")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.legend()
    plt.title("1D Poisson Equation Solved by PINN")
    plt.show()