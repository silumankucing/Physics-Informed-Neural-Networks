import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Define the neural network
class PINN2D(nn.Module):
    def __init__(self):
        super(PINN2D, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1)
        )

    def forward(self, x, y):
        inputs = torch.cat((x, y), dim=1)
        return self.hidden(inputs)

# Define the physics-informed loss
def physics_loss(model, x, y):
    x.requires_grad_(True)
    y.requires_grad_(True)
    u = model(x, y)
    
    # Compute gradients
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    
    # Poisson equation residual
    f = -2 * (np.pi**2) * torch.sin(np.pi * x) * torch.sin(np.pi * y)
    return torch.mean((u_xx + u_yy - f) ** 2)

# Define the boundary loss
def boundary_loss(model, x_boundary, y_boundary):
    u_boundary = model(x_boundary, y_boundary)
    return torch.mean(u_boundary ** 2)

# Training the PINN
def train_pinn():
    # Generate training points
    x_train = torch.rand(1000, 1)
    y_train = torch.rand(1000, 1)

    # Generate boundary points
    x_boundary = torch.cat((torch.zeros(250, 1), torch.ones(250, 1), torch.rand(500, 1)))
    y_boundary = torch.cat((torch.rand(250, 1), torch.rand(250, 1), torch.zeros(500, 1)))

    # Initialize the model, optimizer, and loss function
    model = PINN2D()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(5000):
        optimizer.zero_grad()

        # Compute losses
        p_loss = physics_loss(model, x_train, y_train)
        b_loss = boundary_loss(model, x_boundary, y_boundary)
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
    X, Y = torch.meshgrid(x_test, y_test, indexing='ij')
    x_test_flat = X.reshape(-1, 1)
    y_test_flat = Y.reshape(-1, 1)

    u_pred = model(x_test_flat, y_test_flat).detach().numpy().reshape(50, 50)

    # Exact solution
    u_exact = np.sin(np.pi * X.numpy()) * np.sin(np.pi * Y.numpy()) / (np.pi**2)

    # Plot the results
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].imshow(u_pred, extent=[0, 1, 0, 1], origin='lower', cmap='viridis')
    ax[0].set_title("PINN Prediction")
    ax[1].imshow(u_exact, extent=[0, 1, 0, 1], origin='lower', cmap='viridis')
    ax[1].set_title("Exact Solution")
    plt.show()