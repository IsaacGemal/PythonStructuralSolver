import torch

# Data for Elements and Nodes
info = torch.tensor([
    [1, 2, 3, 4],
    [1, 2, 9, 10],
    [3, 4, 9, 10],
    [3, 4, 5, 6],
    [5, 6, 7, 8],
    [5, 6, 9, 10],
    [7, 8, 9, 10],
    [7, 8, 11, 12],
    [9, 10, 11, 12],
], dtype=torch.long)

ele = torch.tensor([
    [1, 0, 0, 10, 20],
    [2, 0, 0, 20, 10],
    [3, 10, 20, 20, 10],
    [4, 16, 20, 20, 20],
    [5, 20, 20, 30, 20],
    [6, 20, 20, 20, 10],
    [7, 30, 20, 20, 10],
    [8, 30, 20, 40, 0],
    [9, 20, 10, 40, 0],
], dtype=torch.float)

# Supports
supports = torch.tensor([1, 2, 12], dtype=torch.long)

# Forces at Nodes
forces = torch.tensor([
    [7, 20],
    [10, -50],
], dtype=torch.float)

A = 10.0  # in^2
E = 29000.0  # ksi

# Preparing the stiffness matrix
enum = info.shape[0]
max_dof = torch.max(info).item()
K_global = torch.zeros((max_dof, max_dof), dtype=torch.float)

# Setting up forces
F_global = torch.zeros(max_dof, dtype=torch.float)
F_global[forces[:, 0].long() - 1] = forces[:, 1]

# Element stiffness calculations
for i in range(enum):
    dx = ele[i, 3] - ele[i, 1]
    dy = ele[i, 4] - ele[i, 2]
    c = torch.sqrt(dy**2 + dx**2)
    dy /= c
    dx /= c
    T = torch.tensor([[dx, dy, 0, 0], [0, 0, dx, dy]], dtype=torch.float)
    kee = (A * E) / (c * 12) * torch.tensor([[1.0, -1.0], [-1.0, 1.0]], dtype=torch.float)
    ke = T.t() @ kee @ T

    # Assembly into the global stiffness matrix
    indices = (info[i, :] - 1).long()
    K_global[torch.meshgrid(indices, indices, indexing='ij')] += ke

# Applying boundary conditions
mask = torch.ones(max_dof, dtype=torch.bool)
mask[supports - 1] = False
K_mod = K_global[mask][:, mask]
F_mod = F_global[mask]

# Solving for displacements
displacements = torch.linalg.solve(K_mod, F_mod)

# Initialize the full displacement array with zeros
full_displacements = torch.zeros(max_dof, dtype=torch.float)
# Place the calculated displacements into their correct positions
full_displacements[mask] = displacements

# Calculating reactions
reactions = K_global[supports - 1] @ full_displacements

# Output results
print("Displacements:")
print(full_displacements)
print("Reactions:")
print(reactions)