import numpy as np
from scipy.linalg import solve

# Data for Elements and Nodes
info = np.array(
    [
        [1, 2, 3, 4],
        [1, 2, 9, 10],
        [3, 4, 9, 10],
        [3, 4, 5, 6],
        [5, 6, 7, 8],
        [5, 6, 9, 10],
        [7, 8, 9, 10],
        [7, 8, 11, 12],
        [9, 10, 11, 12],
    ]
)

ele = np.array(
    [
        [1, 0, 0, 10, 20],
        [2, 0, 0, 20, 10],
        [3, 10, 20, 20, 10],
        [4, 16, 20, 20, 20],
        [5, 20, 20, 30, 20],
        [6, 20, 20, 20, 10],
        [7, 30, 20, 20, 10],
        [8, 30, 20, 40, 0],
        [9, 20, 10, 40, 0],
    ]
)

# Supports
supports = np.array([1, 2, 12])

# Forces at Nodes
forces = np.array(
    [
        [7, 20],
        [10, -50],
    ]
)

A = 10  # in^2
E = 29000  # ksi

# Preparing the stiffness matrix
enum = info.shape[0]
max_dof = np.max(info)
K_global = np.zeros((max_dof, max_dof))

# Setting up forces
F_global = np.zeros(max_dof)
for force in forces:
    F_global[force[0] - 1] = force[1]

# Element stiffness calculations
for i in range(enum):
    dx = ele[i, 3] - ele[i, 1]
    dy = ele[i, 4] - ele[i, 2]
    c = np.sqrt(dy**2 + dx**2)
    dy /= c
    dx /= c
    T = np.array([[dx, dy, 0, 0], [0, 0, dx, dy]])
    kee = (A * E) / (c * 12) * np.array([[1, -1], [-1, 1]])
    ke = T.T @ kee @ T

    # Assembly into the global stiffness matrix
    indices = (info[i, :] - 1).astype(int)
    K_global[np.ix_(indices, indices)] += ke

# Applying boundary conditions
K_mod = np.delete(np.delete(K_global, supports - 1, axis=0), supports - 1, axis=1)
F_mod = np.delete(F_global, supports - 1)

# Solving for displacements
displacements = solve(K_mod, F_mod)

# Initialize the full displacement array with zeros
full_displacements = np.zeros(max_dof)
# Place the calculated displacements into their correct positions
np.put(full_displacements, np.delete(np.arange(max_dof), supports - 1), displacements)

# Calculating reactions
reactions = K_global[supports - 1, :] @ full_displacements


# Output results
print("Displacements:")
print(full_displacements)
print("Reactions:")
print(reactions)
