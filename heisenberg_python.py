import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns  # For heatmap plotting

# Define parameters
spin1 = 1
spin2 = 1
D1 = 1   # in meV
D2 = 2   # in meV
B = 0    # in Tesla

def heisenberg(J, spin1, spin2, D1, D2, B):
    u = '↑'
    d = '↓'
    h = 1
    if spin1 == 0.5:
        D1 = 0
    if spin2 == 0.5:
        D2 = 0
    
    # Spin 1/2 (2x2 matrices)
    x_1_2 = np.array([[0, 1], [1, 0]])  # Pauli X
    y_1_2 = np.array([[0, -1j], [1j, 0]])  # Pauli Y
    z_1_2 = np.array([[1, 0], [0, -1]])  # Pauli Z
    base1_2 = [['1/2 '], ['-1/2 ']]
    
    # Spin 1 (3x3 matrices)
    sqrt2_inv = 1 / np.sqrt(2)  # 1/sqrt(2)
    x_1 = sqrt2_inv * np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])  # Pauli X for S=1
    y_1 = sqrt2_inv * np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]])  # Pauli Y for S=1
    z_1 = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])  # Pauli Z for S=1
    base_1 = [['1 '], ['0 '], ['-1 ']]
   
    # Spin 3/2 (4x4 matrices)
    sqrt3_inv = 1 / np.sqrt(3)  # 1/sqrt(3)
    x_3_2 = sqrt3_inv * np.array([[0, 1, 0, 0],
                                  [1, 0, 1, 0],
                                  [0, 1, 0, 1],
                                  [0, 0, 1, 0]])
    
    y_3_2 = sqrt3_inv * np.array([[0, -1j, 0, 0],
                                  [1j, 0, -1j, 0],
                                  [0, 1j, 0, -1j],
                                  [0, 0, 1j, 0]])
    
    z_3_2 = np.array([[3/2, 0, 0, 0],
                      [0, 1/2, 0, 0],
                      [0, 0, -1/2, 0],
                      [0, 0, 0, -3/2]])
    base3_2 = [['3/2 '], ['1/2 '], ['-1/2 '], ['-3/2 ']]
   
    # Pauli for S=2
    sqrt6_inv = 1 / np.sqrt(6)  # 1/sqrt(6)

    x_2 = sqrt6_inv * np.array([[0, 1, 0, 0, 0],
                                [1, 0, 1, 0, 0],
                                [0, 1, 0, 1, 0],
                                [0, 0, 1, 0, 1],
                                [0, 0, 0, 1, 0]])
    
    y_2 = sqrt6_inv * np.array([[0, -1j, 0, 0, 0],
                                [1j, 0, -1j, 0, 0],
                                [0, 1j, 0, -1j, 0],
                                [0, 0, 1j, 0, -1j],
                                [0, 0, 0, 1j, 0]])
    
    z_2 = np.array([[2, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, -1, 0],
                    [0, 0, 0, 0, -2]])
    base_2 = [['2 '], ['1 '], ['0 '], ['-1 '], ['-2 ']]
    
    def set_spin_matrices(spin):
        if spin == 0.5:
            x1 = x_1_2
            y1 = y_1_2
            z1 = z_1_2
        elif spin == 1:
            x1 = x_1
            y1 = y_1
            z1 = z_1
        elif spin == 1.5:
            x1 = x_3_2
            y1 = y_3_2
            z1 = z_3_2
        elif spin == 2:
            x1 = x_2
            y1 = y_2
            z1 = z_2
        else:
            raise ValueError("Invalid spin value.")
        return x1, y1, z1

    def base(spin):
        if spin == 0.5:
            base = base1_2
        elif spin == 1:
            base = base_1
        elif spin == 1.5:
            base = base3_2
        elif spin == 2:
            base = base_2
        return base

    base1 = np.array(base(spin1))
    base2 = np.array(base(spin2))
    
    x1, y1, z1 = set_spin_matrices(spin1)
    x2, y2, z2 = set_spin_matrices(spin2)
    
    S1 = [h / 2 * np.kron(x1, np.eye(x2.shape[1])), h / 2 * np.kron(y1, np.eye(x2.shape[1])), h / 2 * np.kron(z1, np.eye(x2.shape[1]))]
    S2 = [h / 2 * np.kron(np.eye(x1.shape[1]), x2), h / 2 * np.kron(np.eye(x1.shape[1]), y2), h / 2 * np.kron(np.eye(x1.shape[1]), z2)]

    Sz_1 = np.dot(np.kron(z1, np.eye(x2.shape[1])), np.kron(np.eye(x2.shape[1]), z1))
    Sx_1 = np.dot(np.kron(x1, np.eye(x2.shape[1])), np.kron(np.eye(x2.shape[1]), x1))
    Sy_1 = np.dot(np.kron(y1, np.eye(x2.shape[1])), np.kron(np.eye(x2.shape[1]), y1))

    Sz_2 = np.dot(np.kron(z2, np.eye(x1.shape[1])), np.kron(np.eye(x1.shape[1]), z2))
    Sx_2 = np.dot(np.kron(x2, np.eye(x1.shape[1])), np.kron(np.eye(x1.shape[1]), x2))
    Sy_2 = np.dot(np.kron(y2, np.eye(x1.shape[1])), np.kron(np.eye(x1.shape[1]), y2))
    
    B = 0.01 * B
    Bsz1 = np.kron(z1, np.eye(x2.shape[1]))
    Bsz2 = np.kron(z2, np.eye(x1.shape[1]))

    # Calculate H
    H = J * (np.dot(S1[0], S2[0]) + np.dot(S1[1], S2[1]) + np.dot(S1[2], S2[2])) + Sz_1 * D1 + Sz_2 * D2 + B * (Bsz1 + Bsz2)

    # Compute eigenvalues and eigenvectors
    E, ket = np.linalg.eigh(H)

    # Round and extract the real part of eigenvalues
    E = np.round(E, 2).real
    ket = np.round(ket, 2).real
    E_diag = np.diag(np.round(E, 1).real)

    def kronecker_product_str(A, B):
        # Get the dimensions of both matrices
        rows_A, cols_A = A.shape
        rows_B, cols_B = B.shape

        # Initialize an empty list to store the result
        result = []

        # Loop through each element of A and B to create the Kronecker product
        for i in range(rows_A):
            for j in range(cols_A):
                # Create a block for the Kronecker product
                block = [[A[i, j] + B[m, n] for n in range(cols_B)] for m in range(rows_B)]
                result.extend(block)

        # Convert result back to a NumPy array for easier manipulation
        result = np.array(result)

        return result

    base_tot = kronecker_product_str(base1, base2)
    
    return E, ket, E_diag, base_tot

# Run the Heisenberg model calculation
result = heisenberg(2, spin1=spin1, spin2=spin2, D1=D1, D2=D2, B=B)
E, ket, E_diag, base_tot = result

# Define the range of J values
JJ = np.linspace(0, 2, 150)
eigenvalues = np.array([heisenberg(j, spin1=spin1, spin2=spin2, D1=D1, D2=D2, B=B)[0] for j in JJ])

# Create DataFrame with named rows and columns
ket_matrix = pd.DataFrame(ket, columns=base_tot, index=base_tot)

# Map base labels to state names (customize this mapping as needed)
base_to_state = {tuple(base): f'State {i}' for i, base in enumerate(base_tot)}

# Extract non-zero entries and their corresponding base labels
results = {}
for col in ket_matrix.columns:
    non_zero_entries = ket_matrix[col][ket_matrix[col] != 0]
    associated_base = [base for base in ket_matrix.index[ket_matrix[col] != 0]]
    results[col] = list(zip(associated_base, non_zero_entries))

# Display results
print("\nStates:")
for col in ket_matrix.columns:
    state_name = base_to_state[tuple(col)]
    print(f"'{state_name}':")
    for base_label, value in results[col]:
        print(f"{base_label}: {value}")

# Create labels for eigenvectors
fila = [f'ψ{i}' for i in range(len(E))]

# Plot the eigenvalues
plt.figure(figsize=(10, 6))
for i in range(eigenvalues.shape[1]):
    plt.plot(JJ, eigenvalues[:, i] - eigenvalues[:, 0], '--', label=f'E-E0 {i}')
plt.xlabel('J')
plt.ylabel('E-E0')
plt.title('Energy as a Function of J')
plt.legend(fila, loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True)
plt.show()

# Create DataFrame for eigenvectors
ket_matrix = pd.DataFrame(ket, columns=fila, index=base_tot)

# Plot the eigenvectors as a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(ket_matrix, cmap='viridis', annot=True, fmt='.2f', cbar=True)
plt.xlabel('States',fontsize=20)
plt.ylabel('Autvectors Basis  /S1,S2>',fontsize=20)

plt.xticks(ticks=np.arange(len(fila)) + 0.5, labels=fila, rotation=45,fontsize=20)
plt.yticks(ticks=np.arange(len(base_tot)) + 0.5, labels=base_tot, rotation=0, fontsize=20)
plt.tight_layout()
plt.show()
