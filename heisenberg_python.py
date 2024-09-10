import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns  

"""
    Heisenberg model  H = gμ(S₁B+S₂B) + D₁Sz₁² + D₂Sz₂² + E₁(Sₓ₁² - Sᵧ₁²) + E₂(Sₓ₂² - Sᵧ₂²) + J S₁S₂
    
    Calculate the spin Hamiltonian of two spin systems S1 and S2 using the Heisenberg model. This script can handle S=1/2,1,3/2 
    with magnetic anisotropy (E,D) and a vectorial magnetic field B. For more advanced calculations of two spin systems, check
    https://github.com/ManishMitharwall/Nickelocene_Spin_Sensor 
    # H= Hamiltonian operator. Take in to account: dim(H)=dim(S1)*dim(S2)
    # S1,S2 spin operators S=(Sx,Sy,Sz)
    # Sx,Sy,Sz axial spin operators from the Pauli matrices
    # J>0 favours antiferromagnetic states of the combined system (Eg. E(↑↑)>E(↑↓))
    # D>0 favours low Sz of a single spin system (Eg. E1(1)>E1(1/2))
    # The script also allows to study a single particle by making one of the spins S=0
    
    Parameters:
    - Jmax: Echange coupling constant
    - spin1: Spin of the first particle
    - spin2: Spin of the second particle
    - D1,E1: Anisotropy parameters for the first particle (in meV) (D=out of plane, E=in plane)
    - D2,E2: Anisotropy parameters for the second particle (in meV)
    - B: vectorial Magnetic field (in Tesla)
    
    Returns:
    - E: Eigenvalues of the Hamiltonian
    - ket: Eigenvectors of the Hamiltonian
    - E_diag: Diagonal form of the energy matrix
    - base_tot: Base labels for the combined system
"""

# Define parameters
spin1 = 1   # spin 1
spin2 = 0.5  # spin 2
D1 = 4   # out of plane magnetic anisotropy 1 in meV
D2 = 4   # out of plane magnetic anisotropy 2 in meV
E1 = 0   # in plane magnetic anisotropy spin 1 in meV
E2 = 0   # in plane magnetic anisotropy spin 2 in meV
B = [0,0,0]    # vectorial magnetic field in Tesla
Jmax=2  # max. exchange coupling in meV

def heisenberg(J, spin1, spin2,D1, D2,E1,E2, B):
    # spin 0
    x0=np.array([[0]])
    y0=np.array([[0]])
    z0=np.array([[0]])
    base0=[['0']]
        
    # Spin 1/2 (2x2 matrices)
    x_1_2 = 0.5*np.array([[0, 1], [1, 0]])  # Pauli X
    y_1_2 = 0.5*np.array([[0, -1j], [1j, 0]])  # Pauli Y
    z_1_2 = 0.5*np.array([[1, 0], [0, -1]])  # Pauli Z
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
        if spin==0:
            x1 = x0
            y1 = y0
            z1 = z0
        elif spin == 0.5:
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
        
        return x1, y1, z1

    def base(spin):
        if spin==0:
            base=base0
        elif spin == 0.5:
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
    
    S1 = [ np.kron(x1, np.eye(x2.shape[1])),  np.kron(y1, np.eye(y2.shape[1])),  np.kron(z1, np.eye(z2.shape[1]))]
    S2 = [ np.kron(np.eye(x1.shape[1]), x2),  np.kron(np.eye(y1.shape[1]), y2),  np.kron(np.eye(z1.shape[1]), z2)]
   
    # Sz*Sz
    Sz_1 = np.dot(np.kron(z1, np.eye(x2.shape[1])), np.kron(z1, np.eye(x2.shape[1])))
    Sz_2 = np.dot(np.kron(np.eye(x1.shape[1]), z2), np.kron(np.eye(x1.shape[1]), z2))
    # Sx*Sx
    Sx_1 = np.dot(np.kron(x1, np.eye(x2.shape[1])), np.kron(x1, np.eye(x2.shape[1])))
    Sx_2 = np.dot(np.kron(np.eye(x1.shape[1]), x2), np.kron(np.eye(x1.shape[1]), x2))
    # Sy*Sy
    Sy_1 = np.dot(np.kron(y1, np.eye(x2.shape[1])), np.kron(y1, np.eye(x2.shape[1])))
    Sy_2 = np.dot(np.kron(np.eye(x1.shape[1]), y2), np.kron(np.eye(x1.shape[1]), y2))
    
    # Zeeman component
    B = 0.06 * np.array(B)*2  #mu*g*B in meV
    B1 = (B[0]*np.kron(x1, np.eye(x2.shape[1]))+B[1]*np.kron(y1, np.eye(x2.shape[1]))+B[2]*np.kron(z1, np.eye(x2.shape[1])))
    B2 = (B[0]*np.kron(x2, np.eye(x1.shape[1]))+B[1]*np.kron(y2, np.eye(x1.shape[1]))+B[2]*np.kron(z2, np.eye(x1.shape[1])))
    
    # Calculate H
    H = J * (np.dot(S1[0], S2[0]) + np.dot(S1[1], S2[1]) + np.dot(S1[2], S2[2])) + Sz_1 * D1 + Sz_2 * D2 +E1*(Sx_1-Sy_1)+E2*(Sx_2-Sy_2)+B1+B2
      

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
result = heisenberg(2, spin1=spin1, spin2=spin2, D1=D1, D2=D2,E1=E1,E2=E2 ,B=B)
E, ket, E_diag, base_tot = result

# Define the range of J values
JJ = np.linspace(0, Jmax, 150)
eigenvalues = np.array([heisenberg(j, spin1=spin1, spin2=spin2, D1=D1, D2=D2,E1=E1,E2=E2, B=B)[0] for j in JJ])

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
print("\nStates (raw):")
for col in ket_matrix.columns:
    state_name = base_to_state[tuple(col)]
    state_values = " ".join([f"{value}({base_label})" for base_label, value in results[col]])
    print(f"{state_name} = {state_values}")


# Create labels for eigenvectors
fila = [f'ψ{i}' for i in range(len(E))]

# Plot the eigenvalues
plt.figure(figsize=(10, 6))
for i in range(eigenvalues.shape[1]):
    plt.scatter(JJ, eigenvalues[:, i] - eigenvalues[:, 0],label=f'E-E0 {i}')
plt.xlabel('J (meV)',fontsize=20)
plt.ylabel('E-E0 (meV)',fontsize=20)
plt.title(r'$B = ' + str(B) + r'\ \ S_1 = ' + str(spin1) + r'\ \ S_2 = ' + str(spin2) + 
          r'\ \ D_1 = ' + str(D1) + r'\ \ D_2 = ' + str(D2) + '$', fontsize=20)
plt.legend(fila, loc='upper left', bbox_to_anchor=(1, 1),fontsize=20)
plt.grid(True)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

# Create DataFrame for eigenvectors
ket_matrix = pd.DataFrame(ket, columns=fila, index=base_tot)

# Plot the eigenvectors 
plt.figure(figsize=(12, 8))
sns.heatmap(np.round(ket_matrix,1), cmap='viridis', annot=True, fmt='.2f', cbar=True,annot_kws={"size": 15})  # remove round to get the raw coefficients
plt.xlabel('States',fontsize=20)
plt.ylabel(r'Autovectors Basis $|S_1, S_2\rangle$', fontsize=20)
plt.title('Rounded coefficients',fontsize=20)

plt.xticks(ticks=np.arange(len(fila)) + 0.5, labels=fila, rotation=45,fontsize=20)
plt.yticks(ticks=np.arange(len(base_tot)) + 0.5, labels=base_tot, rotation=0, fontsize=20)
plt.tight_layout()
plt.show()



