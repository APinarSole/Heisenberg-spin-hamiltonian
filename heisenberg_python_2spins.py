import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns  

"""
    Heisenberg model  H = gμ(S₁B+S₂B) + D₁Sz₁² + D₂Sz₂² + E₁(Sₓ₁² - Sᵧ₁²) + E₂(Sₓ₂² - Sᵧ₂²) + J S₁S₂
    
    Calculate the spin Hamiltonian of two spin systems S1 and S2 using the Heisenberg model. This script can handle any spin value 
    with magnetic anisotropy (E,D) and a vectorial magnetic field B. For more advanced calculations of two spin systems, check
    https://github.com/ManishMitharwall/Nickelocene_Spin_Sensor 
    # H= Hamiltonian operator. Take in to account: dim(H)=dim(S1)*dim(S2)
    # S1,S2 spin operators S=(Sx,Sy,Sz)
    # Sx,Sy,Sz axial spin operators from the Pauli matrices
    # Operators from S1 are multiplied by an outer product with a dim(S2) identity matrix and viceversa
    # J>0 favours antiferromagnetic states of the combined system (Eg. E(↑↑)>E(↑↓))
    # D>0 favours low Sz of a single spin system (Eg. E1(1)>E1(1/2))
    # The script also allows to study a single particle by making one of the spins S=0
    # Apart from plotting, the script prints the states, its eigenvectors and the energies in the console
    
    
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
spin1 = 1.5   # spin 1
spin2 = 1.5  # spin 2
D1 = 4   # out of plane magnetic anisotropy 1 in meV
D2 = 0   # out of plane magnetic anisotropy 2 in meV
E1 = 0   # in plane magnetic anisotropy spin 1 in meV
E2 = 0   # in plane magnetic anisotropy spin 2 in meV
B = [0,0,0]    # vectorial magnetic field in Tesla
Jmax=2  # max. exchange coupling in meV

def heisenberg(J, spin1, spin2,D1, D2,E1,E2, B):
    
     
    def pauli_matrix(s):  # Pauli matrices generator
        s=float(s)
        n = int(2*s+1)
        sx=np.empty([n,n])
        sy=np.empty([n,n],dtype=complex)
        sz=np.empty([n,n])
        for a in range(1,n+1):
            for b in range(1,n+1):
                sx[a-1,b-1]=( 0.5*((a==b+1) + (a+1==b))*np.sqrt((s+1)*(a+b-1)-a*b))
                sy[a-1,b-1] =  1j*(0.5*((a==b+1) - (a+1==b))*np.sqrt((s+1)*(a+b-1)-a*b))
                sz[a-1,b-1] = (s+1-a)*(a==b)
        return sx,sy,sz


    def base(spin):
        base = []
        
        # Generate the list of spin quantum numbers from -spin to +spin with step 1
        spin_values = [spin - i for i in range(int(2 * spin + 1))]
        
        # Convert spin values to strings and format half-integers as fractions
        for s in spin_values:
            if s.is_integer():  # For integers
                base.append([str(int(s))])
            else:  # For half-integers
                base.append([f',{int(s * 2)}/2'])  
        return base
    
    
    base1 = np.array(base(spin1))
    base2 = np.array(base(spin2))
    
        
    x1, y1, z1 = pauli_matrix(spin1)
    x2, y2, z2 = pauli_matrix(spin2)
    
    I1=np.eye(x1.shape[1])
    I2=np.eye(x2.shape[1])
        
    S1 = [ np.kron(x1, I2),  np.kron(y1, I2),  np.kron(z1, I2)]
    S2 = [ np.kron(I1, x2),  np.kron(I1, y2),  np.kron(I1, z2)]
   
    # Sz*Sz
    Sz_1 = np.dot(S1[2], S1[2])
    Sz_2 = np.dot(S2[2],S2[2])
    # Sx*Sx
    Sx_1 = np.dot(S1[0],S1[0])
    Sx_2 = np.dot(S2[0],S2[0])
    # Sy*Sy
    Sy_1 = np.dot(S1[1],S1[1])
    Sy_2 = np.dot(S2[1],S2[1])
    
    #Heisenberg Hamiltonian H
    # Zeeman component
    B = 0.06 * np.array(B)*2  #mu*g*B in meV
    B1 = B[0]*S1[0]+B[1]*S1[1]+B[2]*S1[2]
    B2 = B[0]*S2[0]+B[1]*S2[1]+B[2]*S2[2]
    H_Z=B1+B2
    
    # Magnetic anisotropy component
    H_D=Sz_1 * D1 + Sz_2 * D2
    H_E=E1*(Sx_1-Sy_1)+E2*(Sx_2-Sy_2)
    
    # Exchange component
    
    H_J=J * (np.dot(S1[0], S2[0]) + np.dot(S1[1], S2[1]) + np.dot(S1[2], S2[2]))
    
    # Calculate H
    
    H=H_Z+H_D+H_E+H_J
    
    # Compute eigenvalues and eigenvectors
    E, ket = np.linalg.eigh(H)

    # Round and extract the real part of eigenvalues
    E = np.round(E, 2).real
    ket = np.round(ket, 2).real
    # normalization
    for i in range(ket.shape[1]):  # Iterate over columns
       column = ket[:, i]  # Extract the i-th column
       norm = np.sum(np.abs(column))  # Calculate the sum of the absolute values of the column
       if norm != 0:  # Check to avoid division by zero
        ket[:, i] = column / norm  # Normalize the column
    
    
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
    results[col] = list(zip(associated_base, np.round(non_zero_entries,2)))

# Create labels for eigenvectors

fila = [f'ψ{i}' for i in range(len(E))]

# States
print("\nStates J="+str(Jmax)+' (meV):')
for col in ket_matrix.columns:
    state_name = base_to_state[tuple(col)]
    state_values = " ".join([f"{value}({base_label})" for base_label, value in results[col]])
    print(f"{state_name} = {state_values}")

# Plot the eigenvalues
if ket.shape[1]<=20:               # limits the plot to dimension 20, change it if you need
    plt.figure(figsize=(12, 8))
else:
    plt.figure(figsize=(30, 18))
Ene=[]
for i in range(eigenvalues.shape[1]):
    plt.scatter(JJ, eigenvalues[:, i] - eigenvalues[:, 0],label=f'E-E0 {i}')
    Ene.append((eigenvalues[:, i] - eigenvalues[:, 0])[-1])
plt.xlabel('J (meV)',fontsize=25)
plt.ylabel('E-E0 (meV)',fontsize=25)
plt.title(r'$B = ' + str(B) + r'\ \ S_1 = ' + str(spin1) + r'\ \ S_2 = ' + str(spin2) + 
          r'\ \ D_1 = ' + str(D1) + r'\ \ D_2 = ' + str(D2) + '$', fontsize=25)
if ket.shape[1]<=10:               # limits the plot to dimension 20, change it if you need
    plt.legend(fila, loc='upper left', bbox_to_anchor=(1, 1),fontsize=25)
else:
    plt.legend(fila, loc='upper left', bbox_to_anchor=(1, 1),ncol=3,fontsize=25)

plt.grid(True)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.show()

# Display eigenvalues (E0, E1, etc.)
print("\nEnergies (E-E0) J="+str(Jmax)+' (meV):')
for i, col in enumerate(ket_matrix.columns):
    eigenvalue = np.round(Ene[i],2)  # Assuming `E` contains the eigenvalues
    print(f"E{i} = {eigenvalue} ")


# Create DataFrame for eigenvectors
ket_matrix = pd.DataFrame(ket, columns=fila, index=base_tot)

# Plot the eigenvectors
if ket.shape[1]<=20:               # limits the plot to dimension 20, change it if you need
    plt.figure(figsize=(12, 8))
else:
    plt.figure(figsize=(30, 18)) 
plt.figure(figsize=(15, 10))
sns.heatmap(np.round(ket_matrix,2), cmap='viridis', annot=True, fmt='.2f', cbar=True,annot_kws={"size": 20})  # remove round to get the raw coefficients
plt.xlabel('States',fontsize=22)
plt.ylabel(r'Autovectors Basis $|S_1, S_2\rangle$', fontsize=22)
plt.title('Normalized coefficients J='+str(Jmax)+' (meV):',fontsize=22)

plt.xticks(ticks=np.arange(len(fila)) + 0.5, labels=fila, rotation=45,fontsize=20)
plt.yticks(ticks=np.arange(len(base_tot)) + 0.5, labels=base_tot, rotation=0, fontsize=20)
plt.tight_layout()
plt.show()




