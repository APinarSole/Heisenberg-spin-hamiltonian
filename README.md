# Heisenberg-spin-hamiltonian 
This script obtains the energies and eigenvalues in a two-spin (heisenberg_python_2spins.py) or 3 spin (heisenberg_python_3spins.py)  system from a Heisenberg model, offering the combined base that diagonalizes the Hamiltonian to visualize the components of each spin state as a function of the exchange coupling J and the magnetic anisotropy. It can handle any spin values with in/out of plane magnetic anisotropy (E,D) and a vectorial magnetic field B. In the 3 spin version there are three J that depends exponentially of the spin centers distance.
This is part of a deeper study of two spin systems carried out to understand how the nickelocene molecule couples with another spins in STM (https://www.nature.com/articles/s41557-024-01453-9), but the script works for any spin system. A more complete script to fit this kind of interactions that also include the cotunneling formalism to account for the current effects in STM is available here: https://github.com/ManishMitharwall/Nickelocene_Spin_Sensor
For more info about spin sensors: https://andrespinarsole.wordpress.com/

Example: 

spin1 = 1

spin2 = 0.5

D1 = 4   # in meV

D2 = 0   # in meV

E1 = 0 # in meV

E2 = 0 # in meV

B = [0,0,0]    # in Tesla

Jmax=2  # max. exchange coupling in meV

Output (plus plots):!

States (normalized):

State 0 = 0.28(('1 -1/2 ',)) -0.72(('0 1/2 ',))

State 1 = -0.72(('0 -1/2 ',)) 0.28(('-1 1/2 ',))

State 2 = 0.28(('0 -1/2 ',)) 0.72(('-1 1/2 ',))

State 3 = -0.72(('1 -1/2 ',)) -0.28(('0 1/2 ',))

State 4 = 1.0(('1 1/2 ',))

State 5 = 1.0(('-1 -1/2 ',))
