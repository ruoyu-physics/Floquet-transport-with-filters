import numpy as np
from config import SystemParameters
from Hamiltonian_setup import HamiltonianBuilder
from GreenFunctionSolver import GreenFunctionSolver
from make_plot import plot_dIdV

def main():
    params = SystemParameters(width=60, length=40, lengthf=20)
    hamiltonian = HamiltonianBuilder(params)
    green = GreenFunctionSolver(hamiltonian)

    E_start = -0.2
    E_end = 0.2
    num = 21

    backgate = np.linspace(E_start, E_end, num)

    print("Calculating dI/dV")
    dIdV = np.zeros(num)
    

    
    for i in range(num):
        T = green.transmission(0, -backgate[i])
        dIdV[i] = T

        print(int(i/num*100),"%")

    print(100, "%")

    plot_dIdV(backgate, dIdV)
    
    


if __name__ == "__main__":
    main()