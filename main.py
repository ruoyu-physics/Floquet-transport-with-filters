import numpy as np
from config import SystemParameters
from Hamiltonian_setup import HamiltonianBuilder, DeviceChain
from GreenFunctionSolver import GreenFunctionSolver
from make_plot import plot_dIdV

def main():
    params = SystemParameters(width=30, length=20, lengthf=20)
    hamiltonian = HamiltonianBuilder(params)
    device_chain = DeviceChain(include_filter=False, hamiltonian_builder=hamiltonian)
    green_solver = GreenFunctionSolver(device_chain=device_chain)

    E_start = -0.2
    E_end = 0.2
    num = 21

    backgate = np.linspace(E_start, E_end, num)

    dIdV = np.zeros(num)
    print("Calculating dI/dV")
    for i in range(num):
        T = green_solver.transmission(0, -backgate[i])
        dIdV[i] = T

        print(int(i/num*100),"%")

    print(100, "%")

    plot_dIdV(backgate, dIdV)
    
    


if __name__ == "__main__":
    main()