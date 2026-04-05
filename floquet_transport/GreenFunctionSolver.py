import numpy as np
from numpy import conj
from numpy.linalg import inv

class GreenFunctionSolver:
    """
    Recursive Green's function solver for transport calculations.

    This class computes:
    - Local Green's functions along the device
    - End-to-end Green's function
    - Transmission using the Floquet Landauer formalism

    It operates on a DeviceChain abstraction and is independent
    of the underlying Hamiltonian construction.
    """

    def __init__(self, device_chain):
        self.params = device_chain.params        # SystemParameters object
        self.chain = device_chain         # DeviceChain object
        self.system_length = len(device_chain.slice_types)

    def local_green_function(self, energy, backgate, direction):
        """
        Compute local Green's functions along the device using recursion.

        Parameters
        ----------
        energy : float
            Energy at which the Green's function is evaluated.
        backgate : float
            Backgate potential applied to the system region.
        direction : {'left', 'right'}
            Direction of recursion.

        Returns
        -------
        list of ndarray
            Green's function for each slice.
        """
        if direction not in {"left", "right"}:
            raise ValueError("direction must be 'left' or 'right'")

        green_list = []

        chain = self.chain
        base = chain.base(0, energy, backgate)
        self_energy_left = chain.self_energy_left
        self_energy_right = chain.self_energy_right

        if direction == "left":
            G = inv(base - self_energy_left)
        elif direction == "right":
            G = inv(base - self_energy_right)

        green_list.append(G)

        for n in range(self.system_length-1):
            base = chain.base(n+1, energy, backgate)
            u, u_d = chain.bond(n, n+1)
            if direction == "left":
                G = inv(base - u_d@G@u)
            elif direction == "right":
                G = inv(base - u@G@u_d)
            green_list.append(G)
        
        base = chain.base(n=-1, energy=energy, backgate=backgate)
        u, u_d = chain.bond(n1=-2, n2=-1)
        if direction == "left":
            GNN = inv( base - u_d@green_list[-2]@u - self_energy_right)
        elif direction == "right":
            GNN = inv( base - u@green_list[-2]@u_d - self_energy_left)
        
        green_list[-1] = GNN

        return green_list

    def transport_green_function(self, energy, backgate, direction):
        """
        Compute the end-to-end Green's function G_{1N}.

        Parameters
        ----------
        energy : float
            Energy at which the Green's function is evaluated.
        backgate : float
            Backgate potential.
        direction : {'left', 'right'}
            Direction of propagation
                'left' : from left to right
                'right': from right to left

        Returns
        -------
        ndarray
            Green's function connecting the first and last slices.
        """
        if direction not in {"left", "right"}:
            raise ValueError("direction must be 'left' or 'right'")
        
        params = self.params
        ns = params.ns
        trunc = params.trunc

        tsize = ns*trunc
       
        green_list= self.local_green_function(energy, backgate, direction=direction)

        GNN = green_list.pop()
    
        G1N = np.identity(tsize)
        for n in range(self.system_length-1):
            u, u_d = self.chain.bond(n, n+1)
            if direction == "left":
                G1N = G1N@green_list[n]@u
            elif direction == "right":
                G1N = G1N@green_list[n]@u_d
        G1N = G1N@GNN
        return G1N

    
    def transmission(self, energy, backgate):
        """
        Compute the transmission using the Floquet Landauer formula.

        Parameters
        ----------
        energy : float
            Incident energy.
        backgate : float
            Backgate potential.

        Returns
        -------
        float
            Total transmission coefficient summed over Floquet channels.
        """

        params = self.params
        ns = params.ns
        trunc = params.trunc
        
        midindex = trunc//2

        green_left = self.transport_green_function(energy, backgate, "left")
        green_right = self.transport_green_function(energy, backgate, "right")

        TL = 0
        TR = 0
        SL = np.zeros((ns, ns), dtype=complex)
        SR = np.zeros((ns, ns), dtype=complex)

        SL = 1j*params.gamma_left*np.identity(ns, dtype=complex)
        SR = 1j*params.gamma_right*np.identity(ns, dtype=complex)
        
        for i in range(trunc):
            GL_m = green_left[ns*i:ns*(i+1), ns*midindex:ns*(midindex+1)]
            GR_m = green_right[ns*i:ns*(i+1), ns*midindex:ns*(midindex+1)]

            GL_m_d = conj(GL_m.T)
            GR_m_d = conj(GR_m.T)

            TL = TL + 2*np.trace(GL_m_d@SL@GL_m@SR)
            TR = TR + 2*np.trace(GR_m_d@SR@GR_m@SL)

        return abs(TL+TR)





