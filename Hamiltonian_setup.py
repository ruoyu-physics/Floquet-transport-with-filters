import numpy as np
from config import BOND_ANGLE
from numpy import exp, identity
from scipy.special import jv

class HamiltonianBuilder:
    """
    Constructs and stores all Hamiltonian components required for the simulation.

    This includes:
    - System Hamiltonian in Floquet extended space representation
    - Filter Hamiltonian
    - Coupling matrices between slices
    - Lead self-energies
    - Floquet frequency shift matrix (m * omega)

    The results are stored in a dictionary for flexible reuse.
    """

    def __init__(self, params):
        self.params = params # SystemParameters object
        self.hamiltonian_dict = self.build_hamiltonian_dict()
    
    def h_m(self, m):
        """
        Construct the m-th Floquet harmonic of the Hamiltonian for a single slice.

        Parameters
        ----------
        m : int
            Floquet harmonic index.

        Returns
        -------
        h : ndarray
            On-site Hamiltonian matrix for the slice.
        U : ndarray
            Coupling matrix between neighboring slices.
        """
        ns = self.params.ns
        t1 = self.params.t1

        # pre-define the Bessel function expansion
        def C1(order, vec, sign):
            if(vec==0):
                return 0
            m = -order
            return 1j**m*jv(sign*m, self.params.z1)*exp(-1j*m*BOND_ANGLE[vec-1])

        h = np.zeros((ns,ns), dtype=complex)
        U = np.zeros((ns,ns), dtype=complex)
        for j in range(ns):
            for i in range(j+1):
                if(i==j):
                    s = -np.sign(i%2-0.5) # alternating 1, -1 (-s gives alterning -1, 1)
                    if(m==0):
                        h[i, j] = s*self.params.M

                elif(abs(i-j)==1):
                    index = i%4
                    vech = [3, 2, 1, 2]         # alternating patterns of bonds
                    signh = [1, -1, 1, -1]      # alternating patterns of bonds
                    h[i, j] = t1* C1(m, vech[index], signh[index])
                    h[j, i] = t1* C1(m, vech[index], -signh[index])

                    vecu = [1, 0, 0, 0]
                    U[i, j] = t1* C1(m, vecu[index], 1)

                    vecu = [0, 0, 3, 0]
                    U[j, i] = t1* C1(m, vecu[index], -1)
        return h, U

    def build_h_system(self,):
        """
        Construct the full Floquet Hamiltonian for one slice of the system in extended space.

        This includes all harmonic couplings up to the specified truncation order.

        Returns
        -------
        h_system : ndarray
            Floquet extended space on-site Hamiltonian.
        u_system : ndarray
            Floquet extended space coupling matrix between slices.
        """
        trunc=self.params.trunc
        ns = self.params.ns
        tsize = trunc*ns
        h_system = np.zeros((tsize, tsize), dtype=complex)
        u_system = np.zeros((tsize, tsize), dtype=complex)
        for i in range(trunc):
            for j in range(i+1):
                for m in range(self.params.maxorder+1):
                    if(i-j==m):
                        h, u = self.h_m(m)
                        h_system[ns*i:ns*(i+1), ns*j:ns*(j+1)] = h
                        u_system[ns*i:ns*(i+1), ns*j:ns*(j+1)] = u
                        h, u = self.h_m(-m)
                        h_system[ns*j:ns*(j+1), ns*i:ns*(i+1)] = h
                        u_system[ns*j:ns*(j+1), ns*i:ns*(i+1)] = u
        return h_system, u_system
    
    def build_m_omega(self,):
        """
        Construct the diagonal Floquet energy shift matrix.

        Each Floquet block is shifted by m * omega.

        Returns
        -------
        momega : ndarray
            Block-diagonal matrix containing Floquet frequency shifts.
        """

        trunc = self.params.trunc
        ns = self.params.ns
        tsize = trunc*ns
        omega = self.params.omega
        momega = np.zeros((tsize, tsize), dtype=complex)
        aa = np.arange(-(trunc//2), trunc//2+1)*omega
        for i in range(trunc): 
            momega[ns*i:ns*(i+1), ns*i:ns*(i+1)] = np.identity(ns)*aa[i]
        return momega
    
    def build_self_energy(self):
        """
        Construct the lead self-energy matrices in Floquet space.

        Returns
        -------
        self_energy_left : ndarray
            Left lead self-energy matrix.
        self_energy_right : ndarray
            Right lead self-energy matrix.
        """

        trunc = self.params.trunc
        ns = self.params.ns
        tsize = ns*trunc

        Gamma_left = identity(ns, dtype=complex)*self.params.gamma_left*1j
        Gamma_right = identity(ns, dtype=complex)*self.params.gamma_right*1j
        
        self_energy_left = np.zeros((tsize, tsize), dtype=complex)
        self_energy_right = np.zeros((tsize, tsize), dtype=complex)
        for i in range(trunc):
            self_energy_left[ns*i:ns*(i+1), ns*i:ns*(i+1)] = Gamma_left
            self_energy_right[ns*i:ns*(i+1), ns*i:ns*(i+1)] = Gamma_right
        return self_energy_left, self_energy_right
    
    def build_h_filter(self):
        """
        Construct the Hamiltonian for the filter region and its coupling to the system.

        Returns
        -------
        h_filter : ndarray
            On-site Hamiltonian for the filter region.
        u_filter : ndarray
            Coupling within the filter region.
        u_couple : ndarray
            Coupling between filter and system.
        """
        ns = self.params.ns
        tf = self.params.tf
        t_couple = self.params.t_couple
        trunc = self.params.trunc

        hfil_m = np.zeros((ns,ns), dtype=complex)
        Ufil_m = np.zeros((ns,ns), dtype=complex)
        u_couple_m = np.zeros((ns,ns), dtype=complex)

        for j in range(ns):
            for i in range(j+1):
                if(i==j):
                    
                    Ufil_m[i, j] = tf
                    u_couple_m[i, j] = t_couple
                    
                elif(abs(i-j)==1):
                    hfil_m[i, j] = tf
                    hfil_m[j, i] = tf

        h_filter = np.zeros((trunc*ns,trunc*ns), dtype=complex)
        u_filter = np.zeros((trunc*ns,trunc*ns), dtype=complex)
        u_couple = np.zeros((trunc*ns,trunc*ns), dtype=complex)
        
        for i in range(trunc):
            h_filter[ns*i:ns*(i+1), ns*i:ns*(i+1)] = hfil_m
            u_filter[ns*i:ns*(i+1), ns*i:ns*(i+1)] = Ufil_m
            u_couple[ns*i:ns*(i+1), ns*i:ns*(i+1)] = u_couple_m

        return h_filter, u_filter, u_couple
    
    def build_hamiltonian_dict(self):
        """
        Construct and store all Hamiltonian components in a structured dictionary.

        Returns
        -------
        dict
            Dictionary containing:
            - onsite Hamiltonians
            - coupling matrices
            - self-energies
            - Floquet frequency shifts
        """
        
        h_system, u_system = self.build_h_system()
        h_filter, u_filter, u_couple = self.build_h_filter()

        u_system_d = u_system.conj().T
        u_filter_d = u_filter.conj().T
        u_couple_d = u_couple.conj().T

        m_omega = self.build_m_omega()
        self_energy_left, self_energy_right = self.build_self_energy()

        hamiltonian_dict = {
            "onsite": {
                "filter": h_filter,
                "system": h_system
            },

            "couplings": {
                ("filter", "filter"): (u_filter, u_filter_d),
                ("system", "system"): (u_system, u_system_d),
                ("filter", "system"): (u_couple, u_couple_d),
                ("system", "filter"): (u_couple_d, u_couple)
            },

            "self_energy_left": self_energy_left,
            "self_energy_right": self_energy_right,
            "m_omega": m_omega
        }
        return hamiltonian_dict



class DeviceChain:
    """
    Represents the full device as a sequence of slices.

    This class abstracts the system into a 1D chain of slice types
    (e.g., 'system', 'filter') and provides access to the corresponding
    Hamiltonians and couplings.

    It allows the Green's function solver to operate without knowledge
    of the underlying physical structure.
    """

    def __init__(self, include_filter, hamiltonian_builder):
        self.include_filter = include_filter
        self.params = hamiltonian_builder.params
        hamiltonian_dict = hamiltonian_builder.hamiltonian_dict

        self.onsite = hamiltonian_dict["onsite"]
        self.couplings = hamiltonian_dict["couplings"]
        self.self_energy_left = hamiltonian_dict["self_energy_left"]
        self.self_energy_right = hamiltonian_dict["self_energy_right"]
        self.m_omega = hamiltonian_dict["m_omega"]
        self.slice_types = self.build_slice_types()

        tsize = self.params.trunc*self.params.ns
        self.identity = identity(tsize, dtype=complex)
    
    def build_slice_types(self):
        """
        Construct the sequence of slice types for the device.

        Returns
        -------
        list of str
            List describing the type of each slice ('system' or 'filter').
        """

        if(self.include_filter):
            slice_types = ["filter"]*self.params.lengthf + ["system"]*self.params.length + ["filter"]*self.params.lengthf
        else:
            slice_types = ["system"]*self.params.length
        return slice_types

    def base(self, n, energy, backgate):
        """
        Construct the effective on-site matrix for slice n.

        Parameters
        ----------
        n : int
            Slice index.
        energy : float
            Energy at which the Green's function is evaluated.
        backgate : float
            Backgate potential applied to the system region.

        Returns
        -------
        ndarray
            Effective on-site matrix for the slice.
        """
        w = energy*self.identity
        vb = backgate*self.identity

        kind = self.slice_types[n]
        ham_onsite = self.onsite[kind]
        if kind == "system":
            ham_onsite = ham_onsite + vb
        base = w + self.m_omega - ham_onsite
        return base
    
    def bond(self, n1, n2):
        """
        Retrieve the coupling matrices between two neighboring slices.

        Parameters
        ----------
        n1 : int
            Index of the first slice.
        n2 : int
            Index of the second slice.

        Returns
        -------
        tuple of ndarray
            (forward coupling, backward coupling) matrices.
        """
        slice1 = self.slice_types[n1]
        slice2 = self.slice_types[n2]
        return self.couplings[(slice1, slice2)]
        
        

        