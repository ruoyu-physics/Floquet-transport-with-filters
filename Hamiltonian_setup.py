import numpy as np
from config import BOND_ANGLE
from numpy import exp, identity
from scipy.special import jv

# build and store system, filter and coupling Hamiltonians using
# SystemParamters object
class HamiltonianBuilder:

    def __init__(self, params):
        
        self.params = params # SystemParameters object
        self.set_hamiltonian()
        
        
    # pre-define the Bessel function expansion
    def C1(self, order, vec, sign):
        if(vec==0):
            return 0
        m = -order
        return 1j**m*jv(sign*m, self.params.z1)*exp(-1j*m*BOND_ANGLE[vec-1])

    # returns the mth harmonic of the Floquet Hamiltonian in real space
    # h gives the hamiltonian of one slice. U gives the coupling between neighboring slices
    def h_m(self, m):
        ns = self.params.ns
        t1 = self.params.t1

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
                    h[i, j] = t1* self.C1(m, vech[index], signh[index])
                    h[j, i] = t1* self.C1(m, vech[index], -signh[index])

                    vecu = [1, 0, 0, 0]
                    U[i, j] = t1* self.C1(m, vecu[index], 1)

                    vecu = [0, 0, 3, 0]
                    U[j, i] = t1* self.C1(m, vecu[index], -1)
        return h, U

    # construct the system floquet Hamiltonian of one slice in extended space
    # return: 
    #   h_system: system hamiltonian
    #   u_system: system coupling between slices
    def build_h_system(self,):
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
    
    # construct the matrix containing m*omega in Floquet representation
    def build_m_omega(self,):
        trunc = self.params.trunc
        ns = self.params.ns
        tsize = trunc*ns
        omega = self.params.omega
        momega = np.zeros((tsize, tsize), dtype=complex)
        aa = np.arange(-(trunc//2), trunc//2+1)*omega
        for i in range(trunc): 
            momega[ns*i:ns*(i+1), ns*i:ns*(i+1)] = np.identity(ns)*aa[i]
        return momega
    
    # construct the lead self energy matrix in Floquet representation
    def build_self_energy(self):
        trunc = self.params.trunc
        ns = self.params.ns
        tsize = ns*trunc

        G = identity(ns, dtype=complex)*self.params.gamma*1j
        
        Gamma = np.zeros((tsize, tsize), dtype=complex)
        for i in range(trunc):
            Gamma[ns*i:ns*(i+1), ns*i:ns*(i+1)] = G
        return Gamma

    # construct filter Hamiltonian
    # return: 
    #    h_filter: filter Hamiltonian
    #    u_filter: coupling Hamiltonian in filter
    #    u_couplie: coupling Hamiltoian between filter and system
    def build_h_filter(self):
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
    
    # Construct and store all the matrices needed
    def set_hamiltonian(self):

        # Compute and store the matrices
        h_system, u_system = self.build_h_system()
        self.h_system = h_system
        self.u_system = u_system

        h_filter, u_filter, u_couple = self.build_h_filter()
        self.h_filter = h_filter
        self.u_filter = u_filter
        self.u_couple = u_couple

        # Compute and Store hermitian of the coupling matrices

        self.u_system_d = u_system.conj().T
        self.u_filter_d = u_filter.conj().T
        self.u_couple_d = u_couple.conj().T

        self.m_omega = self.build_m_omega()
        self.self_energy = self.build_self_energy()

        
