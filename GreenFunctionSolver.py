import numpy as np
from numpy import conj, identity
from numpy.linalg import inv


class GreenFunctionSolver:

    def __init__(self, ham):
        self.params = ham.params        # SystemParameters object
        self.ham = ham         # HamiltonianBuilder object

    # recursively computes and stores the local Green's function of each slice
    def local_green_function(self, energy, backgate, direction):
        params = self.params
        ham = self.ham

        ns = params.ns
        trunc = params.trunc
        length = params.length
        lengthf = params.lengthf

        h_system = ham.h_system
        u_system = ham.u_system
        h_filter = ham.h_filter
        u_filter = ham.u_filter
        u_couple = ham.u_couple

        u_system_d = ham.u_system_d
        u_filter_d = ham.u_filter_d
        u_couple_d = ham.u_couple_d

        m_omega = ham.m_omega
        self_energy = ham.self_energy

        tsize = ns*trunc
        w = energy*identity(tsize, dtype=complex)
        vb = backgate*identity(tsize, dtype=complex)

        base_filter = w + m_omega - h_filter
        base_system = w + m_omega - (h_system + vb)

        # stores all the local Green's function
        Green_list = []
        U_list = []
        

        # recursion once
        def step(G_prev, base, U, U_d):
            if direction=='left':
                G_next = np.linalg.inv(base  - U_d@G_prev@U)
            elif direction=='right':
                G_next = np.linalg.inv(base  - U@G_prev@U_d)
            return G_next
        
        # recursion n times
        def propagate(G_prev, n, base, U, U_d):
            for _ in range(n):
                G_prev = step(G_prev, base, U, U_d)
                Green_list.append(G_prev)
                U_list.append((U, U_d))
            return G_prev

        # first slice, couple to lead
        G = inv(base_filter - self_energy)
        Green_list.append(G)

        # propagate in left filter
        G = propagate(G, lengthf-1, base_filter, u_filter, u_filter_d)
        
        # left filter/system coupling
        G = step(G, base_system, u_couple, u_couple_d)
        Green_list.append(G)
        U_list.append((u_couple, u_couple_d))
    
        # propagate in system
        G = propagate(G, length-1, base_system, u_system, u_system_d)

        # system/right filter coupling
        G = step(G, base_filter, u_couple, u_couple_d)
        Green_list.append(G)
        U_list.append((u_couple, u_couple_d))

        # propagate in right filter
        G = propagate(G, lengthf-1, base_filter, u_filter, u_filter_d)

        if direction == "left":
            GNN = inv( w + m_omega - h_filter - u_filter_d@Green_list[-2]@u_filter - self_energy)
        elif direction == "right":
            GNN = inv( w + m_omega - h_filter - u_filter@Green_list[-2]@u_filter_d - self_energy)
        
        Green_list[-1] = GNN
    
        return Green_list, U_list

    # computes the (1, N) elements of the full Green's function of a length N system.
    def transport_green_function(self, energy, backgate, direction):
        params = self.params
        ns = params.ns
        trunc = params.trunc

        tsize = ns*trunc
       
        Green_list, U_list = self.local_green_function(energy, backgate, direction=direction)

        GNN = Green_list.pop()
    
        G1N = np.identity(tsize)
        for green, coupling in zip(Green_list, U_list):
            if direction == "left":
                U = coupling[0]
            elif direction == "right":
                U = coupling[1]
            G1N = G1N@green@U

        G1N = G1N@GNN

        return G1N

    # compute transmisson conductance via Floquet Landauer formula
    def transmission(self, energy, backgate):
        params = self.params
        ns = params.ns
        trunc = params.trunc
        gamma = params.gamma

        midindex = trunc//2

        green_left = self.transport_green_function(energy, backgate, "left")
        green_right = self.transport_green_function(energy, backgate, "right")

        TL = 0
        TR = 0
        SL = np.zeros((ns, ns), dtype=complex)
        SR = np.zeros((ns, ns), dtype=complex)

        SL = 1j*gamma*np.identity(ns, dtype=complex)
        SR = 1j*gamma*np.identity(ns, dtype=complex)
        
        for i in range(trunc):
            GL_m = green_left[ns*i:ns*(i+1), ns*midindex:ns*(midindex+1)]
            GR_m = green_right[ns*i:ns*(i+1), ns*midindex:ns*(midindex+1)]

            GL_m_d = conj(GL_m.T)
            GR_m_d = conj(GR_m.T)

            TL = TL + 2*np.trace(GL_m_d@SL@GL_m@SR)
            TR = TR + 2*np.trace(GR_m_d@SR@GR_m@SL)

        return abs(TL+TR)





