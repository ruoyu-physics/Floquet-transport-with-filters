from dataclasses import dataclass
import numpy as np
from numpy import sqrt, pi


# ============================
# Universal Constants
# ============================

# Graphene bond angles
BOND_ANGLE = np.array([-pi/6, pi/2, 7*pi/6])


# =============================
# Simulation Parameters
# =============================

@dataclass
class SystemParameters:

    # Sample parameters
    a: float = 1            # lattice bond length
    t1: float = 1            # Nearest hopping amplitude
    A0: float = 0.5      # drive amplitude(vector potential)
    omega: float  = 3       # drive frequency
    M: float = 0            # on-site energy
    gamma: float = -1       # lead self energy

    maxorder: int = 5    # maximum order of Bessel functions kept
    trunc: int = 5        # truncation range (trunc=3: m=-1, 0, 1)
    length: int = 40     # system length
    width: int = 60      # system width

    # Filter parameters
    lengthf: int = 20     # Length of the filter

    tf: float = 0.25         # filter hopping amplitude

    @property
    def t_couple(self):
        return sqrt(self.tf*self.t1)       # Coupling between the filter and the system

    @property
    def ns(self):
        return 2*(self.width+1)     # number of sites in one slice
    
    @property
    def z1(self):
        return self.A0*self.a       # Bessel function argument
    

