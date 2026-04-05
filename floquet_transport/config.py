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
    """
    Container for all physical and numerical parameters of the simulation.

    This class defines the lattice geometry, driving field parameters,
    Floquet truncation, system size, and filter properties.
    It also provides convenient derived quantities used throughout
    the Hamiltonian construction.

    Attributes
    ----------
    a : float
        Lattice bond length.
    t1 : float
        Nearest-neighbor hopping amplitude in the system.
    A0 : float
        Amplitude of the external driving field (vector potential).
    omega : float
        Driving frequency.
    M : float
        On-site mass term (staggered potential).
    gamma_left : float
        Self-energy parameter for the left lead.
    gamma_right : float
        Self-energy parameter for the right lead.
    maxorder : int
        Maximum Bessel function order retained in Floquet expansion.
    trunc : int
        Number of harmonic modes retained. 
        MUST BE ODD NUMBER.
    length : int
        Length of the central system (number of slices).
    width : int
        Width of each slice (number of unit cells).
    lengthf : int
        Length of the filter region (if included).
    tf : float
        Hopping amplitude within the filter region.

    Properties
    ----------
    t_couple : float
        Coupling strength between filter and system.
    ns : int
        Number of sites per slice.
    z1 : float
        Argument of the Bessel functions (A0 * a).
    """

    # System parameters
    a: float = 1            
    t1: float = 1            
    A0: float = 0.5      
    omega: float  = 3       
    M: float = 0            
    gamma_left: float = -1       
    gamma_right: float = -1      

    maxorder: int = 5    
    trunc: int = 5        
    length: int = 40     
    width: int = 60      

    # Filter parameters
    lengthf: int = 20     
    tf: float = 0.25         

    def __post_init__(self):
        if self.trunc % 2 == 0:
            raise ValueError(
                f"trunc must be odd (got {self.trunc}) to center Floquet indices"
            )

    @property
    def t_couple(self):
        return sqrt(self.tf*self.t1)       # Coupling between the filter and the system

    @property
    def ns(self):
        return 2*(self.width+1)     # number of sites in one slice
    
    @property
    def z1(self):
        return self.A0*self.a       # Bessel function argument
    
    
    

