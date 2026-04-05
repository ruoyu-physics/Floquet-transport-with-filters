# Floquet Transport Simulation via Recursive Green's Functions

This project implements a numerical framework for simulating quantum transport in a finite lattice system under periodic drive using recursive Green’s function algorithm.

The code is designed with modular structure, separating:
- Hamiltonian construction
- Device geometry abstraction
- Transport solver

---

## Physical Model

The simulation models a driven honeycomb lattice system under a circularly polarized vector potential. The system is coupled to two ideal metallic leads from left and right. When included, filters are inserted between the leads and the system. Filters are designed to have a narrow and isolated band (square lattice in our case) to block photon-assisted transport processes. Differential conductance dI/dV is calculated using Floquet-Landauer formula: 
\[
\frac{dI}{dV} = \sum_m \left( T_{\mathrm{RL}}^{(m)} + T_{\mathrm{LR}}^{(m)} \right)
\]
When filters are applied, we expect a quantized dI/dV in both resonantly and off-resonantly opened gaps. For more details, see our paper: 
Ruoyu Zhang, Frederik Nathan, Netanel H. Lindner, and Mark S. Rudner, 
"Achieving quantized transport in Floquet topological insulators via energy filters, 
"Phys. Rev. B 110, 075428 (2024). 
https://doi.org/10.1103/PhysRevB.110.075428

---

## Code Structure

- `config.py`  
  Defines all physical and numerical parameters using a dataclass.

- `Hamiltonian_setup.py`  
  Builds Floquet Hamiltonian, couplings, and device structure.

- `GreenFunctionSolver.py`  
  Implements recursive Green’s function and transmission calculation.

- `main.py`  
  Runs the simulation and computes dI/dV.

---

## How to Run

```bash
python main.py
