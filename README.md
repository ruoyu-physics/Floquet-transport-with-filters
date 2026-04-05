# Floquet Transport Simulation via Recursive Green's Functions

This project implements a numerical framework for simulating quantum transport in a periodically driven (Floquet) lattice system using recursive Green’s function techniques.

The code is designed with modular structure, separating:
- Hamiltonian construction
- Device geometry abstraction
- Transport solver

---

## 🚀 Features

- Floquet Hamiltonian construction using Bessel expansion
- Recursive Green’s function algorithm (O(N) scaling in system length)
- Support for multi-region devices (system + filter)
- Transmission calculation via Floquet Landauer formalism
- Modular design for extensibility

---

## 🧠 Physical Model

The simulation models a driven lattice system (e.g., graphene-like structure) under a time-periodic vector potential.

Key components:
- Tight-binding Hamiltonian with nearest-neighbor hopping
- Floquet expansion truncated to finite harmonics
- Lead self-energies for open boundary conditions
- Optional filter region to modify transport properties

---

## 🏗️ Code Structure

- `config.py`  
  Defines all physical and numerical parameters using a dataclass.

- `Hamiltonian_setup.py`  
  Builds Floquet Hamiltonian, couplings, and device structure.

- `GreenFunctionSolver.py`  
  Implements recursive Green’s function and transmission calculation.

- `main.py`  
  Runs the simulation and computes dI/dV.

---

## ▶️ How to Run

```bash
python main.py
