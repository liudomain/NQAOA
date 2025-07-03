# NQAOA
Noisy Quantum Approximation Optimization Algorithm for Solving MaxCut Problem

## Background
In the noisy intermediate-scale quantum (NISQ) era, the Quantum Approximate Optimization Algorithm (QAOA) is a key tool for solving NP-hard problems (e.g., combinatorial optimization).
Most QAOA implementations overlook the inherent noise in quantum circuit environments.
## Research Content
Proposes the Noisy Quantum Approximate Optimization Algorithm (NQAOA) and analyzes its performance in noisy conditions via numerical simulations.
Compares the maximum (max AR) and average (mean AR) approximation ratios of various graphs under different optimization strategies and noise conditions.
Evaluates the impacts of circuit depth, nodes, edges, and noise coefficients on Maximum Cut (MaxCut).
## Experimental Results
### Added experimental verification on MindQuantum, Google(Not uploaded), PyQPanda3
Noise reduces the probability of finding optimal MaxCut solutions and makes the algorithm more easily trapped in suboptimal solutions.
Adam and Adagrad optimizers show the best performance in noisy environments.
Approximation ratio (AR) increases with the number of nodes (n).
AR exhibits a wave-like pattern as the number of edges (e) increases, with ARnoise ≤ ARideal in all cases.
## Significance
Systematically integrate multiple quantum noise types into the QAOA framework, quantitatively analyzing the synergistic effects of noise types, optimizer performance, and graph structures.
Establishes linear/exponential theoretical bounds for AR decay with respect to noise coefficients through theoretical derivations and numerical simulations.
Provides critical references for designing noise-robust quantum optimization algorithms on NISQ devices.
