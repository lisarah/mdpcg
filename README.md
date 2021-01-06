# MDPCG
This repository contains python code for the MDP congestion games, a game model for large groups of users with Markov decision process dynamics. Two algorithms are developed for the game: an online learning method to solve the game without explicit knowledge of the game costs, and an incentive design method to encourage constraint satisfaction without explicit knowledge of game costs and dynamics.

## Content
1. MDP dynamic models: 
	* A mock up MDP with 3 x 5 grid states. Each state has 4 actions: left/right/up/down, where each action takes the user to the target neighbouring state with probability 0 < p < 1 and to another neighbouring state with probability 1-p. 
	* Uber drivers' MDP dynamics in Seattle, WA. See [Tolling for Constraint Satisfaction in MDP Congestion Games](https://arxiv.org/pdf/1903.00747.pdf)  for more model details.
	* Uber drivers' MDP dynamics in New York city generated from 17.8 million Uber pick-up data from [Data.World](https://data.world/data-society/uber-pickups-in-nyc)  (under development).
	* Wheatstone MDP dynamics - for demonstrating of Braess paradox in MDP congestion games. See [Sensitivity Analysis for MDP Congestion games](https://arxiv.org/pdf/1909.04167.pdf) for model description and Braess paradox description.
	* Airport gate assignment MDP dynamics. See [overleaf doc](https://www.overleaf.com/read/tnzgddzckbsh
) for description.
2. Game solvers
	* CVXPY 
	* Custom solver - Frank Wolfe + dynamic programming - with automatic step size generation. See [Tolling for Constraint Satisfaction in MDP Congestion Games](https://arxiv.org/pdf/1903.00747.pdf) for convergence guarantees.
3. Incentive solvers
	* Constrained CVXPY
	* Projected dual ascent
	* ADMM
	* Mystic - a nonconvex solver for non-convex constraints (experimental)
4. Data visualization - custom visualization methods for displaying Wardrop equilibrium and online solutions.