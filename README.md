## Purpose: reproducibility
This code, written in [Julia](https://julialang.org/), is provided to reproduce some of the figures in [this](https://hal.science/hal-05176524) paper, namely Figures 3, 4, 7 and Table 2.

## Content
The code attempts to solve for the minimal time needed to implement a quantum gate on noisy qudits of arbitrary dimension in the GKSL framework subject to a constrained coherent control. 

To tackle this problem, a continuation method is combined with gradient descent and a line-search of the step size. The former means that the cost function involves both the distance to the target gate and a penalty on the the time it takes to reach it. By gradually shrinking the penalty to zero, we expect to find the minimal gate time given the constraints.

## Getting started 
Open the terminal and run
```
git clone https://github.com/killianlutz/GKSLgates.git
```

Assuming Julia version `1.9.4` is correctly [installed](https://docs.julialang.org/en/v1/manual/installation/), there are three steps:

- Open a terminal and change directory to this project. Your directory now ends with `./GKSLgates`.

- Download and install the dependencies missing on your machine by running
```
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

- Perform the necessary calculations, stored in the `./applications/sims` directory, by running
```
julia --project=. appplications/optimize.jl
```

- Generate the figures, stored in the `./applications/sims` directory, by running
```
julia --project=. appplications/saveplots.jl
```

## Troubleshooting
Feel free to reach out to me: [Killian Lutz](https://killianlutz.github.io/).