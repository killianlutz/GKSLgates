module QGateDescent

using LinearAlgebra
using SparseArrays
using StaticArrays 
using DifferentialEquations: ODEProblem, solve, Tsit5, RK4 
using Parameters: @with_kw 
using Random 
using JLD2 
using Printf 
using LaTeXStrings 
using Base.Threads 
using GLMakie: mesh!, arrows3d!, scatter!, scatterlines!, lines!, heatmap!, wireframe!, cgrad, rotr90, resize_to_layout!, hidedecorations!, Figure, Axis, Axis3, AxisAspect, Colorbar, Point3, Sphere, cgrad, RGBAf, ylims!, colsize!, Aspect

# gradient_adjoint
export QuantumSystem, VariationalProblem, StateAdjointPair
export gradient_descent!, solve!, _objective_gradient!, showresults, showbloch_orbits, save, accuracy_metrics, rescale!, target!
export validate, mse, fidelity, showerrors, homotopy_descent!
export gatetime, control, state, finalstate, adjointstate
export build_pauli, pauli_span, decoherence
export aprioribounds!, verifybounds
# export gate1b_hadamard, gate1b_not, gate1b_phaseshift, gate1b_rx, gate1b_ry, gate1b_rz
# solvegksl
export simulate_gksl

include("./tools_qsystem.jl")
include("./tools_complex2real.jl")
include("./tools_grad.jl")
include("./tools_pauli_vcoh.jl")
include("./tools_qgates.jl")
include("./tools_dissipators.jl")
include("./tools_solvegksl.jl")
include("./tools_homotopy.jl")
include("./tools_validation.jl")
include("./tools_bounds.jl")

end # module QGateDescent
