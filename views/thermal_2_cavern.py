import streamlit as st

st.set_page_config(layout="wide") 
st.markdown(" ## Problem description")
st.write("Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.")

st.code(
"""
import safeincave as sf
import safeincave.Utils as ut
import safeincave.HeatBC as heatBC
from mpi4py import MPI
import dolfinx as do
import os
import sys
import ufl
import torch as to
import numpy as np
from petsc4py import PETSc
import time
""",
language="python")

st.code(
"""
grid_path = os.path.join("..", "..", "..", "grids", "cavern_regular")
grid = sf.GridHandlerGMSH("geom", grid_path)
""",
language="python")

st.code(
"""
output_folder = os.path.join("output", "case_0")
""",
language="python")

st.code(
"""
t_control = sf.TimeControllerParabolic(n_time_steps=100, initial_time=0, final_time=5, time_unit="year")
""",
language="python")

st.code(
"""
heat_eq = sf.HeatDiffusion(grid)
""",
language="python")

st.code(
"""
solver_heat = PETSc.KSP().create(grid.mesh.comm)
solver_heat.setType("cg")
solver_heat.getPC().setType("asm")
solver_heat.setTolerances(rtol=1e-12, max_it=100)
heat_eq.set_solver(solver_heat)
""",
language="python")

st.code(
"""
mat = sf.Material(heat_eq.n_elems)
""",
language="python")

st.code(
"""
rho = 2000.0*to.ones(heat_eq.n_elems, dtype=to.float64)
mat.set_density(rho)
""",
language="python")

st.code(
"""
cp = 850*to.ones(heat_eq.n_elems, dtype=to.float64)
mat.set_specific_heat_capacity(cp)
""",
language="python")

st.code(
"""
k = 7*to.ones(heat_eq.n_elems, dtype=to.float64)
mat.set_thermal_conductivity(k)
""",
language="python")

st.code(
"""
heat_eq.set_material(mat)
""",
language="python")

st.code(
"""
time_values = [t_control.t_initial, t_control.t_final]
nt = len(time_values)
""",
language="python")

st.code(
"""
km = 1000
dTdZ = 27/km
T_top = 273 + 20
T_gas = 273 + 10
h_conv = 5.0
""",
language="python")

st.code(
"""
bc_handler = heatBC.BcHandler(heat_eq)
bc_top = heatBC.DirichletBC("Top", nt*[T_top], time_values)
bc_handler.add_boundary_condition(bc_top)
bc_bottom = heatBC.NeumannBC("Bottom", nt*[dTdZ], time_values)
bc_handler.add_boundary_condition(bc_bottom)
bc_east = heatBC.NeumannBC("East", nt*[0.0], time_values)
bc_handler.add_boundary_condition(bc_east)
bc_west = heatBC.NeumannBC("West", nt*[0.0], time_values)
bc_handler.add_boundary_condition(bc_west)
bc_south = heatBC.NeumannBC("South", nt*[0.0], time_values)
bc_handler.add_boundary_condition(bc_south)
bc_north = heatBC.NeumannBC("North", nt*[0.0], time_values)
bc_handler.add_boundary_condition(bc_north)
bc_cavern = heatBC.RobinBC("Cavern", nt*[T_gas], h_conv, time_values)
bc_handler.add_boundary_condition(bc_cavern)
heat_eq.set_boundary_conditions(bc_handler)
""",
language="python")

st.code(
"""
fun = lambda x, y, z: T_top - dTdZ*(z - 660)
T0_field = ut.create_field_nodes(heat_eq.grid, fun)
heat_eq.set_initial_T(T0_field)
""",
language="python")

st.code(
"""
output_heat = sf.SaveFields(heat_eq)
output_heat.set_output_folder(output_folder)
output_heat.add_output_field("T", "Temperature (K)")
outputs = [output_heat]
""",
language="python")

st.code(
"""
sim = sf.Simulator_T(heat_eq, t_control, outputs, True)
sim.run()
""",
language="python")
