import sys
import os
import streamlit as st
sys.path.append(os.path.join("libs"))
from Utils import ( equation,
					figure,
					create_fig_tag,
					cite_eq,
					cite_eq_ref,
					save_session_state,
)
from setup import run_setup

run_setup()

st.set_page_config(layout="wide") 



st.markdown(" ## Example 1: Heat diffusion 1D")
st.write("This example is located in our [repository](https://gitlab.tudelft.nl/ADMIRE_Public/safeincave).")

st.markdown(" ## Goals")

st.write(
	"""
	1. Set constitutive models to overburden and salt formations
	2. Calculate lithostatic pressure
	""")


st.markdown(" ## Problem description")

fig_1_cube_geom_bcs = create_fig_tag("fig_1_cube_geom_bcs")

st.write("Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.")

fig_1_cube_geom_bcs = figure(os.path.join("assets", "thermal", "1_cube_geom_bcs.png"), "Geometry and boundary conditions", "fig_1_cube_geom_bcs", size=600)



st.code(
"""
import safeincave as sf
import safeincave.Utils as ut
import safeincave.HeatBC as heatBC
from petsc4py import PETSc
import dolfinx as do
import torch as to
import os
""",
language="python")

st.code(
"""
grid_path = os.path.join("..", "..", "..", "grids", "cube")
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
t_control = sf.TimeControllerParabolic(n_time_steps=50, initial_time=0.0, final_time=5, time_unit="day")
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
bc_east = heatBC.DirichletBC(boundary_name = "EAST", 
						values = [273, 273],
						time_values = [t_control.t_initial, t_control.t_final])
bc_west = heatBC.RobinBC(boundary_name = "WEST", 
						values = [273, 273],
						h = 5.0,
						time_values = [t_control.t_initial, t_control.t_final])
""",
language="python")

st.code(
"""
bc_handler = heatBC.BcHandler(heat_eq)
bc_handler.add_boundary_condition(bc_east)
bc_handler.add_boundary_condition(bc_west)
""",
language="python")

st.code(
"""
heat_eq.set_boundary_conditions(bc_handler)
""",
language="python")

st.code(
"""
fun = lambda x, y, z: 293
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


save_session_state()