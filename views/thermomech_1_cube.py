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

st.markdown(" ## Example 1: Thermoelasticity in a cube")
st.write("This example is located in our [repository](https://gitlab.tudelft.nl/ADMIRE_Public/safeincave).")

st.markdown(" ## Goals")

st.write(
	"""
	1. Set constitutive models to overburden and salt formations
	2. Calculate lithostatic pressure
	""")


st.markdown(" ## Problem description")

fig_1_cube_names = create_fig_tag("fig_1_cube_names")

st.write("Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.")

fig_1_cube_names = figure(os.path.join("assets", "thermomechanics", "1_cube_names.png"), "Geometry and temperature profile", "fig_1_cube_names", size=500)

fig_1_cube_bcs = create_fig_tag("fig_1_cube_bcs")

st.write("Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.")

fig_1_cube_bcs = figure(os.path.join("assets", "thermomechanics", "1_cube_bcs.png"), "Geometry and temperature profile", "fig_1_cube_bcs", size=500)


st.markdown(" ## Implementation")

st.code(
"""
import safeincave as sf
import safeincave.Utils as ut
import safeincave.HeatBC as heatBC
import safeincave.MomentumBC as momBC
from petsc4py import PETSc
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
t_control = sf.TimeControllerParabolic(n_time_steps=100, initial_time=0.0, final_time=10, time_unit="day")
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
bc_east = heatBC.DirichletBC(boundary_name = "EAST", 
						values = [274, 274],
						time_values = time_values)
""",
language="python")

st.code(
"""
bc_handler = heatBC.BcHandler(heat_eq)
bc_handler.add_boundary_condition(bc_east)
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
mom_eq = sf.LinearMomentum(grid, 0.5)
""",
language="python")

st.code(
"""
mom_solver = PETSc.KSP().create(grid.mesh.comm)
mom_solver.setType("bicg")
mom_solver.getPC().setType("asm")
mom_solver.setTolerances(rtol=1e-12, max_it=100)
mom_eq.set_solver(mom_solver)
""",
language="python")

st.code(
"""
E = 102e9*to.ones(mom_eq.n_elems)
nu = 0.3*to.ones(mom_eq.n_elems)
spring_0 = sf.Spring(E, nu, "spring")
""",
language="python")

st.code(
"""
omega_A = grid.region_indices["OMEGA_A"]
omega_B = grid.region_indices["OMEGA_B"]
""",
language="python")

st.code(
"""
alpha = to.zeros(mom_eq.n_elems)
alpha[omega_A] = 44e-6
alpha[omega_B] = 74e-6
thermo = sf.Thermoelastic(alpha, "thermo")
""",
language="python")

st.code(
"""
mat.add_to_elastic(spring_0)
mat.add_to_thermoelastic(thermo)
""",
language="python")

st.code(
"""
mom_eq.set_material(mat)
""",
language="python")

st.code(
"""
g = -9.81
g_vec = [0.0, 0.0, g]
mom_eq.build_body_force(g_vec)
""",
language="python")

st.code(
"""
bc_west_2 = momBC.DirichletBC(boundary_name = "WEST", 
							component = 2,
							values = [0.0, 0.0],
							time_values = time_values)
bc_west_1 = momBC.DirichletBC(boundary_name = "WEST", 
							component = 1,
							values = [0.0, 0.0],
							time_values = time_values)
bc_west_0 = momBC.DirichletBC(boundary_name = "WEST", 
							component = 0,
							values = [0.0, 0.0],
							time_values = time_values)
bc_bottom = momBC.DirichletBC(boundary_name = "BOTTOM", 
							component = 2,
							values = [0.0, 0.0],
							time_values = time_values)
""",
language="python")

st.code(
"""
bc_handler = momBC.BcHandler(mom_eq)
bc_handler.add_boundary_condition(bc_west_0)
bc_handler.add_boundary_condition(bc_west_1)
bc_handler.add_boundary_condition(bc_west_2)
bc_handler.add_boundary_condition(bc_bottom)
""",
language="python")

st.code(
"""
mom_eq.set_boundary_conditions(bc_handler)
""",
language="python")

st.code(
"""
output_mom = sf.SaveFields(mom_eq)
output_mom.set_output_folder(output_folder)
output_mom.add_output_field("u", "Displacement (m)")
output_mom.add_output_field("sig", "Stress (Pa)")
output_mom.add_output_field("p_nodes", "Mean stress (Pa)")
output_mom.add_output_field("p_elems", "Mean stress (Pa)")
output_mom.add_output_field("q_nodes", "Von Mises stress (Pa)")
output_mom.add_output_field("q_elems", "Von Mises stress (Pa)")
""",
language="python")

st.code(
"""
output_heat = sf.SaveFields(heat_eq)
output_heat.set_output_folder(output_folder)
output_heat.add_output_field("T", "Temperature (K)")
""",
language="python")

st.code(
"""
outputs = [output_mom, output_heat]

""",
language="python")

st.code(
"""
sim = sf.Simulator_TM(mom_eq, heat_eq, t_control, outputs, True)
sim.run()
""",
language="python")
