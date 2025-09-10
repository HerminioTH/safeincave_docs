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


st.markdown(" ## Example 2: Cube with two regions")
st.write("This example is located in our [repository](https://gitlab.tudelft.nl/ADMIRE_Public/safeincave).")

st.markdown(" ## Goals")

st.write(
	"""
	1. Define material properties to different regions
	""")

st.markdown(" ## Problem description")

fig_2_cube_regions_geom = create_fig_tag("fig_2_cube_regions_geom")


st.write("Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.")

fig_2_cube_regions_geom = figure(os.path.join("assets", "2_cube_regions_geom.png"), "(a) Boundary names; (b) Region names; (c) Axial load and confining pressure.", "fig_2_cube_regions_geom", size=800)




fig_2_cube_region_model = create_fig_tag("fig_2_cube_region_model")

st.write(f"As illustrated in Fig. {fig_2_cube_region_model}, the constitutive model consists of an elastic element (spring) and a viscoelastic (kelvin) element.")

fig_2_cube_region_model = figure(os.path.join("assets", "2_cube_regions_model.png"), "Constitutive model composition for the triaxial problem.", "fig_2_cube_region_model", size=300)





st.markdown(" ### Implementation")

st.write("Import relevant packages. Note that the only reason to import package *dolfinx* here is to initialize the custom fields to be saved during the simulation, as explained next.")


st.code(
"""
from safeincave import *
import safeincave.Utils as ut
import safeincave.MomentumBC as momBC
from petsc4py import PETSc
import torch as to
import os
import time
""",
language="python")

st.code(
"""
grid_path = os.path.join("..", "..", "..", "grids", "cube")
grid = GridHandlerGMSH("geom", grid_path)
""",
language="python")

st.code(
"""
t_control = TimeController(dt=0.01, initial_time=0.0, final_time=1.0, time_unit="hour")
""",
language="python")

st.code(
"""
mom_eq = LinearMomentum(grid, theta=0.5)
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
mat = Material(mom_eq.n_elems)
""",
language="python")

st.code(
"""
rho = 2000.0*to.ones(mom_eq.n_elems, dtype=to.float64)
mat.set_density(rho)
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
E0 = to.zeros(mom_eq.n_elems)
nu0 = to.zeros(mom_eq.n_elems)
E0[omega_A] = 8*ut.GPa
E0[omega_B] = 10*ut.GPa
nu0[omega_A] = 0.2
nu0[omega_B] = 0.3
spring_0 = Spring(E0, nu0, "spring")
""",
language="python")

st.code(
"""
eta = to.zeros(mom_eq.n_elems)
E1 = to.zeros(mom_eq.n_elems)
nu1 = to.zeros(mom_eq.n_elems)
eta[omega_A] = 105e11
eta[omega_B] = 38e11
E1[omega_A] = 8*ut.GPa
E1[omega_B] = 5*ut.GPa
nu1[omega_A] = 0.35
nu1[omega_B] = 0.28
kelvin = Viscoelastic(eta, E1, nu1, "kelvin")
""",
language="python")

st.code(
"""
mat.add_to_elastic(spring_0)
mat.add_to_non_elastic(kelvin)
mom_eq.set_material(mat)
""",
language="python")

st.code(
"""
g_vec = [0.0, 0.0, 0.0]
mom_eq.build_body_force(g_vec)
""",
language="python")

st.code(
"""
T0_field = 298*to.ones(mom_eq.n_elems)
mom_eq.set_T0(T0_field)
mom_eq.set_T(T0_field)
""",
language="python")

st.code(
"""
bc_west = momBC.DirichletBC(boundary_name = "WEST", 
					component = 0,
					values = [0.0, 0.0],
					time_values = [0.0, t_control.t_final])
bc_bottom = momBC.DirichletBC(boundary_name = "BOTTOM", 
					component = 2,
					values = [0.0, 0.0],
					time_values = [0.0, t_control.t_final])
bc_south = momBC.DirichletBC(boundary_name = "SOUTH", 
					component = 1,
					values = [0.0, 0.0],
					time_values = [0.0, t_control.t_final])
bc_east = momBC.NeumannBC(boundary_name = "EAST",
					direction = 2,
					density = 0.0,
					ref_pos = 0.0,
					values = [5.0*ut.MPa, 5.0*ut.MPa],
					time_values = [0.0, t_control.t_final],
					g = g_vec[2])
bc_north = momBC.NeumannBC(boundary_name = "NORTH",
					direction = 2,
					density = 0.0,
					ref_pos = 0.0,
					values = [5.0*ut.MPa, 5.0*ut.MPa],
					time_values = [0.0, t_control.t_final],
					g = g_vec[2])
bc_top = momBC.NeumannBC(boundary_name = "TOP",
					direction = 2,
					density = 0.0,
					ref_pos = 0.0,
					values = [8.0*ut.MPa, 8.0*ut.MPa],
					time_values = [0.0, t_control.t_final],
					g = g_vec[2])
""",
language="python")

st.code(
"""
bc_handler = momBC.BcHandler(mom_eq)
bc_handler.add_boundary_condition(bc_west)
bc_handler.add_boundary_condition(bc_bottom)
bc_handler.add_boundary_condition(bc_south)
bc_handler.add_boundary_condition(bc_east)
bc_handler.add_boundary_condition(bc_north)
bc_handler.add_boundary_condition(bc_top)
""",
language="python")

st.code(
"""
mom_eq.set_boundary_conditions(bc_handler)
""",
language="python")

st.code(
"""
output_mom = SaveFields(mom_eq)
output_mom.set_output_folder(output_folder)
output_mom.add_output_field("u", "Displacement (m)")
output_mom.add_output_field("eps_tot", "Total strain (-)")
output_mom.add_output_field("sig", "Stress (Pa)")
output_mom.add_output_field("p_elems", "Mean stress (Pa)")
output_mom.add_output_field("q_elems", "Von Mises stress (Pa)")
""",
language="python")

st.code(
"""
sim = Simulator_M(mom_eq, t_control, outputs, True)
sim.run()
""",
language="python")



save_session_state()

