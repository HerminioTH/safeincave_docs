import streamlit as st

st.set_page_config(layout="wide") 
st.markdown(" ## Problem description")
st.write("Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.")

st.code(
"""
import safeincave as sf
import safeincave.Utils as ut
import safeincave.MomentumBC as momBC
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
class LinearMomentumMod(sf.LinearMomentum):
	def __init__(self, grid, theta):
		super().__init__(grid, theta)

	def initialize(self) -> None:
		self.C.x.array[:] = to.flatten(self.mat.C)
		self.Fvp = do.fem.Function(self.DG0_1)
		self.alpha = do.fem.Function(self.DG0_1)
		self.eps_vp = do.fem.Function(self.DG0_3x3)

	def run_after_solve(self):
		try:
			self.eps_vp.x.array[:] = to.flatten(self.mat.elems_ne[-1].eps_ne_k)
			self.Fvp.x.array[:] = self.mat.elems_ne[-1].Fvp
			self.alpha.x.array[:] = self.mat.elems_ne[-1].alpha
		except:
			pass
""",
language="python")

st.code(
"""
grid_path = os.path.join("..", "..", "..", "grids", "cavern_irregular")
grid = sf.GridHandlerGMSH("geom", grid_path)
""",
language="python")

st.code(
"""
output_folder = os.path.join("output", "case_2")
""",
language="python")

st.code(
"""
mom_eq = LinearMomentumMod(grid, theta=0.5)
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
mat = sf.Material(mom_eq.n_elems)
""",
language="python")

st.code(
"""
salt_density = 2000
rho = salt_density*to.ones(mom_eq.n_elems, dtype=to.float64)
mat.set_density(rho)
""",
language="python")

st.code(
"""
E0 = 102*ut.GPa*to.ones(mom_eq.n_elems)
nu0 = 0.3*to.ones(mom_eq.n_elems)
spring_0 = sf.Spring(E0, nu0, "spring")
""",
language="python")

st.code(
"""
eta = 105e11*to.ones(mom_eq.n_elems)
E1 = 10*ut.GPa*to.ones(mom_eq.n_elems)
nu1 = 0.32*to.ones(mom_eq.n_elems)
kelvin = sf.Viscoelastic(eta, E1, nu1, "kelvin")
""",
language="python")

st.code(
"""
A = 1.9e-20*to.ones(mom_eq.n_elems)
Q = 51600*to.ones(mom_eq.n_elems)
n = 3.0*to.ones(mom_eq.n_elems)
creep_0 = sf.DislocationCreep(A, Q, n, "creep")
""",
language="python")

st.code(
"""
mat.add_to_elastic(spring_0)
mat.add_to_non_elastic(kelvin)
mat.add_to_non_elastic(creep_0)
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
T0_field = 298*to.ones(mom_eq.n_elems)
mom_eq.set_T0(T0_field)
mom_eq.set_T(T0_field)
""",
language="python")

st.code(
"""
tc_equilibrium = sf.TimeController(dt=0.5, initial_time=0.0, final_time=10, time_unit="hour")
""",
language="python")

st.code(
"""
bc_west = momBC.DirichletBC(boundary_name = "West", 
					component = 0,
					values = [0.0, 0.0],
					time_values = [0.0, tc_equilibrium.t_final])
bc_bottom = momBC.DirichletBC(boundary_name = "Bottom", 
					component = 2,
					values = [0.0, 0.0],
					time_values = [0.0, tc_equilibrium.t_final])
bc_south = momBC.DirichletBC(boundary_name = "South", 
					component = 1,
					values = [0.0, 0.0],
					time_values = [0.0, tc_equilibrium.t_final])
side_burden = 10.0*ut.MPa
bc_east = momBC.NeumannBC(boundary_name = "East",
					direction = 2,
					density = salt_density,
					ref_pos = 660.0,
					values = [side_burden, side_burden],
					time_values = [0.0, tc_equilibrium.t_final],
					g = g_vec[2])
bc_north = momBC.NeumannBC(boundary_name = "North",
					direction = 2,
					density = salt_density,
					ref_pos = 660.0,
					values = [side_burden, side_burden],
					time_values = [0.0, tc_equilibrium.t_final],
					g = g_vec[2])
over_burden = 10.0*ut.MPa
bc_top = momBC.NeumannBC(boundary_name = "Top",
					direction = 2,
					density = 0.0,
					ref_pos = 0.0,
					values = [over_burden, over_burden],
					time_values = [0.0, tc_equilibrium.t_final],
					g = g_vec[2])
gas_density = 0.082
p_gas = 10.0*ut.MPa
bc_cavern = momBC.NeumannBC(boundary_name = "Cavern",
					direction = 2,
					density = gas_density,
					# density = 0.082,
					ref_pos = 430.0,
					values = [p_gas, p_gas],
					time_values = [0.0, tc_equilibrium.t_final],
					g = g_vec[2])
""",
language="python")

st.code(
"""
bc_equilibrium = momBC.BcHandler(mom_eq)
bc_equilibrium.add_boundary_condition(bc_west)
bc_equilibrium.add_boundary_condition(bc_bottom)
bc_equilibrium.add_boundary_condition(bc_south)
bc_equilibrium.add_boundary_condition(bc_east)
bc_equilibrium.add_boundary_condition(bc_north)
bc_equilibrium.add_boundary_condition(bc_top)
bc_equilibrium.add_boundary_condition(bc_cavern)
""",
language="python")

st.code(
"""
mom_eq.set_boundary_conditions(bc_equilibrium)
""",
language="python")

st.code(
"""
ouput_folder_equilibrium = os.path.join(output_folder, "equilibrium")
""",
language="python")

st.code(
"""
output_mom = sf.SaveFields(mom_eq)
output_mom.set_output_folder(ouput_folder_equilibrium)
output_mom.add_output_field("u", "Displacement (m)")
output_mom.add_output_field("eps_tot", "Total strain (-)")
output_mom.add_output_field("sig", "Stress (Pa)")
output_mom.add_output_field("p_elems", "Mean stress (Pa)")
output_mom.add_output_field("q_elems", "Von Mises stress (Pa)")
outputs = [output_mom]
""",
language="python")

st.code(
"""
sim = sf.Simulator_M(mom_eq, tc_equilibrium, outputs, True)
sim.run()
""",
language="python")

st.markdown("### Operation stage")

st.code(
"""
mu_1 = 5.3665857009859815e-11*to.ones(mom_eq.n_elems)
N_1 = 3.1*to.ones(mom_eq.n_elems)
n = 3.0*to.ones(mom_eq.n_elems)
a_1 = 1.965018496922832e-05*to.ones(mom_eq.n_elems)
eta = 0.8275682807874163*to.ones(mom_eq.n_elems)
beta_1 = 0.0048*to.ones(mom_eq.n_elems)
beta = 0.995*to.ones(mom_eq.n_elems)
m = -0.5*to.ones(mom_eq.n_elems)
gamma = 0.095*to.ones(mom_eq.n_elems)
alpha_0 = 0.0022*to.ones(mom_eq.n_elems)
sigma_t = 5.0*to.ones(mom_eq.n_elems)
desai = sf.ViscoplasticDesai(mu_1, N_1, a_1, eta, n, beta_1, beta, m, gamma, sigma_t, alpha_0, "desai")
""",
language="python")

st.code(
"""
stress_to = ut.numpy2torch(mom_eq.sig.x.array.reshape((mom_eq.n_elems, 3, 3)))
desai.compute_initial_hardening(stress_to, Fvp_0=0.0)
""",
language="python")

st.code(
"""
mom_eq.mat.add_to_non_elastic(desai)
""",
language="python")

st.code(
"""
tc_operation = sf.TimeController(dt=0.1, initial_time=0.0, final_time=24, time_unit="hour")
""",
language="python")

st.code(
"""
bc_west = momBC.DirichletBC(boundary_name = "West", 
					component = 0,
					values = [0.0, 0.0],
					time_values = [0.0, tc_operation.t_final])
bc_bottom = momBC.DirichletBC(boundary_name = "Bottom", 
					component = 2,
					values = [0.0, 0.0],
					time_values = [0.0, tc_operation.t_final])
bc_south = momBC.DirichletBC(boundary_name = "South", 
					component = 1,
					values = [0.0, 0.0],
					time_values = [0.0, tc_operation.t_final])
bc_east = momBC.NeumannBC(boundary_name = "East",
					direction = 2,
					density = salt_density,
					ref_pos = 660.0,
					values =      [side_burden, side_burden],
					time_values = [0.0,            tc_operation.t_final],
					g = g_vec[2])
bc_north = momBC.NeumannBC(boundary_name = "North",
					direction = 2,
					density = salt_density,
					ref_pos = 660.0,
					values =      [side_burden, side_burden],
					time_values = [0.0,            tc_operation.t_final],
					g = g_vec[2])
bc_top = momBC.NeumannBC(boundary_name = "Top",
					direction = 2,
					density = 0.0,
					ref_pos = 0.0,
					values =      [over_burden, over_burden],
					time_values = [0.0,            tc_operation.t_final],
					g = g_vec[2])
bc_cavern = momBC.NeumannBC(boundary_name = "Cavern",
					direction = 2,
					density = gas_density,
					ref_pos = 430.0,
					values =      [10.0*ut.MPa, 7.0*ut.MPa, 7.0*ut.MPa, 10.0*ut.MPa, 10.0*ut.MPa],
					time_values = [0.0, 2.0*ut.hour, 14*ut.hour, 16*ut.hour, 24*ut.hour],
					g = g_vec[2])
""",
language="python")

st.code(
"""
bc_operation = momBC.BcHandler(mom_eq)
bc_operation.add_boundary_condition(bc_west)
bc_operation.add_boundary_condition(bc_bottom)
bc_operation.add_boundary_condition(bc_south)
bc_operation.add_boundary_condition(bc_east)
bc_operation.add_boundary_condition(bc_north)
bc_operation.add_boundary_condition(bc_top)
bc_operation.add_boundary_condition(bc_cavern)
""",
language="python")

st.code(
"""
mom_eq.set_boundary_conditions(bc_operation)
""",
language="python")

st.code(
"""
output_folder_operation = os.path.join(output_folder, "operation")
""",
language="python")

st.code(
"""
output_mom = sf.SaveFields(mom_eq)
output_mom.set_output_folder(output_folder_operation)
output_mom.add_output_field("u", "Displacement (m)")
output_mom.add_output_field("eps_tot", "Total strain (-)")
output_mom.add_output_field("eps_vp", "Viscoplastic strain (-)")
output_mom.add_output_field("alpha", "Hardening parameter (-)")
output_mom.add_output_field("Fvp", "Yield function (-)")
output_mom.add_output_field("p_elems", "Mean stress (Pa)")
output_mom.add_output_field("q_elems", "Von Mises stress (Pa)")
outputs = [output_mom]
""",
language="python")

st.code(
"""
sim = sf.Simulator_M(mom_eq, tc_operation, outputs, False)
sim.run()
""",
language="python")
