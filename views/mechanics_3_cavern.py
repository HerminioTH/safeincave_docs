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







import plotly.express as px
import numpy as np
import pandas as pd
import os
import sys

hour = 60*60
day = 24*hour
MPa = 1e6

def read_mesh():
	if "mesh_on" not in st.session_state:
		mesh_file = os.path.join("assets", "results", "mechanics", "3_cavern", "mesh_on.html")
		with open(mesh_file, "r", encoding="utf-8") as file:
			html_content = file.read()
			st.session_state["mesh_on"] = {
				"data" : html_content,
				"name" : file.name
			}
	if "mesh_off" not in st.session_state:
		mesh_file = os.path.join("assets", "results", "mechanics", "3_cavern", "mesh_off.html")
		with open(mesh_file, "r", encoding="utf-8") as file:
			html_content = file.read()
			st.session_state["mesh_off"] = {
				"data" : html_content,
				"name" : file.name
			}

def read_csv_file(field_name):
	if field_name not in st.session_state:
		file = os.path.join("assets", "results", "mechanics", "3_cavern", f"{field_name}.csv")
		st.session_state[field_name] = {
			"data" : pd.read_csv(file, index_col=0),
			"name" : field_name
		}

read_csv_file("cavern_displacements")
read_csv_file("subsidence")
read_csv_file("gas_pressure")
read_csv_file("convergence")
read_csv_file("stress_path")
read_mesh()

def show_geometry():
	col11.subheader(f"Geometry view")
	col1_11, col2_12 = col11.columns([1,4])
	radio_value = col1_11.radio("Mesh:", ["on","off"], index=1, horizontal=False)
	mesh_state = f"mesh_{radio_value}"
	if mesh_state not in st.session_state:
		col11.warning(f"Upload {mesh_state}.html file.")
	else:
		with col2_12:
			st.components.v1.html(st.session_state[mesh_state]["data"], height=400, width=350, scrolling=True)
	# col11.write("Something in here.")

def plot_subsidence():
	col32.subheader(f"Subsidence")
	if "subsidence" not in st.session_state:
		col32.warning("Upload subsidence.csv file.")
	else:
		df = st.session_state["subsidence"]["data"]
		uz = df["Subsidence"].values
		uz = uz - uz[0]
		time_list = df["Time"].values
		time_id = st.session_state["Time"]["index"]
		current_time = time_list[time_id]

		fig_subs = px.line()
		fig_subs.add_scatter(x=time_list/day, y=uz*100, mode="lines", line=dict(color="#5abcff"), showlegend=False)
		fig_subs.update_layout(xaxis_title="Time (days)", yaxis_title="Subsidence (cm)")

		marker_props = dict(color='white', size=8, symbol='0', line=dict(width=2, color='black'))
		df_scatter = pd.DataFrame({"x": [current_time/day], "y": [uz[time_id]*100]})
		fig_subs.add_scatter(x=df_scatter["x"], y=df_scatter["y"], mode="markers", line=dict(color='white'), marker=marker_props, showlegend=False)

		col32.plotly_chart(fig_subs, theme="streamlit", use_container_width=True)

def plot_convergence():
	col33.subheader(f"Cavern convergence")
	if "convergence" not in st.session_state:
		col33.warning("Upload convergence.csv file.")
	else:
		df = st.session_state["convergence"]["data"]
		volumes = df["Volume"].values
		time_list = df["Time"].values
		time_id = st.session_state["Time"]["index"]
		current_time = time_list[time_id]

		fig_conv = px.line()
		fig_conv.add_scatter(x=time_list/day, y=volumes, mode="lines", line=dict(color="#5abcff"), showlegend=False)
		fig_conv.update_layout(xaxis_title="Time (days)", yaxis_title="Volumetric loss (%)")

		time_id = st.session_state["Time"]["index"]
		marker_props = dict(color='white', size=8, symbol='0', line=dict(width=2, color='black'))
		df_scatter = pd.DataFrame({"x": [current_time/day], "y": [volumes[time_id]]})
		fig_conv.add_scatter(x=df_scatter["x"], y=df_scatter["y"], mode="markers", line=dict(color='white'), marker=marker_props, showlegend=False)

		col33.plotly_chart(fig_conv, theme="streamlit", use_container_width=True)

def plot_gas_pressure():
	col31.subheader(f"Gas pressure")
	if "gas_pressure" not in st.session_state:
		col31.warning("Upload gas_pressure.csv file.")
	else:
		df = st.session_state["gas_pressure"]["data"]
		p_gas = -df["Pressure"].values
		time_list = df["Time"].values
		time_global = st.session_state["Time"]["data"]
		time_id = st.session_state["Time"]["index"]

		current_p = -np.interp(time_global[time_id], df.Time.values, df.Pressure.values)
		current_time = np.interp(time_global[time_id], df.Time.values, df.Time.values)

		fig_conv = px.line()
		fig_conv.add_scatter(x=time_list/day, y=p_gas/MPa, mode="lines", line=dict(color="#FF57AE"), showlegend=False)
		fig_conv.update_layout(xaxis_title="Time (days)", yaxis_title="Gas pressure (MPa)")

		time_id = st.session_state["Time"]["index"]
		marker_props = dict(color='white', size=8, symbol='0', line=dict(width=2, color='black'))
		df_scatter = pd.DataFrame({"x": [current_time/day], "y": [current_p/MPa]})
		fig_conv.add_scatter(x=df_scatter["x"], y=df_scatter["y"], mode="markers", line=dict(color='white'), marker=marker_props, showlegend=False)

		col31.plotly_chart(fig_conv, theme="streamlit", use_container_width=True)



def create_slider():
	col21.markdown("**Select time:**")
	time_list = st.session_state["Time"]["data"]
	time_value = col22.slider("Time", time_list[0], time_list[-1], time_list[0], label_visibility="collapsed")
	diff = abs(time_list - time_value)
	idx = diff.argmin()
	st.session_state["Time"]["index"] = idx
	col23.write(f"Current time: {round(time_list[idx]/day, 2)} day(s)")

def plot_cavern():
	col12.subheader(f"Cavern shape")
	if "cavern_displacements" not in st.session_state:
		col12.warning("Upload cavern_displacements.csv file.")
	else:
		df = st.session_state["cavern_displacements"]["data"]
		time_list = st.session_state["Time"]["data"]
		time_id = st.session_state["Time"]["index"]

		fig_cavern = px.line()
		# fig_cavern.update_xaxes(range=[0, 80])
		fig_cavern.update_layout(yaxis={"scaleanchor": "x", "scaleratio": 1}, xaxis_title="x (m)", yaxis_title="z (m)")

		mask = (df["Time"] == time_list[0])
		fig_cavern.add_scatter(x=df[mask]["dx"], y=df[mask]["dz"], mode="lines+markers", line=dict(color="#5abcff"), marker=dict(size=10))
		fig_cavern.data[1].name = "Initial shape"

		# mask = (df["Time"] == time_list[-1])
		# fig_cavern.add_scatter(x=df[mask]["dx"], y=df[mask]["dz"], mode="lines", line=dict(color='lightcoral'))
		# fig_cavern.data[2].name = "Final shape"

		mask = (df["Time"] == time_list[time_id])
		fig_cavern.add_scatter(x=df[mask]["dx"], y=df[mask]["dz"], mode="lines", line=dict(color="#FF57AE"))
		fig_cavern.data[2].name = "Current shape"

		# event = col12.plotly_chart(fig_cavern, theme=None, on_select="rerun", selection_mode="points", use_container_width=True)
		event = col12.plotly_chart(fig_cavern, theme="streamlit", on_select="rerun", selection_mode="points", use_container_width=True)

		pt = event.selection["points"]
		if len(pt) > 0:
			x = pt[0]["x"]
			y = 0.0
			z = pt[0]["y"]
			st.session_state["selected_point"] = [x, y, z]
			marker_props = dict(color='red', size=8, symbol='0', line=dict(width=2, color='black'))
			fig_cavern.add_scatter(x=[x], y=[z], mode="markers", line=dict(color='red'), marker=marker_props, showlegend=False)
			fig_cavern.update_layout(clickmode="event+select")
			# with col12:
			# 	st.rerun()
			# col12.plotly_chart(fig_cavern, theme="streamlit", use_container_width=True)
			# col12.plotly_chart(fig_cavern, theme="streamlit", on_select="rerun", selection_mode="points", use_container_width=True)
		# if len(pt) > 0:
		# 	x = pt[0]["x"]
		# 	y = pt[0]["y"]
		# 	marker_props = dict(color='red', size=8, symbol='0', line=dict(width=2, color='black'))
		# 	fig_cavern.add_scatter(x=[x], y=[y], mode="markers", line=dict(color='red'), marker=marker_props, showlegend=False)
			# col12.plotly_chart(fig_cavern, theme="streamlit", on_select="rerun", selection_mode="points", use_container_width=True)

		# print(event)
		# print(event.selection)
		# print(event.selection.get("points"))
		# print(pt)

def plot_stress_path():
	col13.subheader(f"Stress path")
	if "stress_path" not in st.session_state:
		col13.warning("Upload stress_path.csv file.")
	else:
		if "selected_point" in st.session_state:
			xp = st.session_state["selected_point"][0]
			yp = st.session_state["selected_point"][1]
			zp = st.session_state["selected_point"][2]
		else:
			xp, yp, zp = 0, 0, 0

		time_list = st.session_state["Time"]["data"]
		time_id = st.session_state["Time"]["index"]
		time = time_list[time_id]

		df = st.session_state["stress_path"]["data"].copy()
		coords = df[["x", "y", "z"]].values
		d = np.sqrt(  (coords[:,0] - xp)**2
			        + (coords[:,1] - yp)**2
			        + (coords[:,2] - zp)**2 )
		idx_min = d.argmin()
		vertex_id = int(df[df["Time"] == 0.0].iloc[idx_min].iloc[0])

		fig_path = px.line()

		mask1 = (df["ID"] == vertex_id)
		fig_path.add_scatter(x=df["p"][mask1]/MPa, y=df["q"][mask1]/MPa, mode="lines", line=dict(color="#5abcff"), showlegend=False)

		marker_props = dict(color='white', size=8, symbol='0', line=dict(width=2, color='black'))
		mask2 = (df["ID"] == vertex_id) & (df["Time"] == time)
		fig_path.add_scatter(x=df["p"][mask2]/MPa, y=df["q"][mask2]/MPa, mode="markers", line=dict(color='red'), marker=marker_props, showlegend=False)

		fig_path.update_layout(xaxis_title="Mean stress, p (MPa)", yaxis_title="Von Mises stress, q (MPa)")

		col13.plotly_chart(fig_path, theme="streamlit", use_container_width=True)


def load_time():
	if "subsidence" in st.session_state:
		df = st.session_state["subsidence"]["data"]
		time_list = df["Time"].values
		st.session_state["Time"] = {
			"data": time_list,
			"index": 0
		}
	else:
		st.session_state["Time"] = {
			"data": np.arange(10),
			"index": 0
		}


st.set_page_config(layout="wide")
st.title("Results Viewer")

col11, col12, col13 = st.columns([1,1,1])
col21, col22, col23 = st.columns([1,8,2])
col31, col32, col33 = st.columns([1,1,1])


load_time()
create_slider()
show_geometry()
plot_cavern()
plot_subsidence()
plot_stress_path()
plot_convergence()
plot_gas_pressure()