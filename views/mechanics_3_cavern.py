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

import plotly.express as px
import numpy as np
import pandas as pd

from setup import run_setup

run_setup()

st.set_page_config(layout="wide") 


st.markdown(" ## Example 3: Salt cavern")
st.write("This example is located in our [repository](https://github.com/ADMIRE-Public/SafeInCave).")

st.markdown(" ## Goals")

st.write(
	"""
	1. Solve equilibrium stage (initial condition)
	2. Solve operation stage
	3. Apply non-uniform Neumann boundary condition
	4. Retrieve stress field from **LinearMomentum** object
	5. Save custom fields.
	""")




st.markdown(" ## Problem description")

fig_3_cavern_geom = create_fig_tag("fig_3_cavern_geom")

st.write(f"In this example we illustrate how to set up a basic simulation for gas storage in a salt cavern. We take advantage of radial symmetry and only consider a quarter of the domain around a single cavern, as illustrated in Fig. {fig_3_cavern_geom}-a, where the dimensions of the mesh are also shown. In Fig. {fig_3_cavern_geom}-b, the boundary and region names are indicated.")

fig_3_cavern_geom = figure(os.path.join("assets", "3_cavern_geom.png"), "(a) Geometry and boundary names; (b) Axial load and confining pressure history; (c) Lists of values informed to the simulator.", "fig_3_cavern_geom", size=500)

fig_3_cavern_bcs = create_fig_tag("fig_3_cavern_bcs")

st.write(f"For this problem, only the salt layer is considered, so a load of 10 MPa is applied on the *Top* boundary to represent the presence of an overburden. The sideburden is applied on faces *East* and *North* as following the lithostatic pressure, as shown in Fig. {fig_3_cavern_bcs}-a. Faces *Bottom*, *West*, and *South* are prevented from normal displacement. The value of the gas pressure is specified at the cavern's roof, and it varies in time according to the graph in Fig. {fig_3_cavern_bcs}-b. Moreover, the gas pressure increases from the roof to the bottom of the cavern according to the gas specific weight.")

st.write(f"For salt cavern simulations, specifying the initial condition of the system (stresses, inelastic strains, etc) is very important and it's not a trivial task. The approach we adopt here is to divide the simulation in two stages. The first stage is referred to as **Equilibrium stage**, which is where the initial conditions are calculated. In the **Equilibrium stage**, the simulation is run considering zero initial stresses and zero initial strains in the entire domain. The domain is subjected to the boundary conditions shown in Fig. {fig_3_cavern_bcs}-a, and to a constant gas pressure, as indicated in Fig. {fig_3_cavern_bcs}-b. For this case, the Equilibrium stage is simulated for 10 hours, during which the whole structure settles down under its own weight and the external loads (boundary conditions). By the end of the Equilibrium stage, the resulting stress and strain fields are taken as initial condition for the subsequent **Operation stage**, where the actual time varying conditions are applied inside the cavern.")



fig_3_cavern_bcs = figure(os.path.join("assets", "mechanics", "3_cavern_bcs.png"), "(a) Geometry and boundary names; (b) Axial load and confining pressure history; (c) Lists of values informed to the simulator.", "fig_3_cavern_bcs", size=600)

fig_3_cavern_model = create_fig_tag("fig_3_cavern_model")

eq_alpha_init = st.session_state["eq"]["eq_alpha_init"]

st.write(f"The constitutive model for this example includes a viscoplastic element to represent transient creep, a viscoelastic element for reverse transient creep, and a dashpot to describe dislocation creep deformation. This is illustrated in Fig. {fig_3_cavern_model}. However, in the Equilibrium stage the initial stresses can achieve excessively high values, making the viscoplastic element to face convergence problems. Therefore, we do not consider the viscoplastic element in the constitutive model during the Equilibrium stage. The viscoplastic element is added to the constitutive model when the Operation stage begins. Moreover, the initial hardening parameter " + r"$\alpha_0$" + f" is calculated using Eq. ({eq_alpha_init}) such that every point in the domain is on the onset of viscoplastic deformation.")


fig_3_cavern_model = figure(os.path.join("assets", "mechanics", "3_cavern_model.png"), "Constitutive model for the operation stage. The viscoplastic element (desai) is not considered during equilibrium stage.", "fig_3_cavern_model", size=400)



st.markdown(" ## Implementation")

st.write("Import the usual packages, including dolfinx. The reason we import dolfinx here is to define a sub class of **LinearMomentum** so that we can save some custom fields, such as viscoplastic strains, yield function, and hardening parameter values.")

st.code(
"""
import safeincave as sf
import safeincave.Utils as ut
import safeincave.MomentumBC as momBC
from petsc4py import PETSc
import dolfinx as do
import torch as to
import os
""",
language="python")

st.write("Define the modified **LinearMomentum** class, where the internal fields *Fvp*, *alpha*, and *eps_vp* are initialized. See [Example 1: Triaxial](mechanics_1_triaxial) for further details on this procedure.")

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

st.write("Define grid path and create grid object.")

st.code(
"""
grid_path = os.path.join("..", "..", "..", "grids", "cavern_irregular")
grid = sf.GridHandlerGMSH("geom", grid_path)
""",
language="python")

st.write("Define output folder where the simulation results will be saved.")

st.code(
"""
output_folder = os.path.join("output", "case_0")
""",
language="python")

st.write(r"Initialize object for the modified momentum balance equation (**LinearMomentumMod**) and choose Crank-Nicolson as a time integration scheme ($\theta=0.5$).")

st.code(
"""
mom_eq = LinearMomentumMod(grid, theta=0.5)
""",
language="python")

st.write("Define solver for momentum balance equation, choose Conjugate Gradient as a linear solver with Additive Schwartz preconditioner, and set this solver to the momentum balance equation object, *mom_eq*.")

st.code(
"""
mom_solver = PETSc.KSP().create(grid.mesh.comm)
mom_solver.setType("cg")
mom_solver.getPC().setType("asm")
mom_solver.setTolerances(rtol=1e-12, max_it=100)
mom_eq.set_solver(mom_solver)
""",
language="python")

st.write("Initialize **Material** object, which will contain all material properties and the constitutive model.")

st.code(
"""
mat = sf.Material(mom_eq.n_elems)
""",
language="python")

st.write(r"Define a uniform salt density of 2000 kg$/$m$^3$ and set it to the **Material** object.")

st.code(
"""
salt_density = 2000
rho = salt_density*to.ones(mom_eq.n_elems, dtype=to.float64)
mat.set_density(rho)
""",
language="python")

st.write(r"Define uniform elastic properties and create a **Spring** element.")

st.code(
"""
E0 = 102*ut.GPa*to.ones(mom_eq.n_elems)
nu0 = 0.3*to.ones(mom_eq.n_elems)
spring_0 = sf.Spring(E0, nu0, "spring")
""",
language="python")

st.write(r"Define uniform viscoelastic properties and create a **Viscoelastic** element.")

st.code(
"""
eta = 105e11*to.ones(mom_eq.n_elems)
E1 = 10*ut.GPa*to.ones(mom_eq.n_elems)
nu1 = 0.32*to.ones(mom_eq.n_elems)
kelvin = sf.Viscoelastic(eta, E1, nu1, "kelvin")
""",
language="python")

st.write(r"Define uniform dislocation creep properties and create a **DislocationCreep** element.")

st.code(
"""
A = 1.9e-20*to.ones(mom_eq.n_elems)
Q = 51600*to.ones(mom_eq.n_elems)
n = 3.0*to.ones(mom_eq.n_elems)
creep_0 = sf.DislocationCreep(A, Q, n, "creep")
""",
language="python")

st.write(r"Add object *spring_0* to the list of elastic elements of *mat*, and objects *kelvin* and *creep_0* to the list of non-elastic elements of *mat*.")

st.code(
"""
mat.add_to_elastic(spring_0)
mat.add_to_non_elastic(kelvin)
mat.add_to_non_elastic(creep_0)
""",
language="python")

st.write("Set material to the momentum equation object.")

st.code(
"""
mom_eq.set_material(mat)
""",
language="python")

st.write("Define gravity acceleration vector and assign it to *mom_eq* so it builds the body force terms.")

st.code(
"""
g = -9.81
g_vec = [0.0, 0.0, g]
mom_eq.build_body_force(g_vec)
""",
language="python")

st.write("Define an uniform temperature field of 298 K and assign it to both initial and current temperature.")

st.code(
"""
T0_field = 298*to.ones(mom_eq.n_elems)
mom_eq.set_T0(T0_field)
mom_eq.set_T(T0_field)
""",
language="python")

st.markdown(" ### Equilibrium stage")

st.write("For the equilibrium stage, we run the simulation at a constant gas pressure of 10 MPa (defined at the cavern's roof) for a period of 10 hours and a time step size of 0.5 hous. For this, we use **TimeController** to create an equally spaced time discretization.")

st.code(
"""
tc_equilibrium = sf.TimeController(dt=0.5, initial_time=0.0, final_time=10, time_unit="hour")
""",
language="python")


st.write("As mentioned in the introduction, zero normal displacement is imposed on faces *West*, *South*, and *Bottom*.")

st.code(
"""
bc_west = momBC.DirichletBC(boundary_name = "West", 
					component = 0,
					values = [0.0, 0.0],
					time_values = [0.0, tc_equilibrium.t_final])
bc_south = momBC.DirichletBC(boundary_name = "South", 
					component = 1,
					values = [0.0, 0.0],
					time_values = [0.0, tc_equilibrium.t_final])
bc_bottom = momBC.DirichletBC(boundary_name = "Bottom", 
					component = 2,
					values = [0.0, 0.0],
					time_values = [0.0, tc_equilibrium.t_final])
""",
language="python")

st.write("The sideburden is applied on faces *East* and *North* following the lithostatic pressure, that is,")

eq_litho = equation(
	r"p(z) = \text{values} + \text{density} * g * (\text{ref\_pos} - z) = 10\text{ MPa} + 2000*9.81*(660 - z)",
	"eq_litho"
)

st.write(f"According to Eq. ({eq_litho}), the lithostatic pressure increases with the *z* direction. This is informed to **NeumannBC** class by the argument *direction=2*. Additionally, we emphasize that *ref_pos* is the position where *values* is specified. In this case, we specify 10 MPa (*values*) at position *z=660* m (*ref_pos*).")

st.code(
"""
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
""",
language="python")

st.write(f"For the *Top* boundary, a uniform overburden of 10 MPa is imposed. Since this boundary is located at constant *z=660* m, Eq. {eq_litho} will always return 10 MPa. But this is only true if the argument *direction* is set to 2 below. If the user inadvertently changes *direction* to 0 or 1, then the overburden will vary in space. To make sure this does not happen, we set *density* to 0 for the **NeumannBC** on the *Top* boundary.")

st.code(
"""
over_burden = 10.0*ut.MPa
bc_top = momBC.NeumannBC(boundary_name = "Top",
					direction = 2,
					density = 0.0,
					ref_pos = 0.0,
					values = [over_burden, over_burden],
					time_values = [0.0, tc_equilibrium.t_final],
					g = g_vec[2])
""",
language="python")

st.write(f"For the equilibrium stage, a constant gas pressure of 10 MPa is applied on the *Cavern* walls. As with the sideburden, the gas pressure also increases with the *z* direction (i.e., *direction=2*), but according to its own specific weight (*gas_density*).")

st.code(
"""
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

st.write("Create a **BcHandler** object and add the above defined boundary conditions to it.")

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

st.write("Set the **BcHandler** object to the momentum balance equation object, *mom_eq*.")

st.code(
"""
mom_eq.set_boundary_conditions(bc_equilibrium)
""",
language="python")

st.write("Define output folder to save the results of the equilibrium stage simulation.")

st.info(
	r"**_NOTE:_** The equilibrium stage results are not particularly useful, since the only purpose of this stage is to calculate the initial condition for the operation stage. Therefore, saving the equilibrium stage results is is optional."
)

st.code(
"""
ouput_folder_equilibrium = os.path.join(output_folder, "equilibrium")
""",
language="python")

st.write("Choose fields to be saved during the **Equilibrium stage** simulation.")

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

st.write("Create the mechanical simulator **Simulator_M** and run the simulation for the **Equilibrium stage**.")

st.code(
"""
sim = sf.Simulator_M(mom_eq, tc_equilibrium, outputs, compute_elastic_response=True)
sim.run()
""",
language="python")




st.markdown("### Operation stage")

st.write("As mentioned before, the viscoplastic element is included in the constitutive model to run the **Operation stage**. Below, we create an object of class **ViscoplasticDesai**.")

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

st.write(f"When creating the object *desai* above, a uniform initial hardening parameter field *alpha_0* was set 0.0022. However, now we want to calculate a distribution of *alpha_0* that makes the stresses at each element of the mesh to be on the onset of vicoplasticity. This is performed according to Eq. ({eq_alpha_init}), as mentioned before. For this purpose we need to retrieve the initial stress field, which is stored inside object *mom_eq* under the attribute *sig*. The attribute *sig* is a *dolfinx.fem.Function* structure, and we use method '*.x.array*' to convert it to a numpy array and we reshape it to (n_elems, 3, 3). That is, each element of the mesh stores a 3x3 matrix representing its stress tensor. Next, we use function *numpy2torch* to convert the numpy array to a *pytorch tensor*. The pytorch stress field *stress_to* is then passed to method *compute_initial_hardening*, which calculates the alpha_0 field such that *Fvp=0* (that is, all points are on the yield surface).")


st.code(
"""
stress_to = ut.numpy2torch(mom_eq.sig.x.array.reshape((mom_eq.n_elems, 3, 3)))
desai.compute_initial_hardening(stress_to, Fvp_0=0.0)
""",
language="python")

st.write("Now that the **ViscoplasticDesai** object is created, add it to the non-elastic list of the **Material** object, *mat*, belonging to *mom_eq*.")

st.code(
"""
mom_eq.mat.add_to_non_elastic(desai)
""",
language="python")

st.write("Create equally spaced time discretization with a fixed time step size of 0.1 hour and a final time of 24 hours.")

st.code(
"""
tc_operation = sf.TimeController(dt=0.1, initial_time=0.0, final_time=24, time_unit="hour")
""",
language="python")

st.write("Impose zero normal displacement on boundaries *Bottom*, *South*, and *East*.")

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
""",
language="python")

st.write("")

st.write("Apply the same sideburden as in the **Equilibrium stage**, that is, following lithostatic pressure.")

st.info(
	r"**_NOTE:_** Although the boundary conditions applied to *East*, *North*, and *Top* in the **Equilibrium stage** are the same for the **Operation stage**, the same objects *bc_east*, *bc_north*, and *bc_top*, defined before, cannot be reused here because the *time_values* are different."
)

st.code(
"""
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
""",
language="python")

st.write(f"Apply the gas pressure on the *Cavern* walls according to the pressure schedule shown in Fig. {fig_3_cavern_bcs}-b.")

st.code(
"""
bc_cavern = momBC.NeumannBC(boundary_name = "Cavern",
					direction = 2,
					density = gas_density,
					ref_pos = 430.0,
					values =      [10.0*ut.MPa, 7.0*ut.MPa, 7.0*ut.MPa, 10.0*ut.MPa, 10.0*ut.MPa],
					time_values = [0.0, 2.0*ut.hour, 14*ut.hour, 16*ut.hour, 24*ut.hour],
					g = g_vec[2])
""",
language="python")

st.write("Create a **BcHandler** object and add the above defined boundary conditions to it.")

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

st.write("Set the **BcHandler** object to the momentum balance equation object, *mom_eq*.")

st.code(
"""
mom_eq.set_boundary_conditions(bc_operation)
""",
language="python")

st.write("Define output folder to save the results of the **Operation stage** simulation.")

st.code(
"""
output_folder_operation = os.path.join(output_folder, "operation")
""",
language="python")

st.write("Choose fields to be saved during the **Operation stage** simulation.")

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

st.write("Create the mechanical simulator **Simulator_M** and run the simulation for the **Operation stage**.")

st.info(
	r"**_NOTE:_** For the **Operation stage** it is not necessary to calculate the purely elastic response before the simulation begins, hence *compute_elastic_response=False*. This is because the initial condition has been already calculated in the **Equilibrium stage**. Calculating the elastic response here would overwrite the stress field calculated in the **Equilibrium stage**."
)

st.code(
"""
sim = sf.Simulator_M(mom_eq, tc_operation, outputs, compute_elastic_response=False)
sim.run()
""",
language="python")







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
		fig_cavern.update_layout(yaxis={"scaleanchor": "x", "scaleratio": 1}, xaxis_title="x (m)", yaxis_title="z (m)")

		mask = (df["Time"] == time_list[0])
		fig_cavern.add_scatter(x=df[mask]["dx"], y=df[mask]["dz"], mode="lines+markers", line=dict(color="#5abcff"), marker=dict(size=10))
		fig_cavern.data[1].name = "Initial shape"

		mask = (df["Time"] == time_list[time_id])
		fig_cavern.add_scatter(x=df[mask]["dx"], y=df[mask]["dz"], mode="lines", line=dict(color="#FF57AE"))
		fig_cavern.data[2].name = "Current shape"

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


st.markdown(" ## Results Viewer")

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

save_session_state()