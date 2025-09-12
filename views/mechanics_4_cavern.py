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



st.markdown(" ## Example 4: Salt cavern with overburden")
st.write("This example is located in our [repository](https://gitlab.tudelft.nl/ADMIRE_Public/safeincave).")

st.markdown(" ## Goals")

st.write(
	"""
	1. Set constitutive models to overburden and salt formations
	2. Calculate lithostatic pressure
	3. Specify a non-uniform temperature distribution
	""")


st.markdown(" ## Problem description")

fig_4_cavern_geom = create_fig_tag("fig_4_cavern_geom")

st.write(f"This problem simulates gas storage in a cavern constructed in a salt layer under a non-salt overburden, as shown in Fig. {fig_4_cavern_geom}-a. The temperature distribution follows the geothermal gradient, as also indicated in Fig. {fig_4_cavern_geom}-a. The operation condition of the cavern is defined based on the local lithostatic pressure, such that the minimum gas pressure is 20% of lithostatis pressure, and the maximum gas pressure is 80% of lithostatic pressure. The pressure schedule imposed on the cavern is illustrated in Fig. {fig_4_cavern_geom}-b, which also shows the **Equilibrium** and **Operation** stages.")

fig_4_cavern_geom = figure(os.path.join("assets", "mechanics", "4_cavern_geom.png"), "Problem setup details.", "fig_4_cavern_geom", size=700)

fig_4_cavern_names = create_fig_tag("fig_4_cavern_names")

st.write(f"The region names and boundary names are indicated in Fig. {fig_4_cavern_names}.")

fig_4_cavern_names = figure(os.path.join("assets", "4_cavern_names.png"), "Region names and boundary names.", "fig_4_cavern_names", size=700)

fig_4_cavern_model = create_fig_tag("fig_4_cavern_model")

st.write(f"In this example, we consider the overburden to be a purely elastic medium. For simplicity, salt creep is modeled by only dislocation creep. The constitutive models for these two regions are illustrated in Fig. {fig_4_cavern_model}.")

fig_4_cavern_model = figure(os.path.join("assets", "mechanics", "4_cavern_model.png"), "Constitutive models for salt and overburden regions.", "fig_4_cavern_model", size=400)



st.markdown(" ## Implementation")

st.write("Import the usual packages. Notice that we only import *day*, *GPa*, and *create_field_elems* from *safeincave.Utils*. Function *create_field_elems* is convenient to define any scalar field at the elements of a mesh. In this case, it will be useful the define the temperature profile.")

st.code(
"""
import safeincave as sf
from safeincave.Utils import day, GPa, create_field_elems
import safeincave.MomentumBC as momBC
from mpi4py import MPI
from petsc4py import PETSc
import torch as to
import os
import sys
""",
language="python")

st.write("Define grid path and create grid object.")

st.code(
"""
grid_path = os.path.join("..", "..", "..", "grids", "cavern_overburden_coarse")
grid = sf.GridHandlerGMSH("geom", grid_path)
""",
language="python")

st.write("Define output folder where the simulation results will be saved.")

st.code(
"""
output_folder = os.path.join("output", "case_0")
""",
language="python")

st.write(r"Initialize object for the momentum balance equation (**LinearMomentum**) and choose the fully-implicit time integration scheme ($\theta=0.0$).")

st.code(
"""
mom_eq = sf.LinearMomentum(grid, theta=0.0)
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

st.write("Extract lists of indices belonging to regions *Salt* and *Overburden*. Notice that attribute *region_indices* is a dictionary with as many keys as the number of regions in the mesh.")

st.code(
"""
ind_salt = grid.region_indices["Salt"]
ind_ovb = grid.region_indices["Overburden"]
""",
language="python")

st.write(r"Define the density field by setting 2800 kg$/$m$^3$ for the elements in the *Overburden* region, and 2200 kg$/$m$^3$ for the elements in the *Salt* region. Finally, the density vector *rho* to the **Material** object, *mat*.")

st.code(
"""
salt_density = 2200
ovb_density = 2800
rho = to.zeros(mom_eq.n_elems, dtype=to.float64)
rho[ind_salt] = salt_density
rho[ind_ovb] = ovb_density
mat.set_density(rho)
""",
language="python")

st.write(r"Define gas density in kg$/$m$^3$. In this case, we consider the hydrogen density in surface conditions.")

st.code(
"""
gas_density = 0.082
""",
language="python")

st.write(r"Define the elastic properties both regions. In this case, we consider E=102 GPa for the salt rock (*ind_salt*), and E=180 for the overburden (*ind_ovb*). Poisson's ratio is 0.3 for both regions. Finally, create a **Spring** object.")

st.code(
"""
E0 = to.zeros(mom_eq.n_elems)
E0[ind_salt] = 102*GPa
E0[ind_ovb] = 180*GPa
nu0 = 0.3*to.ones(mom_eq.n_elems)
spring_0 = sf.Spring(E0, nu0, "spring")
""",
language="python")

eq_eps_rate_ds_0 = st.session_state["eq"]["eq_eps_rate_ds_0"]

st.write(f"As indicated in Fig. {fig_4_cavern_model}, the overburden is considered to be purely elastic. Therefore, dislocation creep must take place only in the salt layer. By making the coefficient *A* of Eq. ({eq_eps_rate_ds_0}) to be equal to zero for all the elements in the *Overburden* region, we ensure that no dislocation creep will occur in this layer. Next, we create a **DislocationCreep** object.")

st.code(
"""
A = to.zeros(mom_eq.n_elems)
A[ind_salt] = 1.9e-20
A[ind_ovb] = 0.0
Q = 51600*to.ones(mom_eq.n_elems)
n = 3.0*to.ones(mom_eq.n_elems)
creep_0 = sf.DislocationCreep(A, Q, n, "creep")
""",
language="python")

st.write("Add both of the above defined elements to the **Material** object, *mat*.")

st.code(
"""
mat.add_to_elastic(spring_0)
mat.add_to_non_elastic(creep_0)
""",
language="python")

st.write("Set the material *mat* to the momentum equation object, *mom_eq*.")

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

st.write(r"The temperature profile must follow the geothermal gradient. The temperature at the surface is prescribed to be $20^o$C (293 K), and it increases with depth by 27 $^o$C/km. A convenient way of specifying this is by using function *create_field_elems*, which loops over the elements of the mesh (using the **Grid** object passed as input), evaluate a function (also passed as an input) at the element centroid coordinates (x,y,z), and associate the function value to the element. Function *create_field_elems* returns a list of function values associated to the elements of the mesh. Below, we first define a function *T_field_fun* that receives the centroid coordinates (x,y,z) of the element and calculates the corresponding temperature. Next, we execute function *create_field_elems* to generate vector *T0_field*, which is then assigned as initial and current temperature profiles to *mom_eq*.")

st.info(
	r"**_NOTE:_** This same structure can be used to defined any other quantity associated to the elements, such as elastic properties, creep properties, and so on. This offers great flexibility for working with highly heterogeneous media."
)

st.code(
"""
def T_field_fun(x,y,z):
	km = 1000
	dTdZ = 27/km
	T_surface = 20 + 273
	return T_surface - dTdZ*z
T0_field = create_field_elems(grid, T_field_fun)
mom_eq.set_T0(T0_field)
mom_eq.set_T(T0_field)
""",
language="python")

st.write(f"As illustrated in Fig. {fig_4_cavern_geom}-b, the **Equilibrium stage** is run for 5 days. However, in this example, we choose a non-uniform time discretization for the **Equilibrium stage**. Specifically, we want to perform 20 time steps with the time step size to increasing as a geometric progression from 0 until the final time (10 days). This is achieved with class **TimeControllerParabolic**.")

st.code(
"""
tc_eq = sf.TimeControllerParabolic(n_time_steps=20, initial_time=0.0, final_time=5, time_unit="day")
""",
language="python")

st.write("Initialize the boundary handler (**BcHandler**) class to hold the boundary conditions.")

st.code(
"""
bc_equilibrium = momBC.BcHandler(mom_eq)
""",
language="python")


st.write("Apply Dirichlet boundary conditions. For this, create a list of tuples containing the boundary name and the corresponding displacement component to be prescribed. Loop over that list and apply zero displacement to the corresponding boundaries and displacement components.")

st.code(
"""
boundaries = [("West_salt", 0), ("West_ovb", 0), ("East_salt", 0), ("East_ovb", 0), ("South_salt", 1), ("South_ovb", 1), ("North_salt", 1), ("North_ovb", 1), ("Bottom", 2)]
for b_name, component in boundaries:
	bc = momBC.DirichletBC(boundary_name=b_name, component=component, values=[0.0, 0.0], time_values=[0.0, tc_eq.t_final])
	bc_equilibrium.add_boundary_condition(bc)
""",
language="python")


fig_4_cavern_litho_p = create_fig_tag("fig_4_cavern_litho_p")

st.write(f"Now, we want to define boundary condition on the cavern walls. Since the gas pressure depends on the lithostatic pressure at cavern depth, as indicated in Fig. {fig_4_cavern_geom}-b, we first need to calculate the lithostatic pressure at the cavern's roof. The procedure for this calculation is illustrated in Fig. {fig_4_cavern_litho_p}, and it requires us to know the overburden thicknesss and the hanging wall." + r" Next, we calculate the lithostatic pressure $p_1$ at the interface between the overburden and the salt layer. Finally, the lithostatic pressure at the cavern's roof, $p_\text{roof}$ is calculated as also " + f"indicated in Fig. {fig_4_cavern_litho_p}.")

fig_4_cavern_litho_p = figure(os.path.join("assets", "mechanics", "4_cavern_litho_p.png"), "Lithostatic pressure calculation.", "fig_4_cavern_litho_p", size=600)

st.write("To retrieve the values of overburden thickness and hanging wall, we simply read the Gmsh geometry (.geo) file contained in the grid folder. This is done by the function *get_geometry_parameters* define below.")

st.code(
"""
def get_geometry_parameters(path_to_grid):
	f = open(os.path.join(path_to_grid, "geom.geo"), "r")
	data = f.readlines()
	ovb_thickness = float(data[10][len("ovb_thickness = "):-2])
	hanging_wall = float(data[12][len("hanging_wall = "):-2])
	return ovb_thickness, hanging_wall
ovb_thickness, hanging_wall = get_geometry_parameters(grid_path)
""",
language="python")

st.write(f"The calculations indicated in Fig. {fig_4_cavern_litho_p} are performed below. " + r" Notice that $g=-9.81$ m$/$s$^2$, hence the minus sign multiplying $g$ below.")

st.code(
"""
p_1 = salt_density*(-g)*hanging_wall
p_roof = p_1 + ovb_density*(-g)*ovb_thickness
""",
language="python")

st.write("Now that we know the lithostatic pressure at the cavern's roof, we can specify the gas pressure for the **Equilibrium stage**, which is constant and equal 80% of *p_roof* at the cavern's roof. The position of the cavern's roof is given by the overburden thickness plus the hanging wall.")

st.code(
"""
cavern_roof = ovb_thickness + hanging_wall
bc_cavern = momBC.NeumannBC(boundary_name = "Cavern",
					direction = 2,
					density = gas_density,
					ref_pos = cavern_roof,
					values = [0.8*p_roof, 0.8*p_roof],
					time_values = [0,  tc_eq.t_final],
					g = g_vec[2])
bc_equilibrium.add_boundary_condition(bc_cavern)
""",
language="python")

st.write("Finally, set the **Equilibrium stage** boundary conditions to the momentum equation object.")

st.code(
"""
mom_eq.set_boundary_conditions(bc_equilibrium)
""",
language="python")


st.write("Define output fields to be saved.")

st.code(
"""
output_mom = sf.SaveFields(mom_eq)
output_mom.set_output_folder(os.path.join(output_folder, "equilibrium"))
output_mom.add_output_field("u", "Displacement (m)")
output_mom.add_output_field("eps_tot", "Total strain (-)")
output_mom.add_output_field("p_elems", "Mean stress (Pa)")
output_mom.add_output_field("q_elems", "Von Mises stress (Pa)")
output_mom.add_output_field("p_nodes", "Mean stress (Pa)")
output_mom.add_output_field("q_nodes", "Von Mises stress (Pa)")
outputs = [output_mom]
""",
language="python")

st.write("Run the mechanical simulation for the **Equilibrium stage**.")

st.code(
"""
sim = sf.Simulator_M(mom_eq, tc_eq, outputs, compute_elastic_response=True)
sim.run()
""",
language="python")








st.markdown("### Operation stage")

st.write("For the **Operation stage** we use an equally-spaced time discretization between 0 and 10 days (240 hours). The time step size is of 2 hours.")

st.code(
"""
tc_op = sf.TimeController(dt=2, initial_time=0.0, final_time=240, time_unit="hour")
""",
language="python")

st.write("Initialize **BcHandler** object for **Operation stage**.")

st.code(
"""
bc_operation = momBC.BcHandler(mom_eq)
""",
language="python")

st.write("Apply Dirichlet boundary conditions, which are basically the same as those for **Equilibrium stage** but for another interval (i.e., between 0 and 10 days).")

st.code(
"""
boundaries = [("West_salt", 0), ("West_ovb", 0), ("East_salt", 0), ("East_ovb", 0), ("South_salt", 1), ("South_ovb", 1), ("North_salt", 1), ("North_ovb", 1), ("Bottom", 2)]
for b_name, component in boundaries:
	bc = momBC.DirichletBC(boundary_name=b_name, component=component, values=[0.0, 0.0], time_values=[0.0, tc_op.t_final])
	bc_operation.add_boundary_condition(bc)
""",
language="python")

st.write(f"Apply the gas pressure schedule illustrated in Fig. {fig_4_cavern_geom}-b, varying between 20 and 80% of lithostatic pressure.")

st.code(
"""
bc_cavern = momBC.NeumannBC(boundary_name = "Cavern",
					direction = 2,
					density = gas_density,
					ref_pos = cavern_roof,
					values = [0.8*p_roof, 0.2*p_roof, 0.2*p_roof, 0.8*p_roof, 0.8*p_roof],
					time_values = [0*ut.day,  2*ut.day,  6*ut.day, 8*ut.day, 10*ut.day],
					g = g_vec[2])
bc_operation.add_boundary_condition(bc_cavern)
""",
language="python")

st.write("Set the **Operation stage** boundary conditions to the momentum balance equation object.")

st.code(
"""
mom_eq.set_boundary_conditions(bc_operation)
""",
language="python")

st.write("Choose fields to be saved during the **Operation stage** simulation")

st.code(
"""
output_mom = sf.SaveFields(mom_eq)
output_mom.set_output_folder(os.path.join(output_folder, "operation"))
output_mom.add_output_field("u", "Displacement (m)")
# output_mom.add_output_field("Temp", "Temperature (K)")
output_mom.add_output_field("eps_tot", "Total strain (-)")
output_mom.add_output_field("p_elems", "Mean stress (Pa)")
output_mom.add_output_field("q_elems", "Von Mises stress (Pa)")
output_mom.add_output_field("p_nodes", "Mean stress (Pa)")
output_mom.add_output_field("q_nodes", "Von Mises stress (Pa)")
outputs = [output_mom]
""",
language="python")

st.write("Run the mechanical simulation for the **Operation stage**.")

st.code(
"""
sim = sf.Simulator_M(mom_eq, tc_op, outputs, False)
sim.run()
""",
language="python")

save_session_state()
