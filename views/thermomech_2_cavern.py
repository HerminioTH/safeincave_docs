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

st.markdown(" ## Example 1: Thermal effects in salt caverns")
st.write("This example is located in our [repository](https://github.com/ADMIRE-Public/SafeInCave).")

st.markdown(" ## Goals")

st.write(
	"""
	1. Set constitutive models to overburden and salt formations
	2. Set up Equilibrium stage considering only mechanics
	3. Include heat diffusion equation for the Operation stage
	4. Define pressure solution creep element
	5. Calculate lithostatic pressure
	""")


st.markdown(" ## Problem description")

fig_2_cavern_geom = create_fig_tag("fig_2_cavern_geom")

st.write("In this problem we simulate the thermomechanical behavior of a salt cavern storing a gas at constant pressure of 10 MPa and a constant temperature of . The pressure .")

fig_2_cavern_geom = figure(os.path.join("assets", "thermomechanics", "2_cavern_geom.png"), "Geometry and temperature profile", "fig_2_cavern_geom", size=700)

fig_2_cavern_model = create_fig_tag("fig_2_cavern_model")

st.write("Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.")

fig_2_cavern_model = figure(os.path.join("assets", "thermomechanics", "2_cavern_model.png"), "Geometry and temperature profile", "fig_2_cavern_model", size=400)

st.write("Import relevant packages.")

st.code(
"""
import safeincave as sf
import safeincave.Utils as ut
from safeincave.Utils import GPa, MPa, day, hour, create_field_elems, create_field_nodes
import safeincave.HeatBC as heatBC
import safeincave.MomentumBC as momBC
from petsc4py import PETSc
from mpi4py import MPI
import os
import sys
import torch as to
""",
language="python")

st.write("Build grid and define output folder.")

st.code(
"""
grid_path = os.path.join("..", "..", "..", "grids", "cavern_overburden_coarse")
grid = sf.GridHandlerGMSH("geom", grid_path)
output_folder = os.path.join("output", "case_1")
""",
language="python")

st.write("Extract region indices.")

st.code(
"""
ind_salt = grid.region_indices["Salt"]
ind_ovb = grid.region_indices["Overburden"]
""",
language="python")

st.write("Define momentum equation.")

st.code(
"""
mom_eq = sf.LinearMomentum(grid, theta=0.0)
""",
language="python")

st.write("Define solver.")

st.code(
"""
mom_solver = PETSc.KSP().create(grid.mesh.comm)
mom_solver.setType("cg")
mom_solver.getPC().setType("asm")
mom_solver.setTolerances(rtol=1e-12, max_it=100)
mom_eq.set_solver(mom_solver)
""",
language="python")

st.write("Define material properties.")

st.code(
"""
mat = sf.Material(mom_eq.n_elems)
""",
language="python")

st.write("Set material density and gas density.")

st.code(
"""
gas_density = 0.082
salt_density = 2200
ovb_density = 2800
rho = to.zeros(mom_eq.n_elems, dtype=to.float64)
rho[ind_salt] = salt_density
rho[ind_ovb] = ovb_density
mat.set_density(rho)
""",
language="python")

st.write("Now we start building the constitutive model. Let us start by the elastic element (spring).")

st.code(
"""
E0 = to.zeros(mom_eq.n_elems)
E0[ind_salt] = 102*GPa
E0[ind_ovb] = 180*GPa
nu0 = 0.3*to.ones(mom_eq.n_elems)
spring_0 = sf.Spring(E0, nu0, "spring")
""",
language="python")

st.write("Create Kelvin-Voigt viscoelastic element.")

st.code(
"""
eta = to.zeros(mom_eq.n_elems)
eta[ind_salt] = 105e11
eta[ind_ovb] = 105e20
E1 = 10*GPa*to.ones(mom_eq.n_elems)
nu1 = 0.32*to.ones(mom_eq.n_elems)
kelvin = sf.Viscoelastic(eta, E1, nu1, "kelvin")
""",
language="python")

st.write("Create dislocation creep element.")

st.code(
"""
A = to.zeros(mom_eq.n_elems)
A[ind_salt] = 1.9e-20
A[ind_ovb] = 0.0
Q = 51600*to.ones(mom_eq.n_elems)
n = 3.0*to.ones(mom_eq.n_elems)
creep_ds = sf.DislocationCreep(A, Q, n, "ds_creep")
""",
language="python")

st.write("Create pressure solution creep element.")

st.code(
"""
A = to.zeros(mom_eq.n_elems)
A[ind_salt] = 1.29e-19
A[ind_ovb] = 0.0
Q = 13184*to.ones(mom_eq.n_elems)
d = 0.01*to.ones(mom_eq.n_elems)
creep_ps = sf.PressureSolutionCreep(A, d, Q, "ps_creep")
""",
language="python")

st.write("Create thermo strain element.")

st.code(
"""
alpha = to.zeros(mom_eq.n_elems)
alpha[ind_salt] = 44e-6
alpha[ind_ovb] = 0.0
thermo = sf.Thermoelastic(alpha, "thermo")
""",
language="python")

st.write("Add elements to the constitutive model.")

st.code(
"""
mat.add_to_elastic(spring_0)
mat.add_to_thermoelastic(thermo)
mat.add_to_non_elastic(kelvin)
mat.add_to_non_elastic(creep_ds)
mat.add_to_non_elastic(creep_ps)
""",
language="python")

st.write("Set constitutive model.")

st.code(
"""
mom_eq.set_material(mat)
""",
language="python")

st.write("Set gravity acceleration vector.")

st.code(
"""
g = -9.81
g_vec = [0.0, 0.0, g]
mom_eq.build_body_force(g_vec)
""",
language="python")

st.markdown(" ### Equilibrium stage")

st.write("Set initial temperature field.")

st.code(
"""
km = 1000
dTdZ = 27/km
T_top = 273 + 20
T_field_fun = lambda x,y,z: T_top + dTdZ*(660 - z)
T0_field = create_field_elems(grid, T_field_fun)
mom_eq.set_T0(T0_field)
mom_eq.set_T(T0_field)
""",
language="python")


st.write("Time settings for equilibrium stage.")

st.code(
"""
tc_eq = sf.TimeControllerParabolic(n_time_steps=20, initial_time=0.0, final_time=10, time_unit="day")
""",
language="python")

st.write("Boundary conditions.")

st.code(
"""
bc_west_salt = momBC.DirichletBC(boundary_name="West_salt", component=0, values=[0.0, 0.0], time_values=[0.0, tc_eq.t_final])
bc_west_ovb = momBC.DirichletBC(boundary_name = "West_ovb", component=0, values=[0.0, 0.0], time_values=[0.0, tc_eq.t_final])

bc_east_salt = momBC.DirichletBC(boundary_name="East_salt", component=0, values=[0.0, 0.0], time_values=[0.0, tc_eq.t_final])
bc_east_ovb = momBC.DirichletBC(boundary_name = "East_ovb", component=0, values=[0.0, 0.0], time_values=[0.0, tc_eq.t_final])

bc_bottom = momBC.DirichletBC(boundary_name="Bottom", component=2, values=[0.0, 0.0], time_values=[0.0, tc_eq.t_final])

bc_south_salt = momBC.DirichletBC(boundary_name="South_salt", component=1, values=[0.0, 0.0], time_values=[0.0, tc_eq.t_final])
bc_south_ovb = momBC.DirichletBC(boundary_name="South_ovb", component=1, values=[0.0, 0.0], time_values=[0.0, tc_eq.t_final])

bc_north_salt = momBC.DirichletBC(boundary_name="North_salt", component=1, values=[0.0, 0.0], time_values=[0.0, tc_eq.t_final])
bc_north_ovb = momBC.DirichletBC(boundary_name="North_ovb", component=1, values=[0.0, 0.0], time_values=[0.0, tc_eq.t_final])
""",
language="python")

st.write("Extract geometry dimensions.")

st.code(
"""
def get_geometry_parameters(path_to_grid):
	f = open(os.path.join(path_to_grid, "geom.geo"), "r")
	data = f.readlines()
	ovb_thickness = float(data[10][len("ovb_thickness = "):-2])
	salt_thickness = float(data[11][len("salt_thickness = "):-2])
	hanging_wall = float(data[12][len("hanging_wall = "):-2])
	return ovb_thickness, salt_thickness, hanging_wall

ovb_thickness, salt_thickness, hanging_wall = get_geometry_parameters(grid_path)
cavern_roof = ovb_thickness + hanging_wall
""",
language="python")

st.write("Calculate lithostatic pressure at the cavern's roof.")

st.code(
"""
p_roof = - salt_density*g*hanging_wall - ovb_density*g*ovb_thickness
""",
language="python")

st.write("Apply a constant gas pressure of 80% of lithostatic pressure on the cavern walls.")

st.code(
"""
bc_cavern = momBC.NeumannBC(boundary_name = "Cavern",
					direction = 2,
					density = gas_density,
					ref_pos = cavern_roof,
					values = [0.8*p_roof, 0.8*p_roof],
					time_values = [0*day,  tc_eq.t_final],
					g = g_vec[2])
""",
language="python")

st.write("Add boundary conditions to *BcHandler* object.")

st.code(
"""
bc_equilibrium = momBC.BcHandler(mom_eq)
bc_equilibrium.add_boundary_condition(bc_west_salt)
bc_equilibrium.add_boundary_condition(bc_west_ovb)
bc_equilibrium.add_boundary_condition(bc_east_salt)
bc_equilibrium.add_boundary_condition(bc_east_ovb)
bc_equilibrium.add_boundary_condition(bc_bottom)
bc_equilibrium.add_boundary_condition(bc_south_salt)
bc_equilibrium.add_boundary_condition(bc_south_ovb)
bc_equilibrium.add_boundary_condition(bc_north_salt)
bc_equilibrium.add_boundary_condition(bc_north_ovb)
bc_equilibrium.add_boundary_condition(bc_top)
bc_equilibrium.add_boundary_condition(bc_cavern)
""",
language="python")

st.write("Set *BcHandler* object to *mom_eq*.")

st.code(
"""
mom_eq.set_boundary_conditions(bc_equilibrium)
""",
language="python")

st.write("Equilibrium output folder.")

st.code(
"""
ouput_folder_equilibrium = os.path.join(output_folder, "equilibrium")
""",
language="python")

st.write("Create output handlers.")

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

st.write("Run simulation for equilibrium stage using the purely mechanical simulator, that is, *Simulator_M* class.")

st.code(
"""
sim = sf.Simulator_M(mom_eq, tc_eq, outputs, True)
sim.run()
""",
language="python")

st.markdown(" ### Operation stage")

st.write("Time settings for operation stage.")

st.code(
"""
tc_op = sf.TimeController(dt=0.5, initial_time=0.0, final_time=240, time_unit="day")
""",
language="python")

st.write("Define heat diffusion equation.")

st.code(
"""
heat_eq = sf.HeatDiffusion(grid)
""",
language="python")

st.write("Define solver for heat diffusion equation.")

st.code(
"""
solver_heat = PETSc.KSP().create(grid.mesh.comm)
solver_heat.setType("cg")
solver_heat.getPC().setType("asm")
solver_heat.setTolerances(rtol=1e-12, max_it=100)
heat_eq.set_solver(solver_heat)
""",
language="python")

st.write("Set specific heat capacity.")

st.code(
"""
cp = 850*to.ones(heat_eq.n_elems, dtype=to.float64)
mat.set_specific_heat_capacity(cp)
""",
language="python")

st.write("Set thermal conductivity.")

st.code(
"""
k = 7*to.ones(heat_eq.n_elems, dtype=to.float64)
mat.set_thermal_conductivity(k)
""",
language="python")

st.write("Set material properties to heat_equation.")

st.code(
"""
heat_eq.set_material(mat)
""",
language="python")

st.write("Set initial temperature.")

st.code(
"""
T0_field_nodes = create_field_nodes(grid, T_field_fun)
heat_eq.set_initial_T(T0_field_nodes)
""",
language="python")

st.write("Define boundary conditions for the thermal problem.")

st.code(
"""
bc_top = heatBC.DirichletBC("Top", [T_top, T_top], [tc_op.t_initial, tc_op.t_final])
bc_bottom = heatBC.NeumannBC("Bottom", [dTdZ, dTdZ], [tc_op.t_initial, tc_op.t_final])
bc_east_salt = heatBC.NeumannBC("East_salt", [0.0, 0.0], [tc_op.t_initial, tc_op.t_final])
bc_east_ovb = heatBC.NeumannBC("East_ovb", [0.0, 0.0], [tc_op.t_initial, tc_op.t_final])
bc_west_salt = heatBC.NeumannBC("West_salt", [0.0, 0.0], [tc_op.t_initial, tc_op.t_final])
bc_west_ovb = heatBC.NeumannBC("West_ovb", [0.0, 0.0], [tc_op.t_initial, tc_op.t_final])
bc_south_salt = heatBC.NeumannBC("South_salt", [0.0, 0.0], [tc_op.t_initial, tc_op.t_final])
bc_south_ovb = heatBC.NeumannBC("South_ovb", [0.0, 0.0], [tc_op.t_initial, tc_op.t_final])
bc_north_salt = heatBC.NeumannBC("North_salt", [0.0, 0.0], [tc_op.t_initial, tc_op.t_final])
bc_north_ovb = heatBC.NeumannBC("North_ovb", [0.0, 0.0], [tc_op.t_initial, tc_op.t_final])
""",
language="python")

st.write("Define Robin boundary condition on the *Cavern* walls.")

st.code(
"""
T_gas = T_top
h_conv = 5.0
bc_cavern = heatBC.RobinBC("Cavern", [T_gas, T_gas], h_conv, [tc_op.t_initial, tc_op.t_final])
""",
language="python")


st.write("Add boundary conditions to the *BcHandler* object.")

st.code(
"""
bc_handler = heatBC.BcHandler(heat_eq)
bc_handler.add_boundary_condition(bc_top)
bc_handler.add_boundary_condition(bc_bottom)
bc_handler.add_boundary_condition(bc_east_salt)
bc_handler.add_boundary_condition(bc_east_ovb)
bc_handler.add_boundary_condition(bc_west_salt)
bc_handler.add_boundary_condition(bc_west_ovb)
bc_handler.add_boundary_condition(bc_south_salt)
bc_handler.add_boundary_condition(bc_south_ovb)
bc_handler.add_boundary_condition(bc_north_salt)
bc_handler.add_boundary_condition(bc_north_ovb)
bc_handler.add_boundary_condition(bc_cavern)
""",
language="python")

st.write("Finally, set boundary conditions to *HeatDiffusion* object.")

st.code(
"""
heat_eq.set_boundary_conditions(bc_handler)
""",
language="python")

st.write("Define boundary conditions for momentum equations.")

st.code(
"""
bc_west_salt = momBC.DirichletBC(boundary_name="West_salt", component=0, values=[0.0, 0.0], time_values=[0.0, tc_op.t_final])
bc_west_ovb = momBC.DirichletBC(boundary_name = "West_ovb", component=0, values=[0.0, 0.0], time_values=[0.0, tc_op.t_final])

bc_east_salt = momBC.DirichletBC(boundary_name="East_salt", component=0, values=[0.0, 0.0], time_values=[0.0, tc_op.t_final])
bc_east_ovb = momBC.DirichletBC(boundary_name = "East_ovb", component=0, values=[0.0, 0.0], time_values=[0.0, tc_op.t_final])

bc_bottom = momBC.DirichletBC(boundary_name="Bottom", component=2, values=[0.0, 0.0], time_values=[0.0, tc_op.t_final])

bc_south_salt = momBC.DirichletBC(boundary_name="South_salt", component=1, values=[0.0, 0.0], time_values=[0.0, tc_op.t_final])
bc_south_ovb = momBC.DirichletBC(boundary_name="South_ovb", component=1, values=[0.0, 0.0], time_values=[0.0, tc_op.t_final])

bc_north_salt = momBC.DirichletBC(boundary_name="North_salt", component=1, values=[0.0, 0.0], time_values=[0.0, tc_op.t_final])
bc_north_ovb = momBC.DirichletBC(boundary_name="North_ovb", component=1, values=[0.0, 0.0], time_values=[0.0, tc_op.t_final])

bc_top = momBC.NeumannBC(boundary_name="Top", direction=2, density=0.0, ref_pos=z_surface, values=[0, 0], time_values=[0, tc_op.t_final], g=g_vec[2])
""",
language="python")

st.write("Define list of gas pressure values and corresponding time values")

st.code(
"""
p_values = 3*[0.8*p_roof, 0.8*p_roof, 0.2*p_roof, 0.2*p_roof] + [0.8*p_roof]
t_values = [20*day*i for i in range(13)]
""",
language="python")

fig_2_cavern_p_gas = create_fig_tag("fig_2_cavern_p_gas")

st.write(f"such that the prescribed gas pressure follows the cyclic pattern shown in Fig. {fig_2_cavern_p_gas}")

fig_2_cavern_p_gas = figure(os.path.join("assets", "thermomechanics", "2_cavern_p_gas.png"), "Geometry and temperature profile", "fig_2_cavern_p_gas", size=400)

st.write("Include the above mentioned gas pressure schedule to the boundary condition on the *Cavern* walls.")

st.code(
"""
bc_cavern = momBC.NeumannBC(boundary_name = "Cavern",
					direction = 2,
					density = gas_density,
					ref_pos = cavern_roof,
					values = p_values,
					time_values = t_values,
					g = g_vec[2])
""",
language="python")

st.write("Create the *BcHandler* object, add the boundary conditions, and set it to *mom_eq*.")

st.code(
"""
bc_operation = momBC.BcHandler(mom_eq)
bc_operation.add_boundary_condition(bc_west_salt)
bc_operation.add_boundary_condition(bc_west_ovb)
bc_operation.add_boundary_condition(bc_east_salt)
bc_operation.add_boundary_condition(bc_east_ovb)
bc_operation.add_boundary_condition(bc_bottom)
bc_operation.add_boundary_condition(bc_south_salt)
bc_operation.add_boundary_condition(bc_south_ovb)
bc_operation.add_boundary_condition(bc_north_salt)
bc_operation.add_boundary_condition(bc_north_ovb)
bc_operation.add_boundary_condition(bc_top)
bc_operation.add_boundary_condition(bc_cavern)

mom_eq.set_boundary_conditions(bc_operation)
""",
language="python")

st.write("Define output folder for the operation stage.")

st.code(
"""
output_folder_operation = os.path.join(output_folder, "operation")
""",
language="python")

st.write("Create output handlers for both momentum and heat equations.")

st.code(
"""
output_mom = sf.SaveFields(mom_eq)
output_mom.set_output_folder(output_folder_operation)
output_mom.add_output_field("u", "Displacement (m)")
output_mom.add_output_field("p_elems", "Mean stress (Pa)")
output_mom.add_output_field("q_elems", "Von Mises stress (Pa)")

output_heat = sf.SaveFields(heat_eq)
output_heat.set_output_folder(output_folder_operation)
output_heat.add_output_field("T", "Temperature (K)")

outputs = [output_mom, output_heat]
""",
language="python")

st.write("Solve the thermo-mechanical problem.")

st.code(
"""
sim = sf.Simulator_TM(mom_eq, heat_eq, tc_op, outputs, False)
sim.run()
""",
language="python")

save_session_state()
