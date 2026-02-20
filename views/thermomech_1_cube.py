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
st.write("This example is located in our [repository](https://github.com/ADMIRE-Public/SafeInCave).")

st.markdown(" ## Goals")

st.write(
	"""
	1. Solve a one-way coupled thermoelasticity model in a cube
	2. Introduce the thermo-mechanical simulator *Simulator_TM*
	3. Specify different thermal properties for different regions of the domain
	""")


st.markdown(" ## Problem description")

fig_1_cube_names = create_fig_tag("fig_1_cube_names")

st.write(f"For this example, we use the cube with two regions *OMEGA_A* and *OMEGA_B*, and with the boundary names shown in Fig. {fig_1_cube_names}. The problem consists of solving the thermo-mechanical behavior of the cube as a response to temperature disturbance. The cube is composed of a thermo-elastic material, as represented by the constitutive model shown in Fig. {fig_1_cube_names}-c.")

fig_1_cube_names = figure(os.path.join("assets", "thermomechanics", "1_cube_bcs_model.png"), "Name tags and constitutive model.", "fig_1_cube_names", size=700)

fig_1_cube_bcs = create_fig_tag("fig_1_cube_bcs")

st.write(f"The mechanical boundary conditions are shown in Fig. {fig_1_cube_bcs}-a. Boundary *WEST* is fixed (i.e., Dirichlet for all displcement components), normal displacement on boundary *BOTTOM* is set to zero, and all the remaining boundaries are stress free. The initial temperature of the cube is uniform and equal to 293 K. All faces are isolated, except for boundary *EAST*, where a temperature of 274 K is imposed (Dirichlet). The material is initially undeformed and a mechanical response starts to develop due to the thermal strain element (see Fig. {fig_1_cube_names}-c) as the temperature front penetrates the cube. Additionally, we specify different thermal expansion coefficient values for regions *OMEGA_A* and *OMEGA_B*, so that it generates a non-symmetric mechanical response in the horizontal plane (that is, the cube will bend around the *z* axis).")

fig_1_cube_bcs = figure(os.path.join("assets", "thermomechanics", "1_cube_bcs.png"), "(a) Mechanical and (b) thermal boundary conditions.", "fig_1_cube_bcs", size=500)


st.markdown(" ## Implementation")

st.write("For this case, we need to import both *HeatBC* and *MomentumBC* modules for applying boundary conditions in both models.")

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

st.write("Create *GridHandlerGMSH* object and define output folder.")

st.code(
"""
grid_path = os.path.join("..", "..", "..", "grids", "cube")
grid = sf.GridHandlerGMSH("geom", grid_path)
output_folder = os.path.join("output", "case_0")
""",
language="python")

st.write("Define *TimeController* object. In this case, we specify the initial time as 0 and final time as 10 days. Additionally, we want the time step size to increase following a geometric progression (parabola) with 100 time steps. For this, we use class *TimeControllerParabolic*.")

st.code(
"""
t_control = sf.TimeControllerParabolic(n_time_steps=100, initial_time=0.0, final_time=10, time_unit="day")
""",
language="python")

st.write("Initialize *HeatDiffusion* equation object.")

st.code(
"""
heat_eq = sf.HeatDiffusion(grid)
""",
language="python")

st.write("Build linear system solver, choosing Conjugate Gradient method and Additive Schwartz Method as a preconditioner.")

st.code(
"""
heat_solver = PETSc.KSP().create(grid.mesh.comm)
heat_solver.setType("cg")
heat_solver.getPC().setType("asm")
heat_solver.setTolerances(rtol=1e-12, max_it=100)
heat_eq.set_solver(heat_solver)
""",
language="python")

st.write("Initialize *Material* object to hold all the thermal **and** mechanical properties.")

st.code(
"""
mat = sf.Material(heat_eq.n_elems)
""",
language="python")

st.write(r"Define uniform density distribution (in kg$/$m$^3$).")

st.code(
"""
rho = 2000.0*to.ones(heat_eq.n_elems, dtype=to.float64)
mat.set_density(rho)
""",
language="python")

st.write(r"Define uniform specific heat capacity field (in J kg$^{-1}$K$^{-1}$).")

st.code(
"""
cp = 850*to.ones(heat_eq.n_elems, dtype=to.float64)
mat.set_specific_heat_capacity(cp)
""",
language="python")

st.write(r"Specify uniform thermal conductivity distribution (in W$/$m$^3$).")

st.code(
"""
k = 7*to.ones(heat_eq.n_elems, dtype=to.float64)
mat.set_thermal_conductivity(k)
""",
language="python")

st.write("At this point, the *Material* object already has all the necessary information to be used in the *HeatDiffusion* equation object. So we assign *mat* to *heat_eq*.")

st.code(
"""
heat_eq.set_material(mat)
""",
language="python")

st.write("Apply Dirichlet boundary condition to face *EAST*. Note that *t_control.t_initial = 0* and *t_control.t_final = 10* days.")

st.code(
"""
bc_east = heatBC.DirichletBC("EAST", [274, 274], [t_control.t_initial, t_control.t_final])
""",
language="python")

st.write("")

st.write("Add the boundary condition to the *BcHandler* object, and set it to the *HeatDiffusion* object.")

st.code(
"""
bc_handler = heatBC.BcHandler(heat_eq)
bc_handler.add_boundary_condition(bc_east)
heat_eq.set_boundary_conditions(bc_handler)
""",
language="python")

st.write("Use a *lambda* function that returns a constant value (293) to create temperature field at the grid nodes using the *create_field_nodes* function and set it as an initial temperature for the *HeatDiffusion* object.")

st.info(
	r"**_IMPORTANT:_** This same initial temperature field will be assigned to the *LinearMomentum* object inside class *Simulator_TM*. Therefore, it is not necessary to explicitly specify initial temperature field for the momentum equation, since it will be overwritten inside *Simulator_TM*."
)

st.code(
"""
fun = lambda x, y, z: 293
T0_field = ut.create_field_nodes(heat_eq.grid, fun)
heat_eq.set_initial_T(T0_field)
""",
language="python")

st.write("Initialize the momentum equation object.")

st.code(
"""
mom_eq = sf.LinearMomentum(grid, 0.5)
""",
language="python")

st.write("Define and set solver to the linear momentum balance equation.")

st.code(
"""
mom_solver = PETSc.KSP().create(grid.mesh.comm)
mom_solver.setType("bicg")
mom_solver.getPC().setType("asm")
mom_solver.setTolerances(rtol=1e-12, max_it=100)
mom_eq.set_solver(mom_solver)
""",
language="python")

st.write("Build a spring element with uniform elastic constants.")

st.code(
"""
E = 102e9*to.ones(mom_eq.n_elems)
nu = 0.3*to.ones(mom_eq.n_elems)
spring_0 = sf.Spring(E, nu, "spring")
""",
language="python")

st.write("Extract indices of elements belonging to regions *OMEGA_A* and *OMEGA_B*.")

st.code(
"""
omega_A = grid.region_indices["OMEGA_A"]
omega_B = grid.region_indices["OMEGA_B"]
""",
language="python")

st.write(r"Build a thermal strain element with $\alpha_{th} = 44\times 10^{-6}$ $\text{K}^{-1}$ for all elements in *OMEGA_A*, and $\alpha_{th} = 74\times 10^{-6}$ $\text{K}^{-1}$ for all elements in *OMEGA_B*.")

st.code(
"""
alpha = to.zeros(mom_eq.n_elems)
alpha[omega_A] = 44e-6
alpha[omega_B] = 74e-6
thermo = sf.Thermoelastic(alpha, "thermo")
""",
language="python")

st.write("Add the spring to the **elastic** list of *mat*, and add the thermal strain element to the **thermoelastic** list of *mat*.")

st.code(
"""
mat.add_to_elastic(spring_0)
mat.add_to_thermoelastic(thermo)
""",
language="python")

st.write("Now the material contains all necessary information for the linear momentum balance equation to be solved. Therefore, set this material to *mom_eq*.")

st.code(
"""
mom_eq.set_material(mat)
""",
language="python")

st.write("Set the gravity acceleration vector to the *LinearMomentum* object.")

st.code(
"""
g = -9.81
g_vec = [0.0, 0.0, g]
mom_eq.build_body_force(g_vec)
""",
language="python")

st.write("Fix boundary *WEST* in all directions.")

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
""",
language="python")

st.write("Fix boundary *BOTTOM* in the normal direction (i.e., *z* direction).")


st.code(
"""
bc_bottom = momBC.DirichletBC(boundary_name = "BOTTOM", 
							component = 2,
							values = [0.0, 0.0],
							time_values = time_values)
""",
language="python")

st.write("Add the boundary condition objects to the boundary handler, and set the boundary handler to the *LinearMomentum* object.")

st.code(
"""
bc_handler = momBC.BcHandler(mom_eq)
bc_handler.add_boundary_condition(bc_west_0)
bc_handler.add_boundary_condition(bc_west_1)
bc_handler.add_boundary_condition(bc_west_2)
bc_handler.add_boundary_condition(bc_bottom)
mom_eq.set_boundary_conditions(bc_handler)
""",
language="python")

st.write("Choose fields to be saved related to the momentum balance equation.")

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

st.write("Choose fields to be saved related to the heat diffusion equation.")

st.code(
"""
output_heat = sf.SaveFields(heat_eq)
output_heat.set_output_folder(output_folder)
output_heat.add_output_field("T", "Temperature (K)")
""",
language="python")

st.write("Combine the two *SaveFields* objects created above in a list.")

st.code(
"""
outputs = [output_mom, output_heat]

""",
language="python")

st.write("Use class *Simulator_TM* to solve the thermomechanical problem. Pass *mom_eq*, *heat_eq*, *t_control*, and *outputs* as input arguments. Notice that for this case we solve the initial elastic response before the transient simulation starts.")

st.code(
"""
sim = sf.Simulator_TM(mom_eq, heat_eq, t_control, outputs, True)
sim.run()
""",
language="python")

save_session_state()
