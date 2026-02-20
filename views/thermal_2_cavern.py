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



st.markdown(" ## Example 2: Heat diffusion in salt cavern")
st.write("This example is located in our [repository](https://github.com/ADMIRE-Public/SafeInCave).")

st.markdown(" ## Goals")

st.write(
	"""
	1. Set up common boundary conditions for heat diffusion in salt cavern
	2. Specify initial temperature distribution following geothermal gradient
	""")


st.markdown(" ## Problem description")

fig_2_cavern_geom_bcs = create_fig_tag("fig_2_cavern_geom_bcs")

st.write(f"This problem considers the same geometry already used in previous examples, as shown in Fig. {fig_2_cavern_geom_bcs}-a. The initial temperature profile follows the geothermal gradient of 27 K/km, as shown in Fig. {fig_2_cavern_geom_bcs}-b, starting from 293 K at the *Top* boundary, where temperature is prescribed (Dirichlet). The geothermal gradient is imposed on the *Bottom* boundary (Neumann). A convective heat transfer is imposed on the *Cavern* walls, with a constant gas temperature of 283 K (thus a cold gas). All the remaining boundaries are isolated.")

fig_2_cavern_geom_bcs = figure(os.path.join("assets", "thermal", "2_cavern_geom_bcs.png"), "Geometry and boundary conditions", "fig_2_cavern_geom_bcs", size=600)


st.markdown(" ### Implementation")

st.write("Import the usual packages. From package *Utils*, we only import function *create_field_nodes*, which is convenient to specify the initial temperature distribution.")



st.code(
"""
import safeincave as sf
from safeincave.Utils import create_field_nodes
import safeincave.HeatBC as heatBC
import safeincave.MomentumBC as momBC
from petsc4py import PETSc
import torch as to
import os
""",
language="python")

st.write("Create *GridHandlerGMSH* object.")

st.code(
"""
grid_path = os.path.join("..", "..", "..", "grids", "cavern_regular")
grid = sf.GridHandlerGMSH("geom", grid_path)
""",
language="python")

st.write("Define output folder, where results are saved.")

st.code(
"""
output_folder = os.path.join("output", "case_0")
""",
language="python")

st.write("Similarly to the previous thermal example, we use time step sizes increasing in a geometric progression with 100 time steps util the final time of 5 years.")

st.code(
"""
t_control = sf.TimeControllerParabolic(n_time_steps=100, initial_time=0, final_time=5, time_unit="year")
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
solver_heat = PETSc.KSP().create(grid.mesh.comm)
solver_heat.setType("cg")
solver_heat.getPC().setType("asm")
solver_heat.setTolerances(rtol=1e-12, max_it=100)
heat_eq.set_solver(solver_heat)
""",
language="python")

st.write("Initialize *Material* object to hold all the thermal properties.")

st.code(
"""
mat = sf.Material(heat_eq.n_elems)
""",
language="python")

st.write(r"Include uniform density distribution (in kg$/$m$^3$).")

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

st.write("The *Material* object is complete. Let's assign it to *HeatDiffusion* equation object.")

st.code(
"""
heat_eq.set_material(mat)
""",
language="python")

st.write("For convenience, let us define a list with the initial and final time values.")

st.code(
"""
time_values = [t_control.t_initial, t_control.t_final]
""",
language="python")

st.write("Also for convenience, we define below the geothermal gradient, temperature at the *Top* boundary, gas temperature, and convective heat transfer coefficient.")

st.code(
"""
km = 1000
dTdZ = 27/km
T_top = 293
T_gas = 283
h_conv = 5.0
""",
language="python")

st.write("Initialize the *BcHandler* object.")

st.code(
"""
bc_handler = heatBC.BcHandler(heat_eq)
""",
language="python")

st.write("Add boundary conditions to the *BcHandler* object.")

st.code(
"""
bc_top = heatBC.DirichletBC("Top", [T_top, T_top], time_values)
bc_handler.add_boundary_condition(bc_top)
bc_bottom = heatBC.NeumannBC("Bottom", [dTdZ, dTdZ], time_values)
bc_handler.add_boundary_condition(bc_bottom)
bc_east = heatBC.NeumannBC("East", [0.0, 0.0], time_values)
bc_handler.add_boundary_condition(bc_east)
bc_west = heatBC.NeumannBC("West", [0.0, 0.0], time_values)
bc_handler.add_boundary_condition(bc_west)
bc_south = heatBC.NeumannBC("South", [0.0, 0.0], time_values)
bc_handler.add_boundary_condition(bc_south)
bc_north = heatBC.NeumannBC("North", [0.0, 0.0], time_values)
bc_handler.add_boundary_condition(bc_north)
bc_cavern = heatBC.RobinBC("Cavern", [T_gas, T_gas], h_conv, time_values)
bc_handler.add_boundary_condition(bc_cavern)
""",
language="python")

st.write("Add the *BcHandler* object to the *HeatDiffusion* equation object.")

st.code(
"""
heat_eq.set_boundary_conditions(bc_handler)
""",
language="python")

st.write("Define a *lambda* function for the geothermal temperature profile, pass it as an argument to the *create_field_nodes* function and assign the resulting array (i.e., temperature field) as an initial temperature to the *HeatDiffusion* equation object.")

st.code(
"""
fun = lambda x, y, z: T_top - dTdZ*(z - 660)
T0_field = ut.create_field_nodes(heat_eq.grid, fun)
heat_eq.set_initial_T(T0_field)
""",
language="python")

st.write("Create a *SaveFields* object and select the tempetarure field *T* to be saved during simulation.")

st.code(
"""
output_heat = sf.SaveFields(heat_eq)
output_heat.set_output_folder(output_folder)
output_heat.add_output_field("T", "Temperature (K)")
outputs = [output_heat]
""",
language="python")

st.write("Build the thermal simulator (*Simulator_T*) and run the simulation.")

st.code(
"""
sim = sf.Simulator_T(heat_eq, t_control, outputs, True)
sim.run()
""",
language="python")

save_session_state()
