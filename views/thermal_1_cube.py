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
st.write("This example is located in our [repository](https://github.com/ADMIRE-Public/SafeInCave).")

st.markdown(" ## Goals")

st.write(
	"""
	1. Set up heat diffusion equation
	2. Apply Dirichlet boundary condition
	3. Apply Robin boundary condition
	4. Plot results using **PostProcessingTools** and [matplotlib](https://matplotlib.org/)
	""")


st.markdown(" ## Problem description")

fig_1_cube_geom_bcs = create_fig_tag("fig_1_cube_geom_bcs")

st.write(f"Consider a cube with boundaries named as indicated in Fig. {fig_1_cube_geom_bcs}-a. Although the geometry is three-dimensional, we set up a one-dimensional problem by imposing zero heat flux (Neumann) on boundaries *BOTTOM*, *NORTH*, *TOP*, and *SOUTH. Temperature is prescribed (Dirichlet) on boundary *WEST*, and convective heat flux (Robin) is applied to boundary *EAST*, as shown in Fig. {fig_1_cube_geom_bcs}-b. The prescribed values and thermal properties are also indicated in Fig. {fig_1_cube_geom_bcs}-b. The initial temperature field is uniform and equal to 293 K.")

fig_1_cube_geom_bcs = figure(os.path.join("assets", "thermal", "1_cube_geom_bcs.png"), "Geometry and boundary conditions", "fig_1_cube_geom_bcs", size=600)


st.markdown(" ### Implementation")

st.write("Import the usual packages. Notice that, for the heat diffusion equation, boundary conditions are applied using module *safeincave.HeatBC*.")

st.code(
"""
import safeincave as sf
import safeincave.HeatBC as heatBC
from petsc4py import PETSc
import torch as to
import os
""",
language="python")

st.write("Create *GridHandlerGMSH* object.")

st.code(
"""
grid_path = os.path.join("..", "..", "..", "grids", "cube")
grid = sf.GridHandlerGMSH("geom", grid_path)
""",
language="python")

st.write("Define output folder, where results are saved.")

st.code(
"""
output_folder = os.path.join("output", "case_0")
""",
language="python")

st.write("Define *TimeController* object. In this case, we specify the initial time as 0 and final time as 5 days. Additionally, we want the time step size to increase following a geometric progression (parabola) with 50 time steps. For this, we use class *TimeControllerParabolic*.")

st.code(
"""
t_control = sf.TimeControllerParabolic(n_time_steps=50, initial_time=0.0, final_time=5, time_unit="day")
""",
language="python")

st.write("Initialize *HeatDiffusion* equation object.")

st.code(
"""
heat_eq = sf.HeatDiffusion(grid)
""",
language="python")

st.write("Build linear system solver, choosing Conjugate Gradient method and incomplete LU decomposition as a preconditioner.")

st.code(
"""
solver_heat = PETSc.KSP().create(grid.mesh.comm)
solver_heat.setType("cg")
solver_heat.getPC().setType("ilu")
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

st.write("Create *DirichletBC* object for boundary *EAST*, where a constant temperature of 273 K is imposed.")

st.code(
"""
bc_east = heatBC.DirichletBC(boundary_name = "EAST", 
						values = [273, 273],
						time_values = [t_control.t_initial, t_control.t_final])
""",
language="python")

st.write(r"Create *RobinBC* object for boundary *WEST*, where convective heat transfer is imposed. The far field temperature $T_\infty$ is 273 K and the convective heat transfer coefficient is 5.0 W$/$m$^2/$K.")

st.code(
"""
bc_west = heatBC.RobinBC(boundary_name = "WEST", 
						values = [273, 273],
						h = 5.0,
						time_values = [t_control.t_initial, t_control.t_final])
""",
language="python")

st.write("Create a *BcHandler* object and add the two above defined boundary condition objects.")

st.code(
"""
bc_handler = heatBC.BcHandler(heat_eq)
bc_handler.add_boundary_condition(bc_east)
bc_handler.add_boundary_condition(bc_west)
""",
language="python")

st.write("Finally, set the *BcHandler* object to the *HeatDiffusion* object.")

st.code(
"""
heat_eq.set_boundary_conditions(bc_handler)
""",
language="python")

st.write("Specify an uniform initial temperature distribution in the cube.")

st.code(
"""
T0_field = 293*to.ones(heat_eq.n_elems, dtype=to.float64)
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



st.markdown(" ### Plot results")

fig_1_cube_plot = create_fig_tag("fig_1_cube_plot")

st.write(f"The results can be readily visualized in [Paraview](https://www.paraview.org/). However, it is often useful to post-process the results on our own. In this section, we explain how to use SafeInCave *PostProcessingTools* to read and plot results such as the ones shown in Fig. {fig_1_cube_plot}. Figure {fig_1_cube_plot}-b shows the temperature profiles for different time steps along the edge shared by faces *TOP* and *SOUTH* of the cube, as highlighted in Fig. {fig_1_cube_plot}-a. Figure {fig_1_cube_plot}-c shows the temperature evolution with time at the probe point (0.5, 0.0, 1.0), also highlighted in Fig. {fig_1_cube_plot}-a.")


fig_1_cube_plot = figure(os.path.join("assets", "thermal", "1_cube_plot.png"), "Temperature results for the cube problem.", "fig_1_cube_plot", size=800)


st.write("First, import *safeincave.PostProcessingTools*, [matplotlib](https://matplotlib.org/), and [numpy](https://numpy.org/).")

st.code(
"""
import safeincave.PostProcessingTools as post
import matplotlib.pyplot as plt
import numpy as np
import os
""",
language="python")

st.write("Define the folder location where the results are saved.")

st.code(
"""
results_folder = os.path.join("output", "case_0")
""",
language="python")

st.write("We want to read the temperature field *T*, which is a **scalar** quantity stored at the grid **nodes**. For this, we need to use function *read_node_scalar*, which returns:")

st.markdown(
	"""
	- *points*: an array of dimension (n,3) storing the coordinates (x,y,z) for the grid nodes, with *n* denoting the number of grid nodes;
	- *t*: an array of size *m* containing the time instants of the transient simulation;
	- *T*: an array of dimension (m,n) containing the temperature values at each grid node for all time instants.
	"""
)


st.code(
"""
points, t, T = post.read_node_scalar(os.path.join(results_folder, "T", "T.xdmf"))
""",
language="python")

st.write("Next, we convert the time list values from seconds to days.")

st.code(
"""
t /= (60*60*24)
""",
language="python")

st.write("Extract array **indices** along edge (y,z)=(0,1).")

st.code(
"""
line_idx = np.where((points[:,1] == 0.0) & (points[:,2] == 1.0))[0]
""",
language="python")

st.write("Extract **points** along edge (y,z)=(0,1).")

st.code(
"""
line_points = points[line_idx]
""",
language="python")

st.write("Extract *x* coordinates along the edge of interest.")

st.code(
"""
x0 = line_points[:,0]
""",
language="python")

st.write("Extract temperature profiles along edge.")

st.code(
"""
T0 = T[:,line_idx]
""",
language="python")

st.write("Since we use an unstructured mesh, the points along the edge do not follow any particular ordering. For plotting purposes, it would be useful to sort these points according to increasing values of (in this case) **x** coordinates. We can do this by using *np.argsort* on the array of *x* coordinates. This function returns a list of permutation indices that we can use later to reorder our *x0* and *T0* arrays.")

st.code(
"""
sorted_idx = np.argsort(x0)
""",
language="python")

st.write("Now, reorder *x0* and *T0*.")

st.code(
"""
x_sorted = x0[sorted_idx]
T_sorted = T0[:,sorted_idx]
""",
language="python")

st.write("Define a *matplotlib* figure with two axes.")

st.code(
"""
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))
fig.subplots_adjust(top=0.90, bottom=0.15, left=0.11, right=0.980, hspace=0.35, wspace=0.293)
""",
language="python")

st.write("Plot the temperature profiles at specified time steps (in this case, 0, 3, 15, and 30).")

st.code(
"""
ax1.plot(x_sorted, T_sorted[0,:], ".-", color="#377eb8", label=f"t={round(t[0],2)} day(s)")
ax1.plot(x_sorted, T_sorted[3,:], ".-", color="#ff7f00", label=f"t={round(t[3],2)} day(s)")
ax1.plot(x_sorted, T_sorted[15,:], ".-", color="#4daf4a", label=f"t={round(t[15],2)} day(s)")
ax1.plot(x_sorted, T_sorted[30,:], ".-", color="#f781bf", label=f"t={round(t[30],2)} day(s)")
ax1.set_xlabel("x (m)", fontname="serif", fontsize=12)
ax1.set_ylabel("Temperature (K)", fontname="serif", fontsize=12)
ax1.legend(loc=0, shadow=True, fancybox=True, prop={"size":8})
""",
language="python")

st.write("Plot the transient tempetarure at the probe point (0.5, 0.0, 1.0). This point is the 5th point our array.")

st.code(
"""
ax2.plot(t, T_sorted[:,5])
ax2.set_xlabel("Time (days)", fontname="serif", fontsize=12)
ax2.set_ylabel("Temperature (K)", fontname="serif", fontsize=12)
""",
language="python")

st.write("Show the figure on screen.")

st.code(
"""
plt.show()
""",
language="python")


save_session_state()