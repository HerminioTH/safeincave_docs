import streamlit as st
import os
from setup import clean_cache, run_setup

st.set_page_config(
    page_title="SafeInCave Docs",
    page_icon=os.path.join("assets", "logo.png"),
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items=None,
)
st._config.set_option("theme.base", "light")




pg = st.navigation(
	{
		"Info": [
			st.Page(page="views/land.py", title="Home", icon=":material/home:", default=True), 
			st.Page(page="views/about.py", title="About", icon=":material/info_i:"),
			st.Page(page="views/api.py", title="API Documentation", icon=":material/menu_book:"),
		],
		"Getting started": [
			st.Page(page="views/installation.py", title="Installation")
		],
		"Fundamental theory": [
			st.Page(page="views/constitutive_model.py", title="Constitutive modeling"),
			st.Page(page="views/model_equations.py", title="Governing equations"),
			st.Page(page="views/numerical_formulation.py", title="Numerical formulation"),
		],
		"Implementation": [
			st.Page(page="views/implementation.py", title="Implementation")
		],
		"Mechanics": [
			st.Page(page="views/mechanics_1_triaxial.py", title="Example 1: Triaxial"),
			st.Page(page="views/mechanics_2_cube_regions.py", title="Example 2: Regions"),
			st.Page(page="views/mechanics_3_cavern.py", title="Example 3: Cavern full model"),
			st.Page(page="views/mechanics_4_cavern.py", title="Example 4: Cavern overburden")
		],
		"Thermal": [
			st.Page(page="views/thermal_1_cube.py", title="Example 1: Cube"),
			st.Page(page="views/thermal_2_cavern.py", title="Example 2: Cavern"),
		],
		"Thermo-mechanics": [
			st.Page(page="views/thermomech_1_cube.py", title="Example 1: Cube"),
			st.Page(page="views/thermomech_2_cavern.py", title="Example 1: Cavern"),
		],
	},
	position="sidebar"
)

# pg = st.navigation(
# 	{
# 		"Info": [land_page, about_page],
# 		"Getting started": [installation_page],
# 		"Fundamental theory": [theory_page, implementation_page],
# 		"Implementation": [theory_page, implementation_page],
# 		"Examples: Mechanics": [mechanics_1_triaxial_page, mechanics_2_cube_regions_page, mechanics_3_cavern_page, mechanics_4_cavern_page],
# 		# "Examples: Thermal": [],
# 		# "Examples: Thermo-Mechanics": [],
# 	},
# 	position="sidebar"
# )

st.logo(os.path.join("assets", "logo_safeincave.png"))
# st.sidebar.text("Something here.")

pg.run()