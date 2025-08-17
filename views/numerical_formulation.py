import streamlit as st
import os
import sys
# sys.path.append(os.path.join("..", "libs"))
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
st.markdown(" # Numerical Formulation")
st.markdown(
	"""
	Lorem ipsum.
	"""
)

st.markdown(" ## Heat diffusion")
st.markdown(
	"""
	Let us define the following trial function space for approximating temperature as
	"""
)

eq_trial_T = equation(
	r"""
	\mathcal{S} = \lbrace T : \Omega \rightarrow \mathbb{R}\hspace{1mm} | \hspace{1mm} T \in H^1, T = \bar{T} \hspace{1mm} \text{on} \hspace{1mm} \Gamma_d^T \rbrace,
	""",
	"eq_trial_T"
)

st.markdown(
	r"""
	and the test function space as
	"""
)

eq_test_T = equation(
	r"""
	\mathcal{S}_0 = \lbrace v : \Omega \rightarrow \mathbb{R}\hspace{1mm} | \hspace{1mm} v \in H^1, v = 0 \hspace{1mm} \text{on} \hspace{1mm} \Gamma_d^T \rbrace.
	""",
	"eq_test_T"
)

eq_heat_0 = st.session_state["eq"]["eq_heat_0"]
st.markdown(
	f"""
	The weak form of the heat diffusion equation (Eq. {eq_heat_0}) reads
	"""
)

eq_weak_heat_0 = equation(
	r"""
	\begin{align}
	    \int_\Omega \left( \rho c \frac{\partial T}{\partial t} + k\nabla T \cdot \nabla v \right)\mathrm{d}\Omega 
	    + \int_{\Gamma^q} q'' v \mathrm{d}\Gamma
	    + \int_{\Gamma^h} h\left( T - T_\infty \right) v \mathrm{d}\Gamma
	    = 0.
	\end{align}
	""",
	"eq_weak_heat_0"
)

st.markdown(
	f"""
	Integrating in time between $t$ and $t+\Delta t$, using the fully-implicit (i.e., backward Euler) scheme to evaluate the time integrals, and rearranging the terms, Eq. ({eq_weak_heat_0}) becomes
	"""
)

eq_weak_heat_1 = equation(
	r"""
	\begin{align}
	    \int_\Omega \left(  \frac{\rho c}{\Delta t}T + k\nabla T \cdot \nabla v \right)\mathrm{d}\Omega 
	    + \int_{\Gamma^h} h T v \mathrm{d}\Gamma
	    = 
	    \int_\Omega  \frac{\rho c}{\Delta t} T^t \mathrm{d}\Omega 
	    + \int_{\Gamma^h} h T_\infty v \mathrm{d}\Gamma
	    - \int_{\Gamma^q} q'' v \mathrm{d}\Gamma
	    .
	\end{align}
	""",
	"eq_weak_heat_1"
)

st.markdown(
	r"""
	where $T^t$ refers to the temperature evaluated at the previous time step $t$, while the temperature evaluated at the current time step $t+\Delta t$ carries no superscript to avoid heavy notation ($T$). 
	"""	
)





st.markdown(" ## Linear momentum")
st.markdown(
	"""
	Lorem ipsum.	
	"""
)

st.markdown(" ### Consistent tangent matrix")
st.markdown(
	r"""
	Consider a non-elastic element $i$, with strain rate $\dot{\pmb{\varepsilon}}_i$. Using the $\theta-$method to integrate the strain rate from $t$ to $t+\Delta t$, gives
	"""
)

eq_eps_i_0 = equation(
	r"""
	\pmb{\varepsilon}_i = \pmb{\varepsilon}_i^t + \underbrace{\Delta t \theta}_{\phi_1} \dot{\pmb{\varepsilon}}_i^t + \underbrace{\Delta t (1 -\theta)}_{\phi_2} \dot{\pmb{\varepsilon}}_i.
	""",
	"eq_eps_i_0"
)

st.markdown(
	r"""
	The strain rate $\dot{\pmb{\varepsilon}}_i$, evaluated at the current time level $t+\Delta t$, depends on the stress tensor $\pmb{\sigma}$, which is still an unknown. Therefore, the problem must be linearized and solved iteratively in each time step. For this reason, for every variable at $t+\Delta t$ we indicate the""" + " **previous** iteration level with the superscript $k$ (previous iteration), and to the **current** iteration with $k+1$." + f" Therefore, Eq. ({eq_eps_i_0}) is rewritten as"
)

eq_eps_i_1 = equation(
	r"""
	\pmb{\varepsilon}_i^{k+1} = \pmb{\varepsilon}_i^t + \phi_1 \dot{\pmb{\varepsilon}}_i^t + \phi_2 \dot{\pmb{\varepsilon}}_i^{k+1}
	""", 
	"eq_eps_i_1"
)

eq_stress_1 = st.session_state["eq"]["eq_stress_1"]
eq_ne_strain = st.session_state["eq"]["eq_ne_strain"]
st.markdown(
	f"From Equations ({eq_stress_1}) and ({eq_ne_strain}), it follows that the stress tensor at the current iteration $k+1$ can be expressed as"
)

eq_stress_2 = equation(
	r"\pmb{\sigma}^{k+1} = \mathbb{C}_0 : \left( \pmb{\varepsilon}^{k+1} - \pmb{\varepsilon}_{th} - \pmb{\varepsilon}_{ne}^t - \phi_1 \dot{\pmb{\varepsilon}}_{ne}^t - \phi_2 \dot{\pmb{\varepsilon}}_{ne}^{k+1}\right).",
	"eq_stress_2"
)

st.info(
	r"**_NOTE:_** No superscript is used for the thermal strain $\pmb{\varepsilon}_{th}$ as it only depends on temperature, so the nonlinear iterations do not apply. But keep in mind it refers to the current time level $t+\Delta t$."
)

eq_eps_rate_i_0 = equation(
	r"""
	\dot{\pmb{\varepsilon}}_i^{k+1} = \dot{\pmb{\varepsilon}}_i^{k} 
	+ \frac{\partial \dot{\pmb{\varepsilon}}_i}{\partial \pmb{\sigma}} : \delta \pmb{\sigma}
	+ \frac{\partial \dot{\pmb{\varepsilon}}_i}{\partial \omega_i} \delta \omega_i
	""",
	"eq_eps_rate_i_0"
)



st.markdown(" ### Weak formulation")



save_session_state()