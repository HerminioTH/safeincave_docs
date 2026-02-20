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
	    = 0. \nonumber
	\end{align}
	""",
	"eq_weak_heat_0"
)

st.markdown(
	fr"""
	Integrating in time between $t$ and $t+\Delta t$, using the fully-implicit (i.e., backward Euler) scheme to evaluate the time 
    integrals, and rearranging the terms, Eq. ({eq_weak_heat_0}) becomes
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
	    . \nonumber
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

st.markdown(
	r"where $\delta\pmb{\sigma} = \pmb{\sigma}^{k+1} - \pmb{\sigma}^k$ and $\delta\omega_i = \omega_i^{k+1} - \omega_i^k$. "
)

st.info(
	r"**_NOTE:_** The term $\frac{\partial \dot{\pmb{\varepsilon}}_i}{\partial \pmb{\sigma}}$ is a rank-4 tensor, while $\delta\pmb{\sigma}$ is arank-2, hence the double dot product between them, which results a rank-2 tensor. For further support on tensorial operations, check [here](https://youtu.be/w5KX3F_rdzU?si=QQLVBq1NcrvOiS32), and [here](https://youtu.be/JiN6jwp0RPk?si=K1Qhe3lAxJD4LI5w) for practical examples."
)

st.markdown(
	r"The increment of state variable $\delta\omega_i$ can be obtained by defining a residual equation based on the evolution equation of $\omega_i$ and using Newton-Raphson to drive the residual to zero. Considering the residual equation is of the form $r_i = r_i(\pmb{\sigma}, \omega_i)$, it follows that" 
)

eq_res_0 = equation(
	r"""
	r_i^{k+1} = r_i^k + \frac{\partial r_i}{\partial \pmb{\sigma}} : \delta \pmb{\sigma} + \underbrace{\frac{\partial r_i}{\partial \omega_i}}_{h_i} \delta \omega_i = 0
	\quad \rightarrow \quad
	\delta \omega_i = - \frac{1}{h_i} \left( r_i^k + \frac{\partial r_i}{\partial \pmb{\sigma}} : \delta \pmb{\sigma} \right).
	""",
	"eq_res_0"
)

st.info(
	r"""**_NOTE:_** Currently, only the viscoplastic element uses a state variable (the hardening parameter, $\alpha$). For this case, the residual equation would read $r(\alpha) = \alpha - a_1 \left[ \left( a_1 / \alpha_0 \right)^{1/\eta} + \xi \right]^{-1}$.

	"""
)

st.markdown(
	f"Substituting Eq. ({eq_res_0}) into Eq. ({eq_eps_rate_i_0})" + r" to eliminate $\delta\omega_i$ yields"
)

eq_eps_rate_i_1 = equation(
	r"""
	\dot{\pmb{\varepsilon}}_i^{k+1} = \dot{\pmb{\varepsilon}}_i^{k} 
	+ \underbrace{\left( \frac{\partial \dot{\pmb{\varepsilon}}_i}{\partial \pmb{\sigma}} - \frac{1}{h_i} \frac{\partial \dot{\pmb{\varepsilon}}_i}{\partial \omega_i} \frac{\partial r_i}{\partial \pmb{\sigma}} \right)}_{\mathbb{G}_i} : \delta \pmb{\sigma}
	- \underbrace{\frac{r_i^k}{h_i} \frac{\partial \dot{\pmb{\varepsilon}}_i}{\partial \omega_i}}_{\mathbf{B}_i}
	\quad \rightarrow \quad
	\dot{\pmb{\varepsilon}}_i^{k+1} = \dot{\pmb{\varepsilon}}_i^{k} + \mathbb{G}_i : \delta \pmb{\sigma} - \mathbf{B}_i.
	""",
	"eq_eps_rate_i_1"
)

st.markdown(
	"Considering all non-elastic elements,"
)

eq_eps_rate_ne_0 = equation(
	r"""
	\dot{\pmb{\varepsilon}}_{ne}^{k+1}
	=
	\dot{\pmb{\varepsilon}}_{ne}^{k} + \mathbb{G}_{ne} : \delta \pmb{\sigma} - \mathbf{B}_{ne},
	""",
	"eq_eps_rate_ne_0"
)

st.markdown(
	r"where $\mathbb{G}_{ne} = \sum_{i=1}^{N_{ne}} \mathbb{G}_i$ and $\mathbf{B}_{ne} = \sum_{i=1}^{N_{ne}} \mathbf{B}_i$."
)

st.markdown(
	f"Finally, substituting Eq. ({eq_eps_rate_ne_0}) into Eq. ({eq_stress_2}) leads to"
)

eq_stress_3 = equation(
	r"""
	\pmb{\sigma}^{k+1} = \mathbb{C}_T : \left[ 
		\pmb{\varepsilon}^{k+1} 
		- \pmb{\varepsilon}_{ne}^k
		+ \phi_2 \left( \mathbb{G}_{ne} : \pmb{\sigma}^k + \mathbf{B}_{ne} \right)
	\right]
	""",
	"eq_stress_3"
)

st.markdown(
	r"where $\pmb{\varepsilon}_{ne}^k = \pmb{\varepsilon}_{ne}^t + \phi_1 \dot{\pmb{\varepsilon}}_{ne}^t + \phi_1 \dot{\pmb{\varepsilon}}_{ne}^k$, and the consistent tangent matrix is given by"
)

eq_CT = equation(
	r"\mathbb{C}_T = \left( \mathbb{C}_0^{-1} + \phi_2 \mathbb{G}_{ne} \right)^{-1}", "eq_CT"
)

st.markdown(
	f"We can further simplify Eq. ({eq_stress_3}) by defining"
)

eq_eps_rhs = equation(
	r"""
	\pmb{\varepsilon}^k_\text{rhs} = \pmb{\varepsilon}_{ne}^k - \phi_2 \left( \mathbb{G}_{ne} : \pmb{\sigma}^k + \mathbf{B}_{ne} \right)
	""",
	"eq_eps_rhs"
)

st.markdown(
	"In this manner, the stress tensor can be expressed as"
)

eq_stress_4 = equation(
	r"\pmb{\sigma}^{k+1} = \mathbb{C}_T : \left( \pmb{\varepsilon}^{k+1} - \pmb{\varepsilon}^k_\text{rhs} \right)",
	"eq_stress_4"
)

st.markdown(
	"Finally, the linearized momentum balance equation reads"
)

eq_mom_1 = equation(
	r"\nabla \cdot \mathbb{C}_T : \pmb{\varepsilon}^{k+1} = \mathbf{f} + \nabla \cdot \mathbb{C}_T : \pmb{\varepsilon}_\text{rhs}^k.",
	"eq_mom_1"
)








	
st.markdown(" ### Weak formulation")



save_session_state()