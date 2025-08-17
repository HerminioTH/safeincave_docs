from itertools import count
import json
import streamlit as st

eq_counter = count(1)
fig_counter = count(1)

if "eq" not in st.session_state:
	st.session_state["eq"] = {}

def advance_eq_counter():
	return next(eq_counter)

def advance_fig_counter():
	return next(fig_counter)

def create_eq_tag(tag):
	if tag not in st.session_state["eq"]:
		st.session_state["eq"][tag] = advance_eq_counter()
	return st.session_state["eq"][tag]

def create_fig_tag(tag):
	if tag not in st.session_state["fig"]:
		st.session_state["fig"][tag] = advance_fig_counter()
	return st.session_state["fig"][tag]

def equation(eq, tag):
	eq_num = create_eq_tag(tag)
	st.latex(fr"{eq}" + r"\tag{" + str(eq_num) + r"}")
	return eq_num

def figure(fig, caption, tag, size=500):
	fig_num = create_fig_tag(tag)
	c1, c2, c3 = st.columns([1,2,1])
	c2.image(fig, caption=f"Figure {fig_num}: " + caption, width=size)
	return fig_num

def cite_eq(tag):
	if tag not in st.session_state["eq"]:
		raise Exception(f"Equation with tag '{tag}' does not exist.")
	else:
		return st.session_state["eq"][tag]

def cite_eq_ref(tag):
	return f"({cite_eq(tag)})"


def cite_fig(tag):
	if tag not in st.session_state:
		raise Exception(f"Figure with tag '{tag}' does not exist.")
	else:
		return st.session_state[tag]





def read_json(file_name):
	with open(file_name, "r") as j_file:
		data = json.load(j_file)
	return data

def save_json(data, file_name):
	with open(file_name, "w") as f:
	    json.dump(data, f, indent=4)

def save_session_state():
	data = {"eq": {}, "fig": {}}
	if "eq" in st.session_state:
		data["eq"] = st.session_state["eq"]
	if "fig" in st.session_state:
		data["fig"] = st.session_state["fig"]
	save_json(data, "session_state_cache.json")

