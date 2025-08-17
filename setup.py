import streamlit as st
import json
import os
import sys
sys.path.append("libs")
from Utils import read_json, save_json

def clean_cache():
	if os.path.isfile("session_state_cache.json"):
		data = {"eq": {}, "fig": {}}
		save_json(data, "session_state_cache.json")


def run_setup():
	if os.path.isfile("session_state_cache.json"):
		data = read_json("session_state_cache.json")
		st.session_state["eq"] = data["eq"]
		st.session_state["fig"] = data["fig"]
	else:
		pass


if __name__ == '__main__':
	run_setup()