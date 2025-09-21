import streamlit as st

st.set_page_config(layout="wide") 
st.markdown(" # Installation")

st.write("This section will walk you through the installation process of the SafeInCave simulator on your system. Since SafeInCave is based on FEniCSx, it can only be installed on Ubuntu and MacOS systems. If you use Windows, first install Windows Subsystem for Linux (WSL) and then install Ubuntu on it. Next, install [FEniCSx](https://fenicsproject.org/), and finally the [safeincave](https://test.pypi.org/project/safeincave/) package.")

st.markdown(" ## Install WSL + Ubuntu")
st.info(
	"**_NOTE:_** Skip this step if you are **not** using Windows."
)

st.write("To install WSL, open PowerShell in administrator mode and execute the following commands:")

st.code(
"""Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux
wsl --update""", 
language="powershell")

st.write("To proceed with Ubuntu installation, open PowerShell in administrator mode and execute:")

st.code(
"wsl --install -d Ubuntu-22.04", 
language="powershell")

st.write("Choose a username and a password. You will notice that PowerShell suddenly becomes a Ubuntu terminal.")


st.markdown(" ## Option 1: FEniCSx + SafeInCave")

st.write("Install FEniCSx, as described [here](https://fenicsproject.org/download/). Open Ubuntu terminal and execute:")

st.code(
"""sudo add-apt-repository ppa:fenics-packages/fenics
sudo apt update
sudo apt install fenicsx""", 
language="bash")

st.write("Install SafeInCave by executing:")

st.code(
"""sudo apt install python3-pip
pip install --upgrade pip
pip3 install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple safeincave""", 
language="bash")


st.markdown(" ## Option 2: Conda + FEniCSx + SafeInCave")

st.write("Use conda to avoid conflicts with other packages on your system. Download Miniconda3-py310_25.5.1-0-Linux-x86_64.sh from https://repo.anaconda.com/miniconda/ and save it in your Ubuntu home/user_name directory. In this same directory, execute:")

st.code(
"""bash Miniconda3-py310_25.3.1-1-Linux-x86_64.sh""", 
language="bash")

st.write("Restart the terminal and execute:")

st.code(
"""conda activate""",
language="bash")

st.write("You should notice the tag (base) in your command line. Let's create a new environment:")

st.code(
"""
conda create -n safe python=3.10
conda activate safe""",
language="bash")

st.write("The tag (safe) should now appear in the command line.")
st.write("To install FEniCSx using conda, as described [here](https://fenicsproject.org/download/), execute:")

st.code(
"""
conda install -c conda-forge fenics-dolfinx mpich pyvista
""",
language="bash")

st.write("Install SafeInCave by executing:")

st.code(
"""
sudo apt update
sudo apt install python3-pip
pip install --upgrade pip
pip3 install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple safeincave""", 
language="bash")

st.write("Finally, make sure *libxft2* and *libxinerama1* are installed before running safeincave.")

st.code(
"""
sudo apt-get install libxft2
sudo apt-get install -y libxinerama1
""", 
language="bash")

