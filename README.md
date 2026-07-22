# ADC_Team_5
The main repository for team  5

Go to the ...\ADC_Team_5\Python
Windows Start:

python3 -m venv .venv
.venv\Scripts\activate.bat
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -c "import mujoco; print(mujoco.__version__)"
python3 Simulation_Base.py --model "C:\ADC_Team_5\Model\G1_29dof.xml"

Linux Start:

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -c "import mujoco; print(mujoco.__version__)"
python3 Simulation_Base.py --model "/home/<username>/ADC_Team_5/Model/G1_29dof.xml"
