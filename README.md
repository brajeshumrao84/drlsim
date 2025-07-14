## Setup

drlsim requires Python 3.6 or 3.7. The setup has been tested on Ubuntu 16.04 and 20.04, using Python 3.6 and [`coord-sim v2.1.0`](https://github.com/brajeshumrao84/coord-sim/releases/tag/v2.1.0). For newer versions, only Python 3.7 is supported by the GitHub CI (Python 3.6 is no longer supported). 

### Create a virtual environment (venv)

On your local machine, run the following commands to set up Python 3.6 (if not already installed) and create a virtual environment:

```bash
# Check Python version
python3 --version

# Install Python 3.6 if not installed
sudo apt update
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.6 python3.6-dev python3.6-venv

# Create a virtual environment
python3.6 -m venv ./venv
# Activate the virtual environment
source venv/bin/activate
# Upgrade pip and setuptools
pip install -U pip setuptools
Install Dependencies
To install drlsim, we recommend installing it from the source:

# If you encounter issue while making the files, make sure you have all the dependencies resolved for cmake
sudo apt install build-essential cmake pkg-config libjpeg-dev libtiff-dev libpng-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libgtk-3-dev libatlas-base-dev gfortran python3-dev ninja-build

# Install drlsim from the current repo
pip install -e .
This command also installs the required coord-sim simulation environment.

Usage
Once the environment is set up, you can start training and testing the RL agent. Use the following command to see available options:


$ drlsim -h
Usage: drlsim [OPTIONS] AGENT_CONFIG NETWORK SERVICE SIM_CONFIG STEPS

Options:
  --seed INTEGER               Specify the random seed for the environment and
                               the learning agent.
  -t, --test TEXT              Name of the training run whose weights should
                               be used for testing.
  -w, --weights TEXT           Continue training with the specified weights
                               (similar to testing)
  -a, --append-test            Append a test run of the previously trained
                               agent.
  -v, --verbose                Set console logger level to debug. (Default is
                               INFO)
  -b, --best                   Test the best of the trained agents so far.
  -ss, --sim-seed INTEGER      Set the simulator seed
  -gs, --gen-scenario PATH     Specify a different simulator config file for additional scenario tests
  -h, --help                   Show this message and exit.
Training and Testing Example
To train and then test the agent, use the following command:

drlsim res/config/agent/sample_agent.yaml res/networks/sample_network.graphml res/service_functions/abc.yaml res/config/simulator/sample_config.yaml 10 --append-test
Results are saved under results/ and are organized based on input arguments and the timestamp.

Testing with a Specific Set of Weights
To run the agent with a specific training run, specify the <timestamp_seed>:

drlsim res/config/agent/sample_agent.yaml res/networks/sample_network.graphml res/service_functions/abc.yaml res/config/simulator/sample_config.yaml 200 -t <timestamp_seed> -e 1
Testing with Multiple Scenarios
To test the agent in multiple scenarios using different simulator configurations, use the -gs option:


drlsim res/config/agent/sample_agent.yaml res/networks/sample_network.graphml res/service_functions/abc.yaml res/config/simulator/sample_config.yaml 1000 --append-test -gs res/config/simulator/sample_config.yaml
Visualizing Training Progress with Tensorboard
To monitor the learning progress, you can use Tensorboard:


tensorboard --logdir=./graph
To filter by specific agent configurations and networks:

tensorboard --logdir=./graph/<agent_config>/<network>/<service>/<simulator_config>
Jupyter Notebook for Result Analysis
For analyzing results, use the provided Jupyter notebook eval_example.ipynb:


# Install dependencies
pip install -r eval_requirements.txt
# Start Jupyter Lab
jupyter lab
To run Jupyter on a remote server, use the following:


jupyter notebook --ip 0.0.0.0 --no-browser
Running Parallel Training
For parallelizing training across multiple agents and scenarios, use the following script:

./scripts/run_parallel.sh
Acknowledgement
This project is an upgradation of original project found at https://github.com/RealVNF/DeepCoord)
```
