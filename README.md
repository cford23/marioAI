# marioAI
Final Project for Reinforcement Learning course. The goal of this project is to train a DQN agent to play the Game Boy game Super Mario Land.

## Steps To Run Program
1. [Clone Repository](#clone-repository)
2. [Create/Activate Virtual Environment](#createactivate-virtual-environment)
3. [Install Python Libraries](#install-python-libraries)
4. [Train or Test Agent](#train-or-test-agent)

## Clone Repository
Navigate to the directory where you want the repository and copy the following command to clone the repository.
```
git clone https://github.com/cford23/marioAI.git
```

## Create/Activate Virtual Environment
Navigate into the cloned repository and copy the following command to create the virtual environment.
```
python -m venv venv
```
Once the virtual environment has been created, activate it using one of the following commands depending on your OS.

Mac
```
source venv/bin/activate
```
Windows
```
.\venv\bin\activate
```

## Install Python Libraries
Once the virtual environment has been created and activated, run the following command to install the Python libraries needed for this project to run.
```
pip install -r requirements.txt
```

## Train or Test Agent
### Training
When training the agent, you can modify the model's hyperparameters in the [agentInfo.json](agentInfo.json) file. To train the agent, run the following command.
```
python play.py train
```
The agent will then train for the given number of epochs. Once the training has completed, an image will be available in the plots folder with the file name starting with the timestamp of when it was run. This image will contain a graph of the agent's max level progress over each epoch.

### Testing
To test the agent, first make sure there is a [marioAI.sav](marioAI.sav) file. This document contains all the necessary information to load the agent and have it play. Next, run the following command to test the agent.
```
python play.py test
```
You can modify the lives field in [agentInfo.json](agentInfo.json) to change how many lives the agent starts with. After running the command to test the agent, a window will appear allowing you to watch the agent as it plays.