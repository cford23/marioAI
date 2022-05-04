from agent import Agent
import pickle
import sys
import json


if len(sys.argv) != 2:
    print("Missing run type parameter - either 'train' or 'test'")
    sys.exit()

with open('agentInfo.json', 'r') as params:
    agentInfo = json.load(params)

# Train and save the model
if sys.argv[1] == 'train':
    marioAI = Agent(**agentInfo['training'])
    marioAI.train()
    with open(agentInfo['training']['outputFile'], 'wb') as outputFile:
        pickle.dump(marioAI, outputFile)

# Load and test model
elif sys.argv[1] == 'test':
    with open(agentInfo['testing']['agentFile'], 'rb') as modelFile:
        marioAI = pickle.load(modelFile)
    lives = agentInfo['testing']['lives']
    marioAI.test(lives)

# Incorrect run parameter
else:
    print("Incorrect run type value - either 'train' or 'test'")
    sys.exit()