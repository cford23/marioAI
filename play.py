from agent import Agent
import pickle
import sys
import json


modelFileName = 'marioAI.sav'

if len(sys.argv) != 2:
    print("Missing run type parameter - either 'train' or 'test'")
    sys.exit()

# Train and save the model
if sys.argv[1] == 'train':
    with open('modelParams.json', 'r') as params:
        modelParams = json.load(params)
    marioAI = Agent(**modelParams)
    marioAI.train()
    pickle.dump(marioAI, open(modelFileName, 'wb'))

# Load and test model
elif sys.argv[1] == 'test':
    marioAI = pickle.load(open(modelFileName, 'rb'))
    lives = 1
    marioAI.test(lives)

# Incorrect run parameter
else:
    print("Incorrect run type value - either 'train' or 'test'")
    sys.exit()