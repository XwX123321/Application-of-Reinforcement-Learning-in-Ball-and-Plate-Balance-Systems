# Final project for ball-plate balancing system

# Installation
the `environment.yml` file includes all the dependencies you need to install
Run `conda env create -f environment.yml` in the terminal to get all dependencies
# Files and Folders
The `env` folder includes two file as bellow:
- `__init__.py` : Sets the default import content for the module.
- `ball_plate_env.py`: Python code defining the ball-plate physical environment, based on the OpenAI Gym interface. 

The `models` folder saves the weights of the one with the best training results:
- `best_model.pth`: Weighting parameters saved when the average reward is maximized during training.

The `runs` folder records the loss and reward curves during each training session into the tensorboard to facilitate real-time monitoring during the training process.
Run `tensorboard --logdir your absolute path of sac_ball_balance_sb3` to view the curve.

- `ball.urdf`: URDF model file describing the ball for the physics simulation. 
- `plateform.urdf`: URDF model file describing the platform for the physics simulation.
- `SAC_training.py`: This file includes the Neural network design, SAC algorithm and the main function to train the agent
- `test_agent`: Load the network and the trained weights to test the agent while visualize the process of sumulation
- `environment.yml`: YAML file containing the environment dependencies, including the necessary Python packages and versions, for configuring the virtual environment.
# Train the agent
If you want to train from the beginning, just run the `python SAC_training.py` and wait for a while to see the result. It will be connected with the customized gym. 
# Test the agent
I have already trained the weight parameters. if you want to view the effect of the model, just run the `python test_agent.py` in the terminal. The average reward for each round of testing will be recorded.
