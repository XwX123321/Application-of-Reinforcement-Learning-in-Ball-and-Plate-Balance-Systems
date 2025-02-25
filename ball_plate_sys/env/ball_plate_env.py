import os.path
import gymnasium as gym
import pybullet as p
import pybullet_data
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
import math


class BallBalanceEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'],
                'video.frames_per_second' : 50
                }

    def __init__(self, render=True):
        # Initialize the environment, define the action and observation spaces, and set up the physics simulation
        super(BallBalanceEnv, self).__init__()
        self._observation = []
        # Action space: motor angles on X and Y axes with a range of [-0.1, 0.1]
        self.action_space = spaces.Box(np.array([-0.1, -0.1]), np.array([0.1, 0.1]), dtype=np.float32)  # X, Y
        # Observation space: includes ball coordinates (x, y, z), velocity (vx, vy, vz), and platform angles (angle_x, angle_y)
        self.observation_space = spaces.Box(low=np.array([-0.5, -0.5, 0.4, -1.0, -1.0, -1.0, -math.pi*0.25, -math.pi*0.25]),
                                            high=np.array([0.5, 0.5, 1.0, 1.0, 1.0, 1.0, math.pi*0.25, math.pi*0.25]), dtype=np.float32)


        # Choose to render the environment with GUI or run without visualization (DIRECT)
        if (render):
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)

        # Set additional search paths to locate URDF files in PyBullet
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # Initialize key variables
        self.seed()
        self.angle_x = 0
        self.angle_y = 0
        self.envStepCounter = 0
        self.record_time = 0
        self.eps_reward = 0
        self.ave_reward = 0
        self.platform_size = 0.5
        self.platform_height = 0.5
        self.drop_threshold = 0.07
        self.max_episode_steps = 1500

    def seed(self, seed=None):
        # Initialize the random number generator
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

#
    def step(self, action):
       # Update platform angles based on the action and simulate one time step
       self.compute_angle(action)   # TODO
       p.stepSimulation()
       self ._observation = self.compute_observation()

       # Compute the reward and check if the episode is done
       reward = self.compute_reward()
       done = self.compute_done()

       # Record the step count and rewards
       self.record_time += 1
       if self.envStepCounter == 0:
           self.eps_reward = 0
       self.eps_reward += reward
       truncated = self.envStepCounter >=1500
       self.envStepCounter += 1
       self.ave_reward = self.eps_reward / self.envStepCounter

       # Return the current state, reward, done flag, and other info
       return np.array(self._observation), reward, done, truncated, {}



    def reset(self, seed = None, options = None):
        # Reset the environment, randomize ball position, and initialize the simulation
        if seed is not None:
            self.np_random, seed = seeding.np_random(seed)

        self.angle_x = 0
        self.angle_y = 0
        self.maxAngle = math.pi*0.2
        self.envStepCounter = 0

        # Reset the simulation environment
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(0.01)

        # Load plane and ball URDFs
        planeId = p.loadURDF("plane.urdf")
        random_x = self.np_random.uniform(low=-0.2, high=0.2)
        random_y = self.np_random.uniform(low=-0.2, high=0.2)
        cubeStartPos = [random_x, random_y, 1.5]
        cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
        path = os.path.abspath(os.path.dirname(__file__))

        # Load ball and platform URDFs
        self.botId_1 = p.loadURDF(os.path.join(path, "../ball.urdf"),
                                  cubeStartPos,
                                  cubeStartOrientation)

        self.botId_2 = p.loadURDF(os.path.join(path, "../plateform.urdf"),
                                  [0, 0, 0],
                                  cubeStartOrientation,
                                  useFixedBase=True
                                  )

        # Adjust the platform dynamics properties
        p.changeDynamics(self.botId_2, -1, restitution=0)
        p.changeDynamics(self.botId_2, -1, lateralFriction=0.8)

        # Compute the initial observation
        self._observation = self.compute_observation()
        info = {"reset_seed": seed}
        return np.array(self._observation), info

    def compute_angle(self, action):
        # Compute the new platform angles based on the action and contact status
        contacts = p.getContactPoints(bodyA=self.botId_1, bodyB=self.botId_2)

        if contacts:
            # If the ball is on the platform, update the angles based on the action
            action_x, action_y = action
            angle_x = clamp(self.angle_x + action_x, -self.maxAngle, self.maxAngle)
            angle_y = clamp(self.angle_y + action_y, -self.maxAngle, self.maxAngle)
            self.angle_x = angle_x
            self.angle_y = angle_y
        else:
            # If the ball is not on the platform, keep the platform level
            self.angle_x = 0
            self.angle_y = 0

        # Apply the computed angles to the platform's joints using POSITION_CONTROL
        p.setJointMotorControl2(bodyUniqueId=self.botId_2,
                                jointIndex=0,
                                #controlMode=p.POSITION_CONTROL,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=self.angle_y,
                                force=500
                                )
        p.setJointMotorControl2(bodyUniqueId=self.botId_2,
                                jointIndex=1,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=self.angle_x,
                                force=500
                                )

    def compute_observation(self):
        # Get the ball's position and velocity, and the platform's orientation
        ball_pos, _ = p.getBasePositionAndOrientation(self.botId_1) #get position
        ball_v, _ = p.getBaseVelocity(self.botId_1) #get ball velocity
        _, platform_orn = p.getBasePositionAndOrientation(self.botId_2)
        angle_x, angle_y, _ = p.getEulerFromQuaternion(platform_orn)

        # Construct the observation array
        observation = np.array([
            ball_pos[0], ball_pos[1], ball_pos[2],
            ball_v[0], ball_v[1], ball_v[2],
            angle_x, angle_y, ], dtype=np.float32)
        return observation

# set the reward function
    def compute_reward(self):
        ball_pos, _ = p.getBasePositionAndOrientation(self.botId_1) # get the ball's position
        #distance = np.sqrt(ball_pos[0]**2 + ball_pos[1]**2)
        distance = np.sqrt(ball_pos[0]**2 + ball_pos[1]**2) # compute the distance from the ball to the center of the plate
        dis_reward = -distance
        ball_v, _ = p.getBaseVelocity(self.botId_1)  # get the ball's velocity of three direction
        #speed = np.sqrt(ball_v[0] ** 2 + ball_v[1] ** 2)
        speed = np.sqrt(ball_v[0] ** 2 + ball_v[1] ** 2 + ball_v[2] ** 2) # compute the magnitude of velocity
        speed_reward = -speed*0.1

        platform_pos, platform_orn = p.getBasePositionAndOrientation(self.botId_2)
        angle_x, angle_y, _ = p.getEulerFromQuaternion(platform_orn)
        angle_penalty = - (abs(angle_x)**2 + abs(angle_y)**2) * 0.5

        # Total reward is the sum of the distance, speed, angle penalties, and termination penalty
        reward = dis_reward + speed_reward + angle_penalty# the goal is to make the ball closer to the center and minimum the speed
        if distance <= 0.05:
            reward += 1
        return reward

# if the ball falls out of the plate or the steps of simulation is too long, end this process
    def compute_done(self):
        # Check if the ball is out of bounds or if the maximum number of steps is reached
        ball_pos, _ = p.getBasePositionAndOrientation(self.botId_1)
        return abs(ball_pos[0] > 0.5) or abs(ball_pos[1] > 0.5) or self.envStepCounter >=1500 or ball_pos[2] < self.platform_height - self.drop_threshold

# make a value between min and max
def clamp(a, minn, maxn):
    return(max(min(maxn, a), minn))





