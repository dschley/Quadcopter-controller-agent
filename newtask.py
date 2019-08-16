import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        #self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 
        
        #goal: go to 200m in the air
        # goal of 200 for z axis (self.sim.pose[2])
        self.heightGoals = [50., 100., 150., 200.]
        self.goalsMet = [0. for _ in range(len(self.heightGoals))]
        
        self.state_size = self.action_repeat * (6 + len(self.heightGoals))

    '''
    def get_reward(self):
        """Uses current pose of sim to return reward."""
        reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        return reward
    '''
    
    #reward function with reward clipping
    def get_reward(self, previous_state):
        discount = 1
        if 0. in self.goalsMet:
            currentBestGoal = self.goalsMet.index(0.) - 1
        else:
            print("All goals met")
            return 100
        
        if previous_state[2] > self.sim.pose[2]:
            discount = 0.1 #punish going backwards
#             discount = -0.1 #reaaally punish going backwards
        
#         tierReward = (currentBestGoal + 2) / len(self.goalsMet)
        closeness = (self.sim.pose[2] / self.heightGoals[currentBestGoal + 1])
        upVReward = max((self.sim.v[2] + 1) / 5, 0.1)

        reward = closeness * upVReward
#         reward = closeness * upVReward * discount
#         reward = closeness * tierReward * upVReward * discount
#         print(closeness, tierReward, upVReward, discount, reward)

        reward = (reward * 0.99) + 0.01
        return reward
    
    '''
    #old reward function with possible exploding gradients problem
    def get_reward(self, previous_state):
        discount = 1.
        if 0. in self.goalsMet:
            currentBestGoal = self.goalsMet.index(0.) - 1
        else:
            print("no 0, all goals met?", self.goalsMet)
            return 1000000
        
        if previous_state[2] > self.sim.pose[2]:
            discount = 0.1 #punish going backwards

        closeness = (self.sim.pose[2] / self.heightGoals[currentBestGoal + 1])
        
        tierReward = 2 ** ((currentBestGoal + 1) ** 2)
        reward = tierReward * closeness
        reward = reward * discount
        
        #time penalty. dont want it to think that spending time in a good state is better than going to a better one
        reward = reward - (0.1 / (self.sim.pose[2] + 1))
        
        return reward
    '''
    
    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            p_s = self.sim.pose
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            
            self.updateGoals()
            reward += self.get_reward(p_s) 
            
            currentPose = self.sim.pose
            #print("p1",currentPose,currentPose.shape)
            currentPose = np.append(currentPose, np.array(self.goalsMet))
            #print("p2",currentPose,currentPose.shape)

            pose_all.append(currentPose)
        next_state = np.concatenate(pose_all)
        
        if done:
            #either ran out of time or crashed
#             reward -= 10 
            reward -= 1 
        
        #print("ns",next_state,next_state.shape)
        return next_state, reward, done
    
    def updateGoals(self):
        if 0. in self.goalsMet:
            nextGoal = self.goalsMet.index(0.)
            
            if self.sim.pose[2] >= self.heightGoals[nextGoal]:
                self.heightGoals[nextGoal] = 1.
            elif nextGoal > 1 and self.sim.pose[2] < self.heightGoals[nextGoal - 1]:
                self.heightGoals[nextGoal - 1] = 0.
        
    '''
    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done
    '''
    
    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        self.goalsMet = [0. for _ in self.goalsMet]
        #print(self.sim.pose.shape, np.array(self.goalsMet).shape)
        unrepeatedState = self.sim.pose
        unrepeatedState = np.append(unrepeatedState, np.array(self.goalsMet))
        #print(unrepeatedState,unrepeatedState.shape)
        state = np.concatenate([unrepeatedState] * self.action_repeat) 
        #print(state,state.shape)
        return state
    
    '''
    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state
    '''