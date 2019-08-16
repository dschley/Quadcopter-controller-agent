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
        
        #goal: go to 200m in the air, then do a barrel roll
        # goal of 200 for z axis (self.sim.pose[2])
        # goal of rotating along x euler angle (theta) (self.sim.pose[3])
        '''
        self.g1 = 200. 
        self.g2 = np.pi / 4 
        self.g3 = np.pi / 2
        self.g4 = 3 * np.pi / 4
        self.g4 = np.pi
        self.g5 = np.pi + np.pi / 4
        self.g6 = np.pi + np.pi / 2
        self.g7 = np.pi + 3 * np.pi / 4
        self.g8 = 0
        '''
        self.rotationSteps = 8
        self.rotationStepSize = np.pi / (self.rotationSteps / 2)
        self.barrelRollGoals = [200.]
        self.barrelRollGoals.extend([np.pi * ((r / (self.rotationSteps / 2)) % 2) for r in range(1, self.rotationSteps + 1)])
        self.goalsMet = [0. for _ in range(self.rotationSteps + 1)]
        
        self.state_size = self.action_repeat * (6 + len(self.barrelRollGoals))

    '''
    def get_reward(self):
        """Uses current pose of sim to return reward."""
        reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        return reward
    '''
    
    def get_reward(self, previous_state):
        discount = 1.
        if 0. in self.goalsMet:
            currentBestGoal = self.goalsMet.index(0.) - 1
        else:
            print("no 0, all goals met?", self.goalsMet)
            return 1000000
        
        if self.goalsMet[0] == 0. and previous_state[2] > self.sim.pose[2]:
            discount = 0.1 #punish going backwards
        elif self.goalsMet[0] == 1. and previous_state[3] > self.sim.pose[3]:
            discount = 0.1
                
        closeness = (abs(200. - self.sim.pose[2]) / 200.) \
            if currentBestGoal == -1 else \
            (abs(self.barrelRollGoals[currentBestGoal + 1] - self.sim.pose[3]) / self.rotationStepSize)
        
        tierReward = 2 ** ((currentBestGoal + 1) ** 2)
        reward = tierReward * closeness
        reward = reward * discount
        return reward
    
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
        #print("ns",next_state,next_state.shape)
        return next_state, reward, done
    
    def updateGoals(self):
        if self.goalsMet[0] == 0.:
            if self.sim.pose[2] >= 200.:
                self.goalsMet[0] = 1.
        elif 0. in self.goalsMet:
            nextGoal = self.goalsMet.index(0.)
            if nextGoal == 1 and self.sim.pose[2] < 50:
                self.goalsMet[0] = 0.
            elif self.sim.pose[3] >= self.barrelRollGoals[nextGoal]:
                self.goalsMet[nextGoal] = 1.
            elif nextGoal > 1 and self.sim.pose[3] < self.barrelRollGoals[nextGoal - 1]:
                self.goalsMet[nextGoal - 1] = 0.
        
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