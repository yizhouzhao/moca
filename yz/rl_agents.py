from yz.nav_tools import FrameInfo
from yz.params import *
from yz.rl_policy import PolicyNetwork

from ai2thor.controller import Controller
from collections impoort deque

import numpy as np
import torch

class QueryAgent():
    def __init__(self, scTrueene, target, room_object_types = G_livingroom_objtype):
        # ai2thor
        self.room_object_types = room_object_types
        self.controller = Controller(scene=scene, 
                        renderInstanceSegmentation=True,
                        width=1080,
                        height=1080)
        self.target_type = target

        # register query
        FrameInfo.candidates = self.room_object_types
        self.query_indicates = [True for _ in range(len(FrameInfo.candidates))]

        # Keep track of qa history 
        self.keep_frame_map = False # weather to keep history to avoid duplicated query
        self.history = deque(maxlen = 1000)
        # self.frame_map = {} # (position and rotation) -> FrameInfo

        # RL part
        self.observation = None
        self.last_action = None

        self.episode_done = False

        # reward
        self.time_penalty = -0.1

        self.first_seen = False
        self.first_seen_reward = 1 # reward for finding the object initially

        self.first_in_range = False
        self.first_in_range_reward = 5.0 # object in interaction range

        self.mission_success_reward = 10.0 # object in interaction range and done

        # init event
        self.event = None
        self.step(5) # self.controller.step("Done") 
        self.observation = self.get_observation()

        # training
        self.use_gpu = True

        # policy network
        input_dim = len(self.observation)
        hidden_dim = 64
        output_dim = 6
        self.p1 = PolicyNetwork(input_dim, hidden_dim, output_dim)
        if self.use_gpu:
            self.p1 = self.p1.cuda()

    def ret_scene_and_target(self, scene: str, target: str):
        self.controller.reset(scene=scene)
        self.target_type = target
    
    def take_action(self, epsilon = 0.2):
        current_state = self.observation

        if np.random.rand() < epsilon:
            action_code = np.random.randint(6)
        else:
            current_state_tensor = torch.FloatTensor(current_state).unsqueeze(0)
            if self.use_gpu:
                current_state_tensor = current_state_tensor.to("cuda")
            action_prob = self.p1(current_state_tensor)
            
            action_code = torch.argmax(action_prob, dim = -1)[0].item()

        self.step(action_code)
        next_observation = self.get_observation()
        # calulate reward
        reward = self.time_penalty
        
        frame_info = FrameInfo(self.event)
        if not self.first_seen:
            for obj in frame_info.object_info:
                if self.target_type == obj["objectType"]:
                    reward += self.first_seen_reward
                    self.first_seen = True
                    break
        
        if not self.first_in_range:
            for obj in frame_info.object_info:
                if self.target_type == obj["objectType"] and obj["visible"] == True:
                    reward += self.first_in_range_reward
                    self.first_in_range = True
                    break
        

        for obj in frame_info.object_info:
            if self.target_type == obj["objectType"] and obj["visible"] == True:
                if action_code == 5:
                    reward += self.mission_success_reward
                    self.episode_done = True
                break

        self.history.append([self.observation.copy(), action_code, reward, next_observation, self.episode_done])
        
        self.observation = next_observation


    def step(self, action_code:int):
        self.last_action = action_code
        self.event = self.controller.step(G_action_code2action[action_code])

    def get_observation(self):
        '''
        Get state for RL
        '''
        state = []
        frame_info = FrameInfo(self.event)

        # target encode
        target_encode = [0 for _ in range(len(self.room_object_types))]
        target_index = self.room_object_types.index(self.target_type)
        target_encode[target_index] = 1
        state.extend(target_encode)

        # agent state: last action success encode
        last_action_success = 1 if self.event.metadata["lastActionSuccess"] else -1
        state.append(last_action_success)
        last_action_encode = [0] * len(G_action2code)
        last_action_encode[self.last_action] = 1
        state.extend(last_action_encode)

        # agent head position
        head_pose = round(self.event.metadata["agent"]['cameraHorizon']) // 30
        state.append(head_pose)

        # object state query
        # if self.keep_history
        frame_info = FrameInfo(self.event)
        obj_query_state = frame_info.get_answer_array_for_all_candidates(self.query_indicates)
        state.extend(obj_query_state)

        return state

    def close(self):
        self.controller.stop()

    

