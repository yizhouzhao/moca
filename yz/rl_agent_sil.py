from yz.nav_tools import FrameInfo
from yz.params import *
from yz.rl_policy import PolicyNetwork, ValueNetwork
from yz.utils import soft_update_from_to

from ai2thor.controller import Controller
from collections import deque

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QueryAgentSIL():
    def __init__(self, scene, target, room_object_types = G_livingroom_objtype):
        # ai2thor
        self.scene = scene
        self.target = target
        assert self.target in room_object_types
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
        self.replay_buffer = deque(maxlen = 1000)

        # self.frame_map = {} # (position and rotation) -> FrameInfo

        # RL part
        self.observation = None
        self.last_action = None

        self.episode_done = False
        self.episode_history = []
        

        # reward
        self.time_penalty = -0.01
        self.action_fail_penalty = -0.01

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
        
        # learning
        self.batch_size = 4
        self.learning_rate = 0.001
        self.alpha = 1 # soc temparature
        self.gamma = 0.95 # discount factor

        # record
        self.n_train_steps_total = 0
        self.episode_steps = 0
        self.episode_total_reward = 0
        

        # policy network
        self.state_dim = len(self.observation)
        self.hidden_dim = 64
        self.action_dim = 6
        self.policy = PolicyNetwork(self.state_dim, self.hidden_dim, self.action_dim)
        self.value_net = ValueNetwork(self.state_dim, self.action_dim)
        
        if self.use_gpu:
            self.policy = self.policy.cuda()
            self.value_net = self.value_net.cuda()

        # loss
        self.vf_criterion = nn.MSELoss()


        self.policy_optimizer = optim.Adam(
            self.policy.parameters(),
            lr=self.learning_rate,
        )
        self.value_optimizer = optim.Adam(
            self.value_net.parameters(),
            lr=self.learning_rate,
        )

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
    
    def take_action(self):
        current_state = self.observation

        current_state_tensor = torch.FloatTensor(current_state).unsqueeze(0)
        if self.use_gpu:
            current_state_tensor = current_state_tensor.to("cuda")
        action, log_prob = self.policy.sample_action_with_prob(current_state_tensor)
        
        action_code = torch.argmax(action, dim = -1)[0].item()
        #print(action_code)

        self.step(action_code)
        next_observation = self.get_observation()
        
        # calulate reward
        reward = self.time_penalty
        
        if not self.event.metadata["lastActionSuccess"]:
            reward += self.action_fail_penalty

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

        self.episode_history.append([self.observation.copy(), action_code, reward, log_probs[0], self.episode_done])
        
        self.observation = next_observation

        # for print record
        self.episode_steps += 1
        self.episode_total_reward += reward

    def learn(self):
        '''
        On episode ends, learn somethings
        '''

        # A2C part
        policy_loss = 0
        value_loss = 0

        R = 0
        for i in reversed(range(len(self.episode_history))):
            state = h[0]
            action = h[1]
            reward = h[2]
            log_probs = h[3]

            h = self.episode_history[i]
            R = self.gamma * R + reward

            s0 = torch.FloatTensor(state)
            if self.use_gpu:
                s0 = s0.to("cuda")
            
            value_i = self.value_net(s0.unsqueeze(-1))[0]
            advantage = R - values_i
            value_loss += 0.5 * advantage ** 2

            entropy = - torch.sum(torch.exp(log_probs) * log_probs)

            policy_loss = policy_loss - log_probs[action] * advantage.detach() - self.alpha * entropy
        
        #update parameters
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        

    def reset_scene_and_target(self, scene: str, target: str):
        self.controller.reset(scene=scene)
        self.target_type = target

    def reset_episode(self):
        self.reset_scene_and_target(self.scene, self.target)
        self.episode_done = False
        self.episode_steps = 0
        self.episode_total_reward = 0
        self.episode_policy_loss.clear()
        self.episode_pf1_loss.clear()
        self.episode_pf2_loss.clear()

    def close(self):
        self.controller.stop()

    def save_model(self):
        from datetime import datetime
        now = datetime.now()
        time_str = now.strftime("%H:%M:%S")
        torch.save(self.qf1.state_dict(), "record/qf1_" + time_str + ".pth")
        torch.save(self.qf2.state_dict(), "record/qf2_" + time_str + ".pth")
        torch.save(self.qf2.state_dict(), "record/policy_" + time_str + ".pth")