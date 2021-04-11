from yz.nav_tools import FrameInfo
from yz.params import *
from yz.rl_policy import PolicyNetwork, QNetwork
from yz.utils import soft_update_from_to

from ai2thor.controller import Controller
from collections import deque

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class QueryAgent():
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
        self.history = deque(maxlen = 1000)
        # self.frame_map = {} # (position and rotation) -> FrameInfo

        # RL part
        self.observation = None
        self.last_action = None

        self.episode_done = False

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
        self.soft_target_tau = 0.01
        self.target_update_period = 1 # updatge target network frequency

        # record
        self.n_train_steps_total = 0
        self.episode_steps = 0
        self.episode_total_reward = 0

        # policy network
        self.input_dim = len(self.observation)
        self.hidden_dim = 64
        self.action_dim = 6
        self.policy = PolicyNetwork(self.input_dim, self.hidden_dim, self.action_dim)

        self.qf1 = QNetwork(self.input_dim , self.action_dim, self.hidden_dim)
        self.qf2 = QNetwork(self.input_dim , self.action_dim, self.hidden_dim)
        self.target_q1 = QNetwork(self.input_dim , self.action_dim, self.hidden_dim)
        self.target_q2 = QNetwork(self.input_dim , self.action_dim, self.hidden_dim)
        
        if self.use_gpu:
            self.policy = self.policy.cuda()
            self.qf1 = self.qf1.cuda()
            self.qf2 = self.qf2.cuda()
            self.target_q1 = self.target_q1.cuda()
            self.target_q2 = self.target_q2.cuda()

        # loss
        self.qf_criterion = nn.MSELoss()

        self.update_target_networks()

        self.policy_optimizer = optim.Adam(
            self.policy.parameters(),
            lr=self.learning_rate,
        )
        self.qf1_optimizer = optim.Adam(
            self.qf1.parameters(),
            lr=self.learning_rate,
        )
        self.qf2_optimizer = optim.Adam(
            self.qf2.parameters(),
            lr=self.learning_rate,
        )

        self.update_target_networks()
    
    def update_target_networks(self):
        soft_update_from_to(self.qf1, self.target_q1, self.soft_target_tau)
        soft_update_from_to(self.qf2, self.target_q2, self.soft_target_tau)
    
    def take_action(self, epsilon = 0.2):
        current_state = self.observation

        if np.random.rand() < epsilon:
            action_code = np.random.randint(6)
        else:
            current_state_tensor = torch.FloatTensor(current_state).unsqueeze(0)
            if self.use_gpu:
                current_state_tensor = current_state_tensor.to("cuda")
            action, _ = self.policy.sample_action_with_prob(current_state_tensor)
            
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

        self.history.append([self.observation.copy(), action_code, reward, next_observation, self.episode_done])
        
        self.observation = next_observation

        self.episode_steps += 1
        self.episode_total_reward += reward
        


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

    def learn(self):
        # sample history
        sample_list = random.sample(self.history, self.batch_size)
        s0 = [sample_list[i][0] for i in range(self.batch_size)]
        a = [[1 if j == sample_list[i][1] else 0 for j in range(self.action_dim)] for i in range(self.batch_size)]
        r = [[sample_list[i][2]] for i in range(self.batch_size)]
        s1 = [sample_list[i][3] for i in range(self.batch_size)]
        d = [[int(sample_list[i][4])] for i in range(self.batch_size)]
        
        s0 = torch.FloatTensor(s0)
        a = torch.LongTensor(a)
        r = torch.FloatTensor(r)
        s1 = torch.FloatTensor(s1)
        d = torch.FloatTensor(d)

        if self.use_gpu:
            s0 = s0.to("cuda")
            a = a.to("cuda")
            r = r.to("cuda")
            s1 = s1.to("cuda")
            d = d.to("cuda")

        """
        Policy loss
        """
        new_obs_actions, log_pi = self.policy.sample_action_with_prob(s0)
        log_pi = log_pi.unsqueeze(-1)

        q_new_actions = torch.min(
            self.qf1(s0, new_obs_actions),
            self.qf2(s0, new_obs_actions),
        )

        policy_loss = (self.alpha*log_pi - q_new_actions).mean()

        """
        QF Loss
        """
        q1_pred = self.qf1(s0, a)
        q2_pred = self.qf2(s0, a)
        new_next_actions, new_log_pi = self.policy.sample_action_with_prob(s1)
        new_log_pi = new_log_pi.unsqueeze(-1)

        target_q_values = torch.min(self.target_q1(s1, new_next_actions),self.target_q2(s1, new_next_actions)) - self.alpha * new_log_pi

        q_target = r + (1. - d) * self.gamma * target_q_values

        qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
        qf2_loss = self.qf_criterion(q2_pred, q_target.detach())


        #update parameters
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        self.n_train_steps_total += 1
        if self.n_train_steps_total % self.target_update_period == 0:
            self.update_target_networks()

    def reset_scene_and_target(self, scene: str, target: str):
        self.controller.reset(scene=scene)
        self.target_type = target

    def reset_episode(self):
        self.reset_scene_and_target(self.scene, self.target)
        self.episode_done = False
        self.episode_steps = 0
        self.episode_total_reward = 0

    def close(self):
        self.controller.stop()
