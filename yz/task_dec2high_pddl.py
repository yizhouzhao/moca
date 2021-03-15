import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import json
import collections
import numpy as np

from data.preprocess import Dataset
from tqdm.auto import tqdm

from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

from transformers import EncoderDecoderModel

from transformers import BertConfig, EncoderDecoderConfig, BertModel
from transformers.models.bert.modeling_bert import BertEncoder, BertPooler

import torch
import torch.nn as nn

seq_hidden_encode_dim = 32
seq_hidden_dim = 8
seq_num_attention_heads = 6


def set_up_args():
    # parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    # settings
    parser.add_argument('--seed', help='random seed', default=123, type=int)
    parser.add_argument('--data', help='dataset folder', default='data/json_feat_2.1.0')
    parser.add_argument('--splits', help='json file containing train/dev/test splits', default='data/splits/oct21.json')
    parser.add_argument('--preprocess', help='store preprocessed data to json files', action='store_true')
    parser.add_argument('--pp_folder', help='folder name for preprocessed data', default='pp')
    parser.add_argument('--save_every_epoch', help='save model after every epoch (warning: consumes a lot of space)', action='store_true')
    parser.add_argument('--model', help='model to use', default='seq2seq_im_mask')
    parser.add_argument('--gpu', help='use gpu', action='store_true')
    parser.add_argument('--dout', help='where to save model', default='exp/model:{model}')
    parser.add_argument('--resume', help='load a checkpoint')

    # hyper parameters
    parser.add_argument('--batch', help='batch size', default=4, type=int)
    parser.add_argument('--epoch', help='number of epochs', default=50, type=int)
    parser.add_argument('--lr', help='optimizer learning rate', default=7e-6, type=float)
    parser.add_argument('--decay_epoch', help='num epoch to adjust learning rate', default=10, type=int)
    parser.add_argument('--dhid', help='hidden layer size', default=512, type=int)
    parser.add_argument('--dframe', help='image feature vec size', default=3*7*7, type=int)
    parser.add_argument('--demb', help='language embedding size', default=100, type=int)
    parser.add_argument('--pframe', help='image pixel size (assuming square shape eg: 300x300)', default=300, type=int)
    parser.add_argument('--mask_loss_wt', help='weight of mask loss', default=1., type=float)
    parser.add_argument('--action_loss_wt', help='weight of action loss', default=1., type=float)
    parser.add_argument('--subgoal_aux_loss_wt', help='weight of subgoal completion predictor', default=0.2, type=float)
    parser.add_argument('--pm_aux_loss_wt', help='weight of progress monitor', default=0.2, type=float)

    # dropouts
    parser.add_argument('--zero_goal', help='zero out goal language', action='store_true')
    parser.add_argument('--zero_instr', help='zero out step-by-step instr language', action='store_true')
    parser.add_argument('--lang_dropout', help='dropout rate for language (goal + instr)', default=0., type=float)
    parser.add_argument('--input_dropout', help='dropout rate for concatted input feats', default=0., type=float)
    parser.add_argument('--vis_dropout', help='dropout rate for Resnet feats', default=0.3, type=float)
    parser.add_argument('--hstate_dropout', help='dropout rate for LSTM hidden states during unrolling', default=0.3, type=float)
    parser.add_argument('--attn_dropout', help='dropout rate for attention', default=0., type=float)
    parser.add_argument('--actor_dropout', help='dropout rate for actor fc', default=0., type=float)

    # other settings
    parser.add_argument('--dec_teacher_forcing', help='use gpu', action='store_true')
    parser.add_argument('--temp_no_history', help='use gpu', action='store_true')

    # debugging
    parser.add_argument('--fast_epoch', help='fast epoch during debugging', action='store_true')
    parser.add_argument('--dataset_fraction', help='use fraction of the dataset for debugging (0 indicates full size)', default=0, type=int)

    args = parser.parse_args("")
    return args


def load_task_and_plan_json(args, split_type = "train"):
    '''
    Load task and plan from json:
    only load the task description and high-level ppdl
    '''
    with open(args.splits) as f:
        splits = json.load(f)
        print({k: len(v) for k, v in splits.items()})

    k = split_type
    d = splits[k]

    task2plan_split = []
    for task in tqdm(d):
        # load json file
        json_path = os.path.join(args.data, k, task['task'], 'traj_data.json')
        with open(json_path) as f:
            ex = json.load(f)
            #len_pddl = len(ex['plan']['high_pddl'])
            #print(ex)
            plans = []
            for item in ex['plan']['high_pddl']:
                #print(item)
                action = item['discrete_action']['action']
                action_args = item['discrete_action']['args'][0] if len(item['discrete_action']['args']) > 0 else None
                
                plans.append([action,action_args])
                
            task_desc = []
            for item in ex['turk_annotations']['anns']:
                task_desc.append(item['task_desc'])
            
            
        task2plan_split.append({"task_desc":task_desc, "plans":plans})


    return task2plan_split


class TaskPlanDataset(Dataset):
    def __init__(self, task2plan_split:list):
        '''
        task2plan_split:
        [{"task":... "plans":....}]
        '''
        # stuff
        self.raw_data = task2plan_split

        self.tasks = []
        self.raw_plans = []
        self.plans = []

        self.allword2index = {"<PAD>":0, "<START>":1}
        #self.action2index = {"NoOp":0}
        #self.action_arg2index= {"NoArg":0}
        self.preprocess_raw_data()
        
    def write_vocab(self, write_path):
        with open(write_path, "w") as f:
            for key in self.allword2index:
                f.write(key+"\n")

    
    def preprocess_raw_data(self):
        for item in tqdm(self.raw_data):
            plans = item["plans"]
            plans_codes = []

            for plan in plans:
                action = plan[0]
                action_arg = "NoArg" if plan[1] is None else plan[1]

                plans_codes.append(action)
                plans_codes.append(action_arg)
                
                if action not in self.allword2index:
                    action_code = len(self.allword2index)
                    self.allword2index[action] = action_code
                
                if action_arg not in self.allword2index:
                    action_code = len(self.allword2index)
                    self.allword2index[action_arg] = action_code




                self.raw_plans.append(plans)
            # plan_codes = []
            # for plan in plans:
            #     action = plan[0]
            #     action_arg = "NoArg" if plan[1] is None else plan[1]

            #     if action not in self.action2index:
            #         action_code = len(self.action2index)
            #         self.action2index[action] = action_code
            #     else:
            #         action_code = self.action2index[action]
                
            #     if action_arg not in self.action_arg2index:
            #         action_arg_code = len(self.action_arg2index)
            #         self.action_arg2index[action_arg] = action_arg_code
            #     else:
            #         action_arg_code = self.action_arg2index[action_arg]

            #     plan_codes.append([action_code, action_arg_code])

            for task in item['task_desc']:
                self.tasks.append(task)
                self.plans.append(" ".join(plans_codes))
        
        
    def __getitem__(self, index):
        # stuff
        task = self.tasks[index]
        plan = self.plans[index]
        return task, plan

    def __len__(self):
        return len(self.tasks) # of how many examples(images?) you have

class BasicTokenizer(object):
    def __init__(self, vocab_file):
        self.vocab = self.load_vocab(vocab_file)
    
    def load_vocab(self, vocab_file):
        """Loads a vocabulary file into a dictionary."""
        vocab = collections.OrderedDict()
        index = 0
        with open(vocab_file, "r", encoding='utf8') as reader:
            for token in reader.readlines():
                token = token.strip()
                vocab[token] = index
                index += 1
        return vocab
    
    def tokenize(self, input_string_list):
        tokenized_list = []
        max_length = max([len(item.split()) for item in input_string_list])
        for input_string in input_string_list:
            tokenized_line = [1]
            words = input_string.split(" ")
            if len(words) > max_length:
                max_length =  len(words)
            for word in words:
                tokenized_line.append(self.vocab[word])
            for j in range(max_length - len(words)):
                tokenized_line.append(self.vocab["<PAD>"])
            
            tokenized_list.append(tokenized_line)

        return tokenized_list

def calculate_accuracy(logits, labels):
    acc_maxtrix = torch.argmax(logits, dim = -1) == labels
    return np.mean(acc_maxtrix.cpu().numpy())