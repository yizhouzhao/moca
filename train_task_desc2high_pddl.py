from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel
from yz.task_dec2high_pddl import *

def train():
    args = set_up_args()
    task2plan_train = load_task_and_plan_json(args, "train")

    # Initializing a BERT bert-base-uncased style configuration
    config_encoder = BertConfig()
    config_decoder = BertConfig()

    config_decoder.update({
        "vocab_size": len(decoder_tokenizer.vocab),
        "num_hidden_layers":6,
        "num_attention_heads":6
    })
