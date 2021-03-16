#!/usr/bin/env python
# coding: utf-8

# # 导入数据

# In[ ]:


from yz.task_dec2high_pddl import *


# In[ ]:


args = set_up_args()


# In[ ]:


args.gpu = True


# In[ ]:


task2plan_train = load_task_and_plan_json(args, "train")


# In[ ]:


tpd = TaskPlanDataset(task2plan_train)


# # 写入字典

# In[ ]:


#tpd[np.random.randint(1000)]


# In[ ]:


tpd.write_vocab("yz/vocab.txt")


# # 训练

# In[ ]:


train_dataloader = DataLoader(tpd, batch_size=args.batch, shuffle=True,)


# In[ ]:


tasks, plans = next(iter(train_dataloader))


# In[ ]:


tasks


# In[ ]:


from transformers import BertTokenizer


# In[ ]:


# Load pre-trained model tokenizer (vocabulary)
encoder_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# tokenized_text = encoder_tokenizer(tasks, padding=True, truncation=True, max_length=100, return_tensors="pt").input_ids

decoder_tokenizer = BasicTokenizer("yz/vocab.txt")

#targets = torch.LongTensor(decoder_tokenizer.tokenize(plans))


# In[ ]:


from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel


# # 导入模型 BERT

# In[ ]:


# Initializing a BERT bert-base-uncased style configuration
config_encoder = BertConfig()
config_decoder = BertConfig()

config_decoder.update({
    "vocab_size": len(decoder_tokenizer.vocab),
    "num_hidden_layers":3,
    "num_attention_heads":3
})

config = EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)


# In[ ]:


model = EncoderDecoderModel(config=config)


# In[ ]:


model.encoder = BertModel.from_pretrained('bert-base-uncased')


# In[ ]:


if args.gpu and torch.cuda.is_available():
    model = model.cuda()


# In[ ]:


loss_fun = nn.CrossEntropyLoss(ignore_index=0)


# In[ ]:


optimizer = torch.optim.Adam(model.decoder.parameters(), lr=args.lr)


# In[ ]:


from datetime import datetime

# datetime object containing current date and time
begin_time = datetime.now()
print(begin_time)


# In[ ]:


for epoch in range(args.epoch):
    model.train()
    train_loss_list = []
    train_acc_list = []
    for tasks, plans in tqdm(train_dataloader):
        tokenized_text = encoder_tokenizer(tasks, padding=True, truncation=True, 
                                           max_length=100, return_tensors="pt").input_ids
#         try:
        targets = torch.LongTensor(decoder_tokenizer.tokenize(plans))
#         except:
#             print(tasks)
#             print(plans)
        if args.gpu and torch.cuda.is_available():
            tokenized_text = tokenized_text.to("cuda")
            targets = targets.to("cuda")
        
        outputs = model(input_ids = tokenized_text, decoder_input_ids = targets[:,:-1])
        
        output_dim = outputs.logits.shape[-1]
        outputs.logits = outputs.logits.contiguous().view(-1, output_dim)
        labels = targets[:,1:].contiguous().view(-1)
        
        train_loss = loss_fun(outputs.logits, labels)
        #print(train_loss)
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        train_acc = calculate_accuracy(outputs.logits, labels)
        
        train_loss_list.append(train_loss.item())
        train_acc_list.append(train_acc)
        
        
    mean_train_loss = sum(train_loss_list) / (len(train_loss_list) + 1e-4)
    mean_train_acc = sum(train_acc_list) / (len(train_acc_list) + 1e-4)
    print("epoch: {} train_loss: {:.3f}, train_acc: {:.3f}".format(epoch, mean_train_loss, mean_train_acc))


# In[ ]:


targets.shape


# In[ ]:


plans


# In[ ]:


for p in plans:
    print(len(p.split()))


# In[ ]:


max_length = max([len(item.split()) for item in plans])


# In[ ]:


what = decoder_tokenizer.tokenize(plans)


# In[ ]:


what[0]


# In[ ]:


for w in what:
    print(len(w))


# In[ ]:


decoded_string = ""


# In[ ]:


decoder_tokenizer.vocab_verse = {v:k for k,v in decoder_tokenizer.vocab.items()}


# In[ ]:


for label in what[-3]:
    decoded_string += decoder_tokenizer.vocab_verse[label] + " "


# In[ ]:


decoded_string


# In[ ]:


torch.LongTensor(what)


# In[ ]:


torch.LongTensor(decoder_tokenizer.tokenize(plans))


# In[ ]:


# datetime object containing current date and time
end_time = datetime.now()
print(end_time)


# In[ ]:


torch.save(model.state_dict(), "exp/{}.pth".format(end_time))


# In[ ]:


#model.load_state_dict(torch.load("exp/2021-03-15 20:11:33.176086.pth"))


# In[ ]:




