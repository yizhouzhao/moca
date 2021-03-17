from yz.task_dec2high_pddl import *
from transformers import BertTokenizer
from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel

from datetime import datetime

if __name__ == "__main__":
    args = set_up_args()
    args.gpu = True
    task2plan_train = load_task_and_plan_json(args, "train")
    tpd = TaskPlanDataset(task2plan_train)

    ## 写入字典
    #tpd.write_vocab("yz/vocab.txt")

    # 训练和测试模型
    train_dataloader = DataLoader(tpd, batch_size=args.batch, shuffle=True,)
    
    task2plan_valid_seen = load_task_and_plan_json(args, "valid_seen")
    tpd_valid_seen = TaskPlanDataset(task2plan_valid_seen)
    valid_seen_dataloader = DataLoader(tpd_valid_seen, batch_size=1, shuffle = True)

    # Initializing a BERT bert-base-uncased style configuration
    config_encoder = BertConfig()
    config_decoder = BertConfig()

    encoder_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    decoder_tokenizer = BasicTokenizer("yz/vocab.txt")


    config_decoder.update({
        "vocab_size": len(decoder_tokenizer.vocab),
        "num_hidden_layers":3,
        "num_attention_heads":3
    })

    config = EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)

    # 导入模型 BERT
    model = EncoderDecoderModel(config=config)
    model.encoder = BertModel.from_pretrained('bert-base-uncased')
    if args.gpu and torch.cuda.is_available():
        model = model.cuda()
    
    loss_fun = nn.CrossEntropyLoss()
    #loss_fun = nn.CrossEntropyLoss(ignore_index=0)

    optimizer = torch.optim.Adam(model.decoder.parameters(), lr=args.lr)

    # 记录时间
    begin_time = datetime.now()
    print("Start training BERT: ", begin_time)

    # 开始训练
    for epoch in range(args.epoch):
        model.train()
        train_loss_list = []
        train_acc_list = []
        for tasks, plans in tqdm(train_dataloader):
            tokenized_text = encoder_tokenizer(tasks, padding=True, truncation=True, 
                                            max_length=100, return_tensors="pt").input_ids
            targets = torch.LongTensor(decoder_tokenizer.tokenize(plans))
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

        if epoch % 5 == 0:
            model.eval()
            valid_all_match = []
            for tasks, plans in tqdm(valid_seen_dataloader):
                try:
                    tokenized_text = encoder_tokenizer(tasks, padding=True, truncation=True, max_length=100, return_tensors="pt").input_ids

                    if args.gpu and torch.cuda.is_available():
                            tokenized_text = tokenized_text.to("cuda")

                    output_labels = model.generate(tokenized_text, decoder_start_token_id=1)
                    ouput_array = output_labels.cpu().numpy()[0]

                    targets = decoder_tokenizer.tokenize(plans)

                    all_match = np.all(ouput_array[:len(targets[0])] == targets[0])
                    valid_all_match.append(1 if all_match else 0)
                except:
                    valid_all_match.append(0)
            
            print("epoch: {} valid_all_match: {:.3f}".format(epoch, np.mean(valid_all_match)))

    end_time = datetime.now()
    print("Finished Training:", end_time)
    print("Time elapsed (in hour): {:.2f}".format((end_time - begin_time).total_seconds() / 3600))
    
    # 保存模型
    torch.save(model.state_dict(), "exp/{}.pth".format(end_time))
    config.save_pretrained("exp/{}".format(end_time))