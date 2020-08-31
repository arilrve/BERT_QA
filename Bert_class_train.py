from transformers import AdamW
from Bert_classfication_QA_data import convert_data_to_ids,make_dataset,compute_accuracy,split_dataset
from albert_zh import AlbertConfig, AlbertTokenizer, AlbertForSequenceClassification
import pandas as pd
import torch
import torch.optim
from torch.utils.data import DataLoader

import numpy


tokenizer = AlbertTokenizer.from_pretrained("albert_tiny/vocab.txt")

# model setting
model_config = AlbertConfig
config = model_config.from_pretrained("albert_tiny/config.json",num_labels=130)
model = AlbertForSequenceClassification.from_pretrained("albert_tiny/pytorch_model.bin", from_tf=False, config=config)

# setting device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("using device",device)
model.to(device)

# pandas read excel
df = pd.read_excel("pdQA.xlsx") 
Q = df["question"].str.replace("Q_","")
A = df["answer"].str.replace("A_","")

# convert data by convert_data_to_ids function
data_feature = convert_data_to_ids(Q,A)
input_ids = data_feature['input_ids']
# input_masks = data_feature['input_masks']
input_segment_ids = data_feature['input_segment_ids']
answer_lables = data_feature['answer_lables']
# print(answer_lables)  #確認answer_label是相對應的

# dataset 、 dataloader
dataset = make_dataset(input_ids = input_ids,  answer_lables = answer_lables)

train_dataset, test_dataset = split_dataset(dataset, split_rate = 0.95)
train_dataloader = DataLoader(train_dataset,batch_size=16,shuffle=False)
test_dataloader = DataLoader(test_dataset,batch_size=16,shuffle=False) 


#optimizer
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
optimizer = AdamW(optimizer_grouped_parameters, lr=5e-6, eps=1e-8  )
# scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

model.zero_grad()
for epoch in range(400):
    running_loss_val = 0.0
    running_acc = 0.0
    for batch_index, batch_dict in enumerate(train_dataloader):
        model.train()
        batch_dict = tuple(t.to(device) for t in batch_dict)
        outputs = model(
            input_ids = batch_dict[0],
            # attention_mask=batch_dict[1],
            labels = batch_dict[1]
            )

        loss,logits = outputs[:2]
        loss.sum().backward()
        optimizer.step()
        # scheduler.step()  # Update learning rate schedule
        model.zero_grad()
            
        # compute the loss
        loss_t = loss.item()
        running_loss_val += (loss_t - running_loss_val) / (batch_index + 1)
        # compute the accuracy
        acc_t = compute_accuracy(logits, batch_dict[1])
        running_acc += (acc_t - running_acc) / (batch_index + 1)
        # log
        print("epoch:%2d batch:%4d train_loss:%2.4f train_acc:%3.4f"%(epoch+1, batch_index+1, running_loss_val, running_acc))

    running_loss_val = 0.0
    running_acc = 0.0
    for batch_index, batch_dict in enumerate(test_dataloader):
        model.eval()
        batch_dict = tuple(t.to(device) for t in batch_dict)
        with torch.no_grad():
            outputs = model(
                input_ids = batch_dict[0],
                # attention_mask=batch_dict[1],
                labels = batch_dict[1]
                )
            loss,logits = outputs[:2]
                
            # compute the loss
            loss_t = loss.item()
            running_loss_val += (loss_t - running_loss_val) / (batch_index + 1)

            # compute the accuracy
            acc_t = compute_accuracy(logits, batch_dict[1])
            running_acc += (acc_t - running_acc) / (batch_index + 1)
            # log
            print("epoch:%2d batch:%4d test_loss:%2.4f test_acc:%3.4f"%(epoch+1, batch_index+1, running_loss_val, running_acc))

model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
model_to_save.save_pretrained('trained_model')