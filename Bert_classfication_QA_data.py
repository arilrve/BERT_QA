from albert_zh import AlbertTokenizer
import pandas as pd
import torch
from torch.utils.data import TensorDataset,DataLoader
import pickle
from  torch.utils.data import random_split 
tokenizer = AlbertTokenizer.from_pretrained("albert_tiny/vocab.txt")



#pandas read excel
# df = pd.read_excel("pdQA.xlsx") 
# Q = df["question"].str.replace("Q_","")
# A = df["answer"].str.replace("A_","")

max_seq_len = 256
input_id_list = []
segment_id_list = []
attention_id_list = []

#set question_data
#conver data to ids_embedding
def convert_data_to_ids(Question,Answer):
    for i in range(len(Question)):
        #split single_Q
        input_tokenize = tokenizer.tokenize(Question[i]) 

        #convert token to ids
        input_ids = tokenizer.encode_plus(
                    input_tokenize,
                    max_length=256,
                    padding=True,
                    pad_to_max_legth=True,
                    return_attention_mask=True,
                    # return_tensors='pt'
                )
        input_ids['input_ids'] = tokenizer.build_inputs_with_special_tokens(input_ids['input_ids'])
    #embedding padding (補0)
        while len(input_ids['input_ids']) <= max_seq_len:
            input_ids['input_ids'].append(0)
            input_ids['token_type_ids'].append(0)
            # input_ids['attention_mask'].append(0) (在albert不需要輸入attention_mask 所以這邊就當作註解)

    #list append padding embedding data
        input_id_list.append(input_ids['input_ids'])
        segment_id_list.append(input_ids['token_type_ids'])
        # attention_id_list.append(input_ids['attention_mask']) (在albert不需要輸入attention_mask 所以這邊就當作註解)
          
    #set answer_data
    answer_norepeat = sorted(list(set(Answer)))
    
    #用於查找id或是text的list
    ans_list = []
    for index_a,a in enumerate(answer_norepeat):
        if a != None:
            ans_list.append((index_a,a))
    # print(ans_list)

    #set answer_label
    #設定在訓練時要放進去訓練的label (必須對應每個問題 
    a_lables = []
    for answer in Answer :
        for ans_id,ans_text in ans_list:
            if answer == ans_text :
                # print(answer)
                a_lables.append(ans_id)

    # print(a_lables) #確定answer_label 是相對應的
    answer_lables = a_lables

    data_features ={'input_ids':input_id_list,
                    # 'input_masks':attention_id_list, (在albert不需要輸入attention_mask 所以這邊就當作註解)
                    'input_segment_ids':segment_id_list,
                    'answer_lables':answer_lables,
                    'question_dic':Question,
                    'answer_dic':Answer,
                    'ans_list':ans_list
                    }

 #將資料打包成pickle檔
    output = open('data_features.pkl', 'wb')
    pickle.dump(data_features,output)
    return data_features

#set dataset
def make_dataset(input_ids, answer_lables):
    all_input_ids = torch.tensor([input_id for input_id in input_ids],dtype=torch.long)
    # all_input_masks = torch.tensor([input_mask for input_mask in input_masks],dtype=torch.long)
    # all_input_segment_ids = torch.tensor([input_segment_id for input_segment_id in input_segment_ids],dtype=torch.long)
    all_answer_lables = torch.tensor([answer_lable for answer_lable in answer_lables],dtype=torch.long)
    return TensorDataset(all_input_ids ,  all_answer_lables)

#split dataset
def split_dataset(full_dataset, split_rate):  
    train_size = int(split_rate * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    return train_dataset,test_dataset

def compute_accuracy(y_pred, y_target):
    # 計算正確率
    _, y_pred_indices = y_pred.max(dim=1)
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100