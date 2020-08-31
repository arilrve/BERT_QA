import torch
from albert_zh import AlbertForSequenceClassification,AlbertTokenizer,AlbertConfig
import pickle


pkl_file = pkl_file = open('data_features.pkl', 'rb')
data_features = pickle.load(pkl_file)
answer_dic = data_features['answer_dic']
answer_list = data_features['ans_list']
# print(answer_dic)
# print(answer_list)


#setting model from pretrained
model_config = AlbertConfig
model_file_path = "trained_model/pytorch_model.bin"
config = model_config.from_pretrained("trained_model/config.json",num_labels=130)
model = AlbertForSequenceClassification.from_pretrained(model_file_path, from_tf=False, config=config)
model.eval()

tokenizer = AlbertTokenizer.from_pretrained("albert_tiny/vocab.txt")

q_input=['透析方式','尿毒症狀']
for q in q_input:
    q_tokenize = tokenizer.tokenize(q)
    bert_ids = tokenizer.build_inputs_with_special_tokens(tokenizer.convert_tokens_to_ids(q_tokenize))
    input_ids = torch.LongTensor(bert_ids).unsqueeze(0)
    print(input_ids)
    outputs = model(input_ids)
    predicts = outputs[:1]
    # print(outputs[:1])
    predicts = predicts[0]
    print(predicts)
    max_val = torch.max(predicts)
    print(max_val)
    label = (predicts == max_val).nonzero().numpy()[0][1]

    for ans_id , ans_text in answer_list:
        if label == ans_id:
            print("Question:", q)
            print("Answer(label , answer):", label,ans_text)
            print()