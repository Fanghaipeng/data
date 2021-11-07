## 自动对对联的例子
import sys
import json
import torch
from tqdm import tqdm
import time
from torch.utils.data import Dataset, DataLoader
from bert_seq2seq import Tokenizer, load_chinese_base_vocab
from bert_seq2seq import load_bert

vocab_path = "/home/glt/pretrain/transformer/chinese_wwm_ext_pytorch/vocab.txt" # roberta模型字典的位置
model_name = "roberta" # 选择模型名字
model_path = "bert_poi_model.bin" # roberta模型位置
data_dir = "../data/TestA_Preporcess_public.json"
word2idx = load_chinese_base_vocab(vocab_path)


def read_corpus(dir_path):
    """
    读原始数据
    """
    with open(dir_path, 'r') as f:
        data = json.load(f)
    test_data = {}
    for key, item in data['data'].items():
        text = [x['text'] for x in item['texts']]
        test_data[key] = {
            'src': ','.join(text),
        }

    return test_data
    
class BertDataset(Dataset):
    """
    针对特定数据集，定义一个相关的取数据的方式
    """
    def __init__(self, sents_src, sents_tgt) :
        ## 一般init函数是加载所有数据
        super(BertDataset, self).__init__()
        # 读原始数据
        # self.sents_src, self.sents_tgt = read_corpus(poem_corpus_dir)
        self.sents_src = sents_src
        self.sents_tgt = sents_tgt

        
        self.idx2word = {k: v for v, k in word2idx.items()}
        self.tokenizer = Tokenizer(word2idx)

    def __getitem__(self, i):
        ## 得到单个数据
        # print(i)
        src = self.sents_src[i]
        tgt = self.sents_tgt[i]
        token_ids, token_type_ids = self.tokenizer.encode(src, tgt)
        output = {
            "token_ids": token_ids,
            "token_type_ids": token_type_ids,
        }
        return output

    def __len__(self):

        return len(self.sents_src)
        
def collate_fn(batch):
    """
    动态padding， batch为一部分sample
    """

    def padding(indice, max_length, pad_idx=0):
        """
        pad 函数
        """
        pad_indice = [item + [pad_idx] * max(0, max_length - len(item)) for item in indice]
        return torch.tensor(pad_indice)

    token_ids = [data["token_ids"] for data in batch]
    max_length = max([len(t) for t in token_ids])
    token_type_ids = [data["token_type_ids"] for data in batch]

    token_ids_padded = padding(token_ids, max_length)
    token_type_ids_padded = padding(token_type_ids, max_length)
    target_ids_padded = token_ids_padded[:, 1:].contiguous()

    return token_ids_padded, token_type_ids_padded, target_ids_padded

class Trainer:
    def __init__(self):
        # 加载数据
        self.test_data = read_corpus(data_dir)
        
        # 判断是否有可用GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device: " + str(self.device))
        # 定义模型
        self.bert_model = load_bert(word2idx, model_name=model_name)
        ## 加载预训练的模型参数～
        self.bert_model.load_all_params(model_path)
        # 将模型发送到计算设备(GPU或CPU)
        self.bert_model.set_device(self.device)

    def test(self):
        # 一个epoch的训练
        self.bert_model.eval()
        for id, item in tqdm(self.test_data.items()):
            text = item['src']
            gene_text = self.bert_model.generate(text, beam_size=3)
            gene_text = gene_text.replace(' ', '').replace('[UNK]', '')
            item['tgt'] = gene_text
            print('{} =====> 【{}】'.format(text, gene_text))
        self.to_submit()
    
    def to_submit(self):
        with open('../data/result_example.json', 'r') as f:
            data = json.load(f)
        for id, item in self.test_data.items():
            data[id] = item['tgt']
        with open('../data/submit/submit.json', 'w') as f:
            json.dump(data, f)


if __name__ == '__main__':
    trainer = Trainer()
    trainer.test()
