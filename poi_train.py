## 自动对对联的例子
import sys
import torch
import random
from tqdm import tqdm
import time
from torch.utils.data import Dataset, DataLoader
from bert_seq2seq import Tokenizer, load_chinese_base_vocab
from bert_seq2seq import load_bert
from transformers import AdamW, get_linear_schedule_with_warmup

vocab_path = "/home/glt/pretrain/transformer/chinese_wwm_ext_pytorch/vocab.txt" # roberta模型字典的位置
model_name = "roberta" # 选择模型名字
model_path = "/home/glt/pretrain/transformer/chinese_wwm_ext_pytorch/pytorch_model.bin" # roberta模型位置
recent_model_path = "" # 用于把已经训练好的模型继续训练
model_save_path = "./bert_poi_model_shuffle.bin"
batch_size = 32
lr = 5e-5
data_dir = "../data/translate_dataset.txt"
word2idx = load_chinese_base_vocab(vocab_path)
shuffle = True


def read_corpus(dir_path):
    """
    读原始数据
    """
    sents_src = []
    sents_tgt = []
    test_src = []
    test_tgt = []
    with open(dir_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            split = line.strip('\n').split(',')
            sents_src.append(','.join(split[:-1]))
            sents_tgt.append(split[-1])
    num_test = 20
    test_src = sents_src[:num_test]
    test_tgt = sents_tgt[:num_test]
    sents_src = sents_src[num_test:]
    sents_tgt = sents_tgt[num_test:]
    return sents_src, sents_tgt, test_src, test_tgt
    
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
        if shuffle:
            src_list = src.split(',')
            random.shuffle(src_list)
            src = ','.join(src_list)
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
        self.sents_src, self.sents_tgt, self.test_src, self.test_tgt = read_corpus(data_dir)
        # 判断是否有可用GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device: " + str(self.device))
        # 定义模型
        self.bert_model = load_bert(word2idx, model_name=model_name)
        ## 加载预训练的模型参数～
        self.bert_model.load_pretrain_params(model_path)
        # 将模型发送到计算设备(GPU或CPU)
        self.bert_model.set_device(self.device)
        # 声明需要优化的参数
        self.optim_parameters = list(self.bert_model.parameters())
        self.optimizer = AdamW(self.optim_parameters, lr=lr)
        # 声明自定义的数据加载器
        dataset = BertDataset(self.sents_src, self.sents_tgt)
        self.dataloader =  DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        total_steps = len(self.dataloader) * 10
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, 
                                                    num_warmup_steps=0, # Default value in run_glue.py
                                                    num_training_steps=total_steps)

    def train(self, epoch):
        # 一个epoch的训练
        self.bert_model.train()
        self.iteration(epoch, dataloader=self.dataloader, train=True)
    
    def save(self, save_path):
        """
        保存模型
        """
        self.bert_model.save_all_params(save_path)
        print("{} saved!".format(save_path))

    def iteration(self, epoch, dataloader, train=True):
        total_loss = 0
        start_time = time.time() ## 得到当前时间
        step = 0
        for token_ids, token_type_ids, target_ids in tqdm(dataloader,position=0, leave=True):
            step += 1
            if step % 100 == 0:
                self.bert_model.eval()
                for text, test_label in zip(self.test_src, self.test_tgt):
                    gene_text = self.bert_model.generate(text, beam_size=3)
                    print('{} =====> {} |【{}】'.format(text, test_label, gene_text))
                self.bert_model.train()

            # 因为传入了target标签，因此会计算loss并且返回
            predictions, loss = self.bert_model(token_ids,
                                                token_type_ids,
                                                labels=target_ids,
                                                )
            # 反向传播
            if train:
                # 清空之前的梯度
                self.optimizer.zero_grad()
                # 反向传播, 获取新的梯度
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.bert_model.parameters(), 1.0)
                # 用获取的梯度更新模型参数
                self.optimizer.step()
                self.scheduler.step()

            # 为计算当前epoch的平均loss
            total_loss += loss.item()
        
        end_time = time.time()
        spend_time = end_time - start_time
        # 打印训练信息
        print("epoch: {} loss: {:.4f} lr: {:.6f}".format(epoch, total_loss, self.scheduler.get_last_lr()[0]))
        # 保存模型
        self.save(model_save_path)

if __name__ == '__main__':
    trainer = Trainer()
    train_epoches = 10
    for epoch in range(train_epoches):
        # 训练一个epoch
        trainer.train(epoch)
