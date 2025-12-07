import json

from torch.utils.data import Dataset
import torch
import os

# 为了避免tokenizers的并行化警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 先写dataset类，实现dataset内定的方法
class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(data_path)

    def load_data(self, data_path):
        samples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        # 对文本进行tokenizer编码 获取ids和attention_mask
        # 1. 提取并处理字段
        context = sample.get("context", "").strip()
        input_text = sample.get("input", "").strip()

        # 2. 拼接文本（自定义规则）
        if not context and not input_text:
            raise ValueError(f"样本无有效文本（context/input 均为空）：{sample}")
        full_text = ""
        if context and input_text:
            full_text = f"{context}\n{input_text}"  # 换行分隔
        else:
            full_text = context if context else input_text

        # 对文本进行tokenizer编码 获取ids和attention_mask
        encoding = self.tokenizer(
            # str(sample["text"]),
            full_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0)  # 去掉batch维度
        # [1,1,1,0,0] 表示前三个token是有效的
        loss_mask = input_ids != self.tokenizer.pad_token_id  # 生成loss mask，pad部分为0，其他为1

        # 自回归
        # 自回归训练的核心是「用前 n-1 个 token 预测第 n 个 token」，因此需要对 input_ids 做 “错位切割”：
        # X：输入序列 → 去掉最后一个token（用前max_length-1个token作为模型输入）
        X = torch.tensor(input_ids[:-1], dtype = torch.long)
        # Y：标签序列 → 去掉第一个token（每个位置的标签是下一个token）
        Y = torch.tensor(input_ids[1:], dtype = torch.long)
        # loss_mask：和X长度匹配 → 去掉最后一个（忽略最后一个位置的pad损失）
        loss_mask = torch.tensor(loss_mask[:-1], dtype = torch.long)
        return X, Y, loss_mask