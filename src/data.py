import random
from typing import Union
import os
import gin
import torch
from torch.utils.data import Dataset
import pickle
from common import get_dataset

@gin.configurable
class ItemTokenizer(object):
    def __init__(self, table_path: str="./dataset/amazon_food/item2index.txt"):
        self.path = table_path
        self.unk_token = 0
        self.tokenize_map = {}
        self.detokenize_map = {}
        self._build_map()

    def _build_map(self):
        with open(self.path, 'r') as file:
            while True:
                data = file.readline()
                if data == '':
                    break
                data = data.split(',')
                item_id, item_index = str(data[0]), int(data[1])
                self.tokenize_map[item_id] = item_index
                self.detokenize_map[item_index] = item_id

    def get_vocab_size(self):
        return len(self.detokenize_map) + 1

    def tokenize(self, item: Union[str, int]):
        item = str(item)
        if item in self.tokenize_map:
            return self.tokenize_map[item]
        return self.unk_token

    def detokenize(self, id: int):
        if id in self.detokenize_map:
            return self.detokenize_map[id]
        return "<unk>"

    def in_vocab(self, item: Union[str, int]):
        item = str(item)
        return item in self.tokenize_map

    def save(self, path:str):
        with open(path, 'wb') as f:
            obj = {"tokenize_map": self.tokenize_map, "detokenize_map": self.detokenize_map}
            pickle.dump(obj, f)


@gin.configurable
class CateTokenizer(ItemTokenizer):
    def __init__(self, table_path: str="./dataset/amazon_food/cate2index.txt"):
        super(CateTokenizer, self).__init__(table_path)


def pad_or_truncate(to_pad, max_seq_len, padding_value):
    orig_length = len(to_pad)
    if orig_length > max_seq_len:
        return to_pad[-max_seq_len:], max_seq_len
    else:
        to_pad.extend([padding_value] * (max_seq_len-len(to_pad)))
        return to_pad, orig_length


def preprocess_data(data, max_seq_len, item_tokenizer, cate_tokenizer, use_ar_on_rank_samples, include_target_for_ar, modelling_style="ar"):
    target_item_id = item_tokenizer.tokenize(data['item_id'])
    target_cate_id = cate_tokenizer.tokenize(data['cate_id'])
    click_label = int(data["is_clicked"])
    sequence = data["seq_value"]
    historical_item_ids = []
    historical_cate_ids = []
    item_ar_labels = []
    cate_ar_labels = []
    if sequence:
        item_seq_kvs = [dict([x.split(':') for x in item.split('#') if x]) for item in sequence.split(';')]
        for kvs in item_seq_kvs:
            item_id = item_tokenizer.tokenize(kvs["item__item_id"])
            cate_id = cate_tokenizer.tokenize(kvs["item__cate_id"])
            if item_id == item_tokenizer.unk_token:
                continue
            if historical_item_ids:
                item_ar_labels.append(item_id)
                cate_ar_labels.append(cate_id)
            historical_item_ids.append(item_id)
            historical_cate_ids.append(cate_id)
        if include_target_for_ar and (click_label >= 1):
            item_ar_labels.append(target_item_id)
            cate_ar_labels.append(target_cate_id)
        else:
            item_ar_labels.append(-100)
            cate_ar_labels.append(-100)

    if len(historical_item_ids) < 2:
        return None

    if not use_ar_on_rank_samples and click_label in (0,1):
        item_ar_labels = [-100] * len(item_ar_labels)
        cate_ar_labels = [-100] * len(cate_ar_labels)
    elif modelling_style == "mlm":
        masked_historical_item_ids = [0 if random.random() < 0.3 else x for x in historical_item_ids]
        masked_historical_cate_ids = [0 if random.random() < 0.3 else x for x in historical_cate_ids]
        item_ar_labels = [x if mx==0 and x>0 else -100 for x,mx in zip(historical_item_ids, masked_historical_item_ids)]
        cate_ar_labels = [x if mx==0 and x>0 else -100 for x,mx in zip(historical_cate_ids, masked_historical_cate_ids)]
        historical_item_ids = masked_historical_item_ids
        historical_cate_ids = masked_historical_cate_ids

    historical_item_ids, historical_len = pad_or_truncate(historical_item_ids, max_seq_len, 0)
    historical_cate_ids, _ = pad_or_truncate(historical_cate_ids, max_seq_len, 0)
    item_ar_labels, _ = pad_or_truncate(item_ar_labels, max_seq_len, -100)
    cate_ar_labels, _ = pad_or_truncate(cate_ar_labels, max_seq_len, -100)

    return {
        "target_item_id": torch.tensor(target_item_id),
        "target_cate_id": torch.tensor(target_cate_id),
        "click_label": torch.tensor(click_label),
        "historical_item_ids": torch.tensor(historical_item_ids, dtype=torch.int64),
        "historical_cate_ids": torch.tensor(historical_cate_ids, dtype=torch.int64),
        "historical_len": torch.tensor(historical_len),
        "item_ar_labels": torch.tensor(item_ar_labels),
        "cate_ar_labels": torch.tensor(cate_ar_labels)
    }


@gin.configurable
class SeqRecDataset(Dataset):
    def __init__(self, table_path, max_seq_len=100, modelling_style="ar", use_ar_on_rank_samples=False, include_target_for_ar=True, data_cols="user_id,item_id,cate_id,is_clicked,seq_value,seq_len"):
        self.table_path = table_path
        self.data_cols = data_cols
        self.max_seq_len = max_seq_len
        self.include_target_for_ar = include_target_for_ar
        self.use_ar_on_rank_samples = use_ar_on_rank_samples
        self.modelling_style = modelling_style
        self.item_tokenizer = ItemTokenizer()
        self.cate_tokenizer = CateTokenizer()
        self.data = get_dataset(table_path, col_names=self.data_cols)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        features = preprocess_data(x, max_seq_len=self.max_seq_len, item_tokenizer=self.item_tokenizer, cate_tokenizer=self.cate_tokenizer, modelling_style=self.modelling_style, use_ar_on_rank_samples=self.use_ar_on_rank_samples, include_target_for_ar=self.include_target_for_ar)
        return features

    def save_tokenizers(self, ckpt_dir):
        self.item_tokenizer.save(os.path.join(ckpt_dir, "item.tokenizer.pkl"))
        self.cate_tokenizer.save(os.path.join(ckpt_dir, "cate.tokenizer.pkl"))


if __name__ == '__main__':
    import os
    from torch.utils.data import DataLoader
    dataset = SeqRecDataset("./dataset/amazon_food/pretrain_dataset.txt", modelling_style="ar", max_seq_len=200)
    it = iter(dataset)
    for i in range(1):
        print(next(it))
