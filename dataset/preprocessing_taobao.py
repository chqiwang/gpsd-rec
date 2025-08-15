import _pickle as pkl
import pandas as pd
import random
import numpy as np

RAW_DATA_FILE = './taobao/UserBehavior.csv'
item2index = './taobao/item2index.txt'
cate2index = './taobao/cate2index.txt'
pretrain_train_file = './taobao/pretrain_dataset.txt'
ft_train_file = './taobao/ft_train_dataset.txt'
ft_train_file_new = './taobao/ft_train_dataset_new.txt'
ft_eval_file = './taobao/ft_eval_dataset.txt'
ft_eval_file_new = './taobao/ft_eval_dataset_new.txt'
itemid2cateid_file = './taobao/itemid2cateid.txt'

MAX_LEN_ITEM = 200
def to_df(file_name):
    df = pd.read_csv(RAW_DATA_FILE, header=None, names=['uid', 'iid', 'cid', 'btag', 'time'])
    return df

def remap(df):
    padding_n = 1
    item_key = sorted(df['iid'].unique().tolist())
    item_len = len(item_key)
    item_map = dict(zip(item_key, range(padding_n, item_len + padding_n)))

    user_key = sorted(df['uid'].unique().tolist())
    user_len = len(user_key)
    user_map = dict(zip(user_key, range(padding_n, user_len + padding_n)))

    cate_key = sorted(df['cid'].unique().tolist())
    cate_len = len(cate_key)
    cate_map = dict(zip(cate_key, range(padding_n, cate_len + padding_n)))

    btag_key = sorted(df['btag'].unique().tolist())
    btag_len = len(btag_key)
    btag_map = dict(zip(btag_key, range(padding_n, btag_len + padding_n)))

    itemid2cateid = dict(zip(df['iid'], df['cid']))

    print(item_len, user_len, cate_len, btag_len)
    return df, item_len, user_len + item_len + cate_len + btag_len + 1, item_map, cate_map, itemid2cateid  # +1 is for unknown target btag


def gen_user_item_group(df, item_cnt, feature_size):
    user_df = df.sort_values(['uid', 'time']).groupby('uid')
    item_df = df.sort_values(['iid', 'time']).groupby('iid')

    print("group completed")
    return user_df, item_df


def gen_dataset(user_df, item_df, item_cnt, feature_size, item_map):
    train_sample_list = []
    test_sample_list = []
    user_seq_list = []

    item_voc_list = list(item_map.keys())
    # get each user's last touch point time
    print(len(user_df))

    user_last_touch_time = []
    for uid, hist in user_df:
        user_last_touch_time.append(hist['time'].tolist()[-1])
    print("get user last touch time completed")

    user_last_touch_time_sorted = sorted(user_last_touch_time)
    split_time = user_last_touch_time_sorted[int(len(user_last_touch_time_sorted) * 0.7)]

    cnt = 0
    for uid, hist in user_df:
        cnt += 1
        if cnt % 10000 == 0:
            print(cnt)
        item_hist = hist['iid'].tolist()
        cate_hist = hist['cid'].tolist()
        btag_hist = hist['btag'].tolist()
        target_item_time = hist['time'].tolist()[-1]

        test_flag = (target_item_time > split_time)

        if len(item_hist) < 10:
            continue

        def sample_neg_item(item_hist, cate_hist, indice):
            test_target_item = item_hist[indice]
            test_target_item_cate = cate_hist[indice]
            test_target_item_btag = feature_size
            label = 1

            neg_label = 0
            neg_test_target_item = test_target_item
            while neg_test_target_item == item_hist[indice]:
                neg_test_target_idx = random.randint(0, len(item_voc_list) - 1)  # idx
                neg_test_target_item = item_voc_list[neg_test_target_idx]
                neg_test_target_item_cate = item_df.get_group(neg_test_target_item)['cid'].tolist()[0]
                neg_test_target_item_btag = feature_size
            return test_target_item, test_target_item_cate, label, neg_test_target_item, neg_test_target_item_cate, neg_label

        # randomly sampling negative items
        target_item, target_item_cate, label, neg_item, neg_item_cate, neg_label = sample_neg_item(item_hist, cate_hist, -1)

        # the item history part of the sample
        item_part = []
        for i in range(len(item_hist)):
            item_part.append([uid, item_hist[i], cate_hist[i], btag_hist[i]])

        start_idx = max(0, len(item_part) - MAX_LEN_ITEM - 1)
        item_part_pad = item_part[start_idx:len(item_part)]

        cate_list, item_list = [], []
        for i in range(0, len(item_part_pad) - 1):
            item_list.append(item_part_pad[i][1])
            cate_list.append(item_part_pad[i][2])

        if test_flag == 0:
            train_sample_list.append((uid, target_item, target_item_cate, label, item_list, cate_list))
            train_sample_list.append((uid, neg_item, neg_item_cate, neg_label, item_list, cate_list))
        else:
            test_sample_list.append((uid, target_item, target_item_cate, label, item_list, cate_list))
            test_sample_list.append((uid, neg_item, neg_item_cate, neg_label, item_list, cate_list))

        tmp_user_seq_list, tmp_user_cate_list = [], []
        for i in range(0, len(item_part_pad) - 2):
            tmp_user_seq_list.append(item_part_pad[i][1])
            tmp_user_cate_list.append(item_part_pad[i][2])

        user_seq_item = item_part_pad[len(item_part_pad) - 2][1]
        user_seq_item_cate = item_df.get_group(user_seq_item)['cid'].tolist()[0]
        user_seq_list.append((uid, user_seq_item, user_seq_item_cate, tmp_user_seq_list, tmp_user_cate_list))

    random.shuffle(train_sample_list)
    random.shuffle(test_sample_list)
    random.shuffle(user_seq_list)
    print("length", len(train_sample_list), len(test_sample_list))
    return train_sample_list, test_sample_list, user_seq_list


def write_to_local_file(item_map, cate_map, train_sample_list, test_sample_list, user_seq_list, itemid2cateid):
    with open(itemid2cateid_file, 'w') as i2c_f:
        for key, value in itemid2cateid.items():
            i2c_f.write(str(key) + ',' + str(value) + '\n')

    with open(item2index, 'w') as i2i_f:
        for key, value in item_map.items():
            i2i_f.write(str(key) + ',' + str(value) + '\n')

    with open(cate2index, 'w') as c2i_f:
        for key, value in cate_map.items():
            c2i_f.write(str(key) + ',' + str(value) + '\n')

    # data format: "user_id,item_id,cate_id,is_clicked,seq_value,seq_len"
    with open(ft_train_file, 'w') as ft_train_f:
        for item in train_sample_list:
            basic_info = ','.join(map(str, [item[0], item[1], item[2], item[3]]))
            tmp_item_list = []
            for tmp_item, tmp_cate in zip(item[4], item[5]):
                tmp_item_list.append("item__item_id:" + str(tmp_item) + "#" + "item__cate_id:" + str(tmp_cate))
            seq_feature = ';'.join(tmp_item_list)
            seq_len = len(item[4])
            basic_info = basic_info + ',' + seq_feature + ',' + str(seq_len)
            ft_train_f.write(basic_info + '\n')

    with open(ft_eval_file, 'w') as ft_eval_f:
        for item in test_sample_list:
            basic_info = ','.join(map(str, [item[0], item[1], item[2], item[3]]))
            tmp_item_list = []
            for tmp_item, tmp_cate in zip(item[4], item[5]):
                tmp_item_list.append("item__item_id:" + str(tmp_item) + "#" + "item__cate_id:" + str(tmp_cate))
            seq_feature = ';'.join(tmp_item_list)
            seq_len = len(item[4])
            basic_info = basic_info + ',' + seq_feature + ',' + str(seq_len)
            ft_eval_f.write(basic_info + '\n')

    with open(pretrain_train_file, 'w') as pretrain_f:
        for item in user_seq_list:
            user_id, item_id, item_cate = item[0], item[1], item[2]
            basic_info = ','.join(map(str, [user_id, item_id, item_cate, 2]))
            tmp_item_list = []
            for tmp_item, tmp_cate in zip(item[3], item[4]):
                tmp_item_list.append("item__item_id:" + str(tmp_item) + "#" + "item__cate_id:" + str(tmp_cate))
            seq_feature = ';'.join(tmp_item_list)
            seq_len = len(item[3])
            basic_info = basic_info + ',' + seq_feature + ',' + str(seq_len)
            pretrain_f.write(basic_info + '\n')


def main():
    df = to_df(RAW_DATA_FILE)
    df, item_cnt, feature_size, item_map, cate_map, itemid2cateid_map = remap(df)
    print("feature_size", item_cnt, feature_size)

    user_df, item_df = gen_user_item_group(df, item_cnt, feature_size)
    train_sample_list, test_sample_list, user_seq_list = gen_dataset(user_df, item_df, item_cnt, feature_size, item_map)
    print(f"train_sample_list: {len(train_sample_list)}, user_seq_list: {len(user_seq_list)}")

    write_to_local_file(item_map, cate_map, train_sample_list, test_sample_list, user_seq_list, itemid2cateid_map)
    print('Finishing Writing to local file!')

main()