import pandas as pd
import random

log_info_file = './dataset/amazon_food/Grocery_and_Gourmet_Food_5.json'
meta_file = './dataset/amazon_food/meta_Grocery_and_Gourmet_Food.json'
log_df_pickle = './dataset/amazon_food/reviews.pkl'
meta_df_pickle = './dataset/amazon_food/meta.pkl'

item2index = './dataset/amazon_food/item2index.txt'
cate2index = './dataset/amazon_food/cate2index.txt'
pretrain_train_file = './dataset/amazon_food/pretrain_dataset.txt'
ft_train_file = './dataset/amazon_food/ft_train_dataset.txt'
ft_eval_file = './dataset/amazon_food/ft_eval_dataset.txt'

MAX_LEN_ITEM = 200

def to_df(file_path):
    with open(file_path, 'r') as fin:
        df = {}
        i = 0
        for line in fin:
            line = line.replace("true", "True")
            line = line.replace("false", "False")
            df[i] = eval(line)
            i += 1
        df = pd.DataFrame.from_dict(df, orient='index')
    return df

def remap(log_df, meta_df):
    def build_map(df, col_name):
        padding_n = 1
        key = sorted(df[col_name].unique().tolist())
        m = dict(zip(key, range(padding_n, padding_n + len(key))))
        return m, key

    asin_map, asin_key = build_map(meta_df, 'asin')
    cate_map, cate_key = build_map(meta_df, 'categories')
    revi_map, revi_key = build_map(log_df, 'reviewerID')

    meta_df = meta_df.sort_values('asin')
    meta_df = meta_df.reset_index(drop=True)
    meta_df = meta_df[['asin', 'categories']]
    
    log_df = log_df.sort_values(['reviewerID', 'unixReviewTime'])
    log_df = log_df.reset_index(drop=True)
    log_df = log_df[['reviewerID', 'asin', 'unixReviewTime']]

    user_count, item_count, cate_count, example_count =\
        len(revi_map), len(asin_map), len(cate_map), log_df.shape[0]

    df = log_df.merge(meta_df, how='left', left_on='asin', right_on='asin')

    # rename
    df = df.rename(columns={'reviewerID': 'uid', 'asin': 'iid', 'unixReviewTime': 'time', 'categories': 'cid'})

    print("user_count: {}, item_count: {}, cate_count: {}".format(user_count, item_count, cate_count))
    return df, asin_map, cate_map

def gen_user_item_group(df):
    user_df = df.sort_values(['uid', 'time']).groupby('uid')
    item_df = df.sort_values(['iid', 'time']).groupby('iid')

    return user_df, item_df

def gen_dataset(user_df, item_df, item_map):
    train_sample_list = []
    test_sample_list = []
    user_seq_list = []

    item_voc_list = list(item_map.keys())

    user_last_touch_time = []
    for uid, hist in user_df:
        user_last_touch_time.append(hist['time'].tolist()[-1])

    user_last_touch_time_sorted = sorted(user_last_touch_time) 
    split_time = user_last_touch_time_sorted[int(len(user_last_touch_time_sorted) * 0.7)] # split by user last touch time

    invalid_cnt = 0
    cnt = 0
    for uid, hist in user_df:
        cnt += 1
        
        item_hist = hist['iid'].tolist()
        cate_hist = hist['cid'].tolist()
        target_item_time = hist['time'].tolist()[-1]

        test_flag = (target_item_time > split_time) # as test

        # Exclude short sequences
        if len(item_hist) < 4:
            invalid_cnt += 1
            continue

        def sample_neg_item(item_hist, cate_hist, indice):
            test_target_item = item_hist[indice]
            test_target_item_cate = cate_hist[indice]
            label = 1

            neg_label = 0
            neg_test_target_item = test_target_item
            while neg_test_target_item == item_hist[indice]:
                neg_test_target_idx = random.randint(0, len(item_voc_list)-1) # idx
                neg_test_target_item = item_voc_list[neg_test_target_idx]
                neg_test_target_item_cate = item_df.get_group(neg_test_target_item)['cid'].tolist()[0]
            return test_target_item, test_target_item_cate, label, neg_test_target_item, neg_test_target_item_cate, neg_label

        # Sample a negative item
        target_item, target_item_cate, label, neg_item, neg_item_cate, neg_label = sample_neg_item(item_hist, cate_hist, -1)

        # the item history part of the sample
        item_seq = []
        for i in range(len(item_hist)):
            item_seq.append([uid, item_hist[i], cate_hist[i]])
        
        # Truncate the sequence
        start_idx = max(0, len(item_seq)-MAX_LEN_ITEM-1)
        item_seq_truc = item_seq[start_idx:len(item_seq)]
        
        # Build datasets for discriminative training
        cate_list, item_list = [], []
        for i in range(0, len(item_seq_truc)-1):
            item_list.append(item_seq_truc[i][1])
            cate_list.append(item_seq_truc[i][2])
        
        if test_flag == 0:
            train_sample_list.append((uid, target_item, target_item_cate, label, item_list, cate_list))
            train_sample_list.append((uid, neg_item, neg_item_cate, neg_label, item_list, cate_list))
        else:
            test_sample_list.append((uid, target_item, target_item_cate, label, item_list, cate_list))
            test_sample_list.append((uid, neg_item, neg_item_cate, neg_label, item_list, cate_list))

        # Build datasets for pretraining
        tmp_user_seq_list, tmp_user_cate_list = [], []
        for i in range(0,len(item_seq_truc)-2):
            tmp_user_seq_list.append(item_seq_truc[i][1])
            tmp_user_cate_list.append(item_seq_truc[i][2])
        
        target_item = item_seq_truc[len(item_seq_truc)-2][1]
        target_item_cate = item_df.get_group(target_item)['cid'].tolist()[0]
        user_seq_list.append((uid, target_item, target_item_cate, tmp_user_seq_list, tmp_user_cate_list))
    
    random.shuffle(train_sample_list)
    random.shuffle(test_sample_list)
    random.shuffle(user_seq_list)
    print("train size:", len(train_sample_list))
    print("test size:", len(test_sample_list))
    print("invalid user count: ", invalid_cnt)
    return train_sample_list, test_sample_list, user_seq_list

def write_to_local_file(item_map, cate_map, train_sample_list, test_sample_list, user_seq_list):
    with open(item2index, 'w') as i2i_f:
        for key,value in item_map.items():
            i2i_f.write(str(key) + ',' + str(value) + '\n')

    with open(cate2index, 'w') as c2i_f:
        for key,value in cate_map.items():
            c2i_f.write(str(key) + ',' + str(value) + '\n')

    with open(ft_train_file , 'w') as ft_train_f:
        for item in train_sample_list:
            basic_info = ','.join(map(str, [item[0], item[1], item[2], item[3]]))
            tmp_item_list = []
            # Construct the sequence
            for tmp_item, tmp_cate in zip(item[4], item[5]):
                tmp_item_list.append("item__item_id:"+str(tmp_item)+"#"+"item__cate_id:"+str(tmp_cate))
            seq_feature = ';'.join(tmp_item_list)
            seq_len = len(item[4])
            basic_info = basic_info + ',' + seq_feature + ',' + str(seq_len)
            ft_train_f.write(basic_info + '\n')
    
    with open(ft_eval_file , 'w') as ft_eval_f:
        for item in test_sample_list:
            basic_info = ','.join(map(str, [item[0], item[1], item[2], item[3]]))
            tmp_item_list = []
            # Construct the sequence
            for tmp_item, tmp_cate in zip(item[4], item[5]):
                tmp_item_list.append("item__item_id:"+str(tmp_item)+"#"+"item__cate_id:"+str(tmp_cate))
            seq_feature = ';'.join(tmp_item_list)
            seq_len = len(item[4])
            basic_info = basic_info + ',' + seq_feature + ',' + str(seq_len)
            ft_eval_f.write(basic_info + '\n')
    
    with open(pretrain_train_file , 'w') as pretrain_f:
        for item in user_seq_list:
            basic_info = ','.join(map(str, [item[0], item[1], item[2], '2']))
            tmp_item_list = []
            # Construct the sequence
            for tmp_item, tmp_cate in zip(item[3], item[4]):
                tmp_item_list.append("item__item_id:"+str(tmp_item)+"#"+"item__cate_id:"+str(tmp_cate))
            seq_feature = ';'.join(tmp_item_list)
            seq_len = len(item[3])
            basic_info = basic_info + ',' + seq_feature + ',' + str(seq_len)
            pretrain_f.write(basic_info + '\n')

def main():
    reviews_df = to_df(log_info_file)
    meta_df = to_df(meta_file)
    meta_df = meta_df[meta_df['asin'].isin(reviews_df['asin'].unique())]
    meta_df = meta_df.reset_index(drop=True)

    reviews_df = reviews_df[['reviewerID', 'asin', 'unixReviewTime']]
    meta_df = meta_df[['asin', 'category']]
    meta_df.rename(columns={'category': 'categories'}, inplace=True)
    
    meta_df['categories'] = meta_df['categories'].apply(lambda x: '&&'.join(x))
    meta_df['categories'] = meta_df['categories'].str.replace('[ ,:#]', '', regex=True)

    df, item_map, cate_map = remap(reviews_df, meta_df)

    user_df, item_df = gen_user_item_group(df)
    train_sample_list, test_sample_list, user_seq_list = gen_dataset(user_df, item_df, item_map)

    write_to_local_file(item_map, cate_map, train_sample_list, test_sample_list, user_seq_list)
    print('Finish processing data!')

main()