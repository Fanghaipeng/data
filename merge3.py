import json
import pickle
import difflib

with open('../data/submit/bert_poi_model_shuffle_cls_reg_ep10_images_cleandata_2fc_rawimg_transmutilhead.json', 'r') as f:
    data1 = json.load(f)

with open('../data/submit/submit_shuffle_nezha.json', 'r') as f:
    data2 = json.load(f)
with open('./scores.pkl', 'rb') as f:
    scores = pickle.load(f)

data_dir = "../data/TestA_Preporcess_public.json"
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

def string_similar(s1, s2):
    return difflib.SequenceMatcher(None, s1, s2).quick_ratio()

test_data = read_corpus(data_dir)
keys = list(data1.keys())

final = {}
cnt = {x: 0 for x in range(15)}

for key in keys:
    score = scores[key]
    cls_res = data1[key]
    gen_res = data2[key]
    simlarity = string_similar(cls_res, gen_res)
    print('-'*60)
    print(test_data[key]['src'])
    print(score)
    print('cls: ', cls_res)
    print('gen: ', gen_res)
    print(simlarity)

    if len(cls_res) == 0:  # 100%
        final[key] = gen_res
        print('0 fin: ', final[key])
        cnt[0] += 1
        continue

    # cls 中某item出现了2次以上，使用gen  100%
    if len(cls_res) > len(''.join(score.keys())):
        final[key] = gen_res
        print('1 fin: ', final[key])
        cnt[1] += 1
        continue

    if simlarity < 0.1:  # 100%
        final[key] = cls_res
        print('4 fin: ', final[key])
        cnt[4] += 1
        continue
    if len(cls_res) <= 3:  # 100%
        final[key] = gen_res
        print('7 fin: ', final[key])
        cnt[7] += 1
        continue
    # cls == gen
    if cls_res == gen_res:
        final[key] = cls_res
        print('12 fin: ', final[key])
        cnt[12] += 1
        continue
    # cls 中以 gen结尾，则选择cls
    if cls_res.endswith(gen_res):
        final[key] = cls_res
        print('2 fin: ', final[key])
        cnt[2] += 1
        continue
    # cls 中以 gen 为开头，则选择gen
    if cls_res.startswith(gen_res):
        final[key] = gen_res
        print('9 fin: ', final[key])
        cnt[9] += 1
        continue
    # gen 中以 cls 为开头，则选择gen
    if gen_res.startswith(cls_res):
        final[key] = gen_res
        print('10 fin: ', final[key])
        cnt[10] += 1
        continue
    # gen 中以 cls 为结尾，则选择gen
    if gen_res.endswith(cls_res):
        final[key] = gen_res
        print('11 fin: ', final[key])
        cnt[11] += 1
        continue
    # cls 中有score非常低的，则选择gen
    if min(score.values()) < 0.6:
        final[key] = gen_res
        print('3 fin: ', final[key])
        cnt[3] += 1
        continue
    # 得到二者的 sim
    # sim = 0: cls
    # sim > 0.7: gen
    if simlarity > 0.8:
        final[key] = cls_res
        print('5 fin: ', final[key])
        cnt[5] += 1
        continue
    # len(cls) > len(gen) + 2: cls
    if len(cls_res) > len(gen_res) + 2:
        final[key] = cls_res
        print('6 fin: ', final[key])
        cnt[6] += 1
        continue
    final[key] = gen_res
    print('8 fin: ', final[key])
    cnt[8] += 1
    continue

print(cnt)
with open('../data/submit/merge3.json', 'w') as f:
    json.dump(final, f)