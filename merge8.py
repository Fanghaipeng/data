# 733 glt

import json
import pickle
import difflib

with open('/data/fanghaipeng/mydata/nezha_cls_reg_ep10_sortdata.json', 'r') as f:
    data0 = json.load(f)
with open('/data/fanghaipeng/mydata/nezha_cls_reg_ep10_sortdata_filter_th01.json', 'r') as f:
    data1 = json.load(f)
with open('/data/fanghaipeng/mydata/submit_sort_nezha_wo.json', 'r') as f:
    data2 = json.load(f)
with open('/data/fanghaipeng/mydata/submit_clsreg_0point1_nezha_wo.json', 'r') as f:
    data3 = json.load(f)
with open('/data/fanghaipeng/mydata/submit_clsreg_0point2_nezha_wo.json', 'r') as f:
    data4 = json.load(f)
with open('/data/fanghaipeng/mydata/scores_regen_sort.pkl', 'rb') as f:
    scores = pickle.load(f)

data_dir = "/data/fanghaipeng/POIdata/TestA_Preporcess_public.json"
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
count = 0
for key in keys:
    count += 1
    if(count > 50) :
        break
    score = scores[key]
    cls_res = data0[key]
    cls_res_th01 = data1[key]
    gen_res = data2[key]
    regen_res = data3[key]
    regen_res_th02 = data4[key]
    simlarity = string_similar(cls_res, gen_res)
    print('-'*60)
    print(test_data[key]['src'])
    print(score)
    print('cls: ', cls_res)
    print('gen: ', gen_res)
    print('reg: ', regen_res)
    print(simlarity)
    score = {x: v for x, v in score}
    if len(cls_res) == 0:  # 100%
        final[key] = gen_res
        print('0 fin: ', final[key])
        cnt[0] += 1
        continue

    # cls 中某item出现了2次以上，使用gen  100%
    if len(cls_res) > len(''.join(score.keys())):
        final[key] = regen_res
        print('1 fin: ', final[key])
        cnt[1] += 1
        continue

    if simlarity < 0.1:  # 100%
        final[key] = cls_res_th01
        print('4 fin: ', final[key])
        cnt[4] += 1
        continue
    if len(cls_res) <= 3:  # 100%
        if len(gen_res) <= 3:
            final[key] = regen_res
        else:
            final[key] = gen_res
        print('7 fin: ', final[key])
        cnt[7] += 1
        continue
    # cls == gen
    if cls_res_th01 == regen_res:
        final[key] = cls_res_th01
        print('12 fin: ', final[key])
        cnt[12] += 1
        continue
    # cls 中以 gen结尾，则选择cls
    if cls_res_th01.endswith(regen_res):
        final[key] = cls_res_th01
        print('2 fin: ', final[key])
        cnt[2] += 1
        continue
    if cls_res_th01.endswith(gen_res):
        final[key] = cls_res_th01
        print('2 fin: ', final[key])
        cnt[2] += 1
        continue
    if cls_res.endswith(regen_res):
        final[key] = cls_res
        print('2 fin: ', final[key])
        cnt[2] += 1
        continue
    if cls_res.endswith(gen_res):
        final[key] = cls_res
        print('2 fin: ', final[key])
        cnt[2] += 1
        continue
    # cls 中以 gen 为开头，则选择gen
    if cls_res_th01.startswith(regen_res) or cls_res.startswith(regen_res):
        final[key] = regen_res
        print('9 fin: ', final[key])
        cnt[9] += 1
        continue
    if cls_res_th01.startswith(gen_res) or cls_res_th01.startswith(gen_res):
        final[key] = gen_res
        print('9 fin: ', final[key])
        cnt[9] += 1
        continue
    # gen 中以 cls 为开头，则选择gen
    if regen_res.startswith(cls_res_th01) or regen_res.startswith(cls_res):
        final[key] = regen_res
        print('10 fin: ', final[key])
        cnt[10] += 1
        continue
    if gen_res.startswith(cls_res_th01) or gen_res.startswith(cls_res):
        final[key] = gen_res
        print('10 fin: ', final[key])
        cnt[10] += 1
        continue
    # gen 中以 cls 为结尾，则选择gen
    if regen_res.endswith(cls_res_th01) or regen_res.endswith(cls_res):
        final[key] = regen_res
        print('11 fin: ', final[key])
        cnt[11] += 1
        continue
    if gen_res.endswith(cls_res_th01) or gen_res.endswith(cls_res):
        final[key] = gen_res
        print('11 fin: ', final[key])
        cnt[11] += 1
        continue
    # cls 中有score非常低的，则选择gen
    if min(score.values()) < 0.6:
        final[key] = regen_res
        print('3 fin: ', final[key])
        cnt[3] += 1
        continue
    if simlarity > 0.8:
        final[key] = cls_res_th01
        print('5 fin: ', final[key])
        cnt[5] += 1
        continue
    # len(cls) > len(gen) + 2: cls
    if len(cls_res) > len(gen_res) + 2:
        final[key] = cls_res_th01
        print('6 fin: ', final[key])
        cnt[6] += 1
        continue
    final[key] = regen_res
    print('8 fin: ', final[key])
    cnt[8] += 1
    continue

print(cnt)
with open('/data/fanghaipeng/mydata/merge8.json', 'w') as f:
    json.dump(final, f)