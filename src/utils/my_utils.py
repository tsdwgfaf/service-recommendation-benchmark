import json
import os
import string
from typing import List
import torch
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from contractions import contractions_dict
import numpy as np
from torch import nn

average_word2vec = [9.23808376e-02, -8.33334521e-02, -4.74280455e-05, 1.36091554e-01, -1.11752324e-02, -8.99234279e-03,
                    8.04370020e-02, -1.01534897e-01, -4.50805886e-02, 6.10845049e-01, -1.13657723e-01, 3.47111115e-03,
                    1.00918116e-01, -1.08997056e-01, -8.33621646e-02, -1.21096042e-01, 8.74309468e-02, -2.68108681e-02,
                    2.45264840e-02, 5.65145424e-02, 4.17319617e-02, -6.88751415e-02, -2.08641354e-01, -1.06936927e-01,
                    1.64321090e-01, -1.77382827e-02, -1.67868030e-02, 2.83146807e-02, 7.04033445e-02, -5.70690312e-02,
                    -2.60383594e-02, -1.84562827e-01, 9.58827049e-02, -1.21240050e-01, 4.57531164e-01, -3.04205911e-02,
                    -7.29277557e-02, -1.26595399e-02, 6.19921405e-02, 3.61088305e-02, 4.24100106e-02, 8.48461234e-02,
                    4.51797330e-02, -1.89532025e-01, -2.90695551e-02, -2.75481958e-02, -6.76743539e-02, -1.16799021e-01,
                    8.04972147e-02, -7.29645087e-02, -1.84058839e-02, 8.43595530e-02, 1.34550491e-02, -2.69625743e-02,
                    2.98314064e-02, -1.27335157e-01, 5.58416258e-02, -1.25067229e-01, -9.21245292e-02, 1.24383514e-01,
                    -9.40693807e-02, -5.71038633e-02, -2.03708314e-01, 4.97911051e-02, 6.30723289e-02, 2.55479761e-01,
                    -8.01348858e-02, -5.61313659e-02, -1.47185127e-02, 5.67639291e-02, -3.55590870e-02, -1.98987710e-02,
                    -1.78036819e-02, -5.30447878e-02, 1.07484707e-01, -2.28992263e-02, -1.24939310e-01, -1.05359128e-01,
                    1.01757551e-01, 2.64407862e-02, 2.89461950e-02, 6.77808611e-02, -4.23378316e-03, -1.17672184e-01,
                    -7.51891585e-02, -1.97259998e-03, 5.61320113e-02, -1.24114693e-01, 6.38650966e-03, -2.51534427e-02,
                    7.35455381e-02, -1.23097367e-01, 1.72607668e-01, 6.08207385e-02, -1.92689996e-02, -1.88154097e-02,
                    1.81382794e-01, -2.18553349e-02, -3.61989762e-02, 2.79732726e-01, 1.74813260e-02, -1.97856217e-02,
                    -1.82405363e-02, 5.25282401e-02, 3.15587194e-02, 9.57461932e-03, -1.37746916e-01, -6.69914745e-02,
                    8.45852682e-02, -7.56560517e-02, 9.94373185e-03, 1.94307289e-01, 5.62558295e-02, 8.98335660e-02,
                    -1.06941628e-01, -1.08840842e-01, 9.01051125e-02, -8.61959480e-02, -1.36264104e-02, 2.46705223e-01,
                    3.48930472e-02, 3.60505095e-02, -2.84199812e-02, 2.15288677e-02, 4.54887880e-02, 7.82933490e-03,
                    -2.65201629e-03, -7.96693493e-02, -5.14674314e-02, -2.86113586e-03, 5.68288941e-03, -1.54271216e-01,
                    -4.83912687e-02, 4.67413624e-02, -3.46997881e-03, 2.80795185e-02, -5.93713245e-03, 2.77160089e-02,
                    -1.06073800e-03, -1.01578374e-01, 4.95304726e-02, -3.87987236e-02, -7.66712199e-02, -1.13672438e-01,
                    2.71455713e-01, -8.96379938e-03, -8.34819967e-02, 5.23370872e-03, 1.34065536e-01, -3.42112472e-03,
                    -3.61707576e-01, 8.81232374e-03, -6.46509766e-02, 1.38155682e-02, -2.79683636e-01, -5.25893849e-02,
                    1.26041254e-01, 1.87358619e-02, -6.54088733e-02, 5.14817640e-02, -1.57615607e-01, 8.11078842e-02,
                    -4.72711351e-02, 3.03989619e-02, 8.42918687e-02, -7.92461592e-02, 1.39181823e-02, -4.13944561e-02,
                    2.43480194e-02, -1.75854917e-01, -1.10805305e-01, -1.59476957e-02, 3.63058938e-01, -7.12143829e-02,
                    -3.16085887e-03, -5.10861570e-02, 2.66880442e-02, -2.45827383e-02, 5.53645577e-02, -2.75470106e-01,
                    -6.10489127e-02, -1.19412291e-01, -7.98876673e-02, -3.06054558e-02, 3.02861795e-02, 1.12967743e-01,
                    2.34407745e-01, -6.50806012e-03, -2.78638930e-02, -8.21134224e-02, 1.53513722e-02, -1.41713336e-01,
                    9.36357689e-03, 3.40281020e-02, 3.28226698e-02, 4.33809903e-02, -1.13317070e-01, 1.07237702e-01,
                    -2.67703249e-02, -6.65319810e-03, -5.04604353e-01, 3.49752076e-02, -1.38339440e-02, 3.14966643e-02,
                    -3.36699651e-02, -4.09862247e-03, 1.06694945e-01, -8.18598150e-02, 2.03122608e-03, 9.92763297e-02,
                    4.53290983e-03, 9.06645498e-02, -6.53008609e-02, 1.25525055e-02, -1.24131227e-01, -6.20327046e-02,
                    -3.20642780e-02, -9.58516589e-02, 2.00287049e-02, -3.53399458e-04, -2.33409949e-01, 9.28285557e-02,
                    8.48394470e-02, -2.44608388e-02, 1.14548231e-02, -2.18968534e-02, -7.14388497e-02, 3.33529291e-02,
                    3.87423033e-02, 4.57710564e-02, -1.25987611e-02, -7.83917755e-03, -1.12510742e-02, 5.84910895e-02,
                    -1.40727402e-01, -5.62050277e-02, -5.34312455e-02, -4.50937342e-02, 6.88280380e-02, -6.75163652e-02,
                    5.90525547e-02, -3.55978975e-02, -2.76973308e-01, -2.36740571e-02, 3.50863686e-01, 1.67888118e-02,
                    -1.27298348e-01, 1.04443644e-01, 2.80098412e-02, 6.60937237e-04, -6.23325201e-02, 4.02930992e-02,
                    4.96635274e-02, 3.40309479e-02, -3.12685522e-01, 9.29385075e-02, 3.56945878e-02, 1.51344436e-01,
                    -7.24692754e-02, -7.90923158e-02, 3.09726127e-02, 1.14359339e-01, -1.02786961e-01, -7.39974855e-02,
                    5.11928221e-02, 1.73081626e-02, 7.33908062e-02, -2.21396721e-02, -2.36993343e-02, -4.35658508e-02,
                    3.62323864e-02, 4.76493653e-02, -9.43683567e-02, -3.68518950e-02, -1.80413252e-02, -7.73990237e-02,
                    6.67913742e-01, -1.68882165e-02, -2.71679201e-01, -6.27591949e-02, 1.45924292e-01, 5.73932119e-02,
                    -2.15114111e-02, 3.01602511e-02, 1.08308361e-02, -1.88530631e-01, 6.71421331e-02, -1.21531496e-01,
                    -3.74377282e-02, 1.40142804e-02, 4.98320855e-02, 1.68101625e-01, 9.40173773e-02, -3.89945136e-02,
                    -1.09531039e-01, -2.94610210e-01, 2.17316869e-03, 2.18728182e-01, 1.81845193e-01, -6.70226964e-02]


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def text_processes(text: str) -> List[str]:
    tokens = word_tokenize(text)
    # filter punctuation
    tokens = [token for token in tokens if token not in string.punctuation]
    # filter stop words
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    # replace abbreviation
    tokens = [contractions_dict[token] if token in contractions_dict.keys() else token for token in tokens]
    # lemmatization
    wnl = WordNetLemmatizer()
    tags = pos_tag(tokens)
    res = []
    for tag in tags:
        wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
        res.append(wnl.lemmatize(tag[0], pos=wordnet_pos))
    return res


def parse_glove_6b_300d():
    """
    处理词向量模型，保存为单词+向量两个文件
    :return:
    """
    if os.path.exists('data/glove/glove.6B.300d.word_list.npy'):
        return
    vec_dict = {}
    with open('data/glove/raw/glove.6B.300d.txt', 'r', encoding='utf-8') as f:
        for line in f:
            val = line.split(' ')
            word = val[0]
            vec = np.asarray(val[1:], 'float32')
            vec_dict[word] = vec
    np.save('data/glove/glove.6B.300d.word_list.npy', np.array(list(vec_dict.keys())))
    np.save('data/glove/glove.6B.300d.vector_list.npy', np.array(list(vec_dict.values())), dtype='float32')


def save_user_item_interaction_matrix():
    with open('../../data/api_mashup/raw/active_apis_data.txt', 'r', encoding='utf-8') as f:
        api_dict_list = json.load(f)
    api_list = [api_dict['title'] for api_dict in api_dict_list]
    with open('../../data/api_mashup/raw/active_mashups_data.txt', 'r', encoding='utf-8') as f:
        mashup_dict_list = json.load(f)
    user_vec_list = []
    for mashup_dict in mashup_dict_list:
        index = torch.LongTensor([[api_list.index(api['title']) for api in mashup_dict['related_apis']]])
        user_vec = torch.zeros(1, len(api_list)).scatter(1, index, 1)
        user_vec_list.append(user_vec)
    user_item_inter_matrix = torch.cat(user_vec_list, dim=0)
    np.save('../../data/api_mashup/user_item_interaction_matrix.npy', user_item_inter_matrix.numpy())


def calculate_average_word2vec():
    total_vec = np.zeros((300,))
    num_word = 0
    with open('../../data/glove/raw/glove.6B.300d.txt', 'r', encoding='utf-8') as f:
        for line in f:
            val = line.split(' ')
            vec = np.asarray(val[1:], 'float32')
            total_vec += vec
            num_word += 1
    return total_vec / num_word


def save_mashup_title():
    with open('../../data/api_mashup/raw/active_mashups_data.txt', 'r', encoding='utf-8') as f:
        mashup_list = json.load(f)
    title_list = [mashup['title'] for mashup in mashup_list]
    with open('../../data/api_mashup/mashup_title.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(title_list, ensure_ascii=False))


def save_mashup_description():
    with open('../../data/api_mashup/raw/active_mashups_data.txt', 'r', encoding='utf-8') as f:
        mashup_list = json.load(f)
    description_list = [text_processes(mashup['description']) for mashup in mashup_list]
    with open('../../data/api_mashup/mashup_description.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(description_list, ensure_ascii=False))


def save_api_description():
    with open('../../data/api_mashup/raw/active_apis_data.txt', 'r', encoding='utf-8') as f:
        api_list = json.load(f)
    description_list = [text_processes(api['description']) for api in api_list]
    with open('../../data/api_mashup/api_description.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(description_list, ensure_ascii=False))


def save_mashup_related_api():
    with open('../../data/api_mashup/raw/active_mashups_data.txt', 'r', encoding='utf-8') as f:
        mashup_list = json.load(f)
    related_apis_list = [[api['title'] for api in mashup['related_apis']] for mashup in mashup_list]
    with open('../../data/api_mashup/mashup_related_api.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(related_apis_list, ensure_ascii=False))


def save_mashup_description_word2vec():
    word2vec_dict = {}
    with open('../../data/glove/raw/glove.6B.300d.txt', 'r', encoding='utf-8') as f:
        for line in f:
            val = line.strip().split(' ')
            word = val[0]
            vec = val[1:]
            word2vec_dict[word] = list(map(float, vec))
    with open('../../data/api_mashup/mashup_description.json', 'r', encoding='utf-8') as f:
        des_list = json.load(f)
    des_word2vec_list = []
    for word_list in des_list:
        vec_list = [word2vec_dict[each_word] if each_word in word2vec_dict.keys() else [0.0 for i in range(300)] for
                    each_word in
                    word_list]
        des_word2vec_list.append(vec_list)
    # 补全不等长的 description 序列
    max_len = max([len(vec_list) for vec_list in des_word2vec_list])
    for vec_list in des_word2vec_list:
        for i in range(max_len - len(vec_list)):
            vec_list.append([0.0 for n in range(300)])
    with open('../../data/api_mashup/mashup_description_word2vec.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(des_word2vec_list, ensure_ascii=False))


def save_api_description_word2vec():
    word2vec_dict = {}
    with open('../../data/glove/raw/glove.6B.300d.txt', 'r', encoding='utf-8') as f:
        for line in f:
            val = line.strip().split(' ')
            word = val[0]
            vec = val[1:]
            word2vec_dict[word] = list(map(float, vec))
    with open('../../data/api_mashup/api_description.json', 'r', encoding='utf-8') as f:
        des_list = json.load(f)
    des_word2vec_list = []
    for word_list in des_list:
        vec_list = [word2vec_dict[each_word] if each_word in word2vec_dict.keys() else [0.0 for i in range(300)] for
                    each_word in
                    word_list]
        des_word2vec_list.append(vec_list)
    # 补全不等长的 description 序列
    max_len = max([len(vec_list) for vec_list in des_word2vec_list])
    for vec_list in des_word2vec_list:
        for i in range(max_len - len(vec_list)):
            vec_list.append([0.0 for n in range(300)])
    with open('../../data/api_mashup/api_description_word2vec.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(des_word2vec_list, ensure_ascii=False))


def collate_fn(batch_list):
    """
    补全不等长的 description 序列
    :param batch_list: __get_item__返回的元组 (x, label) 组成的batch列表
    :return:
    """
    padding_batch_list = []
    max_len = max([len(item) for item in batch_list])
    for item in batch_list:
        padding_len = max_len - item.shape[0]
        pad = nn.ZeroPad2d(padding=(0, 0, 0, padding_len))
        padding_batch_list.append(pad)
    return torch.tensor(padding_batch_list, dtype=torch.float32)


def save_api_title():
    with open('../../data/api_mashup/raw/active_apis_data.txt', 'r', encoding='utf-8') as f:
        api_list = json.load(f)
    title_list = [api['title'] for api in api_list]
    with open('../../data/api_mashup/api_title.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(title_list, ensure_ascii=False))


def delete_unused_api():
    with open('../../data/api_mashup/raw/active_apis_data.txt', 'r', encoding='utf-8') as f:
        api_list = json.load(f)
    with open('../../data/api_mashup/mashup_related_api.json', 'r', encoding='utf-8') as f:
        related_api_list = json.load(f)
    related_api_set = set()
    for related_apis in related_api_list:
        for related_api in related_apis:
            related_api_set.add(related_api)
    used_api = []
    for api in api_list:
        if api['title'] in related_api_set:
            used_api.append(api)
    with open('../../data/api_mashup/raw/active_apis_data.txt', 'w', encoding='utf-8') as f:
        f.write(json.dumps(used_api))


if __name__ == '__main__':
    with open('../../data/api_mashup/raw/active_apis_data.txt', 'r', encoding='utf-8') as f:
        l = json.load(f)
    print(len(l))

