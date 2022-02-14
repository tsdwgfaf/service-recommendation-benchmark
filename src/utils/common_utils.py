import json
import os
import string
import sys
from typing import List

import gensim
import torch
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from contractions import contractions_dict
import numpy as np
from torch import nn
from gensim import similarities


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
    """
    process the raw text
    """
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
    for t in tags:
        wordnet_pos = get_wordnet_pos(t[1]) or wordnet.NOUN
        res.append(wnl.lemmatize(t[0], pos=wordnet_pos))
    return res


def create_user_item_interaction_matrix(
    mashup_path: str,
    api_path: str,
    is_save: bool,
    save_dir: str = '../../data/api_mashup',
):
    with open(api_path, 'r', encoding='utf-8') as f:
        api_dict_list = json.load(f)
    api_list = [api_dict['title'] for api_dict in api_dict_list]
    with open(mashup_path, 'r', encoding='utf-8') as f:
        mashup_dict_list = json.load(f)
    user_vec_list = []
    for mashup_dict in mashup_dict_list:
        index = torch.tensor([[api_list.index(api['title']) for api in mashup_dict['related_apis']]], dtype=torch.int64)
        user_vec = torch.zeros(1, len(api_list)).scatter(1, index, 1)
        user_vec_list.append(user_vec)
    user_item_inter_matrix = torch.cat(user_vec_list, dim=0)
    if is_save:
        save_path = os.path.join(save_dir, 'user_item_interaction_matrix.pt')
        torch.save(user_item_inter_matrix.int(), save_path)
    return user_item_inter_matrix


def create_mashup_title(
    mashup_path: str,
    is_save: bool,
    save_dir: str = '../../data/api_mashup',
):
    with open(mashup_path, 'r', encoding='utf-8') as f:
        mashups = json.load(f)
    titles = [mashup['title'] for mashup in mashups]
    if is_save:
        save_path = os.path.join(save_dir, 'mashup_title.json')
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(titles, ensure_ascii=False))
    return titles


def create_mashup_category(
    mashup_path: str,
    is_save: bool,
    save_dir: str = '../../data/api_mashup'
):
    with open(mashup_path, 'r', encoding='utf-8') as f:
        mashups = json.load(f)
    categories = [mashup['categories'] for mashup in mashups]
    if is_save:
        save_path = os.path.join(save_dir, 'mashup_category.json')
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(categories, ensure_ascii=False))
    return categories


def create_mashup_tag(
    mashup_path: str,
    is_save: bool,
    save_dir: str = '../../data/api_mashup'
):
    with open(mashup_path, 'r', encoding='utf-8') as f:
        mashups = json.load(f)
    tags = [mashup['tags'] for mashup in mashups]
    if is_save:
        save_path = os.path.join(save_dir, 'mashup_tag.json')
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(tags, ensure_ascii=False))
    return tags


def create_mashup_description(
    mashup_path: str,
    is_save: bool,
    save_dir: str = '../../data/api_mashup'
):
    with open(mashup_path, 'r', encoding='utf-8') as f:
        mashups = json.load(f)
    descriptions = [text_processes(mashup['description']) for mashup in mashups]
    if is_save:
        save_path = os.path.join(save_dir, 'mashup_description.json')
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(descriptions, ensure_ascii=False))
    return descriptions


def create_mashup_related_api(
    mashup_path: str,
    is_save: bool,
    save_dir: str = '../../data/api_mashup'
):
    with open(mashup_path, 'r', encoding='utf-8') as f:
        mashups = json.load(f)
    related_apis = [[api['title'] for api in mashup['related_apis']] for mashup in mashups]
    if is_save:
        save_path = os.path.join(save_dir, 'mashup_related_api.json')
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(related_apis, ensure_ascii=False))
    return related_apis


# def create_mashup_mashup_vec(
#     glove_path: str,
#     mashup_des_path: str,
#     is_save: bool,
#     save_dir: str = '../../data/api_mashup'
#  ):
#     word2vec = {}
#     with open(glove_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             val = line.strip().split(' ')
#             word = val[0]
#             vec = val[1:]
#             word2vec[word] = list(map(float, vec))
#     with open(mashup_des_path, 'r', encoding='utf-8') as f:
#         descriptions = json.load(f)
#     mashup_vectors = []
#     for words in descriptions:
#         vectors = [word2vec[w] if w in word2vec.keys() else [0.0 for i in range(300)] for w in words]
#         mashup_vectors.append(vectors)
#     # 补全不等长的 description 序列
#     max_len = max([len(vec_list) for vec_list in mashup_vectors])
#     for vectors in mashup_vectors:
#         for i in range(max_len - len(vectors)):
#             vectors.append([0.0 for n in range(300)])
#
#     with open('../../data/api_mashup/mashup_description_word2vec.json', 'w', encoding='utf-8') as f:
#         f.write(json.dumps(mashup_vectors, ensure_ascii=False))


def create_api_title(
    api_path: str,
    is_save: bool,
    save_dir: str = '../../data/api_mashup'
):
    with open(api_path, 'r', encoding='utf-8') as f:
        apis = json.load(f)
    titles = [api['title'] for api in apis]
    if is_save:
        save_path = os.path.join(save_dir, 'api_title.json')
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(titles, ensure_ascii=False))
    return titles


def create_api_description(
    api_path: str,
    is_save: bool,
    save_dir: str = '../../data/api_mashup'
):
    with open(api_path, 'r', encoding='utf-8') as f:
        apis = json.load(f)
    descriptions = [text_processes(mashup['description']) for mashup in apis]
    if is_save:
        save_path = os.path.join(save_dir, 'api_description.json')
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(descriptions, ensure_ascii=False))
    return descriptions


def delete_illegal_mashups(
    mashup_path: str,
    api_path: str,
    is_save: bool,
    save_dir: str = '../../data/api_mashup/raw',
):
    with open(mashup_path, 'r', encoding='utf-8') as f:
        mashups = json.load(f)
    apis = create_api_title(api_path, False)

    legal_mashups = []
    for ma in mashups:
        is_legal = True
        for api in ma['related_apis']:
            if api['title'] not in apis:
                is_legal = False
        if is_legal:
            legal_mashups.append(ma)

    if is_save:
        save_path = os.path.join(save_dir, 'active_mashups_data.txt')
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(legal_mashups, ensure_ascii=False))
    return legal_mashups


def delete_unused_apis(
    mashup_path: str,
    api_path: str,
    is_save: bool,
    save_dir: str = '../../data/api_mashup/raw',
):
    with open(api_path, 'r', encoding='utf-8') as f:
        apis = json.load(f)
    # with open(related_apis_path, 'r', encoding='utf-8') as f:
    #     related_apis = json.load(f)
    related_apis = create_mashup_related_api(mashup_path, False)
    used_apis_name = set()
    for r_apis in related_apis:
        for a in r_apis:
            used_apis_name.add(a)
    used_apis = []
    for api in apis:
        if api['title'] in used_apis_name:
            used_apis.append(api)
    if is_save:
        save_path = os.path.join(save_dir, 'active_apis_data.txt')
        # with open('../../data/api_mashup/raw/active_apis_data.txt', 'w', encoding='utf-8') as f:
        #     f.write(json.dumps(used_apis))
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(used_apis))
    return used_apis


# def create_mashup_document(combination: int = 1):
#     with open('../../data/api_mashup/mashup_description.json', 'r', encoding='utf-8') as f:
#         des_list = json.load(f)
#     if combination == 2:
#         with open('../../data/api_mashup/mashup_title.json', 'r', encoding='utf-8') as f:
#             title_list = json.load(f)
#         with open('../../data/api_mashup/mashup_category.json', 'r', encoding='utf-8') as f:
#             category_list = json.load(f)
#         with open('../../data/api_mashup/mashup_related_api.json', 'r', encoding='utf-8') as f:
#             api_list = json.load(f)
#         corpus = []
#         for i in range(len(des_list)):
#             corpus.append(
#                 title_list[i] + ' ' +
#                 ' '.join(category_list[i]) + ' ' +
#                 ' '.join(api_list[i]) + ' ' +
#                 ' '.join(des_list[i])
#             )
#     else:
#         corpus = [' '.join(des) for des in des_list]
#     with open('../../data/api_mashup/mashup_document.json', 'w', encoding='utf-8') as f:
#         f.write(json.dumps(corpus, ensure_ascii=False))
#     return corpus

def create_glove_embedding(
    descriptions: List[List[str]],
    glove_path: str = '../../data/glove/raw/glove.6B.300d.txt',
):
    word2vec = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            val = line.strip().split(' ')
            word = val[0]
            vec = val[1:]
            word2vec[word] = list(map(float, vec))
    embedding = []
    for des in descriptions:
        vectors = [word2vec[word] if word in word2vec.keys() else [0.0 for i in range(300)] for word in des]
        vectors = torch.tensor(vectors, dtype=torch.float32)
        embedding.append(vectors.mean(dim=0).unsqueeze(dim=0))
    embedding = torch.cat(embedding, dim=0)
    return embedding


def create_feature(
    des_path: str,
    model: str,
    is_save: bool,
    res_dir: str = '../../data/api_mashup',
    res_name: str = 'feature'
):
    with open(des_path, 'r', encoding='utf-8') as f:
        descriptions = json.load(f)
    if model == 'glove':
        embedding = create_glove_embedding(descriptions)
    elif model == 'bert':
        pass
    else:
        ex = Exception('Illegal model!')
        raise ex
    if is_save:
        save_path = os.path.join(res_dir, f'{res_name}.pt')
        torch.save(embedding, save_path)
    return embedding


# def description2embedding(descriptions: List[List[str]], num_token: int, res_dir: str, res_name: str):
#     word2vec = {}
#     with open('../../data/glove/raw/glove.6B.300d.txt', 'r', encoding='utf-8') as f:
#         for line in f:
#             val = line.strip().split(' ')
#             word = val[0]
#             vec = val[1:]
#             word2vec[word] = list(map(float, vec))
#     embedding = []
#     for des in descriptions:
#         vectors = [word2vec[word] if word in word2vec.keys() else [0.0 for i in range(300)] for word in des]
#         if len(vectors) < num_token:
#             vectors.extend([[0.0 for i in range(300)] for j in range(num_token - len(vectors))])
#         else:
#             vectors = vectors[:num_token]
#         embedding.append(vectors)
#     embedding = torch.tensor(embedding, dtype=torch.float32)
#     res_path = os.path.join(res_dir, f'{res_name}.pt')
#     torch.save(embedding, res_path)
#
#
# def create_mashup_embedding_word2vec(
#     num_token: int = 100,
#     data_path: str = '../../data/api_mashup/mashup_description.json',
#     res_dir: str = '../../data/api_mashup',
#     res_name: str = 'mashup_embedding_word2vec'
# ):
#     # res: (num_mashup, num_token, dim_embedding)
#     with open(data_path, 'r', encoding='utf-8') as f:
#         descriptions = json.load(f)
#     description2embedding(descriptions, num_token, res_dir, res_name)
#
#
# def create_api_embedding_word2vec(
#     num_token: int = 100,
#     data_path: str = '../../data/api_mashup/api_description.json',
#     res_dir: str = '../../data/api_mashup',
#     res_name: str = 'api_embedding_word2vec'
# ):
#     # res: (num_api, num_token, dim_embedding)
#     with open(data_path, 'r', encoding='utf-8') as f:
#         descriptions = json.load(f)
#     description2embedding(descriptions, num_token, res_dir, res_name)


def create_service_domain_embedding(
    num_token: int = 1000,
):
    # res: (num_domain, num_token, dim_embedding)
    with open('../../data/api_mashup/raw/active_apis_data.txt', 'r', encoding='utf-8') as f:
        apis = json.load(f)
    # 数据中包含没有tag的api，暂时将其 category 赋值为 undefined，抛弃这部分数据，后续需要清洗数据
    categories = list(set([api['tags'][0] if len(api['tags']) > 0 else 'undefined' for api in apis]))
    categories.remove('undefined')
    indexes = [[] for i in range(len(categories))]
    for i in range(len(apis)):
        api = apis[i]
        if len(api['tags']) == 0:
            continue
        cat_index = categories.index(api['tags'][0])
        indexes[cat_index].append(i)
    with open('../../data/api_mashup/mashup_description.json', 'r', encoding='utf-8') as f:
        descriptions = json.load(f)
    word2vec_dict = {}
    with open('../../data/glove/raw/glove.6B.300d.txt', 'r', encoding='utf-8') as f:
        for line in f:
            val = line.strip().split(' ')
            word = val[0]
            vec = val[1:]
            word2vec_dict[word] = list(map(float, vec))
    domains = []
    for domain_indexes in indexes:
        domain = []
        for index in domain_indexes:
            domain.extend([word2vec_dict[word] if word in word2vec_dict.keys() else [0.0 for i in range(300)] for word
                           in descriptions[index]])
        num_domain_token_len = len(domain)
        if num_domain_token_len < num_token:
            domain.extend([[0.0 for i in range(300)] for j in range(num_token - num_domain_token_len)])
        else:
            domain = domain[:num_token]
        domains.append(domain)
    embedding = torch.tensor(domains, dtype=torch.float32)
    torch.save(embedding, '../../data/api_mashup/domain_embedding_word2vec.pt')


# def save_tfidf_matrix(combination: int = 1):
#     """
#     tfidf_matrix: (N x M)，N个mashup，M个word
#     :param combination:
#     :return:
#     """
#     corpus = save_mashup_document(combination)
#     vectorizer = CountVectorizer()
#     X = vectorizer.fit_transform(corpus)
#     transformer = TfidfTransformer()
#     tfidf_matrix = transformer.fit_transform(X).toarray()
#     words = vectorizer.get_feature_names_out()
#     np.save('../../data/api_mashup/mashup_service_document_tfidf_matrix_1.npy', tfidf_matrix)
#     with open('../../data/api_mashup/mashup_service_document_tfidf_word_1.json', 'w', encoding='utf-8') as f:
#         f.write(json.dumps(words.tolist(), ensure_ascii=False))

def train_lda_model(
    corpus: List[List[str]],
    res_model_path: str,
    num_topics: int,
    passes: int = 1000,
):
    dic = gensim.corpora.Dictionary(corpus)
    doc_bow = [dic.doc2bow(doc) for doc in corpus]
    lda_model = gensim.models.ldamodel.LdaModel(doc_bow, num_topics=num_topics, id2word=dic, passes=passes)
    lda_model.save(res_model_path)


def train_hdp_model(
    corpus: List[List[str]],
    res_model_path: str,
):
    dic = gensim.corpora.Dictionary(corpus)
    doc_bow = [dic.doc2bow(doc) for doc in corpus]
    hdp_model = gensim.models.hdpmodel.HdpModel(doc_bow, id2word=dic)
    hdp_model.save(res_model_path)


if __name__ == '__main__':
    # # train hdp model
    # with open('../../data/api_mashup/mashup_description.json', 'r', encoding='utf-8') as f:
    #     corpus = json.load(f)
    # with open('../../data/api_mashup/api_description.json', 'r', encoding='utf-8') as f:
    #     corpus.extend(json.load(f))
    # dic = gensim.corpora.Dictionary(corpus)
    # doc_bow = [dic.doc2bow(doc) for doc in corpus]
    # # train_hdp_model(corpus, '../../data/pre_model/hdp/model.wv')
    # hdp: gensim.models.HdpModel = gensim.models.HdpModel.load('../../data/pre_model/hdp/model.wv')
    #
    # index = similarities.MatrixSimilarity(hdp[doc_bow])
    # sims = index[hdp[doc_bow[0]]]
    # sims = sorted(enumerate(sims), key=lambda item: -item[1])
    # print(sims)

    # create_feature('../../data/api_mashup/mashup_description.json', 'glove', True, '../../data/api_mashup', 'mashup_feature')

    mashup_path = '../../data/api_mashup/raw/active_mashups_data.txt'
    api_path = '../../data/api_mashup/raw/active_apis_data.txt'

    with open(mashup_path, 'r', encoding='utf-8') as f:
        mashups = json.load(f)
    print(len(mashups))