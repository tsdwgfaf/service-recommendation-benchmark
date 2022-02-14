import json
from typing import List, Optional, Set

import numpy as np
import gensim
import torch
import torchmetrics


def topic_list2dic(topic_list: List[tuple]) -> dict:
    topic_dict = {}
    for topic in topic_list:
        topic_dict[topic[0]] = topic[1]
    return topic_dict


def get_directly_linked_number(idx: int, msn: np.ndarray):
    """
    get the number of linked mashup service document at Mashup Service Network to given mashup
    :param idx: the index of given mashup
    :param msn: Mashup Service Network
    :return: the number
    """
    n = 0
    for ms in msn[idx]:
        if ms > 0.0:
            n += 1
    return n


def create_msn(param: float):
    """
    create Mashup Service Network
    :param param: users' preferences lambda_1
    :return: A weight matrix, (N, N), N represent the number of mashups
    """
    with open('../../data/api_mashup/mashup_related_api.json') as f:
        api_list = json.load(f)
    with open('../../data/api_mashup/mashup_category.json') as f:
        tag_list = json.load(f)
    n = len(api_list)
    w_matrix = np.zeros(shape=(n, n))
    for i in range(n):
        for j in range(n):
            api_set1 = set(api_list[i])
            api_set2 = set(api_list[j])
            tag_set1 = set(tag_list[i])
            tag_set2 = set(tag_list[j])
            w1 = param * (len(api_set1 & api_set2) / len(api_set1 | api_set2)) if len(api_set1 | api_set2) != 0 else 0.0
            w2 = (1 - param) * (len(tag_set1 & tag_set2) / len(tag_set1 | tag_set2)) if len(tag_set1 | tag_set2) != 0\
                else 0.0
            w_matrix[i][j] = w1 + w2
    # np.save('../../data/icnc/msn_0.5.npy', w_matrix)
    return w_matrix


def create_incorporation_matrix(param: float, msn: np.ndarray):
    """
    :param param: one probability that random walk stops at the current node
    :param msn: Mashup Service Network
    :return: incorporation matrix, (N, N), N represent the number of mashups
    """
    # create initial probability distribution vector M
    N = msn.shape[0]
    M = np.array([1.0 / get_directly_linked_number(i, msn) for i in range(N)]).reshape(-1, 1)
    P_j = ((1 - param) * (np.linalg.inv(np.eye(N) - param * msn) @ M)).T
    # replace negative value with zero
    P_j = list(map(lambda x: x if x >= 0.0 else 0.0, list(P_j.reshape(-1,))))
    P_j = np.array(list(P_j)).reshape(1, -1)
    return np.tile(P_j, (N, 1))


# def create_network_level_topic_distribute_matrix():
#     """
#     :param msn_path: the path of Mashup Service Network data
#     :return: A distribute_matrix, (N x M), N represent the number of mashups, M represent the number of topics
#     """
#     msn = np.load('../../data/icnc/msn_0.5.npy')
#     dis_matrix = []
#     for m in msn:
#         a = np.random.dirichlet(m, 1)
#         dis_matrix.append(np.random.dirichlet(m, 1))
#


def train_lda_model(doc_bow):
    lda = gensim.models.ldamodel.LdaModel(doc_bow, num_topics=30, id2word=dic, passes=2000)
    lda.save('../../data/icnc/lad_model.wv')


def extract_topics(corpus: List[List[str]], lda: gensim.models.ldamodel.LdaModel) -> List[dict]:
    dic = gensim.corpora.Dictionary(corpus)
    doc_bow = [dic.doc2bow(doc) for doc in corpus]
    topics = []
    for doc in doc_bow:
        topic_list = lda.get_document_topics(doc)
        topics.append(topic_list2dic(topic_list))
    return topics


def calculate_mashup_similarity(topics: List[dict]) -> np.ndarray:
    N = len(topics)
    similarity_matrix = np.zeros(shape=(N, N))
    for i in range(N):
        set1 = set(topics[i].keys())
        for j in range(N):
            D_JS = 0.0
            set2 = set(topics[j].keys())
            common_topics = set1 & set2
            # print(common_topics)
            for topic in common_topics:
                common_denominator = (topics[i][topic] + topics[j][topic]) / 2
                D_JS += topics[i][topic] * np.log(topics[i][topic] / common_denominator)\
                        + topics[j][topic] * np.log(topics[j][topic] / common_denominator)
            D_JS = D_JS * 0.5
            similarity_matrix[i][j] = D_JS
    np.save('../../data/icnc/similarity_matrix.npy', similarity_matrix)
    return similarity_matrix


def cluster_mashup(ms_topics: List[dict], threshold: float) -> List[set]:
    N = len(ms_topics)
    # sim_matrix = calculate_similarity(ms_topics)
    # TODO 删除直接读取
    sim_matrix = np.load('../../data/icnc/similarity_matrix.npy')
    avesim_list = list(np.average(sim_matrix, axis=1))
    # build atom-clusters
    ASS = set(range(N))
    cluster_list = []
    while len(ASS) > 0:
        # print(len(ASS))
        i = max(ASS, key=lambda idx: avesim_list[idx])
        cluster = set()
        cluster.add(i)
        ASS.remove(i)
        new_ASS = set(filter(lambda j: sim_matrix[j][i] <= avesim_list[i], ASS))
        cluster = cluster | (ASS - new_ASS)
        cluster_list.append(cluster)
        ASS = new_ASS
    # construct cluster similarity matrix
    num_cluster = len(cluster_list)
    cluster_sim_matrix = np.zeros(shape=(num_cluster, num_cluster))
    for x in range(num_cluster):
        for y in range(num_cluster):
            P = len(cluster_list[x])
            Q = len(cluster_list[y])
            cluster_sim = 0.0
            for p in range(P):
                for q in range(Q):
                    cluster_sim += sim_matrix[p][q]
            cluster_sim_matrix[x][y] = cluster_sim / (P * Q)
    row_list, col_list = np.where(cluster_sim_matrix == np.max(cluster_sim_matrix))
    x = row_list[0]
    y = row_list[0]
    while len(cluster_list) == 1 and cluster_sim_matrix[x][y] >= threshold:
        # merge cluster_x and cluster_y into a new cluster
        new_cluster = cluster_list[x] | cluster_list[y]
        cluster_list.pop(x)
        cluster_list.pop(y)
        x_vector = cluster_sim_matrix[x].reshape(1, -1)
        y_vector = cluster_sim_matrix[y].reshape(1, -1)
        cluster_sim_matrix = np.delete(cluster_sim_matrix, [x, y], axis=0)
        cluster_sim_matrix = np.delete(cluster_sim_matrix, [x, y], axis=1)
        insert_row = (x_vector + y_vector) / 2.0
        insert_col = np.insert(insert_row, len(insert_row), 0, axis=1).T
        cluster_sim_matrix = np.insert(cluster_sim_matrix, cluster_sim_matrix.shape[0], insert_row, axis=0)
        cluster_sim_matrix = np.insert(cluster_sim_matrix, cluster_sim_matrix.shape[1], insert_col, axis=1)
    return cluster_list


def create_cluster_api_invocation_relationship_matrix(
    mashup_api_inv_matrix: np.ndarray,
    clusters: List[set],
):
    num_cluster = len(clusters)
    num_api = mashup_api_inv_matrix.shape[1]
    cluster_api_inv_matrix = np.zeros(shape=(num_cluster, num_api))
    for i in range(num_cluster):
        for mashup in clusters[i]:
            cluster_api_inv_matrix[i] += mashup_api_inv_matrix[mashup]
    cluster_api_inv_matrix = cluster_api_inv_matrix / cluster_api_inv_matrix.sum(axis=1).reshape(-1, 1)
    return cluster_api_inv_matrix


def create_api_similarity_matrix(cluster_api_inv_matrix: np.ndarray):
    num_cluster = cluster_api_inv_matrix.shape[0]
    num_api = cluster_api_inv_matrix.shape[1]
    api_inv_avg = np.average(cluster_api_inv_matrix, axis=0)
    api_sim_matrix = np.zeros(shape=(num_api, num_api))
    for i in range(num_api):
        for j in range(num_api):
            de = 0.0
            ne1 = 0.0
            ne2 = 0.0
            for m in range(num_cluster):
                de += (cluster_api_inv_matrix[m][i] - api_inv_avg[i]) * (cluster_api_inv_matrix[m][j] - api_inv_avg[j])
                ne1 += (cluster_api_inv_matrix[m][i] - api_inv_avg[i]) ** 2
                ne2 += (cluster_api_inv_matrix[m][j] - api_inv_avg[j]) ** 2
            if ne1 == 0 or ne2 == 0:
                api_sim_matrix[i][j] = 0.0
            else:
                api_sim_matrix[i][j] = de / (np.sqrt(ne1) * np.sqrt(ne2))
    np.save('../../data/icnc/api_similarity_matrix.npy', api_sim_matrix)
    return api_sim_matrix


def calculate_mashup_cluster_topic(clusters: List[set], topics: List[dict]) -> List[dict]:
    mashup_cluster_topics = []
    for cluster in clusters:
        cluster_topics = {}
        for mashup in cluster:
            for key in topics[mashup]:
                if key not in cluster_topics:
                    cluster_topics[key] = topics[mashup][key]
                else:
                    cluster_topics[key] = cluster_topics[key] + topics[mashup][key]
        num_mashup = len(cluster)
        for key in cluster_topics:
            cluster_topics[key] = cluster_topics[key] / num_mashup
        mashup_cluster_topics.append(cluster_topics)
    return mashup_cluster_topics


def recommend_api(
    mashup_requirement_text_bow: list,
    clusters: List[set],
    cluster_api_inv_matrix: np.ndarray,
    mashup_topics: List[dict],
    cluster_topics: List[dict],
    lda_model: gensim.models.ldamodel.LdaModel,
    top_k: int
) -> List[int]:
    requirement_topic = topic_list2dic(lda_model.get_document_topics(mashup_requirement_text_bow))
    # calculate similarity between requirement and cluster
    requirement_cluster_sim_list = []
    set1 = set(requirement_topic.keys())
    for each_cluster_topics in cluster_topics:
        D_JS = 0.0
        set2 = set(each_cluster_topics.keys())
        common_topics = set1 & set2
        for topic in common_topics:
            common_denominator = (requirement_topic[topic] + each_cluster_topics[topic]) / 2
            D_JS += requirement_topic[topic] * np.log(requirement_topic[topic] / common_denominator) \
                    + each_cluster_topics[topic] * np.log(each_cluster_topics[topic] / common_denominator)
        D_JS = D_JS * 0.5
        requirement_cluster_sim_list.append(D_JS)
    # identify the highest similarity
    max_sim_cluster_idx = requirement_cluster_sim_list.index(max(requirement_cluster_sim_list))
    rank_apis = np.argsort(cluster_api_inv_matrix[max_sim_cluster_idx], kind='quicksort')
    return list(rank_apis[-1: -1 - top_k: -1])


if __name__ == '__main__':
    with open('../../data/api_mashup/mashup_document.json', 'r', encoding='utf-8') as f:
        corpus = json.load(f)
    corpus = [doc.split(' ') for doc in corpus]
    dic = gensim.corpora.Dictionary(corpus)
    doc_bow_list = [dic.doc2bow(doc) for doc in corpus]

    # train_lda_model(doc_bow_list)

    lda: gensim.models.ldamodel.LdaModel = gensim.models.ldamodel.LdaModel.load('../../data/icnc/lad_model.wv')
    mashup_topics = extract_topics(corpus, lda)
    clusters = cluster_mashup(mashup_topics, 10)
    mashup_api_inv_matrix = torch.load('../../data/api_mashup/user_item_interaction_matrix.pt').numpy()
    cluster_api_inv_matrix = create_cluster_api_invocation_relationship_matrix(mashup_api_inv_matrix, clusters)
    api_sim_matrix = np.load('../../data/icnc/api_similarity_matrix.npy')
    cluster_topics = calculate_mashup_cluster_topic(clusters, mashup_topics)
    top_k_apis = recommend_api(
        doc_bow_list[5], clusters, cluster_api_inv_matrix, mashup_topics, cluster_topics, lda, 5
    )
    # test
    num_test = 300
    preds = torch.LongTensor([recommend_api(
        doc_bow, clusters, cluster_api_inv_matrix, mashup_topics, cluster_topics, lda, 5
    ) for doc_bow in doc_bow_list[:num_test]])
    num_api = mashup_api_inv_matrix.shape[1]
    preds = torch.zeros(num_test, num_api).scatter(dim=1, index=preds, src=torch.ones(size=(num_test, 5), dtype=torch.float32))


    pre = torchmetrics.Precision(top_k=5)
    recall = torchmetrics.Recall(top_k=5)
    ndcg = torchmetrics.RetrievalNormalizedDCG(k=5)
    # map = torchmetrics.RetrievalMAP()

    p = pre(preds, torch.from_numpy(mashup_api_inv_matrix[:num_test]).long())
    re = recall(preds, torch.from_numpy(mashup_api_inv_matrix[:num_test]).long())
    nd = ndcg(preds, torch.from_numpy(mashup_api_inv_matrix[:num_test]).long(), torch.zeros(size=preds.shape, dtype=torch.int64))
    # m = map(preds, torch.from_numpy(mashup_api_inv_matrix[:num_test]).long(), torch.zeros(size=preds.shape, dtype=torch.int64))
    print(f'precision: {p}')
    print(f'recall: {re}')
    print(f'NDCG: {nd}')
    # print(f'mAP: {m}')
