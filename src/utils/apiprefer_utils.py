import json
from typing import List

import gensim.corpora

def train_lda_model(
    corpus: List[List[str]],
    res_model_path: str = '../../data/api_prefer/lda_model.wv',
    num_topics: int = 200,
    passes: int = 1000,
):
    dic = gensim.corpora.Dictionary(corpus)
    doc_bow = [dic.doc2bow(doc) for doc in corpus]
    lda_model = gensim.models.ldamodel.LdaModel(doc_bow, num_topics=num_topics, id2word=dic, passes=passes)
    lda_model.save(res_model_path)

if __name__ == '__main__':
