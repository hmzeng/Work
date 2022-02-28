from typing import Dict, List
from itertools import chain

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

DF = pd.DataFrame
Series = pd.Series


def check_na(df: DF):
    result = []
    dtype_series = df.dtypes
    # 检查NA和空格
    for col, dtype in zip(dtype_series.index, dtype_series.values):
        col_series: Series = df[col]
        col_str_series = col_series.astype(str).str.replace(" ", '')
        na_mask: Series = col_series.isnull() | (col_str_series == '')
        na_count: int = na_mask.sum()
        na_ratio = na_count / df.shape[0]
        result.append([col, dtype, na_count, na_ratio])
    return pd.DataFrame(result, columns=['col', 'dtype', 'na_count', 'na_ratio'])


def split_and_count(x: Series, sep=':'):
    code_count = x.fillna('None').str.split(sep).values
    code_count = pd.Series(chain(*code_count)).value_counts(normalize=True)
    return code_count


class TopicEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, map_dict: Dict[str, int], n_topics: int = 5):
        self.map_dict = map_dict
        self.n_topics = n_topics
        self.vec = CountVectorizer(vocabulary=self.map_dict)
        self.lda = LatentDirichletAllocation(n_components=self.n_topics, n_jobs=-1)

    def fit(self, X: Series, y=None):
        sparse = self.vec.transform(X)
        self.lda.fit(sparse)
        return self

    def transform(self, X, copy=None):
        sparse = self.vec.transform(X)
        return self.lda.transform(sparse)


def get_topN_topic_representation(array_components: np.ndarray, colnames: List[str], topN: int):
    max_components_array = (array_components * -1).argsort(axis=1)[:, :topN]
    colname_array = np.array(colnames)

    n_topics, _ = array_components.shape
    result = []
    for idx in range(n_topics):
        top_n_comp = colname_array[max_components_array[idx, :]]
        result.append(top_n_comp)

    return pd.DataFrame(result,
                        columns=[f'rank{i+1}_component' for i in range(topN)],
                        index=[f'topic{i+1}' for i in range(n_topics)])
