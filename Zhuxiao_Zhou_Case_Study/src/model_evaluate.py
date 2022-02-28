from typing import Dict, List

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

from src.utils import get_topN_topic_representation

Array = np.ndarray


class ModelEval:
    def __init__(self, best_params: Dict, ct: ColumnTransformer):
        self.params = best_params

        gbm = LGBMClassifier(**best_params)
        self.pipe = Pipeline([('ct', ct), ('lgb', gbm)])

    def eval_on_test_data(self,
                          X_train: Array,
                          X_test: Array,
                          y_train: Array,
                          y_test: Array):
        _ = self.pipe.fit(X_train, y_train, lgb__categorical_feature=[10])
        y_pred_train = self.pipe.predict_proba(X_train)
        y_pred_test = self.pipe.predict_proba(X_test)
        print('train auc =', roc_auc_score(y_train, y_pred_train[:, 1]))
        print('test auc =', roc_auc_score(y_test, y_pred_test[:, 1]))

        return self

    def feature_importance(self, feature_names: List[str]):
        lgb_model = self.pipe.named_steps['lgb']

        return pd.Series(lgb_model.feature_importances_, index=feature_names).sort_values(ascending=False)

    def analysis_topic_col(self, col_name: str, top_n=10):
        ct_model = self.pipe.named_steps['ct']
        lda_product = ct_model.named_transformers_[col_name].lda
        map_dict = ct_model.named_transformers_[col_name].map_dict
        colnames = list(map_dict.keys())

        return get_topN_topic_representation(lda_product.components_, colnames, top_n)
