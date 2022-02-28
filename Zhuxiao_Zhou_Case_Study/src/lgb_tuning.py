from typing import Dict, List, Union

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer

Array = np.ndarray


class LGBTuner:
    def __init__(self,
                 init_lgb_params: Dict[str, Union[float, str]],
                 ct: ColumnTransformer,
                 X_train: Array,
                 y_train: Array):
        self.lgb_params = init_lgb_params
        self.ct = ct
        self.X = X_train
        self.y = y_train

    def lgb_tuning(self, params_list: List[Dict]):
        for params in params_list:
            gbm = LGBMClassifier(**self.lgb_params)
            pipe = Pipeline([('ct', self.ct), ('lgb', gbm)])
            gsearch = GridSearchCV(pipe, param_grid=params, scoring='roc_auc', cv=3, n_jobs=-1)
            gsearch.fit(self.X, self.y, lgb__categorical_feature=[10])

            for k, v in gsearch.best_params_:
                k = k.replace("lgb__", '')
                self.lgb_params[k] = v
            print('bset params = {}'.format(gsearch.best_params_))
            print('best score = {}'.format(gsearch.best_score_))

        return self.lgb_params
