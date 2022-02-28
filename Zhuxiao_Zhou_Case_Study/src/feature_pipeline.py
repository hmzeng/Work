from typing import Optional, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


from .utils import split_and_count, TopicEncoder

DF = pd.DataFrame
Array = np.ndarray


class FeaturePipeline:
    def __init__(self, df_merged: DF, feature_cols: List[str]):
        self.df = df_merged.copy()
        self.ct: Optional[ColumnTransformer] = None
        self.feature_cols = feature_cols

    def make_feature(self):
        # one-hot
        self.df['days'] = (self.df['days'] > 5).astype(int)
        self.df['policy_pending_status'] = self.df['policy_pending_status'].map({'FAILED': 0, 'PASSED': 1})
        self.df['document_uploaded'] = self.df['document_uploaded'].map({'PASSED': 1, 'FAILED-DCK': 0})
        self.df['paid_by_echannel'] = self.df['paid_by_echannel'].map({'Y': 1, 'N': 0})
        self.df['Gender'] = self.df['Gender'].map({'M': 1, 'F': 0})

        # ordinal
        self.df['Education_Level'] = self.df['Education_Level'].map({
            'Primary_or_below': 1,
            'Secondary': 2,
            'Tertiary': 3
        })

        # time_encoding
        self.df['app_month_sin'] = np.sin(self.df['policy_application_date'].dt.month / 12 * 2 * np.pi)
        self.df['app_month_cos'] = np.cos(self.df['policy_application_date'].dt.month / 12 * 2 * np.pi)
        self.df['close_month_sin'] = np.sin(self.df['policy_close_date'].dt.month / 12 * 2 * np.pi)
        self.df['close_month_cos'] = np.cos(self.df['policy_close_date'].dt.month / 12 * 2 * np.pi)

        # replace ':' with ' '
        self.df['product'] = self.df['product'].str.replace(':', ' ')
        self.df['policy_pending_code'] = self.df['policy_pending_code'].str.replace(':', ' ')

        return self

    def make_topic_pipeline(self):
        # topic_model columns transformer
        map_product = {value: idx for idx, value in
                       enumerate(split_and_count(self.df['product'], sep=' ').index.str.lower().sort_values())
                       }
        map_policy = {value: idx for idx, value in
                      enumerate(split_and_count(self.df['policy_pending_code'], sep=' ').index.str.lower().sort_values())
                      }
        self.ct = ColumnTransformer([
            ('product', TopicEncoder(map_product), 'product'),
            ('pending_code', TopicEncoder(map_policy), 'policy_pending_code')
        ],
            remainder='passthrough'
        )

        return self.ct

    def get_total_data(self):
        return self.df[self.feature_cols], self.df['days'].values

    def get_train_test_split(self):
        return train_test_split(
            self.df[self.feature_cols], self.df['days'].values,
            test_size=0.2, random_state=123, stratify=self.df['days'].values
        )
