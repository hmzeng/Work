from typing import Optional

import pandas as pd

from .utils import check_na, split_and_count

DF = pd.DataFrame


class DataPreprocess:
    def __init__(self, df1: DF, df2: DF, df3: DF):
        self.df1 = df1
        self.df2 = df2
        self.df3 = df3

        self.df1_cleaned: Optional[DF] = None
        self.df2_cleaned: Optional[DF] = None
        self.df3_cleaned: Optional[DF] = None

        self.df_merge_final: Optional[DF] = None

    def explore_data(self):
        # explore df1
        print(check_na(self.df1))
        print(self.df1['policy_pending_status'].value_counts(normalize=True))
        print(self.df1['document_uploaded'].value_counts(normalize=True))
        print(self.df1['paid_by_echannel'].value_counts(normalize=True))
        print(self.df1['agency_code'].value_counts(normalize=True))
        print(self.df1['agent_code'].value_counts(normalize=True))
        print(split_and_count(self.df1['product']))

        # explore df2
        print(check_na(self.df2))
        print(split_and_count(self.df2['policy_pending_code']))

        # explore df3
        print(check_na(self.df3))
        print(self.df3['Gender'].value_counts(normalize=True))
        print(self.df3['Education_Level'].value_counts(normalize=True))

        id_count = self.df3['owner_or_insured_id'].value_counts()
        print((id_count >= 2).sum() / id_count.size)

        mask = self.df3['owner_or_insured_id'].str.replace("[0-9]", '', regex=True).str.len() > 0
        print(self.df3[mask])

        days = (self.df2['policy_close_date'] - self.df2['policy_application_date']).dt.days
        self.df2['days'] = days

        return self

    def cleansing_data(self):
        # df1
        mask1 = self.df1['policy_pending_status'].isnull() | (
                    self.df1['policy_pending_status'].str.replace(" ", '') == '')
        mask2 = self.df1['document_uploaded'].isnull() | (self.df1['document_uploaded'].str.replace(" ", '') == '')
        mask = mask1 | mask2
        self.df1_cleaned = self.df1.loc[~mask, :].assign(product=lambda x: x['product'].fillna('None'))
        print(check_na(self.df1_cleaned))

        # self.df2
        mask = self.df2['policy_owner_id'].isnull() | (self.df2['days'] < 0)
        self.df2_cleaned = self.df2.loc[~mask, :].assign(
            policy_pending_code=lambda x: x['policy_pending_code'].fillna('None'),
            policy_owner_id=lambda x: x['policy_owner_id'].astype(int))
        print(check_na(self.df2_cleaned))

        # self.df3
        self.df3_cleaned = self.df3.drop_duplicates(subset='owner_or_insured_id')
        mask = self.df3['owner_or_insured_id'].str.replace("[0-9]", '', regex=True).str.len() > 0
        self.df3_cleaned = self.df3_cleaned.loc[~mask, :].assign(
            owner_or_insured_id=lambda x: x['owner_or_insured_id'].astype(int))

        return self

    def merge_data(self):
        # df2 + df1
        df_merge = self.df2_cleaned.merge(self.df1_cleaned, how='inner', on='policy_id')

        # + df3
        df_merge_1 = df_merge.merge(self.df3_cleaned, how='inner',
                                    left_on='policy_owner_id', right_on='owner_or_insured_id')
        df_merge_2 = df_merge.merge(self.df3_cleaned, how='inner',
                                    left_on='policy_insured_id', right_on='owner_or_insured_id')
        self.df_merge_final = pd.concat([df_merge_1, df_merge_2], axis=0) \
            .drop_duplicates(subset=df_merge.columns, keep='first')

        return self

    @property
    def get_df_merge(self):
        return self.df_merge_final

