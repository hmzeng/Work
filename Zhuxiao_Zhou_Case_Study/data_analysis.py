import pandas as pd

from src.data_preporcess import DataPreprocess
from src.feature_pipeline import FeaturePipeline
from src.lgb_tuning import LGBTuner
from src.model_evaluate import ModelEval

PATH_DF1 = 'D:/data/raw/case_study_data1.xlsx'
PATH_DF2 = 'D:/data/raw/case_study_data2.xlsx'
PATH_DF3 = "D:/data/raw/case_study_data3.csv"
KWARGS_FOR_DF2 = {'parse_dates': ['policy_application_date', 'policy_close_date']}
FEATURE_COLNAMES = ['policy_pending_code', 'product', 'agency_code', 'policy_pending_status',
                    'document_uploaded', 'paid_by_echannel', 'Age', 'Gender', 'Education_Level',
                    'app_month_sin', 'app_month_cos', 'close_month_sin', 'close_month_cos']

LGB_PARAMS_INIT = {
    'objective': 'binary',
    'metric': 'binary_logloss,auc',
    'max_depth': 6,
    'num_leaves': 40,
    'learning_rate': 0.1,
    'colsample_bytree': 0.7,
    'min_child_samples': 21,
    'min_child_weight': 0.001,
    'subsample': 1,
    'subsample_freq': 2,
    'reg_alpha': 0.001,
    'reg_lambda': 8,
    'cat_smooth': 0,
    'num_iterations': 200
}
LGB_PARAMS_TO_TUNE = [
    {
        'lgb__max_depth': [4, 6, 8],
        'lgb__num_leaves': [20, 30, 40]
    },
    {
        'lgb__colsample_bytree': [0.6, 0.7, .8],
    },
    {
        'lgb__subsample': [0.7, 0.8, 0.9, 1],
        'lgb__subsample_freq': [2, 3, 4],
    },
    {
        'lgb__reg_alpha': [1e-5, 1e-3, 1e-1, 10],
        'lgb__reg_lambda': [4, 8, 16, 32],
    },
    {
        'lgb__cat_smooth': [0, 10, 20, 40, 80],
    },
    {
        'lgb__learning_rate': [1e-3, 1e-2, .1],
        'lgb__num_iterations': [100, 200, 300]
    }

]

FEATURE_NAME_TRANSFORMED = [
    'pending_code_topic1', 'pending_code_topic2', 'pending_code_topic3', 'pending_code_topic4', 'pending_code_topic5',
    'product_topic1', 'product_topic2', 'product_topic3', 'product_topic4', 'product_topic5',
    'agency_code', 'policy_pending_status', 'document_uploaded', 'paid_by_echannel', 'Age', 'Gender', 'Education_Level',
    'app_month_sin', 'app_month_cos', 'close_month_sin', 'close_month_cos'
]


def main():
    # load raw data
    df1 = pd.read_excel(PATH_DF1)
    df2 = pd.read_excel(PATH_DF2, **KWARGS_FOR_DF2)
    df3 = pd.read_csv(PATH_DF3)

    # preprocess
    dp = DataPreprocess(df1, df2, df3)
    df_merged = dp.explore_data().cleansing_data().merge_data().get_df_merge()

    # feature pipeline
    fp = FeaturePipeline(df_merged, FEATURE_COLNAMES)
    ct = fp.make_feature().make_topic_pipeline()
    X_train, X_test, y_train, y_test = fp.get_train_test_split()

    # lgb tuning
    lgbt = LGBTuner(LGB_PARAMS_INIT, ct, X_train, y_train)
    best_params = lgbt.lgb_tuning(LGB_PARAMS_TO_TUNE)

    # model evaluate
    me = ModelEval(best_params, ct)
    me.eval_on_test_data(X_train, X_test, y_train, y_test)
    print(me.feature_importance(FEATURE_NAME_TRANSFORMED))
    print(me.analysis_topic_col('product'))


if __name__ == '__main__':
    main()
