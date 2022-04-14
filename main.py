'''
    TODO: Add the module docsting.
'''

# Import Statements
import numpy as np
import pandas as pd
import xgboost as xgb
from varclushi import VarClusHi
from sklearn.modelselection import train_test_split


class FeatureSelection:
    '''
        TODO: Add the class docstring.
    '''

    selected_features = None

    supported_error_metrics = {
        'regression_metrics': ['mae'],
        'classification': ['auc']
    }


    def __init__(
        self,
        df,
        x_vars,
        y_var,
        error_metric,
        # test_size=0.30
    ):
        self.df = df
        self.x_vars = x_vars
        self.y_var = y_var
        # self.test_size = test_size
        self.error_metric = error_metric


        assert error_metric in (
            self.supported_error_metrics["regression"]
            + self.supported_error_metrics["classification"]
        ), f'The error_metric "{error_metric}" is not supported\n The supported metrics are: {self.supported_error_metrics}'


    def split_data_train_test(self, subset=None):
        '''
            TODO: Add the method docstring.
        '''
        if subset is None:
            required_vars = self.x_vars
        else:
            required_vars = subset

        x_train, x_test, y_train, y_test = train_test_split(
            self.df[required_vars],
            self.df[self.target],
            test_size=self.test_size
            random_state=1024
        )
        return x_train, x_test, y_train, y_test


    def variable_clustering(
        self, top_n_features=1, maxeigval2=1, max_clus=None, column_subset=None
    ):
        '''
            TODO: Add the method docstring.
        '''
        if column_subset == None:
            vc_params = VarClusHi(
                self.df[self.x_vars], maxeigval2=maxeigval2, max_clus=max_clus
            )
        else:
            vc_params = VarClusHi(
                self.df[column_subset], maxeigval2=maxeigval2, max_clus=max_clus
            )
        variable_cluster_model = vc_params()

        variable_cluster_info = variable_cluster_model.info()
        vc_rsquare = variable_cluster_model.rsquare
        vc_rsquare.sort_values(by=["Cluster", "RS_Own"], ascending=False, inplace=True)

        top_n_variables = (
            vc_rsquare.groupby(["Cluster"]).head(top_n_features).reset_index()
        )

        self.seleted_features = set(top_n_variables["Variable"])


    @staticmethod
    def clean_feature_importance(estimator, variable_names):
        '''
            TODO: Add method docstring.
        '''
        feature_importance = (
            pd.DataFrame({
                'variables': variable_names,
                'feature_importance': estimator.feature_importances_
            }).sort_values(by=['feature_importance'], ascending=True)
            .reset_index(drop=True)
        )
        return feature_importance


    def remove_zero_importances(self, subset=None, verbose=True):
        '''
            TODO: Add method docsting.
        '''
        if subset is None:
            required_cols = self.x_vars
        else:
            required_cols = subset

        if self.error_metric in self.supported_error_metrics['regression']:
            estimator = xgb.XGBRegressor(seed=1024)
        elif self.error_metric in self.supported_error_metrics['classification']:
            estimator = xgb.XGBClassifier(seed=1024)

        # x_train, x_test, y_train, y_test = self.split_data_train_test(subset=required_cols)

        base_model = estimator.fit(self.df[required_cols], self.df[self.y_var])

        feature_importance = self.clean_feature_importance(
            estimator=estimator, variable_names=self.x_train.columns
        )

        for variable_index in range(len(required_cols)):
            if feature_importance['feature_importance'].min() == 0:
                feature_importance = feature_importance[feature_importance['feature_importance']>0]
                selected_features = feature_importance['variables'].tolist()
                base_model = estimator.fit(self.df[selected_features], self.df[self.y_var])
                feature_importance = self.clean_feature_importance(
                    estimator=estimator, variable_names=self.x_train.columns
                )
                if verbose:
                    print(feature_importance)

        feature_importance = self.clean_feature_importance(
            estimator=estimator, variable_names=self.x_train.columns
        )
        feature_importance = feature_importance[feature_importance['feature_importance']>0]
        self.selected_features = feature_importance['variables'].tolist()

        return feature_importance['variables'].tolist()