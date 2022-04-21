import pandas as pd
import numpy as np
import xgboost as xgb
from varclushi import VarClusHi
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import VarianceThreshold

def filter_missing_rate(df, threshold=0.15):
    missing_report = (
        df
        .isnull()
        .sum()
        .rename('MISSING_COUNT')
        .reset_index()
        .rename(columns={"index":"VARIABLE_NAME"})
        .assign(
            MISSING_PERCENTAGE = lambda x: x['MISSING_COUNT']/df.shape[0]
        )
    )
    required_columns = missing_report['VARIABLE_NAME'][
        missing_report['MISSING_PERCENTAGE']<=threshold
    ]
    return required_columns, df[required_columns]

def filter_constant_columns(df, threshold=0.05):
    constant_filter = VarianceThreshold(threshold=threshold).fit(df)
    filtered_df = pd.DataFrame(
        constant_filter.transform(df),
        columns=constant_filter.get_feature_names_out()
    )
    return constant_filter.get_feature_names_out(), filtered_df

def cluster_variables(
    df,
    top_n_features=1,
    maxeigenval2=1,
    max_clus=None,
    column_subset=None
):
    if column_subset is None:
        variable_cluster_obj = VarClusHi(
            df,
            maxeigval2=maxeigenval2,
            maxclus=max_clus
        )
    else:
        variable_cluster_obj = VarClusHi(
            df[column_subset],
            maxeigval2=maxeigenval2,
            maxclus=max_clus
        )

    variable_cluster_model = variable_cluster_obj.varclus()
    
    vc_rsquare = variable_cluster_model.rsquare
    vc_rsquare.sort_values(
        by=["Cluster", "RS_Own"], ascending=False, inplace=True
    )
    top_n_variables = (
        vc_rsquare.groupby(["Cluster"]).head(top_n_features).reset_index()
    )

    selected_features = list(set(top_n_variables["Variable"]))

    return selected_features, vc_rsquare

def clean_feature_importance(estimator, variable_names):
    feature_importance = (
        pd.DataFrame({
            'VARIABLE_NAME': variable_names,
            'FEATURE_IMPORTANCE': estimator.feature_importances_
        }).sort_values(by=['FEATURE_IMPORTANCE'], ascending=True)
        .reset_index(drop=True)
    )
    return feature_importance

def remove_zero_importances_regression(
    x_train,
    y_train,
    column_subset=None,
    verbose=True
):

    if column_subset != None:
        x_train = x_train[column_subset]

    estimator = xgb.XGBRegressor(seed=1024)
    base_model = estimator.fit(x_train, y_train)

    feature_importance = clean_feature_importance(
        estimator=base_model,
        variable_names = x_train.columns
    )

    selected_features = feature_importance['VARIABLE_NAME'].to_list()
    
    required_columns = x_train.columns

    for variable_index in range(len(required_columns)):
        if feature_importance['FEATURE_IMPORTANCE'].min() == 0:
            feature_importance = feature_importance[feature_importance['FEATURE_IMPORTANCE']>0]
            selected_features = feature_importance['VARIABLE_NAME'].tolist()
            base_model = estimator.fit(x_train[selected_features], y_train)
            feature_importance = clean_feature_importance(
                estimator=base_model,
                variable_names=selected_features
            )
            if verbose:
                print(feature_importance)

    feature_importance = clean_feature_importance(
        estimator=base_model,
        variable_names=selected_features
    )

    feature_importance = feature_importance[feature_importance['FEATURE_IMPORTANCE']>0]
    selected_features = feature_importance['VARIABLE_NAME'].tolist()

    print(feature_importance)

    return selected_features

def select_var_importance(
    x_train,
    y_train,
    x_test,
    y_test,
    column_subset=None
):
    if column_subset != None:
        x_train = x_train[column_subset]

    model_features = x_train.columns

    estimator = xgb.XGBRegressor(seed=1024)
    base_model = estimator.fit(x_train, y_train)

    thresholds = np.sort(base_model.feature_importances_)
    output = []

    for threshold in thresholds:
        selection = SelectFromModel(base_model, threshold=threshold, prefit=True)
        selected_x_train = selection.transform(x_train)
        selection_model = xgb.XGBRegressor(seed=1024)
        selection_model.fit(selected_x_train, y_train)
        selected_x_test = selection.transform(x_test)
        y_pred = selection_model.predict(selected_x_test)
        selected_error = mean_absolute_error(selected_x_test, y_pred)
        temp_df = pd.DataFrame({
            "THRESHOLD": [threshold],
            "VARIABLE_COUNT": [selected_x_train.shape[1]],
            "ERROR": [selected_error]
        })
        output.append(temp_df)

        output = pd.merge(
            right=pd.concat(output, axis=0).reset_index(drop=True),
            left=clean_feature_importance(base_model, model_features),
            how='outer',
            left_index=True,
            right_index=True
        ).drop(columns=['THRESHOLD', 'VARIABLE_COUNT'])

        selected_features = set(output.loc[
            output['THRESHOLD']>=output['THRESHOLD'].min()
        ]['VARIABLE_NAME'])

    return list(selected_features), output

def select_best_model(
    x_train,
    y_train,
    x_test,
    y_test,
    parameter_grid={
        'objective': ['reg:linear'],
        'learning_rate': [0.03, 0.05, 0.07]
    },
    column_subset=None,
    k_cv = 5
):
    if column_subset != None:
        x_train = x_train[column_subset]
        x_test = x_test[column_subset]

    estimator = xgb.XGBRegressor(seed=1024)

    gridsearch = GridSearchCV(
        estimator=estimator,
        parameters=parameter_grid,
        cv=k_cv,
        n_jobs=-1,
        verbose=True
    )

    gridsearch.fit(x_train, y_train)

    return gridsearch.best_estimator_, gridsearch.best_params_
