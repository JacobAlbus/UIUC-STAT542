# Jacob Albus (albus2 On-Campus) Ashish Pabba (apabba2 Online MCS)
# Jacob worked on XGBoost model and Ashish worked on Elasticnet model

import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV

def elastic_preprocess_data(data, train_columns=None, up_quantile = None):
    #Winsorize features
    winsor_features = ["Lot_Frontage", "Lot_Area", "Mas_Vnr_Area", "BsmtFin_SF_2", "Bsmt_Unf_SF", 
                          "Total_Bsmt_SF", "Second_Flr_SF", 'First_Flr_SF', "Gr_Liv_Area", 
                          "Garage_Area", "Wood_Deck_SF", "Open_Porch_SF", "Enclosed_Porch", 
                          "Three_season_porch", "Screen_Porch", "Misc_Val"]
    # If train_data, compute quantile, if test_data, use quantile saved from train_data preprocessing
    if up_quantile is None:
        up_quantile = data[winsor_features].quantile(0.92)
    
    # replace top 8% of data with 92nd quantile values
    for feature in winsor_features:
        data[feature] = np.where(data[feature] > up_quantile[feature], up_quantile[feature], data[feature])
 
    #dropping columns suggested by professor, also drop PID
    data = data.drop(columns=["PID","Street", "Utilities", "Condition_2", "Roof_Matl", "Heating", "Pool_QC",
                                            "Misc_Feature", "Low_Qual_Fin_SF", "Pool_Area", "Longitude","Latitude"])

    #use get_dummies to encode categorical variables
    data = pd.get_dummies(data,drop_first=False)
    
    #Impute missing values with column mean
    data = data.fillna(data.mean())

    #if processing test data, use train data columns to ensure that the shape and columns of training and test data are identical
    if train_columns is not None:
        missing_columns = set(train_columns) - set(data.columns)
        new_data = pd.DataFrame({column: [0] * len(data) for column in missing_columns})
        data = pd.concat([data, new_data], axis=1)
        # Ensure the order of column in the test set is in the same order than in train set
        data = data[train_columns]
    
    #return just processed data if test data, else return train data processed and quantile information
    if train_columns is not None:
        return data
    else:
        return data, up_quantile


def elastic_fit_and_predict():
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')

    # Dropping outliers as suggested by data documentation page, taking a more aggressive approach i.e. 3500 limit vs 4000
    train_data = train_data[train_data['Gr_Liv_Area'] <= 3800]

    y_train = np.log(train_data['Sale_Price'])

    train_data.drop(columns=["Sale_Price"], inplace=True)

    # Preprocess train data
    x_train_pp, up_quantile = elastic_preprocess_data(train_data, None, None)

    # Preprocess test data using columns of preprocessed train data and quantile information from train_data
    x_test_pp = elastic_preprocess_data(test_data, x_train_pp.columns, up_quantile)

    # Standardizing test and train data before regression
    scaler1 = StandardScaler()
    x_train_pp = scaler1.fit_transform(x_train_pp)
    
    
    #Fit elasticnet cv model on processed training data to find optimal parameters in provided grid
    model_elastic = ElasticNetCV(l1_ratio = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1],
                              alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 
                                        0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6], 
                              max_iter = 50000, cv = 10).fit(x_train_pp, y_train)
    
    # Use scaler object fitted on train_data to tranform test data as well, this doesn't involve backwards propagration of
    # of information or data leakage, similar to how a PCA object would be used for transforming training and test data
    x_test_pp = scaler1.transform(x_test_pp)
    
    #make predictions and compute RMSE
    elastic_pred = model_elastic.predict(x_test_pp)
    
    #Save predictions
    submission_df = pd.DataFrame({
        'PID': test_data['PID'],
        'Sale_Price': np.exp(elastic_pred)
    })
    
    submission_df.to_csv('mysubmission1.txt', index=False)


def preprocess_xgb_features(data, train_columns=None):
    # Some values in Mas_Vnr_Type and Misc_feature have a Nan float value instead of just a "None" string
    data = data.fillna('None')
    # Some homes don't have garages so replace their "None" string with 0
    data["Garage_Yr_Blt"] = data["Garage_Yr_Blt"].replace("None", 0)

    try:
        y = np.log(data["Sale_Price"])
    except KeyError:
        y = None

    # Select all features except Sales Price
    best_features = list(data.columns)[1:-1]
    data = data[best_features]

    # One hot encoding of nominal features
    X = pd.get_dummies(data[best_features])

    # Handle column mismatch from one hot encoding
    if train_columns is not None:
        missing_columns = set(train_columns) - set(X.columns)
        for column in missing_columns:
            X[column] = 0
        # Ensure the order of column in the test set is in the same order than in train set
        X = X[train_columns]

    return X, y

def train_xgb_model():
  X_train, y_train = preprocess_xgb_features(pd.read_csv("train.csv"))

  cv_eta = 0.05
  cv_T = 5000
  
  X_test, _ = preprocess_xgb_features(pd.read_csv("test.csv"), X_train.columns)  

  clf = xgb.XGBRegressor(n_estimators=cv_T, learning_rate=cv_eta, 
                        max_depth=6, colsample_bytree=0.5, tree_method="exact")
  clf.fit(X_train, y_train)
  yhat = clf.predict(X_test)

  test_PID = pd.read_csv(f"test.csv")["PID"]
  output = pd.DataFrame({"PID" : test_PID, "Sale_Price" : np.exp(yhat) })
  output.to_csv("mysubmission2.txt", index=False)

# elastic_fit_and_predict()
train_xgb_model()

# test_y = np.log(pd.read_csv("test_y.csv")["Sale_Price"])
# lasso_y = np.log(pd.read_csv("mysubmission1.txt")["Sale_Price"])

# print(np.sqrt(np.mean((test_y - lasso_y)**2)))

# boost_y = np.log(pd.read_csv("mysubmission2.txt")["Sale_Price"])

# print(np.sqrt(np.mean((test_y - boost_y)**2)))