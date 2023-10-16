#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import modules
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV

random_seed = 42
np.random.seed(random_seed)


# In[2]:


#Load and preprocess data
def lasso_preprocess_data(data, train_columns=None, up_quantile = None):
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


# In[3]:


def lasso_fit_and_predict():
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    y_test= pd.read_csv('test_y.csv')

    # Dropping outliers as suggested by data documentation page, taking a more aggressive approach i.e. 3500 limit vs 4000
    train_data = train_data[train_data['Gr_Liv_Area'] <= 3800]

    y_train = np.log(train_data['Sale_Price'])
    y_test = np.log(y_test["Sale_Price"])

    train_data.drop(columns=["Sale_Price"], inplace=True)

    # Preprocess train data
    x_train_pp, up_quantile = lasso_preprocess_data(train_data, None, None)

    # Preprocess test data using columns of preprocessed train data and quantile information from train_data
    x_test_pp = lasso_preprocess_data(test_data, x_train_pp.columns, up_quantile)

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
    elastic_rmse = np.sqrt(mean_squared_error(elastic_pred, y_test))

    print("ElasticNet RMSE", elastic_rmse)
    
    #Save predictions
    submission_df = pd.DataFrame({
        'PID': test_data['PID'],
        'Sale_Price': np.exp(elastic_pred)
    })
    
    submission_df.to_csv('mysubmission1.txt', index=False)

lasso_fit_and_predict()