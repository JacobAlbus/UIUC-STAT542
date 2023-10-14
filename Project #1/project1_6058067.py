import pandas as pd
import numpy as np
import xgboost as xgb

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
    best_features = list(data.columns)[:-1]
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
  X_train, y_train = preprocess_xgb_features(pd.read_csv(f"train.csv"))

  cv_eta = 0.025
  cv_T = 10000

  X_test, _ = preprocess_xgb_features(pd.read_csv("test.csv"), X_train.columns)

  clf = xgb.XGBRegressor(n_estimators=cv_T, learning_rate=cv_eta, 
                        max_depth=6, subsample=0.5, tree_method="exact")
  clf.fit(X_train, y_train)
  yhat = clf.predict(X_test)

  output = pd.DataFrame({"PID" : X_test["PID"], "Sale_Price" : yhat })
  output.to_csv("mysubmission2.txt", index=False)

train_xgb_model()