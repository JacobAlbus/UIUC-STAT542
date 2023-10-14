import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler,OneHotEncoder, LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold

def preprocess_lasso_data(train_data, data):
    # neighborhood binning mapping
    neighborhood_median_prices = train_data.groupby('Neighborhood')['Sale_Price'].median().reset_index()
    neighborhood_median_prices.sort_values(by='Sale_Price', inplace=True)
    neighborhood_mapping = {neighborhood: (label//4) for label, neighborhood in enumerate(neighborhood_median_prices['Neighborhood'])}

    # MS_SubClass mapping
    subclass_median_prices = train_data.groupby('MS_SubClass')['Sale_Price'].median().reset_index()
    subclass_median_prices.sort_values(by='Sale_Price', inplace=True)
    subclass_mapping = {subclass: (label//4) for label, subclass in enumerate(subclass_median_prices['MS_SubClass'])}

    pool_qc_mapping = {"No_Pool" : 0,"Fair" : 1, "Good" : 1,"Typical" : 1, "Excellent" : 3}
    quality_order  = ["Very_Poor","Poor", "Fair","Below_Average","Average","Above_Average", "Good", "Very_Good", "Excellent", "Very_Excellent"]
    quality_mapping = {categorical: i for i, categorical in enumerate(quality_order)}
    sale_condition_mapping = {"AdjLand" : 1, "Abnorml" : 2,"Family" : 2,"Alloca" : 3,"Normal" : 3, "Partial" : 4}
    lot_config_mapping = {"FR2" : 1, "Inside" : 1, "Corner" : 1, "FR3" : 2, "CulDSac" : 3}
    bldg_type_mapping = {"TwoFmCon": 1 , "Twnhs" : 1, "Duplex" : 1, "OneFam" : 2 , "TwnhsE" : 2.5}
    roof_style_mapping = {"Gambrel":1, "Gable" : 2 ,"Mansard" : 2, "Flat" :2.5 , "Hip" : 2.5, "Shed" : 3.5 } 
    roof_matl_mapping = {"ClyTile" : 0, "Roll" : 0.8, "CompShg" : 1,"Tar&Grv" : 1,"Metal" : 1, "Membran" : 2, "WdShngl": 2.25, "WdShake" : 2.25}
    mas_mapping = {"None": 0,"BrkCmn" : 1, "CBlock" : 1.5, "BrkFace" : 2, "Stone" : 3}
    lot_shape_mapping = {"Regular" : 1, "Slightly_Irregular" : 2, "Moderately_Irregular" : 2.75, "Irregular" : 3.25}
    util_mapping  = {"ELO" : 0,"NoSeWa" : 0.5,"NoSewr" : 1.25, "AllPub" : 2}
    land_slope_mapping = {"Gtl" : 1, "Mod" : 2, "Sev" : 3}
    ext_qual_mapping = {"Poor" : 1, 'Fair': 2, 'Typical': 3, 'Good': 4, 'Excellent': 5.5}
    ext_cond_mapping = {"Poor" : 1, 'Fair': 2, 'Typical': 3.5, 'Good': 4, 'Excellent': 5}
    foundation_mapping = {"Slab" : 1, "Stone" : 1,"BrkTil" : 1,"CBlock" : 1, "PConc" : 1.75, "Wood" : 2}
    bsmt_qual_mapping = {"Poor" : 1 , "Fair" : 1, "No_Basement" : 1.5, "Typical" : 2, "Good" : 2.5, "Excellent" : 3.25}
    bsmt_cond_mapping = {"Poor" : 1 , "Fair" : 1, "No_Basement" : 1.5, "Typical" : 2, "Good" : 2.5, "Excellent" : 3.25}

    high_corr_features = ["Bsmt_Cond", "Bsmt_Qual","MS_SubClass", "Pool_QC", "Lot_Config","Lot_Shape", 
                          "Utilities","Land_Slope","Exter_Qual","Exter_Cond","Bldg_Type","Roof_Style",
                          "Roof_Matl", "Mas_Vnr_Type","Sale_Condition", "Overall_Qual", "Gr_Liv_Area", 
                          "Total_Bsmt_SF", "First_Flr_SF", "Full_Bath", "TotRms_AbvGrd","Neighborhood",
                          "Garage_Area", "Foundation", "Misc_Val", "Lot_Frontage","Lot_Area", "Mas_Vnr_Area",
                          "BsmtFin_SF_1", "BsmtFin_SF_2","Bsmt_Unf_SF","Wood_Deck_SF"]
    data_selected = data[high_corr_features]

    #Replace ordinal/categorical variables using numerical mappings
    data_selected["Overall_Qual"] = data_selected["Overall_Qual"].replace(quality_mapping)
    data_selected["Bsmt_Cond"] = data_selected["Bsmt_Cond"].replace(bsmt_cond_mapping)
    data_selected["Bsmt_Qual"] = data_selected["Bsmt_Qual"].replace(bsmt_qual_mapping)
    data_selected["MS_SubClass"] = data_selected["MS_SubClass"].replace(subclass_mapping)
    data_selected["Pool_QC"] = data_selected["Pool_QC"].replace(pool_qc_mapping)
    data_selected["Lot_Config"] = data_selected["Lot_Config"].replace(lot_config_mapping)
    data_selected["Lot_Shape"] = data_selected["Lot_Shape"].replace(lot_shape_mapping)
    data_selected["Utilities"] = data_selected["Utilities"].replace(util_mapping)
    data_selected["Land_Slope"] = data_selected["Land_Slope"].replace(land_slope_mapping)
    data_selected["Exter_Qual"] = data_selected["Exter_Qual"].replace(ext_qual_mapping)
    data_selected["Exter_Cond"] = data_selected["Exter_Cond"].replace(ext_cond_mapping)
    data_selected["Bldg_Type"] = data_selected["Bldg_Type"].replace(bldg_type_mapping)
    data_selected["Roof_Style"] = data_selected["Roof_Style"].replace(roof_style_mapping)
    data_selected["Roof_Matl"] = data_selected["Roof_Matl"].replace(roof_matl_mapping)
    data_selected["Mas_Vnr_Type"] = data_selected["Mas_Vnr_Type"].replace(mas_mapping)
    data_selected['Mas_Vnr_Type'] = data_selected['Mas_Vnr_Type'].fillna(0)
    data_selected["Sale_Condition"] = data_selected["Sale_Condition"].replace(sale_condition_mapping)
    data_selected["Neighborhood"] = data_selected["Neighborhood"].replace(neighborhood_mapping)
    data_selected["Foundation"] = data_selected["Foundation"].replace(foundation_mapping)
    
    #Adding in house age
    data_selected["Age"] = data["Year_Sold"] - data["Year_Built"]

    #adding in boolean for remodel/no remodel
    data_selected['Remodel Yes/No'] = data.apply(lambda row: '1' if row['Year_Built'] != row['Year_Remod_Add'] else '0', axis=1)

    return data_selected

def train_lasso_model():
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')

    # dropping high area outliers as suggested by dataset source document
    x_train = train_data.drop(columns=['PID'])
    x_test = test_data.drop(columns=['PID'])

    y_train = np.log(train_data['Sale_Price'])

    x_train_pp = preprocess_lasso_data(x_train, x_train)
    x_test_pp = preprocess_lasso_data(x_train, x_test)

    x_test_pp = x_test_pp.reindex(columns=x_train_pp.columns)

    #LASSO
    lasso = Lasso(alpha = 0.01)
    lasso.fit(x_train_pp, y_train)

    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

    # define grid
    grid = dict()
    grid['alpha'] = np.arange(0, 1, 0.01)

    # define search
    search = GridSearchCV(lasso, grid, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
    results = search.fit(x_train_pp, y_train)

    lasso_pred = lasso.predict(x_test_pp)

    submission_df = pd.DataFrame({
        'PID': test_data['PID'],
        'Sale_Price': np.exp(lasso_pred)
    })

    # Save the new DataFrame to a text file (comma-separated)
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

  output = pd.DataFrame({"PID" : X_test["PID"], "Sale_Price" : np.exp(yhat) })
  output.to_csv("mysubmission2.txt", index=False)

train_lasso_model()
train_xgb_model()