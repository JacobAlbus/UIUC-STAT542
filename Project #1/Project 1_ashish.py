#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler,OneHotEncoder, LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold


# import matplotlib.pyplot as plt # remember to comment this out


# from sklearn.model_selection import train_test_split


# In[12]:


#LOAD DATA

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
#dropping high area outliers as suggested by dataset source document
# train_data = train_data[train_data['Gr_Liv_Area'] <= 4000]

x_train = train_data.drop(columns=['PID', 'Sale_Price'])
x_test = test_data.drop(columns=['PID'])




y_train = np.log(train_data['Sale_Price'])

y_test = pd.read_csv('test_y.csv')
y_test = y_test.drop(columns=['PID'])
y_test = np.log(y_test["Sale_Price"])


#neighborhood binning mapping
neighborhood_median_prices = train_data.groupby('Neighborhood')['Sale_Price'].median().reset_index()
neighborhood_median_prices.sort_values(by='Sale_Price', inplace=True)
neighborhood_mapping = {neighborhood: (label//4) for label, neighborhood in enumerate(neighborhood_median_prices['Neighborhood'])}

#MS_SubClass mapping
subclass_median_prices = train_data.groupby('MS_SubClass')['Sale_Price'].median().reset_index()
subclass_median_prices.sort_values(by='Sale_Price', inplace=True)
subclass_mapping = {subclass: (label//4) for label, subclass in enumerate(subclass_median_prices['MS_SubClass'])}

#Pool Quality mapping
pool_qc_mapping = {"No_Pool" : 0,"Fair" : 1, "Good" : 1,"Typical" : 1, "Excellent" : 3}

#Overall Quality mapping
quality_order  = ["Very_Poor","Poor", "Fair","Below_Average","Average","Above_Average", "Good", "Very_Good", "Excellent", "Very_Excellent"]
quality_mapping = {categorical: i for i, categorical in enumerate(quality_order)}

#Sale Condition mapping
sale_condition_mapping = {"AdjLand" : 1, "Abnorml" : 2,"Family" : 2,"Alloca" : 3,"Normal" : 3, "Partial" : 4}

#Lot_Config mapping
lot_config_mapping = {"FR2" : 1, "Inside" : 1, "Corner" : 1, "FR3" : 2, "CulDSac" : 3}

#Bldg_Type mapping
bldg_type_mapping = {"TwoFmCon": 1 , "Twnhs" : 1, "Duplex" : 1, "OneFam" : 2 , "TwnhsE" : 2.5}

#Roof_Style mapping
roof_style_mapping = {"Gambrel":1, "Gable" : 2 ,"Mansard" : 2, "Flat" :2.5 , "Hip" : 2.5, "Shed" : 3.5 } 

#Roof_Matl mapping
roof_matl_mapping = {"ClyTile" : 0, "Roll" : 0.8, "CompShg" : 1,"Tar&Grv" : 1,"Metal" : 1, "Membran" : 2, "WdShngl": 2.25, "WdShake" : 2.25}

#Mas_Vnr_Type mapping
mas_mapping = {"None": 0,"BrkCmn" : 1, "CBlock" : 1.5, "BrkFace" : 2, "Stone" : 3}

#lot_shape_mapping
lot_shape_mapping = {"Regular" : 1, "Slightly_Irregular" : 2, "Moderately_Irregular" : 2.75, "Irregular" : 3.25}

#Utilities mapping
util_mapping  = {"ELO" : 0,"NoSeWa" : 0.5,"NoSewr" : 1.25, "AllPub" : 2}

#Land_Slope mapping
land_slope_mapping = {"Gtl" : 1, "Mod" : 2, "Sev" : 3}

#Exterior Quality mapping
ext_qual_mapping = {"Poor" : 1, 'Fair': 2, 'Typical': 3, 'Good': 4, 'Excellent': 5.5}

#Exterior condition mapping
ext_cond_mapping = {"Poor" : 1, 'Fair': 2, 'Typical': 3.5, 'Good': 4, 'Excellent': 5}

#Foundation mapping
foundation_mapping = {"Slab" : 1, "Stone" : 1,"BrkTil" : 1,"CBlock" : 1, "PConc" : 1.75, "Wood" : 2}

#Bsmt quality mapping
bsmt_qual_mapping = {"Poor" : 1 , "Fair" : 1, "No_Basement" : 1.5, "Typical" : 2, "Good" : 2.5, "Excellent" : 3.25}

#Bsmt_Cond mapping
bsmt_cond_mapping = {"Poor" : 1 , "Fair" : 1, "No_Basement" : 1.5, "Typical" : 2, "Good" : 2.5, "Excellent" : 3.25}



# high_corr_features = ["Bsmt_Cond", "Bsmt_Qual","MS_SubClass", "Pool_QC", "Lot_Config","Lot_Shape", "Utilities","Land_Slope","Exter_Qual","Exter_Cond","Bldg_Type","Roof_Style","Roof_Matl", "Mas_Vnr_Type","Sale_Condition", "Overall_Qual", "Gr_Liv_Area", "Total_Bsmt_SF", "First_Flr_SF", "Full_Bath", "TotRms_AbvGrd","Neighborhood","Garage_Area", "Foundation"]
# selected_train_data = train_data[high_corr_features]


# nan_counts = selected_train_data.isna().sum()

# # Print the counts
# print(nan_counts)


# df_filtered = train_data[train_data['Gr_Liv_Area'] > 4000]
# print(df_filtered)


# In[13]:


# # Group the data by neighborhood



# neighborhood_stats = train_data.groupby('Sale_Type')['Sale_Price'].agg(['median', 'mean'])

# neighborhood_stats = neighborhood_stats.sort_values(by='median')

# # Plot the median sales prices
# plt.figure(figsize=(12, 6))  # Adjust the figure size as needed
# plt.bar(neighborhood_stats.index, neighborhood_stats['median'])

# # Customize the plot
# plt.title('Median Sales Price by Neighborhood (Sorted by Mean Sales Price)')
# plt.xlabel('Neighborhood')
# plt.ylabel('Median Sales Price')
# plt.xticks(rotation=90)  # Rotate x-axis labels for better readability

# # Show the plot
# plt.show()


# In[14]:


#PREPROCESS DATA

def preprocess_data(data):
    high_corr_features = ["Bsmt_Cond", "Bsmt_Qual","MS_SubClass", "Pool_QC", "Lot_Config","Lot_Shape", "Utilities","Land_Slope","Exter_Qual","Exter_Cond","Bldg_Type","Roof_Style","Roof_Matl", "Mas_Vnr_Type","Sale_Condition", "Overall_Qual", "Gr_Liv_Area", "Total_Bsmt_SF", "First_Flr_SF", "Full_Bath", "TotRms_AbvGrd","Neighborhood","Garage_Area", "Foundation", "Misc_Val", "Lot_Frontage","Lot_Area", "Mas_Vnr_Area","BsmtFin_SF_1", "BsmtFin_SF_2","Bsmt_Unf_SF","Wood_Deck_SF"]
    data_selected = data[high_corr_features]
    # nan_df = data_selected.isna()
    # print(nan_df)

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

    # scaler = StandardScaler()
    
    # # Fit the scaler on the training data and transform both training and test data
    # data_selected = scaler.fit_transform(data_selected)

    



    
    
    
    
    
    
    # OHE_data = pd.get_dummies(data_selected, columns = ['Neighborhood']) 
    # neigh_cols = [col for col in OHE_data.columns if col.startswith('Neighborhood_')]
    
    # extra_cols = set(neigh_cols_full) - set(neigh_cols)
    
    # if extra_cols:
    #     for extra_column in extra_cols:
    #         OHE_data[extra_column] = False

    # neigh_cols = [col for col in OHE_data.columns if col.startswith('Neighborhood_')]
    
    # scaler = StandardScaler()
    # data_scaled = scaler.fit_transform(data_selected)
    return data_selected


# In[15]:


x_train_pp = preprocess_data(x_train)
x_test_pp = preprocess_data(x_test)

x_test_pp = x_test_pp.reindex(columns=x_train_pp.columns)

print(x_train_pp.shape)
print(x_test_pp.shape)

# cols_diff = set([col for col in x_train_pp.columns]) - set([col for col in x_test_pp.columns])
# cols_diff2 = set([col for col in x_test_pp.columns]) - set([col for col in x_train_pp.columns])

# print(set([col for col in x_train_pp.columns]))
# print(cols_diff)
# print(cols_diff2)

# feature_names_train = x_train_pp.columns.tolist()
# feature_names_test = x_test_pp.columns.tolist()

# print(feature_names_train)
# print(feature_names_test)
x_train_pp.head()



# In[16]:


#LASSO

lasso = Lasso(alpha = 0.01)
lasso.fit(x_train_pp, y_train)

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# define grid
grid = dict()
grid['alpha'] = np.arange(0, 1, 0.01)
# define search
search = GridSearchCV(lasso, grid, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
# perform the search
results = search.fit(x_train_pp, y_train)
# summarize
print('MAE: %.3f' % results.best_score_)
print('Config: %s' % results.best_params_)


# In[17]:


lasso_pred = lasso.predict(x_test_pp)
# print(lasso_pred.shape)
# print(y_test)

mse = mean_squared_error(lasso_pred, y_test)
rmse = np.sqrt(mse)
print(rmse)







# In[18]:


#Save predictions

submission_df = pd.DataFrame({
    'PID': test_data['PID'],
    'Sale_Price': np.exp(lasso_pred)
})

print(submission_df)
# Save the new DataFrame to a text file (comma-separated)
submission_df.to_csv('mysubmission1.txt', index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




