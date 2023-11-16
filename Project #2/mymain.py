nib = "MOSTYPE + MAANTHUI + MGEMOMV + MGEMLEEF + MOSHOOFD + MGODRK + MGODPR + MGODOV + MGODGE + MRELGE + MRELSA + MRELOV + MFALLEEN + MFGEKIND + MFWEKIND + MOPLHOOG + MOPLMIDD + MOPLLAAG + MBERHOOG + MBERZELF + MBERBOER + MBERMIDD + MBERARBG + MBERARBO + MSKA + MSKB1 + MSKB2 + MSKC + MSKD + MHHUUR + MHKOOP + MAUT1 + MAUT2 + MAUT0 + MZFONDS + MZPART + MINKM30 + MINK3045 + MINK4575 + MINK7512 + MINK123M + MINKGEM + MKOOPKLA + PWAPART + PWABEDR + PWALAND + PPERSAUT + PBESAUT + PMOTSCO + PVRAAUT + PAANHANG + PTRACTOR + PWERKT + PBROM + PLEVEN + PPERSONG + PGEZONG + PWAOREG + PBRAND + PZEILPL + PPLEZIER + PFIETS + PINBOED + PBYSTAND + AWAPART + AWABEDR + AWALAND + APERSAUT + ABESAUT + AMOTSCO + AVRAAUT + AAANHANG + ATRACTOR + AWERKT + ABROM + ALEVEN + APERSONG + AGEZONG + AWAOREG + ABRAND + AZEILPL + APLEZIER + AFIETS + AINBOED + ABYSTAND"
print(len(nib.split(" +")))
# import pandas as pd
# import numpy as np
# import statsmodels.api as sm
# import patsy

# # Contributors:
# # Ashish Pabba (apabba2 MCS-DS) 
# # Jacob Albus (albus2 On-Campus)
# # We both independently implemented the linear regression approach from campuswire. 
# # Ashish then added the post prediction shift adjustment function, and Jacob added the SVD/PCA smoothing function.

# def myeval():
#     file_path = 'data/test_with_label.csv'
#     test_with_label = pd.read_csv(file_path)

#     file_path = 'mypred.csv'
#     test_pred = pd.read_csv(file_path)
    
#     # Left join with the test data
#     new_test = test_pred.merge(test_with_label, on=['Date', 'Store', 'Dept'], how='left')

#     # Compute the Weighted Absolute Error
#     actuals = new_test['Weekly_Sales']
#     preds = new_test['Weekly_Pred']
#     weights = new_test['IsHoliday_x'].apply(lambda x: 5 if x else 1)

#     weights2 = new_test['IsHoliday_y'].apply(lambda x: 5 if x else 1)
#     print(sum(weights2 * abs(actuals - preds)) / sum(weights2))

#     return sum(weights * abs(actuals - preds)) / sum(weights)

# def preprocess(data):
#     tmp = pd.to_datetime(data['Date'])
#     data['Wk'] = tmp.dt.isocalendar().week
#     data['Yr'] = tmp.dt.year
#     data['Wk'] = pd.Categorical(data['Wk'], categories=[i for i in range(1, 53)])  # 52 weeks 
#     return data

# def PCATransform(train, d=8):
#     train_svd = pd.DataFrame()

#     for dept in train["Dept"].unique():
#         filtered_train = train[train['Dept'] == dept]
#         selected_columns = filtered_train[['Store', 'Date', 'Weekly_Sales']]
        
#         train_dept_ts = selected_columns.pivot(index='Date', columns='Store', values='Weekly_Sales').reset_index().T
#         train_dept_ts = train_dept_ts.fillna(0)

#         sales_data = train_dept_ts.iloc[1:].to_numpy()
#         store_mean = np.mean(sales_data, axis=1)
#         centered_data = (sales_data.T - store_mean).T
#         centered_data = centered_data.astype(float)

#         try:
#             U, S, Vh = np.linalg.svd(centered_data)
#             S = np.diag(S)
#             smooth_sales = np.dot(U[:, :d], np.dot(S[:d, :d], Vh[:d, :]))
#             smooth_sales = (smooth_sales.T + store_mean).T
        
#             train_dept_ts.iloc[1:] = smooth_sales
#             train_dept_ts = train_dept_ts.T
            
#             nib = pd.melt(train_dept_ts, id_vars=['Date'])
#             nib["Dept"] = dept
            
#             train_svd = pd.concat([train_svd, nib], ignore_index=True)
#         except ValueError:
#             continue
            
#     train_svd = train_svd.rename(columns={"value": "Weekly_Sales"})
#     train_svd["Weekly_Sales"] = train_svd["Weekly_Sales"].astype(float)
#     train_svd["Store"] = train_svd["Store"].astype(int)
    
#     return train_svd

# def post_prediction_adjustment(test_pred, shift=1/7):
#     # Define the critical weeks
#     critical_weeks = ['2011-12-16', '2011-12-23', '2011-12-30', '2012-01-06', '2012-01-13']
#     test_pred['Date'] = pd.to_datetime(test_pred['Date'])
#     test_pred['Wk'] = test_pred['Date'].dt.isocalendar().week

#     # average sales for weeks 49, 50, and 51
#     avg_sales_49_51 = test_pred[test_pred['Date'].isin(['2011-12-02', '2011-12-09', '2011-12-16'])].groupby(['Store', 'Dept'])['Weekly_Pred'].mean().reset_index()

#     # average sales for weeks 48 and 52
#     avg_sales_48_52 = test_pred[test_pred['Date'].isin(['2011-11-25', '2011-12-30'])].groupby(['Store', 'Dept'])['Weekly_Pred'].mean().reset_index()

#     merged_avg = pd.merge(avg_sales_49_51, avg_sales_48_52, on=['Store', 'Dept'], how='inner', suffixes=('_49_51', '_48_52'))

#     # departments with sales bulge
#     bulge_depts = merged_avg[merged_avg['Weekly_Pred_49_51'] > 1.1 * merged_avg['Weekly_Pred_48_52']]

    
#     for date in critical_weeks:
#         for _, row in bulge_depts.iterrows():
#             store, dept = row['Store'], row['Dept']
#             current_week_sales = test_pred[(test_pred['Date'] == date) & (test_pred['Store'] == store) & (test_pred['Dept'] == dept)]['Weekly_Pred']
            
#             if not current_week_sales.empty:
#                 test_pred.loc[(test_pred['Date'] == date) & (test_pred['Store'] == store) & (test_pred['Dept'] == dept), 'Weekly_Pred'] *= (1 - shift)
                
#                 next_week = (pd.to_datetime(date) + pd.Timedelta(weeks=1)).strftime('%Y-%m-%d')
#                 test_pred.loc[(test_pred['Date'] == next_week) & (test_pred['Store'] == store) & (test_pred['Dept'] == dept), 'Weekly_Pred'] += current_week_sales.values[0] * shift

#     return test_pred

# def generate_splits(train, test):
#   # Get store/dept pairs that appear in both train and test
#   train_pairs = train[['Store', 'Dept']].drop_duplicates(ignore_index=True)
#   test_pairs = test[['Store', 'Dept']].drop_duplicates(ignore_index=True)
#   unique_pairs = pd.merge(train_pairs, test_pairs, how = 'inner', on =['Store', 'Dept'])

#   # Create design matrix for each store/dept pair
#   train_split = unique_pairs.merge(train, on=['Store', 'Dept'], how='left')
#   train_split = preprocess(train_split)
#   y, X = patsy.dmatrices('Weekly_Sales ~ Weekly_Sales + Store + Dept + Yr  + Wk + I(Yr**2)', 
#                           data = train_split, 
#                           return_type='dataframe')
#   train_split = dict(tuple(X.groupby(['Store', 'Dept'])))

#   test_split = unique_pairs.merge(test, on=['Store', 'Dept'], how='left')
#   test_split = preprocess(test_split)
#   y, X = patsy.dmatrices('Yr ~ Store + Dept + Yr  + Wk + I(Yr**2)', 
#                           data = test_split, 
#                           return_type='dataframe')
#   X['Date'] = test_split['Date']
#   test_split = dict(tuple(X.groupby(['Store', 'Dept'])))

#   return train_split, test_split

# def remove_collinearities(X_train, X_test):
#   # Drop const columns
#   cols_to_drop = X_train.columns[(X_train == 0).all()]
#   X_train = X_train.drop(columns=cols_to_drop)
#   X_test = X_test.drop(columns=cols_to_drop)

#   # Find linearly dependent columns
#   cols_to_drop = []
#   for i in range(len(X_train.columns) - 1, 1, -1):  # Start from the last column and move backward
#     col_name = X_train.columns[i]
#     # Extract the current column and all previous columns
#     tmp_Y = X_train.iloc[:, i].values
#     tmp_X = X_train.iloc[:, :i].values

#     coefficients, residuals, rank, s = np.linalg.lstsq(tmp_X, tmp_Y, rcond=None)
#     if np.sum(residuals) < 1e-10:
#             cols_to_drop.append(col_name)
  
#   # Drop linearly dependent columns
#   return X_train.drop(columns=cols_to_drop), X_test.drop(columns=cols_to_drop)

# if __name__ == "__main__":
#   # pre-allocate a pd to store the predictions
#   test_pred = pd.DataFrame()

#   train = pd.read_csv('train.csv')
#   train = PCATransform(train, d=8) # Smooth data using PCA

#   test = pd.read_csv('test.csv')

#   train_split, test_split = generate_splits(train, test) 
#   keys = list(train_split)

#   # Build model for each store/dept pair
#   for key in keys:
#       X_train = train_split[key]
#       X_test = test_split[key]

#       Y = X_train['Weekly_Sales']
#       X_train = X_train.drop(['Weekly_Sales','Store', 'Dept'], axis=1)

#       X_train, X_test = remove_collinearities(X_train, X_test)

#       tmp_pred = X_test[['Store', 'Dept', 'Date']]
#       X_test = X_test.drop(['Store', 'Dept', 'Date'], axis=1)
      
#       # Build model and predict
#       model = sm.OLS(Y, X_train).fit()
#       mycoef = model.params.fillna(0)
      
#       tmp_pred['Weekly_Pred'] = np.dot(X_test, mycoef)
#       test_pred = pd.concat([test_pred, tmp_pred], ignore_index=True)

#   test_pred['Weekly_Pred'].fillna(0, inplace=True)
#   test_pred = post_prediction_adjustment(test_pred, shift=1/7)
#   test_pred = test_pred.drop(columns=['Wk'])

#   test['Date'] = pd.to_datetime(test['Date'])
#   new_test = test_pred.merge(test, on=['Date', 'Store', 'Dept'], how='left')

#   file_path = 'mypred.csv'
#   new_test.to_csv(file_path, index=False)
#   print(myeval())