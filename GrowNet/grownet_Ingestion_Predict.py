#!/usr/bin/env python
import torch
import argparse
import numpy as np
import pandas as pd
from data.sparseloader import DataLoader
from data.data import LibSVMRegData
from models.mlp import MLP_2HL
from models.dynamic_net import DynamicNet, ForwardType
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

def clean_data (df):
    for col in df: #Loop through each column
        df = df[df[col] != '#NV'] # Eliminate rows with '#NV'
        if col != 'Time':
            df[col] = df[col].astype(float) # Float them on down the river

    for i in range (0, np.shape(df)[0]): # Replace values that violate physics
        if df['Ch2_Watts'].iloc[i] > 5: df['Ch2_Watts'].iloc[i] = 5
        elif df['Ch2_Watts'].iloc[i] < 0: df['Ch2_Watts'].iloc[i] = 0
        if df['Ch3_Watts'].iloc[i] > 5: df['Ch3_Watts'].iloc[i] = 5
        elif df['Ch3_Watts'].iloc[i] < 0: df['Ch3_Watts'].iloc[i] = 0

    # df = df.drop(df.index[69501:], axis = 0) # Subset data to range where reactor is critical
    # df = df.drop(df.index[:20000], axis = 0)
    
    return  (df)

def normalize(df):
    meta_mean = {}  # Create dictionaries for mean and standard deviation values
    meta_stdev = {}

    for item in df:
        if item != 'Time':
            mean = df[item].astype(float).mean()  # calculate mean for each variable
            stdev = df[item].astype(float).std()  # calculate standard deviation for each variable
            meta_mean[item] = mean  # Populate dictionary
            meta_stdev[item] = stdev  # Populate dictionary
            if stdev != 0:
                df[item] = (df[item] - mean) / stdev  # Normalize data w/ [(x - mean)/standard deviation]
            else:
                df[item] = 0  # Set the normalized values to 0 if standard deviation is 0

    vars = []  # create empty lists that will be used for creating meta dataframe
    mean = []
    stdev = []
    for item in meta_mean:  # populate dataframe lists
        vars.append(item)
        mean.append(meta_mean[item])
        stdev.append(meta_stdev[item])

    meta = pd.DataFrame(zip(vars, mean, stdev), columns=['vars', 'mean', 'stdev'])  # push lists to dataframe
    meta['R2'] = 'NaN'  # Populate empty scoring column

    return df, meta


def fill_na_with_mean(df):
    for col in df.columns:
        if df[col].isna().any():
            df[col].fillna(df[col].mean(), inplace=True)
    return df


parser = argparse.ArgumentParser()
parser.add_argument('--feat_d', type=int, default=4)
parser.add_argument('--hidden_d', type=int, default=32)
parser.add_argument('--boost_rate', type=float, default=0.5)
parser.add_argument('--model_path', type=str, default='./ckptFin/ML_Ready_Data_FCR_cm_cls.pt')
parser.add_argument('--test_data', type=str, default='./dataPredicting/ML_Ready_Incoming_data_FCR_cm.npz')
parser.add_argument('--normalization', default=True, type=lambda x: (str(x).lower() == 'true')) 
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--sparse', action='store_true')
parser.add_argument('--cv', default=True, type=lambda x: (str(x).lower() == 'true')) 
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--target_variable', type=str, default='FCR_cm')
opt = parser.parse_args()

datapath = '/home/trevea/growNet/GrowNet/Regression/dataPredicting'
df = pd.read_csv(datapath+'/test.csv')
df = clean_data(df) # Clean the data with data cleaning function
print("After clean_data\n", df)
df, meta = normalize(df) # normalize the data with data normalization function
print("After normalization\n", df)
#df, meta = normalize(df) # normalize the data with data normalization function
df = fill_na_with_mean(df) # fill NaN values with the mean of the respective columns


# for col in df:
#     if col != 'Time':
#         y = pd.DataFrame()
#         y[col] = df[col] # populate y with col variable
#         x = df.drop([col, 'Time'], axis = 1) # drop y variable and time variable from x data
#         #print(x)
#         np.savez(opt.test_data.format(col), features=x, labels=y)
for col in df:
    if col != 'Time' and col not in ['inv_period', 'Ch1_CPS']:
        row_index = meta.loc[meta['vars'] == col].index[0] # establish the row index for col variable in the meta data file
        y = pd.DataFrame()
        y[col] = df[col] # populate y with col variable
        x = df.drop([col, 'inv_period', 'Ch1_CPS'], axis = 1) # drop y variable, inv_period, and Ch1_CPS from x data
        x = x.drop(['Time'], axis = 1) # drop time variable from x data
        np.savez(opt.test_data.format(col), features=x, labels=y)
target_variable = opt.target_variable

# Modify the code to select the target variable based on the provided argument
y = pd.DataFrame()
y[target_variable] = df[target_variable] # populate y with target_variable
x = df.drop([target_variable, 'Time', 'inv_period', 'Ch1_CPS'], axis=1) # drop y variable and time variable from x data

# Save the test data according to the chosen target variable
np.savez(opt.test_data.format(target_variable), features=x, labels=y)


# Load the test dataset incoming reactor data 
test = LibSVMRegData(opt.test_data, opt.feat_d, opt.normalization)
test_loader = DataLoader(test, opt.batch_size, shuffle=False, drop_last=False, num_workers=2)

# Load the trained model
net_ensemble = DynamicNet.from_file(opt.model_path, lambda stage: MLP_2HL.get_model(stage, opt))

if opt.cuda:
    net_ensemble.to_cuda()

# Set the model to evaluation mode
net_ensemble.to_eval()

# Make predictions on the test data
y_true = []
y_pred = []
print("len of test loader",len(test_loader))
for x, y in test_loader:
   # x = x[:, :-1]  # Drop the last column from x
    print("x test variable",  x)

    print("y test variable", y)
    if opt.cuda:
        x = x.cuda()

    with torch.no_grad():
        _, out = net_ensemble.forward(x)
    y_true.append(y.cpu().numpy().reshape(len(y), 1))
    out = out.cpu().numpy().reshape(len(y), 1)
    y_pred.append(out)

print("y_true:", y_true)
y_true = np.concatenate(y_true, axis=0)
print("y_pred:", y_pred)
y_pred = np.concatenate(y_pred, axis=0)
#print("Ypred Item", y_pred.item())
# These are the normalzied predictions for each row
# I need to retrain removing some features code is commented on reg_train_test_split.py

# Calculate performance metrics
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

print(f'RMSE: {rmse:.5f}, R2: {r2:.5f}')