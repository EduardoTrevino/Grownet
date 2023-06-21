#!/usr/bin/env python
import numpy as np
import argparse
import copy
import torch
import torch.nn as nn
import time
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from joblib import dump
import warnings
from data.sparseloader import DataLoader
from data.data import LibSVMData, LibCSVData, LibSVMRegData
from data.sparse_data import LibSVMDataSp
from models.mlp import MLP_1HL, MLP_2HL, MLP_3HL
from models.dynamic_net import DynamicNet, ForwardType
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.optim import SGD, Adam
from sklearn.metrics import r2_score



parser = argparse.ArgumentParser()
parser.add_argument('--feat_d', type=int, required=False, default=6)
parser.add_argument('--hidden_d', type=int, required=False, default=32)
parser.add_argument('--boost_rate', type=float, required=False, default=0.5)
parser.add_argument('--lr', type=float, required=False, default=0.0036)
parser.add_argument('--num_nets', type=int, required=False, default=40)
parser.add_argument('--data', type=str, required=False, default="ML_Ready_Data_FCR_cm")
parser.add_argument('--tr', type=str, required=False, default="/home/trevea/growNet/ML_scrips_grownet/Regression/data/ML_Ready_Data_FCR_cm_tr.npz")
parser.add_argument('--te', type=str, required=False, default="/home/trevea/growNet/ML_scrips_grownet/Regression/data/ML_Ready_Data_FCR_cm_te.npz")
parser.add_argument('--out_f', type=str, required=False, default="/home/trevea/growNet/ML_scrips_grownet/Regression/model/GrowNetModel_ML_Ready_Data_FCR_cm.pt")
parser.add_argument('--batch_size', type=int, required=False, default=1024)
parser.add_argument('--epochs_per_stage', type=int, required=False, default=1)
parser.add_argument('--correct_epoch', type=int ,required=False, default=1)
parser.add_argument('--L2', type=float, required=False, default=.0e-3)
parser.add_argument('--sparse', action='store_true')
parser.add_argument('--normalization', default=True, type=lambda x: (str(x).lower() == 'true')) 
parser.add_argument('--cv', default=True, type=lambda x: (str(x).lower() == 'true')) 
parser.add_argument('--cuda', action='store_false')

opt = parser.parse_args()

if not opt.cuda:
    torch.set_num_threads(16)

# prepare the dataset
def get_data():
    if opt.data in ['ML_Ready_Data_Ch1_CPS', 'ML_Ready_Data_CCR_cm', 'ML_Ready_Data_Ch2_Watts', 'ML_Ready_Data_Ch3_Watts', 'ML_Ready_Data_FCR_cm', 'ML_Ready_Data_inv_period', 'ML_Ready_Data_Temp']:
        train = LibSVMRegData(opt.tr, opt.feat_d, opt.normalization)
        test = LibSVMRegData(opt.te, opt.feat_d, opt.normalization)
        val = []
        if opt.cv:
            val = copy.deepcopy(train)
            print('Creating Validation set! \n')
            indices = list(range(len(train)))
            cut = int(len(train)*0.95)
            np.random.shuffle(indices)
            train_idx = indices[:cut]
            val_idx = indices[cut:]

            train.feat = train.feat[train_idx]
            train.label = train.label[train_idx]
            val.feat = val.feat[val_idx]
            val.label = val.label[val_idx]
    else:
        raise ValueError(f"Unsupported data option: {opt.data}")

    if opt.normalization:
        scaler = StandardScaler()
        scaler.fit(train.feat)
        train.feat = scaler.transform(train.feat)
        test.feat = scaler.transform(test.feat)
        if opt.cv:
            val.feat = scaler.transform(val.feat)
    print(f'#Train: {len(train)}, #Val: {len(val)} #Test: {len(test)}')
    return train, test, val


def get_optim(params, lr, weight_decay):
    optimizer = Adam(params, lr, weight_decay=weight_decay)
    #optimizer = SGD(params, lr, weight_decay=weight_decay)
    return optimizer


def root_mse(net_ensemble, loader):
    loss = 0
    total = 0
    y_true = []
    y_pred = []
 
    for x, y in loader:
        if opt.cuda:
            x = x.cuda()

        with torch.no_grad():
            _, out = net_ensemble.forward(x)
        y_true.append(y.cpu().numpy().reshape(len(y), 1))
        out = out.cpu().numpy().reshape(len(y), 1)
        y_pred.append(out)
        loss += mean_squared_error(y_true[-1], out)* len(y)
        total += len(y)
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    rmse = np.sqrt(loss / total)
    variance_y = np.var(y_true)
    r2 = 1 - (rmse / variance_y)
    return rmse, r2



def init_gbnn(train):
    positive = negative = 0
    for i in range(len(train)):
        if train[i][1] > 0:
            positive += 1
        else:
            negative += 1
    blind_acc = max(positive, negative) / (positive + negative)
    print(f'Blind accuracy: {blind_acc}')
    #print(f'Blind Logloss: {blind_acc}')
    return float(np.log(positive / negative))

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

    df = df.drop(df.index[69501:], axis = 0) # Subset data to range where reactor is critical
    df = df.drop(df.index[:20000], axis = 0)
    
    return  (df)

def normalize (df):
    meta_mean = {} # Create dictionaries for mean and standard deviation values
    meta_stdev = {}

    for item in df:
        if item != 'Time':
            mean = df[item].astype(float).mean() # calclulate mean for each variable
            stdev = df[item].astype(float).std()# calclulate standard deviation for each variable
            meta_mean[item] = mean # Populate dictionary
            meta_stdev[item] = stdev # Populate dictionary
            df[item] = (df[item] - mean) / stdev # Normalize data w/ [(x - mean)/standard deviation]
            
    vars = [] #create empty lists that will be used for creating meta dataframe
    mean = []
    stdev = []
    for item in meta_mean: # populate dataframe lists
        vars.append(item)
        mean.append(meta_mean[item])
        stdev.append(meta_stdev[item])

    meta = pd.DataFrame(zip(vars, mean, stdev), columns = ['vars', 'mean', 'stdev']) # push lists to dataframe
    meta['R2'] = 'NaN' # Populate empty scoring column

    return (df, meta)

if __name__ == "__main__":
    datapath = '/home/trevea/growNet/ML_scrips_grownet/Regression/data'
    df = pd.read_csv(datapath+'/ML_Ready_Data.csv')
    df = clean_data(df) # Clean the data with data cleaning function
    df, meta = normalize(df) # normalize the data with data normalization function
    meta_df = pd.DataFrame(columns=['vars', 'mean', 'stdev', 'R2', 'RMSE'])
    warnings.filterwarnings("ignore") # ignore all warning messages  

    for col in df:
        if col != 'Time':
            row_index = meta.loc[meta['vars'] == col].index[0] # establish the row index for col variable in the meta data file

            y1 = pd.DataFrame()
            y1[col] = df[col] # populate y with col variable
            x1 = df.drop([col], axis = 1) # drop y variable from x data
            x1 = x1.drop(['Time'], axis = 1) # drop time variable from x data
            x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size = 0.2, random_state=8675309) # split the data
            np.savez(datapath + '/ML_Ready_Data_{}_tr.npz'.format(col), features=x_train, labels=y_train)
            np.savez(datapath + '/ML_Ready_Data_{}_te.npz'.format(col), features=x_test, labels=y_test)
            opt.data = "ML_Ready_Data_" + col
            opt.tr = "/home/trevea/growNet/ML_scrips_grownet/Regression/data/ML_Ready_Data_" + col + "_tr.npz"
            opt.te = "/home/trevea/growNet/ML_scrips_grownet/Regression/data/ML_Ready_Data_" + col + "_te.npz"
            opt.out_f = "/home/trevea/growNet/ML_scrips_grownet/Regression/model/GrowNetModel_ML_Ready_Data_" + col + ".pt"
            print(opt.data)
            print(opt.tr)
            print(opt.te)
            print(opt.out_f)

            train, test, val = get_data()
            N = len(train)
            print(opt.data + ' training and test datasets are loaded!')
            train_loader = DataLoader(train, opt.batch_size, shuffle=True, drop_last=False, num_workers=2)
            test_loader = DataLoader(test, opt.batch_size, shuffle=False, drop_last=False, num_workers=2)
            if opt.cv:
                val_loader = DataLoader(val, opt.batch_size, shuffle=True, drop_last=False, num_workers=2)
            best_rmse = pow(10, 6)
            val_rmse = best_rmse
            best_stage = opt.num_nets-1
            c0 = np.mean(train.label)  #init_gbnn(train)
            net_ensemble = DynamicNet(c0, opt.boost_rate)
            #net_ensemble.cuda()
            loss_f1 = nn.MSELoss()
            loss_models = torch.zeros((opt.num_nets, 3))
            #loss_models = loss_models.cuda()
            for stage in range(opt.num_nets):
                t0 = time.time()
                model = MLP_2HL.get_model(stage, opt)  # Initialize the model_k: f_k(x), multilayer perception v2
                if opt.cuda:
                    model.cuda()

                optimizer = get_optim(model.parameters(), opt.lr, opt.L2)
                net_ensemble.to_train() # Set the models in ensemble net to train mode
                stage_mdlloss = []
                for epoch in range(opt.epochs_per_stage):
                    for i, (x, y) in enumerate(train_loader):
                        
                        if opt.cuda:
                            x= x.cuda()
                            y = torch.as_tensor(y, dtype=torch.float32).cuda().view(-1, 1)
                        middle_feat, out = net_ensemble.forward(x)
                        out = torch.as_tensor(out, dtype=torch.float32).cuda().view(-1, 1)
                        #y = y.cuda()
                        #out = out.to('cuda')
                        grad_direction = -(out-y)

                        _, out = model(x, middle_feat)
                        out = torch.as_tensor(out, dtype=torch.float32).cuda().view(-1, 1)
                        out = out.cuda()
                        loss = loss_f1(net_ensemble.boost_rate*out, grad_direction)  # T

                        model.zero_grad()
                        #model = model.to('cuda')
                        loss.backward()
                        optimizer.step()
                        stage_mdlloss.append(loss.item()*len(y))

                net_ensemble.add(model)
                sml = np.sqrt(np.sum(stage_mdlloss)/N)
                


                lr_scaler = 3
                # fully-corrective step
                stage_loss = []
                if stage > 0:
                    # Adjusting corrective step learning rate 
                    if stage % 15 == 0:
                        #lr_scaler *= 2
                        opt.lr /= 2
                        opt.L2 /= 2
                    optimizer = get_optim(net_ensemble.parameters(), opt.lr / lr_scaler, opt.L2)
                    for _ in range(opt.correct_epoch):
                        stage_loss = []
                        for i, (x, y) in enumerate(train_loader):
                            if opt.cuda:
                                x, y = x.cuda(), y.cuda().view(-1, 1)
                            _, out = net_ensemble.forward_grad(x)
                            out = torch.as_tensor(out, dtype=torch.float32).cuda().view(-1, 1)
                            out = out.to('cuda')
                            loss = loss_f1(out, y) 
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            stage_loss.append(loss.item()*len(y))
                elapsed_tr = time.time()-t0
                sl = 0
                if stage_loss != []:
                    sl = np.sqrt(np.sum(stage_loss)/N)

                print(f'Stage - {stage}, training time: {elapsed_tr: .1f} sec, model MSE loss: {sml: .5f}, Ensemble Net MSE Loss: {sl: .5f}')

                net_ensemble.to_file(opt.out_f)
                net_ensemble = DynamicNet.from_file(opt.out_f, lambda stage: MLP_2HL.get_model(stage, opt))

                if opt.cuda:
                    net_ensemble.to_cuda()
                net_ensemble.to_eval() # Set the models in ensemble net to eval mode

                # Train
                tr_rmse, tr_r2 = root_mse(net_ensemble, train_loader)
                if opt.cv:
                    val_rmse, val_r2 = root_mse(net_ensemble, val_loader) 
                    if val_rmse < best_rmse:
                        best_rmse = val_rmse
                        best_stage = stage

                te_rmse, te_r2 = root_mse(net_ensemble, test_loader)

                print(f'Stage: {stage}  RMSE@Tr: {tr_rmse:.5f} R2@Tr: {tr_r2:.5f}, RMSE@Val: {val_rmse:.5f} R2@Val: {val_r2:.5f}, RMSE@Te: {te_rmse:.5f} R2@Te: {te_r2:.5f}')


                loss_models[stage, 0], loss_models[stage, 1] = tr_rmse, te_rmse

            tr_rmse, te_rmse = loss_models[best_stage, 0], loss_models[best_stage, 1]
            print(f'Best validation stage: {best_stage}  RMSE@Tr: {tr_rmse:.5f}, final RMSE@Te: {te_rmse:.5f}')
            print(f'R2@Tr: {tr_r2:.5f}, final R2@Te: {te_r2:.5f}')
            loss_models = loss_models.detach().cpu().numpy()
            fname =  './results/' + opt.data +'_rmse.csv'
            np.savetxt(fname, loss_models, delimiter=",")
    #         meta_df = meta_df.append({
    #             'vars': col,
    #             'mean': mean,
    #             'stdev': stdev,
    #             'R2': R2,
    #             'RMSE': RMSE
    #         }, ignore_index=True)
    # meta_df.to_csv('/home/trevea/growNet/ML_scrips_grownet/Regression/GrowNetmeta.csv', index=False)

