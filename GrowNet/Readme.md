- Data loading and creating dataloader are handled in GrowNet/Regression/data/data.py. If you want to try new data please check the LibSVMRegdata function in data.py for the right format. 

- Individual model class and ensemble architecture are in GrowNet/Reg/models:  mlp.py and dynamic_net.py. 
You can increase number of hidden layers or change activation function from here: mlp.py