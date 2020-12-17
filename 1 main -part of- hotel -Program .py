
import pandas as pd  
from feature_engineering import feature_engineering
from Modeling1 import Modeling1

    
DATA_PATH = r'C:\Users\Davis\Desktop\Dat and aud Hotel'
dft = pd.read_csv(DATA_PATH+r'\train.csv', header=0) 


#functions recall
X_train,y_train =feature_engineering(dft)
feature_imp=Modeling1(X_train,y_train)










