import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def feature_engineering(dft):
 
      
                   
    dft['id']=dft['id'].fillna(method='ffill')
    
    
    dft['age'].fillna(dft['age'].median(),inplace=True)
    dft['age'].loc[(dft['age']>90) | (dft['age']<18)] = dft['age'].median()
    
    dft['first_affiliate_tracked'].interpolate(method='ffill',inplace=True)
    dft['first_affiliate_tracked'].replace(np.nan, 0, inplace=True)
    dft['first_affiliate_tracked'].drop( 0, axis=0, inplace=True)
    
    dg=pd.get_dummies(dft['gender'])
    dg.drop(['-unknown-','OTHER'],axis=1,inplace=True)
    
    dft= dft.join(dg)
    
    dft.drop(['gender'],axis=1, inplace =True)
    
    dsm=dft['signup_method'].astype(str)
    le1 = LabelEncoder()
    dft['signup_method']=le1.fit_transform(dsm)
    
    dc=dft['country_destination'].astype(str)
    le2 = LabelEncoder()
    dft['country_destination']=le2.fit_transform(dc)
    
    dl=dft['language'].astype(str)
    le3 = LabelEncoder()
    dft['language']=le3.fit_transform(dl)
    
    daf=dft['affiliate_channel'].astype(str)
    le4 = LabelEncoder()
    dft['affiliate_channel']=le4.fit_transform(daf)
    
    dafp=dft['affiliate_provider'].astype(str)
    le5 = LabelEncoder()
    dft['affiliate_provider']=le5.fit_transform(dafp)
    
    fat=dft['first_affiliate_tracked'].astype(str)
    le6 = LabelEncoder()
    dft['first_affiliate_tracked']=le6.fit_transform(fat)
    
    sapp=dft['signup_app'].astype(str)
    le7 = LabelEncoder()
    dft['signup_app']=le7.fit_transform(sapp)
    
    fdt=dft['first_device_type'].astype(str)
    le8 = LabelEncoder()
    dft['first_device_type']=le8.fit_transform(fdt)
    
      
    le9 = LabelEncoder()
    dft['first_browser']=le9.fit_transform(dft['first_browser'])
    
    dft['date_account_created']=pd.to_datetime(dft.date_account_created)
    dft['date_account_created'].fillna(method='ffill')
    
    dft['Month_created']=dft.date_account_created.dt.month
    
    dft['Day_created']=dft.date_account_created.dt.day
    
    
    dft['date_first_booking']=pd.to_datetime(dft.date_first_booking)
    dft['date_first_booking'].fillna(method='ffill')
    
    dft['Month_booking']=dft.date_first_booking.dt.month
    
    
    dft['Day_boking']=dft.date_first_booking.dt.day
    
    
    dft.drop(['date_first_booking','first_affiliate_tracked'],axis=1, inplace =True)
    
    DATA_PATH = r'C:\Users\Davis\Desktop\Dat and aud Hotel'
    dfs = pd.read_csv(DATA_PATH+r'\sessions.csv' , header=0)
    
    dfs.rename(columns = {'user_id': 'id'}, inplace=True)
    dfs['action'].fillna('unknown', inplace=True)
    dfs['action_type'].fillna('unknown', inplace=True)
    dfs['action_detail'].fillna('unknown', inplace=True)
    dfs['secs_elapsed'].fillna(dfs['secs_elapsed'].mean(), inplace=True)
    dfs['id']=dfs['id'].fillna(method='ffill')
    
    
    
    
    dfs_Group=dfs.groupby('id')['secs_elapsed'].sum()
    
  
    
    
       
    merg=dft.merge(dfs_Group, on='id', how='left')
    merg['date_account_created']=pd.to_datetime(merg['date_account_created'])
    merg['date_account_created'].fillna(method='ffill')   
    merg['Month_created']=merg.date_account_created.dt.month        
    merg['Day_created']=merg.date_account_created.dt.day    
    merg.drop(['date_account_created'],axis=1, inplace =True)
    
    
    
    merg['Month_booking'].fillna(method='ffill', inplace=True)
    merg['Day_boking'].fillna(method='ffill', inplace=True)
    merg['secs_elapsed'].fillna(merg['secs_elapsed'].median(), inplace=True)
    
    
    
    merg.drop('id',axis=1, inplace=True)
    
    
    features=merg.drop( 'country_destination',axis=1)
    target=merg['country_destination']
    
    
    X = features
    y = target
    print(X.shape,len(y))  
    
    X_train,X_valid, y_train,y_valid = train_test_split(X,y,shuffle=True,random_state=10, test_size=0.1)
    print(X_train.shape,X_valid.shape,len(y_train),len(y_valid))
    
    X.shape,len(y)
    
    X_train,X_test, y_train,y_test = train_test_split(X,y,shuffle=True,random_state=10, test_size=0.1)
    print(X_train.shape,X_test.shape,len(y_train),len(y_test))
    
    X_train.reset_index(drop=True,inplace=True)
    X_test.reset_index(drop=True,inplace=True)
    y_train.reset_index(drop=True,inplace=True)
    y_test.reset_index(drop=True,inplace=True)
    
      
    return X_train, y_train
