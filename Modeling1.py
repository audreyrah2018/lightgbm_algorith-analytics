import seaborn as sns
import pandas as pd  
import lightgbm as lgb
import operator
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
 

def Modeling1(X_train,y_train):
        
    
    model_lgb = lgb.LGBMClassifier(
                n_jobs=4,
                n_estimators=100000,
                objective='multiclass', 
                metric='multi_logloss', 
                boost_from_average='false',
                num_class=11,## we added
                learning_rate=0.01,
                leaves=64,
                max_depth=-1,
                tree_learner = "serial",
                feature_fraction = 0.7,
                bagging_freq = 5,
                bagging_fraction = 0.7,
                min_data_in_leaf=100,
                silent=-1,
                verbose=-1,
                max_bin = 255,
                bagging_seed = 11,
                )
    kf = StratifiedKFold(n_splits=5,shuffle=True,random_state=10)
    
    
    
        
    
    
    
    models =[]
    for i, (train_idx,valid_idx) in enumerate(kf.split(X_train,y_train)):
        print('...... training {}th fold \n'.format(i+1))
        tr_X = X_train.loc[train_idx]
        tr_y = y_train.loc[train_idx]
        
        va_X = X_train.loc[valid_idx]
        va_y = y_train.loc[valid_idx]
        
        model = model_lgb
        model.fit(tr_X, tr_y, eval_set=[(tr_X, tr_y), (va_X, va_y)], eval_metric = 'multi_logloss', verbose=500, early_stopping_rounds = 300)
        models.append(model)
        
     
     
    fts = X_train.columns.values 
    len(fts)
    
    fts_imp = dict(zip(fts,models[1].feature_importances_))
    fts_imp = sorted(fts_imp.items(), key=operator.itemgetter(1),reverse=True)
    fts_imp[:18]    
    feature_imp = pd.DataFrame({'Value':models[1].feature_importances_,'Feature':X_train.columns})
        
        
    plt.figure(figsize=(40, 20))
    sns.set(font_scale = 5)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[:18])
    
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances-01.png')
    plt.show()

   
    
    return 





