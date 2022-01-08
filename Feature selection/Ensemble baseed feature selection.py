import numpy as np
import pandas as pd 
from sklearn.feature_selection import SelectKBest, f_classif ,mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef,accuracy_score,auc,precision_score,recall_score,f1_score,roc_curve,make_scorer
import pymrmr  

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def notin(lst1, lst2):
    lst3 = [value for value in lst1 if value not in lst2]
    return lst3

def flatten_list(_2d_list):
    flat_list = []
    # Iterate through the outer list
    for element in _2d_list:
        if type(element) is list:
            # If the element is of type list, iterate through the sublist
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list


X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=0.15, random_state=333,stratify=y_train)


Y_mrmr=pd.DataFrame( Y_train,index=X_train.index,columns=['Y'])
X_mrmr=pd.concat([Y_mrmr,X_train], axis=1)
mrmr_sf=pymrmr.mRMR(X_mrmr, 'MIQ', 500)


anova_fit = SelectKBest(score_func=f_classif , k=500).fit(X_train, Y_train)
mi_fit=SelectKBest(score_func=mutual_info_classif , k=500).fit(X_train, Y_train)

anova_sf=flatten_list(x_train.columns[anova_fit.get_support()])
mi_sf=flatten_list(x_train.columns[mi_fit.get_support()])



sf=intersection(mrmr_sf,intersection(anova_sf,mi_sf))


X_train1=x_train.loc[:,sf]
X_test1=X_test.loc[:,sf]