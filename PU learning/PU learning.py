import numpy as np
import pandas as pd 
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score,cross_validate 
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.svm import SVC,LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef,accuracy_score,auc,precision_score,recall_score,f1_score,roc_curve,make_scorer

x_pos_n=x_pos.loc[notin(x_pos.index,intersection(X_test.index,x_pos.index)),:]
x_neg_n=x_neg.loc[notin(x_neg.index,intersection(X_test.index,x_neg.index)),:]

x_train_m=pd.concat([x_pos_n,x_neg_n])
y_m=np.repeat([1,0], [x_pos_n.shape[0],x_neg_n.shape[0]])
y_m=pd.DataFrame( y_m,index=x_train_m.index)



num_oob_m = pd.DataFrame(np.zeros(shape = y_m.shape), index = y_m.index)
sum_oobRF_m = pd.DataFrame(np.zeros(shape = y_m.shape), index = y_m.index)
sum_oobSVM_m = pd.DataFrame(np.zeros(shape = y_m.shape), index = y_m.index)
sum_oobLR_m = pd.DataFrame(np.zeros(shape = y_m.shape), index = y_m.index)
sum_oobX_m = pd.DataFrame(np.zeros(shape = y_m.shape), index = y_m.index)

iP_m = y_m[y_m.iloc[:,0] > 0].index
iU_m = y_m[y_m.iloc[:,0] <= 0].index

results_m = pd.DataFrame({
    
    'label'      : y_m.iloc[:,0]                        
    
}, columns = [ 'label'])


n_estimators = 100
estimatorRF = RandomForestClassifier(
    n_estimators = 1000,  # 1000 trees
    n_jobs = -1           # Use all CPU cores
)
estimatorSVM=SVC(kernel='linear', probability=True)
estimatorLR=LogisticRegression()
estimatorX=XGBClassifier(eval_metric='mlogloss')

for _ in range(n_estimators):
    # Get a bootstrap sample of unlabeled points for this round
    ib = np.random.choice(iU_m, replace = True, size = len(iP_m))
    #ib = np.random.choice(iU, replace = True, size = int(len(iU)/3))
    # Find the OOB data points for this round
    i_oob = list(set(iU_m) - set(ib))

    # Get the training data (ALL positives and the bootstrap
    # sample of unlabeled points) and build the tree
    Xb = x_train_m[y_m.iloc[:,0] > 0].append(x_train_m.loc[ib])
    yb = y_m[y_m.iloc[:,0] > 0].append(y_m.loc[ib])
    estimatorRF.fit(Xb, yb.iloc[:,0])
    estimatorSVM.fit(Xb, yb.iloc[:,0])
    estimatorLR.fit(Xb, yb.iloc[:,0])
    estimatorX.fit(Xb, yb.iloc[:,0])
    # Record the OOB scores from this round
    sum_oobRF_m.loc[i_oob, 0] += estimatorRF.predict_proba(x_train_m.loc[i_oob])[:,1]
    sum_oobSVM_m.loc[i_oob, 0] += estimatorSVM.predict_proba(x_train_m.loc[i_oob])[:,1]
    sum_oobLR_m.loc[i_oob, 0] += estimatorLR.predict_proba(x_train_m.loc[i_oob])[:,1]
    sum_oobX_m.loc[i_oob, 0] += estimatorX.predict_proba(x_train_m.loc[i_oob])[:,1]
    num_oob_m.loc[i_oob, 0] += 1

results_m['num_oob'] =  num_oob_m
results_m['output_bag_RF'] = sum_oobRF_m / num_oob_m
results_m['output_bag_SVM'] = sum_oobSVM_m / num_oob_m
results_m['output_bag_LR'] = sum_oobLR_m / num_oob_m
results_m['output_bag_X'] = sum_oobX_m / num_oob_m
results_m['output_bag_ALL']=(sum_oobRF_m+sum_oobSVM_m+sum_oobLR_m+sum_oobX_m)/(num_oob_m*4)


x_neg_n1=x_neg_n.loc[results_m['output_bag_ALL'][results_m['output_bag_ALL']<0.5].index,:]
x_neg1_=x_neg_n1.loc[ni1,:]
x_neg_n1=x_neg_n1.loc[notin(results_m['output_bag_ALL'][results_m['output_bag_ALL']<0.5].index,ni1),:]
x_neg2_=x_neg_n1.iloc[np.random.choice(range(x_neg_n1.shape[0]), size = (x_pos_n.shape[0])-len(ni1)),]
x_neg_r=pd.concat([x_neg1_,x_neg2_])


X=pd.concat([x_pos_n,x_neg_r])
y=np.repeat([1,0], [x_pos_n.shape[0],x_neg_r.shape[0]])