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

def get_stacking():
	# define the base models
	level0 = list()
	level0.append(('lr', LogisticRegression()))
	level0.append(('nn', MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(10, 50,10), random_state=1)))
	level0.append(('rf', RandomForestClassifier(random_state=3)))
# 	level0.append(('svm', SVC()))
	level0.append(('knn', KNeighborsClassifier()))
# 	level0.append(('XGB', XGBClassifier(eval_metric='mlogloss')))
	# define meta learner model
	level1 = LogisticRegression()
# 	level1 = XGBClassifier()
# 	level1 =  SVC()
	# define the stacking ensemble
	model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5,stack_method='predict_proba')
	return model


def get_models():
	models = dict()
	models['lr'] = LogisticRegression()
	models['nn'] = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 50,10), random_state=1)
	models['rf'] = RandomForestClassifier(random_state=3)
# 	models['svm'] = SVC()
# 	models['XGB'] = XGBClassifier(eval_metric='mlogloss')
	models['knn'] = KNeighborsClassifier()   
	models['stacking'] = get_stacking()
# 	models['voting'] = get_voting()
	return models

# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
	scoring = ['accuracy', 'recall','precision', 'f1' , 'roc_auc']
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1,)
	scores = cross_validate(model, X, y, scoring=scoring, cv=cv, n_jobs=-1, error_score='raise')
	return scores

def evaluate_model_r(model, X, y,r):
	scoring = ['accuracy', 'recall','precision', 'f1' , 'roc_auc']
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=r,)
	scores = cross_validate(model, X, y, scoring=scoring, cv=cv, n_jobs=-1, error_score='raise')
	return scores

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)

# get the models to evaluate
scores_all=pd.DataFrame()
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
	scores = evaluate_model(model, X_2, y_2)
	results.append(scores['test_accuracy'])
	names.append(name)
	for s in scores:
		if (s=='fit_time' or s=='score_time'):
			continue
#		print('>%s %s %.3f (%.3f)' % (name,s, mean(scores[s]), std(scores[s])))
		scores_all.loc[name,s]= np.round(mean(scores[s]),4)

print(scores_all)

# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()



poind_test=np.where(Y_test==1)[0]
negind_test=np.where(Y_test==0)[0]
X_test_neg=X_test1.iloc[negind_test,:]
X_test_pos=X_test1.iloc[poind_test,:]
X_test_r=X_test_neg.iloc[np.random.choice(range(X_test_neg.shape[0]), size = X_test_pos.shape[0]),]


for i in range(50):
    X_test_r=X_test_neg.iloc[np.random.choice(range(X_test_neg.shape[0]), size = X_test_pos.shape[0]),]
    X_test2=pd.concat([X_test_pos,X_test_r])
    Y_test2=np.repeat([1,0], [X_test_pos.shape[0],X_test_r.shape[0]])
    scoring_all_valid=pd.DataFrame()
    preds=pd.DataFrame()
    models = get_models()
    # evaluate the models and store results
    results, names = list(), list()
    for name, model in models.items():
        if i==0:
            sc=scoring_all_valid
        model.fit(X1, y1)
        y_pred=model.predict(X_test2)
        mes=perf_measure(Y_test2,y_pred)
        mcc= matthews_corrcoef(y_true= Y_test2, y_pred= y_pred)
        f1=f1_score(y_true= Y_test2, y_pred= y_pred)
        acc=accuracy_score(y_true=Y_test2, y_pred= y_pred)
        recall=recall_score(y_true=Y_test2, y_pred= y_pred)
        pre=precision_score(y_true=Y_test2, y_pred= y_pred)
        fpr1, tpr1, thresholds = roc_curve( Y_test2, y_pred)
        auc1=auc(fpr1, tpr1)
    #     scoring_all_valid.loc[name,'feature method']=st
        scoring_all_valid.loc[name,'# of features']=X.shape[1]
        scoring_all_valid.loc[name,'# of positive']=sum(Y_test2==1)
        scoring_all_valid.loc[name,'# of negtive']=sum(Y_test2==0)
        scoring_all_valid.loc[name,'TP']=mes[0]
        scoring_all_valid.loc[name,'FP']=mes[1]
        scoring_all_valid.loc[name,'TN']=mes[2]
        scoring_all_valid.loc[name,'FN']=mes[3]
        scoring_all_valid.loc[name,'Accuracy']= np.round(acc,4)
        scoring_all_valid.loc[name,'Recall']= np.round(recall,4)
        scoring_all_valid.loc[name,'Precision']= np.round(pre,4)
        scoring_all_valid.loc[name,'F1']= np.round(f1,4)
        scoring_all_valid.loc[name,'AUC']= np.round(auc1,4)
        scoring_all_valid.loc[name,'MCC']= np.round(mcc,4)
        preds[name]=y_pred
    
    print(scoring_all_valid)
    sc=sc+scoring_all_valid
#     sc.add(sc1)
print(sc/50)    

#Testing 

test_data1=pd.read_csv("datasets/test_data.csv",sep=',',header = 'infer',index_col="#")
n_p_t=46
n_n_t=179
test_data2=test_data1.loc[:,sf33]
scaled_st = StandardScaler().fit_transform(test_data2)
x_tee=pd.DataFrame(scaled_st, index=test_data2.index, columns=test_data2.columns)
y_tee=np.repeat([1,0], [n_p_t,n_n_t])


scoring_all_test=pd.DataFrame()
preds=pd.DataFrame()
models = get_models()
auc_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
    model.fit(X_2, y_2)
    y_pred=model.predict(x_tee)
    mes=perf_measure(y_tee,y_pred)
    mcc= matthews_corrcoef(y_true= y_tee, y_pred= y_pred)
    f1=f1_score(y_true= y_tee, y_pred= y_pred)
    acc=accuracy_score(y_true=y_tee, y_pred= y_pred)
    recall=recall_score(y_true=y_tee, y_pred= y_pred)
    pre=precision_score(y_true=y_tee, y_pred= y_pred)
    fpr1, tpr1, thresholds = roc_curve( y_tee, y_pred)
    auc1=auc(fpr1, tpr1)
    auc_table = auc_table.append({'classifiers':name,
                                        'fpr':fpr1, 
                                        'tpr':tpr1, 
                                        'auc':auc1}, ignore_index=True)
#     scoring_all_test.loc[name,'feature method']=st
    scoring_all_test.loc[name,'# of features']=X_2.shape[1]
    scoring_all_test.loc[name,'# of positive']=sum(y_tee==1)
    scoring_all_test.loc[name,'# of negtive']=sum(y_tee==0)
    scoring_all_test.loc[name,'TP']=mes[0]
    scoring_all_test.loc[name,'FP']=mes[1]
    scoring_all_test.loc[name,'TN']=mes[2]
    scoring_all_test.loc[name,'FN']=mes[3]
    scoring_all_test.loc[name,'Accuracy']= np.round(acc,4)
    scoring_all_test.loc[name,'Recall']= np.round(recall,4)
    scoring_all_test.loc[name,'Precision']= np.round(pre,4)
    scoring_all_test.loc[name,'F1']= np.round(f1,4)
    scoring_all_test.loc[name,'AUC']= np.round(auc1,4)
    scoring_all_test.loc[name,'MCC']= np.round(mcc,4)
    preds[name]=y_pred

print(scoring_all_test)




scores=pd.DataFrame()
y_actual=flatten_list(comp.iloc[:,3])
auc_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])
for i in ([4,5,6,7,8,9,10]) :
    y_pred=flatten_list(comp.iloc[:,i])
    mes=perf_measure(y_actual,y_pred)
    mcc= matthews_corrcoef(y_true= y_actual, y_pred= y_pred)
    f1=f1_score(y_true= y_actual, y_pred= y_pred)
    acc=accuracy_score(y_true=y_actual, y_pred= y_pred)
    recall=recall_score(y_true=y_actual, y_pred= y_pred)
    pre=precision_score(y_true=y_actual, y_pred= y_pred)
    fpr1, tpr1, thresholds = roc_curve( y_actual, y_pred)
    auc1=auc(fpr1, tpr1)
    auc_table = auc_table.append({'classifiers':comp.iloc[:,i].name,
                                        'fpr':fpr1, 
                                        'tpr':tpr1, 
                                        'auc':auc1}, ignore_index=True)
    scores.loc[comp.iloc[:,i].name,'# of positive']=sum(pd.DataFrame(y_actual).iloc[:,0]==1)
    scores.loc[comp.iloc[:,i].name,'# of negtive']=sum(pd.DataFrame(y_actual).iloc[:,0]==0)
    scores.loc[comp.iloc[:,i].name,'TP']=mes[0]
    scores.loc[comp.iloc[:,i].name,'FP']=mes[1]
    scores.loc[comp.iloc[:,i].name,'TN']=mes[2]
    scores.loc[comp.iloc[:,i].name,'FN']=mes[3]
    scores.loc[comp.iloc[:,i].name,'Accuracy']=np.round(acc,4)
    scores.loc[comp.iloc[:,i].name,'Recall']=np.round(recall,4)
    scores.loc[comp.iloc[:,i].name,'Precision']=np.round(pre,4)
    scores.loc[comp.iloc[:,i].name,'F1']=np.round(f1,4)
    scores.loc[comp.iloc[:,i].name,'AUC']=np.round(auc1,4)
    scores.loc[comp.iloc[:,i].name,'MCC']=np.round(mcc,4)
print(scores)

fig = plt.figure(figsize=(12,10))

for i in auc_table.index:
    plt.plot(auc_table.loc[i]['fpr'], 
             auc_table.loc[i]['tpr'], linestyle='--' ,lw=1,
             label="{}, AUC={:.4f}".format(auc_table.loc[i]['classifiers'], auc_table.loc[i]['auc']))
    
plt.plot([0,1], [0,1], color='orange', linestyle='--')

plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("Flase Positive Rate", fontsize=15)

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)

plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
plt.legend(prop={'size':13}, loc='lower right')

plt.show()