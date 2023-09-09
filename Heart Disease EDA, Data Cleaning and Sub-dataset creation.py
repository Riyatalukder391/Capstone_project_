#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)


# In[3]:


raw_data = pd.read_csv('Heartdisease (2).csv')
raw_data.head()


# In[5]:


raw_data[raw_data['HADMAM'] == 1].shape


# In[8]:


# Part 1 - dropping columns based on relevance

cols_drop = ['Unnamed: 0', 'FMONTH', 'IDATE', 'IMONTH', 'IDAY', 'IYEAR', 'DISPCODE', 'SEQNO', '_PSU', 'QSTLANG', 'MSCODE', '_STSTR', 'QSTVER', '_STRWT', '_RAWRAKE', 
             '_WT2RAKE', '_CHISPNC', '_CRACE1', '_CPRACE', '_CLLCPWT', '_DUALUSE', '_DUALCOR', '_LLCPWT2', '_LLCPWT', '_ALTETH3', '_FLSHOT7', '_PNEUMO3', 
             '_RFMAM22', '_MAM5023', '_RFPAP35', '_RFPSA23', '_CLNSCPY', '_SGMSCPY', '_SGMS10Y', '_RFBLDS4', '_STOLDNA', '_VIRCOLN', '_SBONTIM', '_CRCREC1', 
             'CTELENM1', 'PVTRESD1', 'COLGHOUS', 'STATERE1', 'CELPHONE', 'LADULT1', 'COLGSEX', 'NUMADULT', 'LANDSEX', 'NUMMEN', 'NUMWOMEN', 'RESPSLCT', 
             'SAFETIME', 'CTELNUM1', 'CELLFON5', 'CADULT1', 'CELLSEX', 'PVTRESD3', 'CCLGHOUS', 'CSTATE1', 'LANDLINE', 'POORHLTH', 'CVDINFR4', 'CVDCRHD4', 
             'DIABAGE3', 'RMVTETH4', 'RENTHOM1', 'NUMHHOL3', 'NUMPHON3', 'CPDEMO1B', 'STOPSMK2', 'ALCDAY5', 'FLSHTMY3', 'FALL12MN', 'FALLINJ4', 'HADMAM',
             'HOWLONG', 'HADPAP2', 'LASTPAP2', 'HPVTEST', 'HPLSTTST', 'HADHYST2', 'PCPSAAD3', 'PCPSADI1', 'PCPSARE1', 'PSATEST1', 'PSATIME', 'PCPSARS1', 
             'COLNSCPY', 'COLNTEST', 'SIGMSCPY', 'SIGMTEST', 'BLDSTOL1', 'LSTBLDS4', 'STOOLDNA', 'SDNATEST', 'VIRCOLON', 'VCLNTEST', 'HIVTSTD3', 'PDIABTST', 
             'PREDIAB1', 'INSULIN1', 'BLDSUGAR', 'FEETCHK3', 'DOCTDIAB', 'CHKHEMO3', 'FEETCHK', 'EYEEXAM1', 'DIABEYE', 'DIABEDU', 'TOLDCFS', 'HAVECFS', 
             'WORKCFS', 'TOLDHEPC', 'TRETHEPC', 'PRIRHEPC', 'HAVEHEPC', 'HAVEHEPB', 'MEDSHEPB', 'HLTHCVR1', 'CIMEMLOS', 'CDHOUSE', 'CDASSIST', 'CDHELP', 
             'CDSOCIAL', 'CDDISCUS', 'CAREGIV1', 'CRGVREL4', 'CRGVLNG1', 'CRGVHRS1', 'CRGVPRB3', 'CRGVALZD', 'CRGVPER1', 'CRGVHOU1', 'CRGVEXPT', 'LCSFIRST', 
             'LCSLAST', 'LCSCTSCN', 'CNCRAGE', 'CNCRTYP1', 'CSRVTRT3', 'CSRVDOC1', 'CSRVSUM', 'CSRVRTRN', 'CSRVINST', 'CSRVINSR', 'CSRVDEIN', 'CSRVCLIN', 
             'CSRVPAIN', 'CSRVCTL2', 'PCPSADE1', 'PCDMDEC1', 'HPVADVC4', 'HPVADSHT', 'TETANUS1', 'IMFVPLA1', 'BIRTHSEX', 'SOMALE', 'SOFEMALE', 'TRNSGNDR', 
             'ACEDEPRS', 'ACEDRINK', 'ACEDRUGS', 'ACEPRISN', 'ACEDIVRC', 'ACEPUNCH', 'ACEHURT1', 'ACESWEAR', 'ACETOUCH', 'ACETTHEM', 'ACEHVSEX', 'RCSGENDR', 
             'RCSRLTN2', 'CASTHDX2', 'CASTHNO2']


# In[9]:


data = raw_data.drop(cols_drop, axis=1)
data.head()


# In[10]:


data.shape


# In[11]:


# Part 2 - dropping feature engineered variables

cols_drop_engineered_vars = ['_URBSTAT', 'ASTHMA3', 'ASTHNOW', '_LTASTH1', '_CASTHM1', 'HAVARTH4', '_DENVST3', 'LASTDEN4', '_PRACE1', '_MRACE1', 
                             '_HISPANC', '_RACE', '_RACEG21', '_RACEGR3', '_RACEPRV', '_AGEG5YR', '_AGE65YR', '_AGE_G', 'HTM4', '_BMI5CAT', '_RFBMI5', 
                             '_RFSMOK3', 'SMOKE100', 'SMOKDAY2', 'LASTSMK2', 'USENOW3', 'ECIGARET', 'ECIGNOW', 'LCSNUMCG', '_RFBING5', 'MAXDRNKS', 
                             '_DRNKDRV', '_RFSEAT2', '_RFSEAT3', 'SEATBELT', 'DRNKDRI2', 'SEXVAR', 'DECIDE', '_MENT14D', '_RFHLTH', '_PHYS14D', 
                             '_HCVU651', 'HLTHPLN1', 'PERSDOC2', 'MEDCOST', 'CHECKUP1', 'EXERANY2', 'CNCRDIFF', 'DIFFDRES', 'DIFFALON', 'FLUSHOT7', 
                             'SHINGLE2', 'PNEUVAC4', '_AIDTST4', 'HIVTST7', 'HIVRISK5', 'USEMRJN2', 'RSNMRJN1', 'EDUCA', 'VETERAN3', 'CHILDREN', 
                             'INCOME2', '_CHLDCNT', 'PREGNANT', 'HEIGHT3']

print('Number of columns to be dropped which belong to engineered variables: ', len(cols_drop_engineered_vars))


# In[12]:


data.drop(cols_drop_engineered_vars, axis=1, inplace=True)

data.shape


# In[13]:


# replace missing data with custom values
data['EMPLOY1'].fillna(0, inplace=True)
data['AVEDRNK3'].fillna(0, inplace=True)
data['DRNK3GE5'].fillna(0, inplace=True)
data['MARIJAN1'].fillna(0, inplace=True)


# In[14]:


# replace missing values with median
replacena_with_median_cols = ['HHADULT', 'GENHLTH', 'PHYSHLTH', 'MENTHLTH', 'SLEPTIM1', 'CVDSTRK3', 'CHCSCNCR', 'CHCOCNCR', 'CHCCOPD2', 'ADDEPEV3', 'CHCKDNY2', 
                              'DIABETE4', 'MARITAL', 'WEIGHT2', 'DEAF', 'BLIND', 'DIFFWALK', '_METSTAT', '_DRDXAR2', 'HTIN4', 'WTKG3', 
                              '_BMI5']

def replacena_with_median(col):
    data[col].fillna(data[col].median(), inplace=True)
    
for col in replacena_with_median_cols:
    replacena_with_median(col)


# In[15]:


# replace non-information values with custom values based on codebook
data = data.replace({'HHADULT': {77: 1, 99: 1},
                    'GENHLTH': {7: data['GENHLTH'].median(), 9: data['GENHLTH'].median()},
                    'PHYSHLTH': {77: 0, 88: 0, 99: data['PHYSHLTH'].median()},
                    'MENTHLTH': {77: 0, 88: 0, 99: data['MENTHLTH'].median()},
                    'SLEPTIM1': {77: data['SLEPTIM1'].median(), 99: data['SLEPTIM1'].median()},
                    'CVDSTRK3': {7: 2, 9: 2},
                    'CHCSCNCR': {7: 2, 9: 2},
                    'CHCOCNCR': {7: 2, 9: 2},
                    'CHCCOPD2': {7: 2, 9: 2},
                    'ADDEPEV3': {7: 2, 9: 2},
                    'CHCKDNY2': {7: 2, 9: 2},
                    'DIABETE4': {3: 2, 4: 2, 7: 2, 9: 2},
                    'MARITAL': {9: data['MARITAL'].median()},
                    'EMPLOY1': {9: 0},
                    'DEAF': {7: 2, 9: 2},
                    'BLIND': {7: 2, 9: 2},
                    'DIFFWALK': {7: 2, 9: 2},
                    'AVEDRNK3': {88:0, 77: 0, 99: 0},
                    'DRNK3GE5': {88:0, 77: 0, 99: 0},
                    'MARIJAN1': {77: 0, 88:0, 99: 0},
                    '_TOTINDA': {9: data['_TOTINDA'].median()}, 
                    '_ASTHMS1': {9: data['_ASTHMS1'].median()},
                    '_EXTETH3': {9: data['_EXTETH3'].median()},
                    '_SEX': {2: 0}, # replacing female (2) with 0
                    '_EDUCAG': {9: data['_EDUCAG'].median()},
                    '_INCOMG': {9: data['_INCOMG'].median()},
                    '_SMOKER3': {9: data['_SMOKER3'].median()},
                    'DRNKANY5': {7: 2, 9: 2},
                    'DROCDY3_': {900: 0},
                    'DRNKWK1': {99900: 0},
                    '_RFDRHV7': {9: data['_RFDRHV7'].median()}
                    })

# change this replacement method, can't use replace with median right after replacing an overlapping value.
# correct this!


# In[16]:


# dropping all rows where _MICHD is missing

drop_ind = data.index[data['_MICHD'].isna()]
data.drop(drop_ind, axis=0, inplace=True)


# In[17]:


data.isnull().sum().sum()


# In[18]:


data.shape


# In[22]:


data.describe()


# ### Group 1 - Pregnant Women

# In[86]:


y = data['_MICHD'].replace({2:0})
X_numeric = data.drop(['_STATE', 'MARITAL', 'EMPLOY1', '_IMPRACE', '_ASTHMS1', '_EDUCAG', '_SMOKER3', 
                   'CVDSTRK3', 'CHCSCNCR', 'CHCOCNCR', 'CHCCOPD2', 'ADDEPEV3', 'CHCKDNY2', 'DEAF', 'BLIND', 
                    'DIFFWALK', '_METSTAT', '_TOTINDA', '_DRDXAR2', '_EXTETH3', '_SEX', 'DRNKANY5', '_RFDRHV7', 
                       'DIABETE4', '_MICHD'], 
                      axis=1)

X_categoric = data[['_STATE', 'MARITAL', 'EMPLOY1', '_IMPRACE', '_ASTHMS1', '_EDUCAG', '_SMOKER3', 
                   'CVDSTRK3', 'CHCSCNCR', 'CHCOCNCR', 'CHCCOPD2', 'ADDEPEV3', 'CHCKDNY2', 'DEAF', 'BLIND', 
                    'DIFFWALK', '_METSTAT', '_TOTINDA', '_DRDXAR2', '_EXTETH3', '_SEX', 'DRNKANY5', '_RFDRHV7', 
                   'DIABETE4']]

print(X_numeric.shape)
print(X_categoric.shape)


# In[45]:


# checking the spread of the numeric variables

X_numeric


# In[46]:


for col in X_numeric:
    sns.boxplot(x=X_numeric[col])
    print(col)
    plt.show()


# In[47]:


data['PHYSHLTH'] = data['PHYSHLTH'].replace({88:0})
data['MENTHLTH'] = data['MENTHLTH'].replace({88:0})
data['WEIGHT2'][data['WEIGHT2'] > 1000] = data['WEIGHT2'].median()

X_numeric['WTKG3'] = np.log(X_numeric['WTKG3'])
X_numeric['_BMI5'] = np.log(X_numeric['_BMI5'])

X_numeric['_DRNKWK1'][X_numeric['_DRNKWK1'] > 80000]  = X_numeric['_DRNKWK1'].median()


# In[48]:


## Final boxplots of transformed and imputed data

for col in X_numeric:
    sns.boxplot(x=X_numeric[col])
    print(col)
    plt.show()


# In[ ]:


X_numeric


# In[49]:


X_numeric.corr().style.background_gradient(cmap='coolwarm')


# In[ ]:


# Some of these features have fairly high linear correlation coefficient
# PCA could be explored


# In[50]:


X_categoric


# In[51]:


from scipy.stats import chi2_contingency

row_pval = []

df_chi2 = pd.DataFrame()

for col_a in X_categoric.columns:
    row_pval = []
    
    for col_b in X_categoric.columns:
        
        ct_table_ind = pd.crosstab(X_categoric[col_a], X_categoric[col_b]).values
        chi2_stat, p, dof, expected = chi2_contingency(ct_table_ind)
    
        row_pval.append(p)
    
    row_pval = pd.Series(row_pval)
    df_chi2 = pd.concat([df_chi2, row_pval], axis=1)
    
df_chi2


# In[160]:


df_chi2.style.background_gradient(cmap='coolwarm')


# Some (the colored elements) are correlated

# In[65]:


df_chi2.columns = X_categoric.columns
df_chi2.index = X_categoric.columns
plt.figure(figsize=(12, 8))
sns.heatmap(df_chi2, cmap='coolwarm')

plt.show()


# In[130]:


X_categoric_encoded = pd.get_dummies(X_categoric.astype('object'), drop_first=True)


# In[131]:


X_categoric_encoded.shape


# In[132]:


## merging X_cat and X_num

X_full = pd.concat([X_categoric_encoded, X_numeric], axis=1)
X_full.shape


# ## Base model fit

# In[133]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.metrics import cohen_kappa_score, accuracy_score, classification_report, confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=0.3, random_state=42, stratify=y)

kfold = KFold(n_splits=3, shuffle=True, random_state=42)
params = {'n_estimators': [20, 30, 50, 100], 'max_depth': [5, 8, 10]}

GS_rf = GridSearchCV(estimator=RandomForestClassifier(),
                    param_grid=params,
                    scoring='recall',
                    cv=kfold)

GS_rf.fit(X_train, y_train)

print('Train score (accuracy):', GS_rf.score(X_train, y_train))
print('Test score(accuracy):', GS_rf.score(X_test, y_test))

y_preds_test = GS_rf.predict(X_test)
y_predprobs_test = GS_rf.predict_proba(X_test)


# In[162]:


from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

confusion_matrix(y_test, y_preds_test)


# In[161]:


GS_rf.predict_proba(X_train)


# In[ ]:





# In[163]:


sns.displot(x=GS_rf.predict_proba(X_train)[:, 1], hue=y_train)
plt.xlabel('Probability')
plt.axvline(x=0.5, c='r')
plt.show()


# ## KMeans using Numerical variables to check the cluster properties
# 

# In[149]:


X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=0.3, random_state=42, stratify=y)


# In[150]:


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2)

kmeans_labels = kmeans.fit_predict(X_train)


# In[151]:


cohen_kappa_score(y_train, kmeans_labels)


# In[152]:


accuracy_score(y_train, y_preds)


# In[156]:


print('Confusion matrix from the clustering classified labels.')
print(confusion_matrix(y_train, y_preds))

print('Classification report from the clustering classified labels.')
print(classification_report(y_train, list(y_preds)))


# In[157]:


print('Cluster 1 properties')
X_train[y_preds==0].describe()


# In[158]:


print('Cluster 2 properties')
X_train[y_preds==1].describe()


# In[ ]:


# check the distinction of the KDE plots in the pair p


# ## KModes using categorical variables to check the cluster properties

# In[137]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_categoric_encoded, y, test_size=0.3, 
                                                    random_state=42, stratify=y)


# In[138]:


from kmodes.kmodes import KModes

kmodes = KModes(n_clusters=2, random_state=42)
y_preds = kmodes.fit_predict(X_train)


# In[122]:


a = y_preds


# In[139]:


cohen_kappa_score(y_train, y_preds)


# In[140]:


accuracy_score(y_train, y_preds)


# In[141]:


print('Confusion matrix from the clustering classified labels.')
confusion_matrix(y_train, y_preds)


# In[142]:


print('Classification report from the clustering classified labels.')
print(classification_report(y_train, list(y_preds)))


# In[143]:


print('Cluster 1 properties')
X_train[y_preds==0].describe()


# In[144]:


print('Cluster 2 properties')
X_train[y_preds==1].describe()


# ## Applying RepeatedStratifiedKFold

# In[145]:


from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver='lbfgs', max_iter=250)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(model, X_train, y_train, scoring='roc_auc', cv=cv, n_jobs=-1)

print('Mean ROC AUC: %.3f' % mean(scores))


# In[ ]:





# Questions:
# 
#     1. How should we use clustering approaches?
#     2. Why are the output probabilities from the RF model so low?
#     

# Things to do:
#     
#     1. Pick subsets of the data (Eg. pregnant women, cancer patients) and study attributes related to them with respect to _MICHD
#     2. Apply cost based weightage to the dataset and Stratified KFold (Understand this method)
#     3. Try widening the training dataset's feature size and apply PCA instead. (For eg. include all 5 variables that deal with drinking, all 8 variables that deal with Race, and so on)
#     
#     

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




