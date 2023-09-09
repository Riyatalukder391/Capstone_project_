#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
from google.colab import drive

drive.mount('/content/drive') 


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)


# In[ ]:


# load the dataset
path = "/content/drive/MyDrive/datasets"

# Change the directory
os.chdir(path)

raw_data = pd.read_csv('Heartdisease.csv')  
raw_data.head()


# # Part 1 - Drop all unnecessary columns

# In[ ]:


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


# In[ ]:


data = raw_data.drop(cols_drop, axis=1)
data.head()


# In[ ]:


data.shape


# In[ ]:


# Part 2 - dropping feature engineered variables

cols_drop_engineered_vars = ['_URBSTAT', 'ASTHMA3', 'ASTHNOW', '_LTASTH1', '_CASTHM1', 'HAVARTH4', '_DENVST3', 'LASTDEN4', '_PRACE1', '_MRACE1', 
                             '_HISPANC', '_RACE', '_RACEG21', '_RACEGR3', '_RACEPRV', '_AGEG5YR', '_AGE65YR', '_AGE_G', 'HTM4', '_BMI5CAT', '_RFBMI5', 
                             '_RFSMOK3', 'SMOKE100', 'SMOKDAY2', 'LASTSMK2', 'USENOW3', 'ECIGARET', 'ECIGNOW', 'LCSNUMCG', '_RFBING5', 'MAXDRNKS', 
                             '_DRNKDRV', '_RFSEAT2', '_RFSEAT3', 'SEATBELT', 'DRNKDRI2', 'SEXVAR', 'DECIDE', '_MENT14D', '_RFHLTH', '_PHYS14D', 
                             '_HCVU651', 'HLTHPLN1', 'PERSDOC2', 'MEDCOST', 'CHECKUP1', 'EXERANY2', 'CNCRDIFF', 'DIFFDRES', 'DIFFALON', 'FLUSHOT7', 
                             'SHINGLE2', 'PNEUVAC4', '_AIDTST4', 'HIVTST7', 'HIVRISK5', 'USEMRJN2', 'RSNMRJN1', 'EDUCA', 'VETERAN3', 'CHILDREN', 
                             'INCOME2', '_CHLDCNT', 'PREGNANT', 'HEIGHT3']

print('Number of columns to be dropped which belong to engineered variables: ', len(cols_drop_engineered_vars))


# In[ ]:


data.drop(cols_drop_engineered_vars, axis=1, inplace=True)

data.shape


# # Part 2 - Univariate analysis
# Variable distributions, frequency of classes, etc

# In[ ]:


data.info()


# In[ ]:


plt.rcParams['figure.figsize'] = [20, 15]


# ### Replacing missing values

# In[ ]:


data.isnull().sum()*100/len(data)


# In[ ]:


# replace missing data with custom values
data['EMPLOY1'].fillna(0, inplace=True)
data['AVEDRNK3'].fillna(0, inplace=True)
data['DRNK3GE5'].fillna(0, inplace=True)
data['MARIJAN1'].fillna(0, inplace=True)


# In[ ]:


# replace missing values with median
replacena_with_median_cols = ['HHADULT', 'GENHLTH', 'PHYSHLTH', 'MENTHLTH', 'SLEPTIM1', 'CVDSTRK3', 'CHCSCNCR', 'CHCOCNCR', 'CHCCOPD2', 'ADDEPEV3', 'CHCKDNY2', 
                              'DIABETE4', 'MARITAL', 'WEIGHT2', 'DEAF', 'BLIND', 'DIFFWALK', '_METSTAT', '_DRDXAR2', 'HTIN4', 'WTKG3', 
                              '_BMI5']

def replacena_with_median(col):
    data[col].fillna(data[col].median(), inplace=True)
    
for col in replacena_with_median_cols:
    replacena_with_median(col)


# In[ ]:


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


# In[ ]:


# dropping all rows where _MICHD is missing

drop_ind = data.index[data['_MICHD'].isna()]
data.drop(drop_ind, axis=0, inplace=True)


# In[ ]:


data.isnull().sum().sum()


# In[ ]:


# Observations for feature engineering, value replacement and other things



# DROCDY3_ - drinks occassions per day, value technically cannot be above 30, each _MICHD percentage differs with different buckets.
# data['DROCDY3_'].plot(kind='hist', bins=10)

#_MICHD - replace 2 with 0 (NO CHD)


# In[ ]:





# In[ ]:


drink_day = pd.DataFrame({'DROCDY3_': data['DROCDY3_'], 'Group': pd.cut(data['DROCDY3_'], bins=10).values, '_MICHD': data['_MICHD']})


# In[ ]:


drink_day = pd.crosstab(drink_day['Group'], drink_day['_MICHD'])
drink_day['Pct_MICHD'] = drink_day[1.0] / drink_day.sum(axis=1)


# In[ ]:


drink_day


# In[ ]:


# As we see, there are too many high values towards even in the outlier regions, we will keep this column as is.


# In[ ]:


data.hist()
plt.tight_layout()
plt.show()


# In[ ]:


data.boxplot()
plt.tight_layout()
plt.xticks(rotation=90)
plt.show()


# In[ ]:





# ## Analysing each variable

# In[ ]:


from statsmodels.stats.api import proportions_ztest


# In[ ]:


def check_proportionality_michd(col, class_names):
    
    class_names = class_names
    df = pd.crosstab(data[col], data['_MICHD'])
    
    proportion_michd = df[1.0].values / df.sum(axis=1).values

    proportion_michd = pd.Series(data=proportion_michd, index=class_names).sort_values()

    sns.barplot(x=proportion_michd.index , y=proportion_michd.values)
    plt.rcParams['figure.figsize'] = [8, 5]
    plt.xticks(rotation=90)
    plt.show()
    
    z_prop, p_val = two_sample_prop_test(df)
    
    if p_val < 0.05: 
        print('The Alternate Hypothesis passed the 2 sample proportion test.')
        print('The two classes in this category are shown to have different proportions of _MICHD')
    else:
        print('The Null Hypothesis failed to be rejected.')
        print('The two classes in this category cannot be shown to have different proportions of _MICHD')
    
    return proportion_michd

def two_sample_prop_test(df):
    
    # Getting rid of unknown values
    if 9.0 in df.index:
        df = df.drop(9.0, axis=0)
    
    if len(df.index) > 2:
        return 0, 0
    
    # Calculating count and nobs for the category 
    count = df[1.0].values
    nobs = df.sum(axis=1)
    
    z_prop, p_val = proportions_ztest(count=count, nobs=nobs, alternative='two-sided')
    
    return z_prop, p_val


# ### Variable 1: _STATE

# In[ ]:


_STATE_names = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 
                        'Delaware', 'District of Columbia ', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 
                        'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 
                        'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 
                        'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 
                        'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 
                        'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 
                        'West Virginia', 'Wisconsin', 'Wyoming', 'Guam', 'Puerto Rico']

sns.countplot(data['_STATE'], x=_STATE_names)
plt.rcParams['figure.figsize'] = [7, 5]
plt.xticks(rotation=90)
plt.show()


# All states have nearly equal representation in the data 
# - Categoric variable with 53 categories
# - Rule out frequency encoding for this variable, despite its fairly high cardinality. 
# - No state is undersampled

# In[ ]:


proportion_michd_state = check_proportionality_michd('_STATE', _STATE_names)


# All the states have varying proportions of _MICHD in the sample. We can verify this with an ANOVA test as well.
# - Will have to be one hot encoded

# ### Variable 2: HHADULT

# In[ ]:


sns.boxplot(data=data, y='HHADULT', x='_MICHD')


# 

# In[ ]:


data['HHADULT'].value_counts()


# - Most households have 1 - 5 adults,
# - very few lie in the 10 - 60 range
# - We could remove outlier beyond 10: would not lose much of the dataset. OR impute this with 10. 
# 

# ### Variable 3: GENHLTH

# In[ ]:


health = ['Excellent', 'Very Good', 'Good', 'Fair', 'Poor']
proportion_michd_genhlth = check_proportionality_michd('GENHLTH', health)


# - Ordinal category with 5 different values
# - All categories have significantly different sample proportions of _MICHD

# ### Variable 4: PHYSHLTH

# In[ ]:


sns.boxplot(data=data, y='PHYSHLTH', x='_MICHD')


# In[ ]:


a = data['PHYSHLTH'].value_counts()
sns.barplot(a.index, a.values)


# - 88.0 needs to be replaced with 0 (how did it get left behind?)
# - Highly right skewed, could be fixed with log transformation (as an option)
# - Could consider splitting this data into  2: people who never fell sick in the past 30 days, people who did.

# ### Variable 5: MENTHLTH

# In[ ]:


a = data['MENTHLTH'].value_counts()
sns.barplot(a.index, a.values)


# - Most of the data falls under 0 
# - Can create 2 classes out of this data - those people who fell sick at least once, those who didn't fall sick at all.

# ### Variable 5: SLEPTIM1

# In[ ]:


sns.boxplot(data=data, y='SLEPTIM1', x='_MICHD')


# - small range
# - quite a few outliers, data can be left as it is.
# - Sleep time is quite similar in terms of population means, we could drop it or pick another variable for the analysis.

# ### Variable 6: CVDSTRK3

# In[ ]:


data['CVDSTRK3'].value_counts()


# In[ ]:


plt.rcParams['figure.figsize'] = [12, 8]
proportion_michd_stroke = check_proportionality_michd('CVDSTRK3', ['Has had stroke', 'Never had stroke'])


# - Categorical variable with 2 classes only
# - replace 2.0 with 0 
# - The Alternate Hypothesis passed the 2 sample proportion test.
# - The two classes in this category are shown to have different proportions of _MICHD

# ### Variable 7: CHCSCNCR

# In[ ]:


data['CHCSCNCR'].value_counts()


# In[ ]:


proportion_michd_skincancer = check_proportionality_michd('CHCSCNCR', ['Has had skin cancer', 'Never had skin cancer'])


# - Categorical variable with 2 classes only
# - replace 2.0 with 0
# - The Alternate Hypothesis passed the 2 sample proportion test.
# - The two classes in this category are shown to have different proportions of _MICHD

# ### Variable 8: CHCOCNCR

# In[ ]:


data['CHCOCNCR'].value_counts()


# In[ ]:


proportion_michd_cancer = check_proportionality_michd('CHCOCNCR', ['Has had some cancer', 'Never had any cancer'])


# - Categorical variable with 2 classes only
# - replace 2.0 with 0 (For NO)
# - The Alternate Hypothesis passed the 2 sample proportion test.
# - The two classes in this category are shown to have different proportions of _MICHD

# ### Variable 9: CHCCOPD2

# In[ ]:


data['CHCCOPD2'].value_counts()


# In[ ]:


proportion_michd_cancer = check_proportionality_michd('CHCCOPD2', ['Has had bronchitis', 'Never had bronchitis'])


# - Categorical variable with 2 classes only
# - replace 2.0 with 0 (For NO)
# - The Alternate Hypothesis passed the 2 sample proportion test.
# - The two classes in this category are shown to have different proportions of _MICHD

# ### Variable 10: ADDEPEV3

# In[ ]:


data['ADDEPEV3'].value_counts()


# In[ ]:


proportion_michd_dep = check_proportionality_michd('ADDEPEV3', ['Has had depressive disorder', 'Never had depressive disorder'])


# - Categorical variable with 2 classes only
# - replace 2.0 with 0 (For NO)
# - The Alternate Hypothesis passed the 2 sample proportion test.
# - The two classes in this category are shown to have different proportions of _MICHD

# ### Variable 11: CHCKDNY2

# In[ ]:


data['CHCKDNY2'].value_counts()


# In[ ]:


proportion_michd_kidney = check_proportionality_michd('CHCKDNY2', ['Has had kidney stones', 'Never had kidney stones'])


# - Categorical variable with 2 classes only
# - replace 2.0 with 0 (For NO)
# - The Alternate Hypothesis passed the 2 sample proportion test.
# - The two classes in this category are shown to have different proportions of _MICHD

# ### Variable 12: DIABETE4

# In[ ]:


data['DIABETE4'].value_counts()


# In[ ]:


proportion_michd_diabetes = check_proportionality_michd('DIABETE4', ['Has had diabetes', 'Never had diabetes'])


# - Categorical variable with 2 classes only
# - replace 2.0 with 0 (For NO)
# - The Alternate Hypothesis passed the 2 sample proportion test.
# - The two classes in this category are shown to have different proportions of _MICHD

# ### Variable 13: MARITAL

# In[ ]:


data['MARITAL'].value_counts()


# In[ ]:


marital_status = ['Married', 'Divorced', 'Widowed', 'Separated', 'Never married', 'Member of unmarried couple']
proportion_michd_diabetes = check_proportionality_michd('MARITAL', marital_status)


# - Categorical variable with 6 categories 
# - Since the proportion of _MICHD is much higher in single persons who have been separated, we could consider making 2 categories: Never married / Married, and Separated
# - Will have to be one hot encoded for now

# ### Variable 14: EMPLOY1

# In[ ]:


data['EMPLOY1'].value_counts()


# In[ ]:


emp_status = ['Refused', 'Employed', 'self-employed', 'Out of work 1 year', 'out of work more than 1 year', 'Homemaker', 'Student', 'Retired', 'Unable to work']
proportion_michd_diabetes = check_proportionality_michd('EMPLOY1', emp_status)


# - can make different categories: people who are employed / not employed
# - Will need to be one hot encoded for now

# ### Variable 15: WEIGHT2

# In[ ]:


sns.boxplot(data=data, y='WEIGHT2', x='_MICHD')


# - Replacements are required for this variable
# - If the variable is between 9000 - 9500, this will need to be converted to x - 9000
# - If the variable is below 9000, convert this to kilograms
# - Replace 7777 with BMI based info
# - 9999 - replace based on BMI data

# ### Variable 16: DEAF

# In[ ]:


data['DEAF'].value_counts()


# In[ ]:


proportion_michd_deaf = check_proportionality_michd('DEAF', ['Deaf', 'Not Deaf'])


# - Categorical variable with 2 classes only
# - replace 2.0 with 0 (For NO)
# - The Alternate Hypothesis passed the 2 sample proportion test.
# - The two classes in this category are shown to have different proportions of _MICHD

# ### Variable 17: BLIND

# In[ ]:


data['BLIND'].value_counts()


# In[ ]:


proportion_michd_blind = check_proportionality_michd('BLIND', ['Blind', 'Not Blind'])


# - Categorical variable with 2 classes only
# - replace 2.0 with 0 (For NO)
# - The Alternate Hypothesis passed the 2 sample proportion test.
# - The two classes in this category are shown to have different proportions of _MICHD

# ### Variable 18: DIFFWALK

# In[ ]:


data['DIFFWALK'].value_counts()


# In[ ]:


proportion_michd_diffwalk = check_proportionality_michd('DIFFWALK', ['Has difficulty walking', 'No Difficulty'])


# - Categorical variable with 2 classes only
# - replace 2.0 with 0 (For NO)
# - The Alternate Hypothesis passed the 2 sample proportion test.
# - The two classes in this category are shown to have different proportions of _MICHD

# ### Variable 19: AVEDRNK3

# In[ ]:


sns.boxplot(data=data, y='AVEDRNK3', x='_MICHD')


# - numeric data, should ideally be between 1 and 30, we could remove outliers that are valued beyond 40 or so.
# - Else data is highly skewed. 
# - The population means don't seem different, could prove using a two sample test, and pick another test based on the statistical result.

# ### Variable 20: DRNK3GE5

# In[ ]:


sns.boxplot(data=data, y='DRNK3GE5', x='_MICHD')


# - numeric data, should ideally be between 1 and 30, we could remove outliers that are valued beyond 40 or so.
# - Else data is highly skewed. 
# - The population means don't seem different, could prove using a two sample test, and pick another test based on the statistical result.

# ### Variable 21: MARIJAN1

# In[ ]:


sns.boxplot(data=data, y='MARIJAN1', x='_MICHD')


# - Although the mode is at 0, we need to consider the variables that are at the other values

# ### Variable 22: _METSTAT

# In[ ]:


data['_METSTAT'].value_counts()


# 

# In[ ]:


proportion_michd_metstat = check_proportionality_michd('_METSTAT', ['Metro counties', 'Non-metro counties'])


# - Categorical variable with 2 classes only
# - replace 2.0 with 0 (For NO)
# - The Alternate Hypothesis passed the 2 sample proportion test.
# - The two classes in this category are shown to have different proportions of _MICHD

# ### Variable 23: _IMPRACE

# In[ ]:


data['_IMPRACE'].value_counts()


# In[ ]:


races = ['White', 'Black', 'Asian', 'American Indian', 'Hispanic', 'Other race']
proportion_michd_races = check_proportionality_michd('_IMPRACE', races)


# - All categories have different proportions of _MICHD in the data.
# - Will need to be ondehot encoded

# ### Variable 24: _TOTINDA

# In[ ]:


data['_TOTINDA'].value_counts()


# In[ ]:


proportion_michd_exercise = check_proportionality_michd('_TOTINDA', ['Had exercise', 'No exercise'])


# - Categorical variable with 2 classes only
# - replace 2.0 with 0 (For NO)
# - The Alternate Hypothesis passed the 2 sample proportion test.
# - The two classes in this category are shown to have different proportions of _MICHD

# ### Variable 25: _ASTHMS1

# In[ ]:


data['_ASTHMS1'].value_counts()


# In[ ]:


proportion_michd_asthma = check_proportionality_michd('_ASTHMS1', ['Current','Former', 'Never has asthma'])


# - This is ideally an ordinal category, but we could one hot encode it too.

# ### Variable 26: _DRDXAR2

# In[ ]:


data['_DRDXAR2'].value_counts()


# In[ ]:


proportion_michd_arthritis = check_proportionality_michd('_DRDXAR2', ['Has Arthritis', 'No Arthritis'])


# - Categorical variable with 2 classes only
# - replace 2.0 with 0 (For NO)
# - The Alternate Hypothesis passed the 2 sample proportion test.
# - The two classes in this category are shown to have different proportions of _MICHD

# ### Variable 27: _EXTETH3

# In[ ]:


data['_EXTETH3'].value_counts()


# In[ ]:


proportion_michd_extteeth = check_proportionality_michd('_EXTETH3', ['No teeth extracted', 'Has had teeth extracted'])


# - Categorical variable with 2 classes only
# - replace 2.0 with 0 (For NO)
# - The Alternate Hypothesis passed the 2 sample proportion test.
# - The two classes in this category are shown to have different proportions of _MICHD

# ### Variable 28: _SEX

# In[ ]:


data['_SEX'].value_counts()


# In[ ]:


proportion_michd_sex = check_proportionality_michd('_SEX', ['Female', 'Male'])


# - Categorical variable with 2 classes only
# - The Alternate Hypothesis passed the 2 sample proportion test.
# - The two classes in this category are shown to have different proportions of _MICHD

# ### Variable 29: _AGE80

# In[ ]:


plt.rcParams['figure.figsize'] = [12, 8]
sns.boxplot(data=data, y='_AGE80', x='_MICHD')
#data['_AGE80'].plot(kind='box', hue='_MICHD')


# - numeric variable
# - not skewed
# - could do a two sample mean test to check whether their population means really differ from each other.

# ### Variable 30: HTIN4

# In[ ]:


sns.boxplot(data=data, y='HTIN4', x='_MICHD')


# - numeric variable
# - not highly skewed
# - Outliers exist, but no treatment may be required.
# 
# - Seems like the population means are not very different, could do a 2 sample z test to check.

# ### Variable 31: WTKG3

# In[ ]:


sns.boxplot(data=data, y='WTKG3', x='_MICHD')


# - contains 2 decimal places, needs to be divided by 100
# - Some outliers exist, but we cannot remove the outliers yet, as the abnormalities may be showing a higher proportion of _MICHD
# 
# - population means seem to vary very little.

# ### Variable 32: _BMI5

# In[ ]:


sns.boxplot(data=data, y='_BMI5', x='_MICHD')


# - contains 2 decimal places, needs to be divided by 100
# - Some outliers exist, but we cannot remove the outliers yet, as the abnormalities may be showing a higher proportion of _MICHD
# - Not very differentiating in terms of population proportion, could try to pick another BMI variable based on the statistic result. 

# ### Variable 33: _EDUCAG

# In[ ]:


data['_EDUCAG'].value_counts()


# In[ ]:


education_level = ['High school dropout', 'High School Graduate', 'Attended College', 'Graduated College']
proportion_michd_college = check_proportionality_michd('_EDUCAG', education_level)


# - Clearly ordinally related to _MICHD in proportionality of _MICHD found in each category.
# - Needs one-hot encoding

# ### Variable 34: _INCOMG

# In[ ]:


data['_INCOMG'].value_counts()


# In[ ]:


income_level = ['< 15K', '15 to 25K', '25 - 35K', '35 - 50K', '> 50K']
proportion_michd_income = check_proportionality_michd('_INCOMG', income_level)


# - Clearly ordinally related to _MICHD in proportionality of _MICHD found in each category.
# - Can keep the variable as is. (Already ordinally encoded)

# ### Variable 35: _SMOKER3

# In[ ]:


data['_SMOKER3'].value_counts()


# In[ ]:


smoker = ['Current Smoker', 'Current Smoker seldom', 'Former Smoker', 'Never Smoked']
proportion_michd_smoke = check_proportionality_michd('_SMOKER3', smoker)


# - We could later split this data into 2 categories - Never smoked / Have smoked.
# - One hot Encoding

# ### Variable 36: DRNKANY5

# In[ ]:


data['DRNKANY5'].value_counts()


# In[ ]:


proportion_michd_drink = check_proportionality_michd('DRNKANY5', ['Not drank', 'Drank in past 30 days'])


# - Categorical variable with 2 classes only
# - replace 2.0 with 0 (For NO)
# - The Alternate Hypothesis passed the 2 sample proportion test.
# - The two classes in this category are shown to have different proportions of _MICHD

# ### Variable 37: DROCDY3_

# In[ ]:


sns.boxplot(data=data, y='DROCDY3_', x='_MICHD')


# - numeric variable
# - Some values don't make sense, need to be checked

# ### Variable 38: _DRNKWK1

# In[ ]:


sns.boxplot(data=data, y='DROCDY3_', x='_MICHD')


# - Feature engineering required.

# ### Variable 39: _RFDRHV7

# In[ ]:


data['_RFDRHV7'].value_counts()


# In[ ]:


proportion_michd_heavydrink = check_proportionality_michd('_RFDRHV7', ['Not a heavy drinker', 'Heavy drinker'])


# - Categorical variable with 2 classes only
# - replace 2.0 with 0 (For NO)
# - The Alternate Hypothesis passed the 2 sample proportion test.
# - The two classes in this category are shown to have different proportions of _MICHD

# In[ ]:


# Part 2: Bivariate analysis for continuous variables
plt.rcParams['figure.figsize'] = [25, 20]
cont_variables = data[['HHADULT', 'PHYSHLTH', 'SLEPTIM1', 'WEIGHT2', 'AVEDRNK3', 'DRNK3GE5', 'MARIJAN1', '_AGE80', 'HTIN4', 'WTKG3', '_BMI5', 'DROCDY3_', '_MICHD']]

sns.pairplot(data=cont_variables, hue='_MICHD')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


data.columns


# # Part 3 - Bivariate analysis

# In[ ]:


# Part 1 - correlation matrix

df_corr = data.corr()


# In[ ]:


df_corr['_MICHD'][abs(df_corr['_MICHD']) > 0.17]


# - Very few variables show high correlation with _MICHD
# - For this reason, we may have to use a complex, non-linear model for prediction.

# Part 4 - Variables exhibiting high correlation

# In[ ]:


plt.rcParams['figure.figsize'] = [25, 18]
sns.heatmap(df_corr, annot=True)


# Part - Statistic tests to check which variables are important in predicting CHD. For all categorical and numeric variables.
# 

# In[ ]:


#Cols to confirm whether or not they contribute to CHD 
#Part a - categoric: All columns that could be added to the analysis. 

# Columns that are not yet dropped, but could be dropped on the basis on the statistical test - 


# Part - Conduct chi2 tests to check independence of variables.

# In[ ]:





# 

# # Feature Engineering and re-encodings needed

# In[ ]:


# replace 2's with 1's, includes the target variable

data = data.replace({'PHYSHLTH': {88: 0},
                     'CVDSTRK3': {2.0: 0},
                     'CHCSCNCR': {2.0: 0},
                     'CHCOCNCR': {2.0: 0},
                     'CHCCOPD2': {2.0: 0},
                     'ADDEPEV3': {2.0: 0},
                     'CHCKDNY2': {2.0: 0},
                     'DIABETE4': {2.0: 0},
                     'DEAF': {2.0: 0}, 
                     'BLIND': {2.0: 0},
                     'DIFFWALK': {2.0: 0},
                     '_METSTAT': {2.0: 0},
                     '_TOTINDA': {2.0: 0},
                     '_DRDXAR2': {2.0: 0}, 
                     '_EXTETH3': {2.0: 0},
                     '_SEX': {2.0: 0},
                     'DRNKANY5': {2.0: 0},
                     '_RFDRHV7': {2.0: 0},
                     '_MICHD': {2.0: 0}
                     })


# In[ ]:


X_numeric = data.drop(['_STATE', 'GENHLTH', 'MARITAL', 'EMPLOY1', '_IMPRACE', '_ASTHMS1', '_EDUCAG', '_SMOKER3'], axis=1)


# In[ ]:


# Columns that need one hot encoding

X_categoric = data[['_STATE', 'GENHLTH', 'MARITAL', 'EMPLOY1', '_IMPRACE', '_ASTHMS1', '_EDUCAG', '_SMOKER3']]

X_categoric = pd.get_dummies(X_categoric.astype('object'), drop_first=True)


# In[ ]:


# concatenating the numeric and categoric variables

df_full = pd.concat([X_categoric, X_numeric], axis=1)
df_full.head()


# In[ ]:


df_full.shape


# 

# 

# In[ ]:


data['WEIGHT2'] = np.where(((data['WEIGHT2'] > 9000) & (data['WEIGHT2'] < 9500)), data['WEIGHT2'] - 9000, data['WEIGHT2'])
data['WEIGHT2'] = np.where((data['WEIGHT2'] < 1000), data['WEIGHT2']*0.453592, data['WEIGHT2'])
data['WEIGHT2'].plot(kind='box')


# In[ ]:


technologies = {
    'Courses':["Spark","PySpark","Python","pandas"],
    'Fee' :[20000,25000,22000,30000],
    'Duration':['30days','40days','35days','50days'],
    'Discount':[1000,2300,1200,2000]
              }
index_labels=['r1','r2','r3','r4']
df = pd.DataFrame(technologies,index=index_labels)
df


# In[ ]:


df['Discount'] = np.where(df['Discount'] > 1200, 15000, df['Discount'])
print(df)


# In[ ]:





# Part 5 - Drop columns exhibiting high correlation or feature engineered variables, even the ones failing the chi2 test.

# In[ ]:





# Part 6 - Check for clustering methods

# In[ ]:


from scipy.cluster.hierarchy import cophenet, dendrogram, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans


# In[ ]:


# scaling the df_full data

from sklearn.preprocessing import StandardScaler

X = df_full.drop('_MICHD', axis=1)
y = df_full['_MICHD']

scale = StandardScaler()

X_std = scale.fit_transform(X)
X_std


# In[ ]:


kmeans = KMeans(n_clusters=2)

y_preds = kmeans.fit_predict(X_std)


# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report

print(classification_report(y, y_preds))


# In[ ]:


#inertia = [] #inertia (within cluster sum square) or total spread

#for k in range(1,10):
#    kmeans = KMeans(n_clusters=k,n_init=20)
#    kmeans.fit(X_std)
#    inertia.append(kmeans.inertia_)
    
# Visualization of k values:

#plt.plot(np.arange(1, 10), inertia)
#plt.ylabel('wcss values')
#plt.xlabel('Num. clusters')
#plt.show()


# In[ ]:


df_full.head()


# In[ ]:


#Part 2: Trying KModes clustering.

X_cat = df_full.drop(['HHADULT', 'PHYSHLTH', 'SLEPTIM1', 'WEIGHT2', 'AVEDRNK3', 'DRNK3GE5', 'MARIJAN1', '_AGE80', 'HTIN4', 'WTKG3', '_BMI5', 'DROCDY3_'], axis=1)
X_cat.head()


# In[ ]:


get_ipython().system('pip install kmodes')


# In[ ]:


#from kmodes.kmodes import KModes

#kmode = KModes(n_clusters=2, random_state=42)
#y_preds = kmode.fit_predict(X_cat)


# In[ ]:


# comparing 
from scipy.stats import chi2_contingency

observed_values = pd.crosstab(df_full['PHYSHLTH'], df_full['CVDSTRK3']).values

test_stat, p, dof, expected_value = chi2_contingency(observed = observed_values, correction = False)
print(p)


# # Fetching columns with the highest correlation to the target.
# 
# - Part 1: Confirming the variables have a very low population mean. 
# - Part 2: Replacing the variables with other more decisive variables, if any.

# In[ ]:


from statsmodels.stats import weightstats as stests

def two_samp_mean_test(col):

  z_score, pval = stests.ztest(x1 = nl_scores, x2 = sgl_scores, value = 0, alternative = 'larger')


# In[ ]:


# 1. Checking Sleep Time and substitutes for sleeptime

sleep = ['SLEPTIM1']


gp_sleep = data['_BMI5'].groupby(data['_MICHD'])

chd = gp_sleep.get_group(1.0)
non_chd = gp_sleep.get_group(0)

z_score, pval = stests.ztest(x1 = chd, x2 = non_chd, value = 0, alternative = 'two-sided')


# In[ ]:


pval


# # Base Model

# In[ ]:


X = df_full.drop('_MICHD', axis=1)
y = df_full['_MICHD']


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
#from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[ ]:


kfold = KFold(n_splits=3, shuffle=True, random_state=42)
params = {'n_estimators': [20, 30, 50, 100], 'max_depth': [5, 8, 10]}

GS_rf = GridSearchCV(estimator=RandomForestClassifier(),
                    param_grid=params,
                    scoring='f1_weighted',
                    cv=kfold)


# In[ ]:


GS_rf.fit(X_train, y_train)


# In[ ]:


print('Train score (accuracy):', GS_rf.score(X_train, y_train))
print('Test score(accuracy):', GS_rf.score(X_test, y_test))

y_preds_test = GS_rf.predict(X_test)
y_predprobs_test = GS_rf.predict_proba(X_test)


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve


# In[ ]:


confusion_matrix(y_test, y_preds_test)


# In[ ]:


plt.rcParams['figure.figsize'] = [12, 8]
fpr, tpr, thresholds = roc_curve(y_test, y_predprobs_test[:, 1])
plt.plot(fpr, tpr)
plt.plot((0, 1), (0, 1))
plt.show()


# In[ ]:


roc_auc_score(y_test, y_predprobs_test[:, 1])


# In[ ]:


pd.DataFrame({'TPR': tpr, 'FPR': fpr, 'Youden\'s Index': tpr-fpr, 'Thresholds': thresholds}).sort_values(by='Youden\'s Index', ascending=False)


# In[ ]:


y_preds_test_new_thresh = [0 if x < 0.097912 else 1 for x in y_predprobs_test[:, 1]]


# In[ ]:


print(classification_report(y_test, y_preds_test_new_thresh))


# In[ ]:


confusion_matrix(y_test, y_preds_test_new_thresh)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

nums = np.array([30, 60, 200])
nums = nums / nums.sum()

sns.barplot(y=nums, x=['Survey Metadata', 'CDC Calculated variables', 'Survey Phone Responses'])
plt.xticks(rotation=45)
plt.show()


# In[ ]:




