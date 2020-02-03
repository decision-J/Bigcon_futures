#############################################
####     Big Contest : Future's          ####
####                   - 2019.09.10      ####
#############################################
import requests as req
from bs4 import BeautifulSoup as bs
import os
from selenium import webdriver
import pandas as pd
from html_table_parser import parser_functions as parser
import numpy as np
import csv
import timeit

import seaborn as sns
import plotnine as p9

import requests
import json
from bs4 import BeautifulSoup
import datetime


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from statsmodels.graphics.mosaicplot import mosaic

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
import lightgbm as lgbm
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import xgboost as xgb
import scipy.stats as st
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

from sklearn import  metrics

from sklearn.neural_network import MLPClassifier
from mlxtend.classifier import StackingClassifier
from mlxtend.classifier import StackingCVClassifier
from bayes_opt import BayesianOptimization

warnings.filterwarnings('ignore')

#############################################
#####      Import & Crawling data       #####
#############################################

df = pd.read_csv('C:/Users/letsd/Desktop/19빅콘/퓨처스/AFSNT.CSV',encoding='CP949')

p9.ggplot(data=df)

# Crawling of jul, sep, aug data
*iframe url 공항 이름
RKSS : 김포
RKPK : 김해
RKPC : 제주
RKTN : 대구
RKPU : 울산
RKTU : 청주
RKJB : 무안
RKJJ : 광주
RKJY : 여수
RKNY : 양양
RKTH : 포항
RKPS : 사천
RKJK : 군산
RKNW : 원주
RKSI : 인천

url_airport = ['RKSS', 'RKPK', 'RKPC','RKTN','RKPU','RKTU','RKJB','RKJJ','RKJY','RKNY','RKTH','RKPS','RKJK','RKNW','RKSI']

*ODP 상대 공항
1:  GMP(김포) / 2: PUS(김해)/ 3: CJU(제주)/  4: TAE(대구) / 5: USN(울산)/
6:    CJJ(청주) / 7: MWX(무안) / 8: KWJ(광주)/ 9: RSU(여수)/ 10: YNY(양양)/
11: KPO(포항) / 12: HIN(사천)/ 13: KUV(군산)/ 14:WJU(원주)/ 15: ICN(인천)

my_dict_ARP = {'url_airport': url_airport,
                "kor": ['GMP(김포)', 'PUS(김해)', 'CJU(제주)', 'TAE(대구)', 'USN(울산)', 'CJJ(청주)', 'MWX(무안)', 'KWJ(광주)', 'RSU(여수)', 'YNY(양양)','KPO(포항)', 'HIN(사천)', 'KUV(군산)', 'WJU(원주)', 'ICN(인천)'],
                'eng': ['ARP' + str(i) for i in range(1,16)]}
my_dict_ARP = pd.DataFrame(my_dict_ARP)

*FLO
A: 아시아나/ B: 에어부산/ F: 이스타항공
I: 진에어/ H: 제주항공
L: 티웨이/ J:대한항공

my_dict_FLO = {'kor': ['아시아나항공', '에어부산', '이스타항공', '진에어', '제주항공', '티웨이항공', '대한항공'], "eng": ['A', 'B', 'F', 'I', 'H', 'L', 'J']}
my_dict_FLO = pd.DataFrame(my_dict_FLO)

*SDT_DY 요일
julyday = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
augustday = ['thu', 'fri', 'sat', 'sun','mon', 'tue', 'wed']
septemberday = ['sun', 'mon', 'tue', 'wed', 'thu', 'fri', 'sat']

*AOD 출도착
AODlist = {'eng' : ['A', 'D'], 'kor' : ['도착', '출발']}
AODlist = pd.DataFrame( AODlist)

base_url = 'http://www.airportal.co.kr/servlet/aips.life.airinfo.RbHanCTL?gubun=c_getList'
result = pd.DataFrame([])
start = timeit.default_timer()

for d in range(0,8) :
    month = '201909' # 월을 바꾸어서 넣어줌
    date = month + format(d+1).zfill(2)

    for k in range(2) :
        depArr = AODlist.eng[k]
        aodkor = AODlist.kor[k]

        for i in range(15) :
            airport = my_dict_ARP.loc[i,'url_airport']

            url = base_url + '&depArr=' + depArr + '&current_date=' + date + '&airport=' + airport + '&al_icao=&fp_id='
            driver = webdriver.Chrome('C:/Users/Danah/Documents/chromedriver_win32/chromedriver.exe')
            driver.implicitly_wait(5)
            driver.get(url)

            html = driver.page_source
            soup = bs(html, 'html.parser')
            temp = soup.find_all('table')
            driver.close()

            p = parser.make2d(temp[1])
            df = pd.DataFrame(p[0:],columns=['FLO_kor', '', 'FLT', '', 'ODP_kor', '', 'STT', '', 'Expected Time', '', 'ATT', '', 'Type', '', 'DLY'])
            df = df.iloc[::2,::2]
            df = df.drop(['Expected Time', 'Type', 'ATT'], axis=1)

            col = list(df.columns)
            df.insert(col.index('ODP_kor'), 'ODP', df['ODP_kor'].map(my_dict_ARP.set_index('kor')['eng']))
            df = df.drop('ODP_kor', axis=1)
            df.insert(col.index('FLO_kor'), 'FLO', df['FLO_kor'].map(my_dict_FLO.set_index('kor')['eng']))
            df = df.drop('FLO_kor', axis=1)
            df.DLY = list([0 if df['DLY'].iloc[i] == aodkor else 1 if df['DLY'].iloc[i] == '지연' else np.nan for i in range(0, len(df))])
            df = df.dropna(axis=0)

            df['ARP'] = np.repeat(my_dict_ARP.loc[i,'eng'], df.shape[0])
            df['SDT_YY'] = np.repeat(2019, df.shape[0])
            df['SDT_MM'] = np.repeat(9, df.shape[0])
            df['SDT_DD'] = np.repeat(d+1, df.shape[0])
            #df['SDT_DY'] = np.repeat(julyday[d % 7], df.shape[0]) # july
            #df['SDT_DY'] = np.repeat(augustday[d % 7], df.shape[0]) # august
            df['SDT_DY'] = np.repeat(septemberday[d % 7], df.shape[0]) # september
            df['AOD'] = np.repeat(depArr, df.shape[0])

            # 정렬
            df = df[['SDT_YY', 'SDT_MM', 'SDT_DD', 'SDT_DY', 'ARP', 'ODP', 'FLO', 'FLT', 'AOD', 'STT', 'DLY']]
            result = result.append(df,  ignore_index=True)
            print([date, depArr, airport, i])

    # 마지막 시행 end 타임
    end = timeit.default_timer()

# 실행 시간 계산(ms)
elapsed = end - start
print("WorkingTime: {} mins".format(elapsed/60))

# result_july csv 저장
result_july = result
result_july.shape
result_july.head()
result_july.to_csv('testset_july.csv', index=False)

# result_august csv 저장
result_august_p1 = result
result_august_p2 = result
result_august_p1.shape[0] + result_august_p2.shape[0]

result_august_p1[result_august_p1['SDT_DD']==15].index.tolist()[0]
result_august_p1.loc[15584]
result_august_p1 = result_august_p1.loc[1:15583]

result_august = result_august_p1.append(result_august_p2, ignore_index=True)
result_august.shape
result_august.tail()
result_august.to_csv('testset_august.csv', index=False)

# result_september csv 저장
result_september = result
result_september.shape
result_september.head()
result_september.to_csv('testset_september.csv', index=False)


#############################################
#####               EDA                 #####
#############################################

# Imbalance
p9.ggplot(df, p9.aes(x='DLY',fill='DLY')) +p9.geom_bar()
p9.ggplot(df, p9.aes(x='SDT_YY',fill='DLY')) +p9.geom_bar()

# Monthly Delay
plt.title("Month Delay")
a=plt.plot(df_17[["SDT_MM","DLY"]].groupby("SDT_MM").count() ,color='#EC2049')
b=plt.plot(df_18[["SDT_MM","DLY"]].groupby("SDT_MM").count() ,color='#F7DB4F')
c=plt.plot(df_19[["SDT_MM","DLY"]].groupby("SDT_MM").count() ,color='#2F9599')
plt.legend(('2017','2018','2019'),loc='best')
plt.show()

# ARP Delay
df_Y = df[df.DLY=='Y']
sorted = df_Y[["ARP","DLY"]].groupby("ARP").count().sort_values('DLY',ascending=True)

# Jeju Delay
df_jeju = df_Y[df.ARP == "ARP3"]
df_jeju17 = df_17[df.ARP == "ARP3"]
df_jeju18 = df_18[df.ARP == "ARP3"]
df_jeju19 = df_19[df.ARP == "ARP3"]

plt.title("jeju")
plt.plot(df_jeju[["SDT_MM","DLY"]].groupby("SDT_MM").count() ,color='red')
plt.plot(df_jeju17[["SDT_MM","DLY"]].groupby("SDT_MM").count() ,color='blue')
plt.plot(df_jeju18[["SDT_MM","DLY"]].groupby("SDT_MM").count() ,color='green')
plt.plot(df_jeju19[["SDT_MM","DLY"]].groupby("SDT_MM").count() ,color='grey')
plt.show()

# Delay Reason
sort_DRR = df_Y[["DRR","DLY"]].groupby("DRR").count().sort_values('DLY',ascending=True)
sort_DRR.plot.bar(figsize=(18,4),color='#45ADA8',title="Delay Reason")

df_not_AC = df_Y[df_Y.DRR != "C02"]
No_AC_sort=df_not_AC[["DRR","DLY"]].groupby("DRR").count().sort_values('DLY',ascending=True)

No_AC_sort.plot.bar(figsize=(18,8),color='#45ADA8',title="Delay Reason")

# IRR
p9.ggplot(train, p9.aes(x='IRR',fill='DLY')) +p9.geom_bar(position='fill')


#############################################
#####       Feature Engineerting        #####
#############################################

# Make a STT_time from STT
def extract_time(row):
    return row.split(':')[0]

df["STT_time"] = df["STT"].apply(extract_time)

# Merge Dataset
## weather
weather = pd.read_csv('C:/Users/letsd/Desktop/19빅콘/퓨처스/weather.csv')

def extract_weahter_date(row):
    return row.split(' ')[0]
def extract_weather_time(row):
    return row.split(' ')[1]
def extract_year(row):
    return row.split('-')[0]
def extract_month(row):
    return row.split('-')[1]
def extract_date(row):
    return row.split('-')[2]
def extract_time(row):
    return row.split(':')[0]

def extract_zero(row):
    return row.lstrip('0')


weather["date"] = weather["SDT"].apply(extract_weahter_date)
weather["time"] = weather["SDT"].apply(extract_weather_time)

weather["SDT_YY"] = weather["date"].apply(extract_year)
weather["SDT_MM"] = weather["date"].apply(extract_month)
weather["SDT_DD"] = weather["date"].apply(extract_date)

weather["SDT_MM"] = weather["SDT_MM"].apply(extract_zero)
weather["SDT_DD"] = weather["SDT_DD"].apply(extract_zero)

weather["STT_time"] = weather["time"].apply(extract_time)

del weather["date"]
del weather["time"]
del weather["SDT"]

weather["SDT_YY"] = weather["SDT_YY"].astype('int64')
weather["SDT_MM"] = weather["SDT_MM"].astype('int64')
weather["SDT_DD"] = weather["SDT_DD"].astype('int64')

df_merge = pd.merge(df, weather, on=["ARP","SDT_YY","SDT_MM","SDT_DD","STT_time"], how="left")

df_merge.dropna(subset=['sight'], how='all', inplace = True)
df_merge.isnull().sum()

df_merge = df_merge.reset_index()
del df_merge["index"]

## airport
airport = pd.read_csv('C:/Users/letsd/Desktop/19빅콘/퓨처스/국내공항시설_현황(20181231).csv')
df_merge = pd.merge(df_merge, airport, on=["ARP"], how="left")

# Making features
## LCC
Nation = ["A","J"]
LCC = ["B","F","I","H","L"]
ETC = ["C","D","E","G","K"]

df_merge['LCC'] = list([1 if df_merge['FLO'][i] in LCC else 0 if df_merge['FLO'][i] in Nation else 2 for i in range(0, len(df_merge))])

## FLO/Area
a = df_merge[["ARP","FLO"]].sort_values(by = ["ARP","FLO"], ascending=True).drop_duplicates(["ARP","FLO"], keep='first')

num_FLO = a.groupby("ARP").count()
num_FLO.rename(columns={'FLO':'num_FLO'},inplace=True)

df_merge = pd.merge(df_merge, num_FLO, on=["ARP"], how="left")
df_merge['FLO/Area'] = round(df_merge['num_FLO'] / df_merge['Area'] * 1000000,0)

## Y_rate_ARP & Y_rate_FLO
a = df_merge[["ARP","Y"]].sort_values(by = ["ARP","Y"], ascending=True)
rate_ARP = round(a.groupby("ARP").sum() / a.groupby("ARP").count(),2)
rate_ARP.rename(columns={'Y':'Y_rate_ARP'},inplace=True)

a = df_merge[["FLO","Y"]].sort_values(by = ["FLO","Y"], ascending=True)
rate_FLO = round(a.groupby("FLO").sum() / a.groupby("FLO").count(),2)
rate_FLO.rename(columns={'Y':'Y_rate_FLO'},inplace=True)

df_merge = pd.merge(df_merge,rate_ARP, on=["ARP"], how="left")
df_merge = pd.merge(df_merge,rate_FLO, on=["FLO"], how="left")


## Holiday
apikey="FzUV7PUBcXXMk17rgEtB1%2FSBi58HuCFoIEKhy87Sn1uAYveJeOiykOVgmirP4p9mXuR9yOlf0YMUNgjDA2sgQw%3D%3D"
solYear=['2017','2018','2019']
solMonth=['01','02','03','04','05','06','07','08','09','10','11','12']
url1 = "http://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService/getRestDeInfo?solYear="
url2 = "&solMonth={solMonth}&"
url3 = "ServiceKey={apikey}"

gong=[]
for i in solYear :
    apiwy = url1+i+url2+url3
    for j in solMonth :
        url=apiwy.format(solMonth=j,apikey=apikey)
        r=requests.get(url)
        soup = BeautifulSoup(r.content, 'html.parser')
        loc=soup.select('locdate')
        for l in loc :
            print(l.text)
            try:
                gong.append(l.text)
            except:
                gong.append('N/A')

gong_date=[]
for g in gong :
    gong_date.append(datetime.datetime.strptime(g,'%Y%m%d').date())

year = []
length=range(0,len(gong_date)-1)
for l in length :
    year.append(gong_date[l].year)

month = []
for l in length :
    month.append(gong_date[l].month)

day = []
for l in length :
    day.append(gong_date[l].day)

HOL = []
for l in length :
    HOL.append(1)

gong_data = pd.DataFrame(list(zip(year,month,day,HOL)),columns=['SDT_YY','SDT_MM','SDT_DD','HOL'])

df_merge = pd.merge(df_merge,gong_data,on=['SDT_YY','SDT_MM','SDT_DD'],how='left')
df_merge['HOL'].fillna(0,inplace=True)


## Drop features
del df_merge["FLT","REG","IRR","STT","ATT","DRR","CNL","CRR"]


#############################################
#####              Modeling             #####
#############################################
# 완성 data 불러오고 편집하기
train_merge_hol = pd.read_csv('C:/Users/YISS/Desktop/HY/train_merge_hol_v2.csv')
valid_merge_hol = pd.read_csv('C:/Users/YISS/Desktop/HY/valid_merge_hol.csv')

df = train_merge_hol.append(valid_merge_hol)

df["STT_time"] = df["STT_time"].astype("int64")

df = pd.get_dummies(df)
df.info()

# validation set 나누기
train, valid = train_test_split(df, test_size=0.2, random_state=0)

X_train = train.loc[:, train.columns != 'Y']
y_train = train.loc[:, train.columns == 'Y']

X_valid = valid.loc[:, valid.columns != 'Y']
y_valid = valid.loc[:, valid.columns == 'Y']

# Imbalanced data 처리 - SMOTE
sm = SMOTE(random_state=0)
X_resampled, y_resampled = sm.fit_sample(X_train,y_train)

print("After OverSampling, counts of label '1': {}".format(sum(y_resampled==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_resampled==0)))

X_resampled = pd.DataFrame(X_resampled)
y_resampled = pd.DataFrame(y_resampled)
X_resampled.head()

y_resampled.columns = ["Y"]
train = pd.concat([X_resampled, y_resampled], axis=1)
train = train.astype('int64')

X_resampled = train.loc[:, train.columns != 'Y']
y_resampled = train.loc[:, train.columns == 'Y']
X_resampled.columns = ['SDT_YY', 'SDT_MM', 'SDT_DD', 'STT_time', 'Area', 'Runways', 'Apron',
       'Passeger Terminal', 'Parking', 'Cargo Terminal', 'Aircraft Movement',
       'Parking Capacity', 'Passenger', 'Parking  Capacity', 'Cargo',
       'Minimum Landing Visibiling', 'Aircraft Movement18', 'Passenger18',
       'Cargo18', 'RVR', 'num_FLO', 'FLO/Area', 'Y_rate_ARP', 'Y_rate_FLO',
       'HOL', 'wind', 'sight', 'cloud', 'temp', 'dew_temp', 'LCC',
       'SDT_DY_fri', 'SDT_DY_mon', 'SDT_DY_sat', 'SDT_DY_sun', 'SDT_DY_thu',
       'SDT_DY_tue', 'SDT_DY_wed', 'ARP_ARP1', 'ARP_ARP10', 'ARP_ARP11',
       'ARP_ARP12', 'ARP_ARP13', 'ARP_ARP14', 'ARP_ARP15', 'ARP_ARP2',
       'ARP_ARP3', 'ARP_ARP4', 'ARP_ARP5', 'ARP_ARP6', 'ARP_ARP7', 'ARP_ARP8',
       'ARP_ARP9', 'ODP_ARP1', 'ODP_ARP10', 'ODP_ARP11', 'ODP_ARP12',
       'ODP_ARP13', 'ODP_ARP14', 'ODP_ARP15', 'ODP_ARP2', 'ODP_ARP3',
       'ODP_ARP4', 'ODP_ARP5', 'ODP_ARP6', 'ODP_ARP7', 'ODP_ARP8', 'ODP_ARP9',
       'FLO_A', 'FLO_B', 'FLO_C', 'FLO_D', 'FLO_E', 'FLO_F', 'FLO_G', 'FLO_H',
       'FLO_I', 'FLO_J', 'FLO_K', 'FLO_L', 'AOD_A', 'AOD_D']

## LightGBM
d_train = lgbm.Dataset(X_resampled, y_resampled)
d_test = lgbm.Dataset(X_valid, y_valid)

n_folds = 5
random_seed=6

# BayesianOptimization
def lgb_eval(num_leaves, max_depth, lambda_l1, lambda_l2, min_split_gain, min_child_weight):
    params = {'application':'binary',
              'num_iterations': 500 ,
              'learning_rate':0.05,
              'early_stopping_round':100,
              "objective" : "binary",
              "num_threads" : 20 ,
             }
    params["num_leaves"] = int(round(num_leaves))
    params['max_depth'] = int(round(max_depth))
    params['lambda_l1'] = max(lambda_l1, 0)
    params['lambda_l2'] = max(lambda_l2, 0)
    params['min_split_gain'] = min_split_gain
    params['min_child_weight'] = min_child_weight
    cv_result = lgbm.cv(params, d_train,
                       nfold=n_folds, seed=random_seed,
                       stratified=True, verbose_eval =200,
                       metrics=["auc"]
                        )
    return max(cv_result['auc-mean'])

lgbBO = BayesianOptimization(lgb_eval, {'num_leaves': (24, 45),
                                        'max_depth': (5, 8.99),
                                        'lambda_l1': (0, 5),
                                        'lambda_l2': (0, 3),
                                        'min_split_gain': (0.001, 0.1),
                                        'min_child_weight': (5, 50)}, random_state=0)
init_round = 1
opt_round = 5
lgbBO.maximize(init_points=init_round, n_iter=opt_round)

lgbBO.max

params2= {'lambda_l1': int(0.007061717901106768),
  'lambda_l2': int(2.5784660897082654),
  'max_depth': int(8.416918245113665),
  'min_child_weight': int(48.0754057699758),
  'min_split_gain': int(0.03664457054269856),
  'num_leaves': int(44.954395892779246)}

lgb2 = lgbm.train(params2, d_train)
lgb_prob = lgb2.predict( X_valid )

valid_scores = metrics.roc_auc_score(y_valid, lgb_prob)
valid_scores

## XGBoost
dtrain = xgb.DMatrix(X_resampled, label=y_resampled)
dtest = xgb.DMatrix(X_valid)

def xgb_eval(max_depth,
              gamma,
              min_child_weight,
              max_delta_step,
              subsample,
              colsample_bytree):
    params = { 'booster' : 'gbtree',
              'max_depth' : int(max_depth),
              'gamma' : gamma,
              'eta' : 0.1,
              'objective' : 'binary:logistic',
              'silent' : True,
              'eval_metric': 'auc',
              'subsample' : max(min(subsample, 1), 0),
              'colsample_bytree' : max(min(colsample_bytree, 1), 0),
              'min_child_weight' : min_child_weight,
              'max_delta_step' : int(max_delta_step),
              'seed' : 1001 }

    cv_result = xgb.cv(
                    params,
                    dtrain,
                    stratified = True,
                    nfold = 5,
                    num_boost_round = 30,
                    verbose_eval = 10,
                    early_stopping_rounds = 20,
                    metrics = "auc")
    return max(cv_result['train-auc-mean'])

xgb_bo = BayesianOptimization(xgb_eval, {'max_depth': (2, 8),
                                         'gamma': (0.001, 10.0),
                                         'min_child_weight': (0, 20),
                                         'max_delta_step': (0, 10),
                                         'subsample': (0.0, 1.0),
                                         'colsample_bytree' :(0.0, 1.0)})

init_round = 1
opt_round = 5
xgb_bo.maximize(init_points=init_round, n_iter=opt_round)

xgb_bo.max

params2= {'colsample_bytree': int(0.8436828535349092),
          'gamma': int(5.445026823724472),
          'max_delta_step': int(4.805277543755815),
          'max_depth': int(4.635537764423709),
          'min_child_weight': int(19.937763933841033),
          'subsample': int(0.3631126541327174)}

xgb2 = xgb.train(params2, dtrain)

xgb_prob = xgb2.predict( dtest )

valid_scores_xgb = metrics.roc_auc_score(y_valid, xgb_prob)
valid_scores_xgb


## RandomForest
def rf_eval(n_estimators, max_features, max_depth, min_samples_leaf, min_samples_split):
    params = {'n_estimators':35,
                'max_features':1,
                'max_depth':None,
                'min_samples_leaf':1,
                'min_samples_split':1
             }
    params["n_estimators"] = int(n_estimators)
    params['max_features'] = int(max_features)
    params['max_depth'] = int(max_depth)
    params['min_samples_leaf'] = int(min_samples_leaf)
    params['min_samples_split'] = int(min_samples_split)

    RFC = RandomForestClassifier()
    RFC.fit(X_resampled, y_resampled)
    cv_result = cross_val_score(RFC, X_valid, y_valid, cv = 10, scoring='roc_auc')
    return cv_result.mean()

rfBO = BayesianOptimization(rf_eval, {'n_estimators':(10,200),
                                        'max_features':(0,10),
                                        'max_depth':(0,20),
                                        'min_samples_leaf':(0,50),
                                        'min_samples_split':(0,50)}, random_state=0)
init_round = 1
opt_round = 5
rfBO.maximize(init_points=init_round, n_iter=opt_round)

rfBO.max

RFC = RandomForestClassifier(max_depth=int(10.976270078546495),
                              max_features= int(7.151893663724195),
                              min_samples_leaf= int(30.138168803582193),
                              min_samples_split= int(27.244159149844844),
                              n_estimators= int(90.4944118743919))
RFC.fit(X_resampled, y_resampled)
rf_prob = RFC.predict( X_valid )

valid_scores = metrics.roc_auc_score(y_valid, rf_prob)
valid_scores


## Stacking
LGB = lgbm.LGBMClassifier(lambda_l1= int(0.007061717901106768),
                            lambda_l2= int(2.5784660897082654),
                            max_depth= int(8.416918245113665),
                            min_child_weight= int(48.0754057699758),
                            min_split_gain= int(0.03664457054269856),
                            num_leaves= int(44.954395892779246))

XGB = XGBClassifier(colsample_bytree= int(0.8436828535349092),
                    gamma= int(5.445026823724472),
                    max_delta_step= int(4.805277543755815),
                    max_depth= int(4.635537764423709),
                    min_child_weight= int(19.937763933841033),
                    subsample= int(0.3631126541327174))

RFC = RandomForestClassifier(max_depth=int(10.976270078546495),
                              max_features= int(7.151893663724195),
                              min_samples_leaf= int(30.138168803582193),
                              min_samples_split= int(27.244159149844844),
                              n_estimators= int(90.4944118743919))

# CAT = CatBoostClassifier()

clf_list = [LGB, XGB, RFC
            # ,CAT
            ]

sclf_LR = StackingClassifier(classifiers=clf_list, meta_classifier=LogisticRegression())
sclf_LR2 = StackingClassifier(classifiers=clf_list, meta_classifier=LogisticRegression(penalty='l2'))
sclf_MLP = StackingClassifier(classifiers=clf_list, meta_classifier=MLPClassifier(hidden_layer_sizes=(100,50)))
sclf_MLP2 = StackingClassifier(classifiers=clf_list, meta_classifier=MLPClassifier(hidden_layer_sizes=(10,5)))
sclf_XGB = StackingClassifier(classifiers=clf_list, meta_classifier=XGBClassifier(n_estimators=100))

sclf_LR2.fit(X_resampled,y_resampled)

valid_pred = sclf_LR2.predict(X_valid)
valid_scores = metrics.roc_auc_score(y_valid, valid_pred)
valid_scores


#############################################
#####        Feature Importance         #####
#############################################

plt.rcParams["figure.figsize"] = (15,15)
lgbm.plot_importance(lgb2)
plt.show()
