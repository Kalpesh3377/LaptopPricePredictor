import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
df=pd.read_csv('laptop_data.csv')
# print(df.head())
# print(df.shape)
# print(df.info())
# print(df.duplicated().sum())
# print(df.isnull().sum())
df.drop(columns=['Unnamed: 0'],inplace=True)
# print(df.head())
df['Ram']=df['Ram'].str.replace('GB','')
df['Weight']=df['Weight'].str.replace('kg','')
# print(df.head())
df['Ram']=df['Ram'].astype('int32')
df['Weight']=df['Weight'].astype('float64')
# print(df.info())
# sns.displot(df['Price'])
# plt.show()
# df['Company'].value_counts().plot(kind='bar')
# plt.show()
# sns.barplot(x=df['Company'],y=df['Price'])
# plt.xticks(rotation='vertical')
# plt.show()
# df['TypeName'].value_counts().plot(kind='bar')
# plt.show()
# sns.barplot(x=df['TypeName'],y=df['Price'])
# plt.xticks(rotation='vertical')
# plt.show()
# sns.displot(df['Inches'])
# plt.show()
# sns.scatterplot(x=df['Inches'],y=df['Price'])
# plt.show()
# print(df['ScreenResolution'].value_counts())
df['Touchscreen']=df['ScreenResolution'].apply(lambda x:1 if 'Touchscreen' in x else 0)
# print(df.head())
# print(df.sample(5))
# df['Touchscreen'].value_counts().plot(kind='bar')
# plt.show()
# sns.barplot(x=df['Touchscreen'],y=df['Price'])
# plt.show()
df['IPS']=df['ScreenResolution'].apply(lambda x:1 if 'IPS' in x else 0)
# print(df.head())
# print(df.sample(5))
# df['IPS'].value_counts().plot(kind='bar')
# plt.show()
# sns.barplot(x=df['IPS'],y=df['Price'])
# plt.show()
new=df['ScreenResolution'].str.split('x',n=1,expand=True)
df['X_res']=new[0]
df['Y_res']=new[1]
# print(df.head())
df['X_res']=df['X_res'].str.replace(',','').str.findall(r'(\d+\.?\d+)').apply(lambda x:x[0])
# print(df.head())
df['X_res']=df['X_res'].astype('int')
df['Y_res']=df['Y_res'].astype('int')
# print(df.info())
# print(df.corr()['Price'])
df['ppi']=(((df['X_res']**2)+(df['Y_res']**2))**0.5/df['Inches']).astype('float')
# print(df.corr()['Price'])
df.drop(columns=['ScreenResolution','X_res','Y_res','Inches'],inplace=True)
# print(df.head())
# print(df['Cpu'].value_counts())
df['Cpu Name']=df['Cpu'].apply(lambda x:" ".join(x.split()[0:3]))
# print(df.head())
def fetch_processor(text):
    if(text=='Intel Core i5' or text=='Intel Core i7' or text=='Intel Core i3'):
        return text
    else:
        if text.split()[0]=='Intel':
            return 'Other Intel Processor'
        else:
            return 'AMD Processor'
df['Cpu Brand']=df['Cpu Name'].apply(fetch_processor)
# print(df.head())
# df['Cpu Brand'].value_counts().plot(kind='bar')
# plt.show()

# sns.barplot(x=df['Cpu Brand'],y=df['Price'])
# plt.xticks(rotation='vertical')
# plt.show()

df.drop(columns=['Cpu','Cpu Name'],inplace=True)
# print(df.head())

# df['Ram'].value_counts().plot(kind='bar')
# plt.show()

# sns.barplot(x=df['Ram'],y=df['Price'])
# plt.xticks(rotation='vertical')
# plt.show()

df['Memory'] = df['Memory'].astype(str).replace('\.0', '', regex=True)
df["Memory"] = df["Memory"].str.replace('GB', '')
df["Memory"] = df["Memory"].str.replace('TB', '000')
new = df["Memory"].str.split("+", n = 1, expand = True)

df["first"]= new[0]
df["first"]=df["first"].str.strip()

df["second"]= new[1]

df["Layer1HDD"] = df["first"].apply(lambda x: 1 if "HDD" in x else 0)
df["Layer1SSD"] = df["first"].apply(lambda x: 1 if "SSD" in x else 0)
df["Layer1Hybrid"] = df["first"].apply(lambda x: 1 if "Hybrid" in x else 0)
df["Layer1Flash_Storage"] = df["first"].apply(lambda x: 1 if "Flash Storage" in x else 0)

df['first'] = df['first'].str.replace(r'\D', '')

df["second"].fillna("0", inplace = True)

df["Layer2HDD"] = df["second"].apply(lambda x: 1 if "HDD" in x else 0)
df["Layer2SSD"] = df["second"].apply(lambda x: 1 if "SSD" in x else 0)
df["Layer2Hybrid"] = df["second"].apply(lambda x: 1 if "Hybrid" in x else 0)
df["Layer2Flash_Storage"] = df["second"].apply(lambda x: 1 if "Flash Storage" in x else 0)

df['second'] = df['second'].str.replace(r'\D', '')

df["first"] = df["first"].astype(int)
df["second"] = df["second"].astype(int)

df["HDD"]=(df["first"]*df["Layer1HDD"]+df["second"]*df["Layer2HDD"])
df["SSD"]=(df["first"]*df["Layer1SSD"]+df["second"]*df["Layer2SSD"])
df["Hybrid"]=(df["first"]*df["Layer1Hybrid"]+df["second"]*df["Layer2Hybrid"])
df["Flash_Storage"]=(df["first"]*df["Layer1Flash_Storage"]+df["second"]*df["Layer2Flash_Storage"])

df.drop(columns=['first', 'second', 'Layer1HDD', 'Layer1SSD', 'Layer1Hybrid',
       'Layer1Flash_Storage', 'Layer2HDD', 'Layer2SSD', 'Layer2Hybrid',
       'Layer2Flash_Storage'],inplace=True)

# print(df.head())

df.drop(columns=['Memory'],inplace=True)
# print(df.head())

# print(df.corr()['Price'])

df.drop(columns=['Hybrid','Flash_Storage'],inplace=True)
# print(df.head())

# print(df['Gpu'].value_counts())
df['Gpu brand'] = df['Gpu'].apply(lambda x:x.split()[0])
# print(df.head())

# print(df['Gpu brand'].value_counts())
df = df[df['Gpu brand'] != 'ARM']
# print(df['Gpu brand'].value_counts())

# sns.barplot(x=df['Gpu brand'],y=df['Price'],estimator=np.median)
# plt.xticks(rotation='vertical')
# plt.show()

df.drop(columns=['Gpu'],inplace=True)
# print(df.head())

# print(df['OpSys'].value_counts())

# sns.barplot(x=df['OpSys'],y=df['Price'])
# plt.xticks(rotation='vertical')
# plt.show()

def cat_os(inp):
    if inp == 'Windows 10' or inp == 'Windows 7' or inp == 'Windows 10 S':
        return 'Windows'
    elif inp == 'macOS' or inp == 'Mac OS X':
        return 'Mac'
    else:
        return 'Others/No OS/Linux'
df['os'] = df['OpSys'].apply(cat_os)
# print(df.head())

df.drop(columns=['OpSys'],inplace=True)

# sns.barplot(x=df['os'],y=df['Price'])
# plt.xticks(rotation='vertical')
# plt.show()

# sns.distplot(df['Weight'])
# plt.show()

# sns.scatterplot(x=df['Weight'],y=df['Price'])
# plt.show()

# print(df.corr()['Price'])
# sns.heatmap(df.corr())
# plt.show()

# sns.distplot(np.log(df['Price']))
# plt.show()

X = df.drop(columns=['Price'])
y = np.log(df['Price'])

# print(X)
# print(y)
#
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.15,random_state=2)
# print(X_train)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score,mean_absolute_error
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor,ExtraTreesRegressor
from sklearn.svm import SVR
# from xgboost import XGBRegressor
# Linear regression
step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = LinearRegression()

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))
# R2 score 0.8073277448418521
# MAE 0.21017827976429174

# Ridge Regression
# step1 = ColumnTransformer(transformers=[
#     ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
# ],remainder='passthrough')
#
# step2 = Ridge(alpha=10)
#
# pipe = Pipeline([
#     ('step1',step1),
#     ('step2',step2)
# ])
#
# pipe.fit(X_train,y_train)
#
# y_pred = pipe.predict(X_test)
#
# print('R2 score',r2_score(y_test,y_pred))
# print('MAE',mean_absolute_error(y_test,y_pred))
# R2 score 0.8127331031311811
# MAE 0.20926802242582954
#
# Lasso Regression
# step1 = ColumnTransformer(transformers=[
#     ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
# ],remainder='passthrough')
#
# step2 = Lasso(alpha=0.001)
#
# pipe = Pipeline([
#     ('step1',step1),
#     ('step2',step2)
# ])
#
# pipe.fit(X_train,y_train)
#
# y_pred = pipe.predict(X_test)
#
# print('R2 score',r2_score(y_test,y_pred))
# print('MAE',mean_absolute_error(y_test,y_pred))
# R2 score 0.8071853945317105
# MAE 0.21114361613472565
#
# KNN
# step1 = ColumnTransformer(transformers=[
#     ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
# ],remainder='passthrough')

# step2 = KNeighborsRegressor(n_neighbors=3)
#
# pipe = Pipeline([
#     ('step1',step1),
#     ('step2',step2)
# ])

# pipe.fit(X_train,y_train)
#
# y_pred = pipe.predict(X_test)
#
# print('R2 score',r2_score(y_test,y_pred))
# print('MAE',mean_absolute_error(y_test,y_pred))
# R2 score 0.8021984604448553
# MAE 0.19319716721521116
# Decision Tree
# step1 = ColumnTransformer(transformers=[
#     ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
# ],remainder='passthrough')

# step2 = DecisionTreeRegressor(max_depth=8)
#
# pipe = Pipeline([
#     ('step1',step1),
#     ('step2',step2)
# ])

# pipe.fit(X_train,y_train)
#
# y_pred = pipe.predict(X_test)

# print('R2 score',r2_score(y_test,y_pred))
# print('MAE',mean_absolute_error(y_test,y_pred))
# R2 score 0.8466456692979233
# MAE 0.1806340977609143
# SVM
# step1 = ColumnTransformer(transformers=[
#     ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
# ],remainder='passthrough')

# step2 = SVR(kernel='rbf',C=10000,epsilon=0.1)
#
# pipe = Pipeline([
#     ('step1',step1),
#     ('step2',step2)
# ])

# pipe.fit(X_train,y_train)
#
# y_pred = pipe.predict(X_test)
#
# print('R2 score',r2_score(y_test,y_pred))
# print('MAE',mean_absolute_error(y_test,y_pred))
# R2 score 0.8083180902257614
# MAE 0.20239059427481307
# Random Forest
# step1 = ColumnTransformer(transformers=[
#     ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
# ],remainder='passthrough')
#
# step2 = RandomForestRegressor(n_estimators=100,
#                               random_state=3,
#                               max_samples=0.5,
#                               max_features=0.75,
#                               max_depth=15)
#
# pipe = Pipeline([
#     ('step1',step1),
#     ('step2',step2)
# ])
#
# pipe.fit(X_train,y_train)
#
# y_pred = pipe.predict(X_test)
#
# print('R2 score',r2_score(y_test,y_pred))
# print('MAE',mean_absolute_error(y_test,y_pred))
# R2 score 0.8873402378382488
# MAE 0.15860130110457718
# ExtraTrees
# step1 = ColumnTransformer(transformers=[
#     ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
# ],remainder='passthrough')
#
# step2 = ExtraTreesRegressor(n_estimators=100,
#                               random_state=3,
#                               max_samples=0.5,
#                               max_features=0.75,
#                               max_depth=15)
#
# pipe = Pipeline([
#     ('step1',step1),
#     ('step2',step2)
# ])
#
# pipe.fit(X_train,y_train)
#
# y_pred = pipe.predict(X_test)
#
# print('R2 score',r2_score(y_test,y_pred))
# print('MAE',mean_absolute_error(y_test,y_pred))
# R2 score 0.8753793123440623
# MAE 0.15979519126758127
# AdaBoost
# step1 = ColumnTransformer(transformers=[
#     ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
# ],remainder='passthrough')
#
# step2 = AdaBoostRegressor(n_estimators=15,learning_rate=1.0)
#
# pipe = Pipeline([
#     ('step1',step1),
#     ('step2',step2)
# ])
#
# pipe.fit(X_train,y_train)
#
# y_pred = pipe.predict(X_test)
#
# print('R2 score',r2_score(y_test,y_pred))
# print('MAE',mean_absolute_error(y_test,y_pred))
# R2 score 0.7929652659237908
# MAE 0.23296532406396742
# Gradient Boost
# step1 = ColumnTransformer(transformers=[
#     ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
# ],remainder='passthrough')
#
# step2 = GradientBoostingRegressor(n_estimators=500)
#
# pipe = Pipeline([
#     ('step1',step1),
#     ('step2',step2)
# ])
#
# pipe.fit(X_train,y_train)
#
# y_pred = pipe.predict(X_test)
#
# print('R2 score',r2_score(y_test,y_pred))
# print('MAE',mean_absolute_error(y_test,y_pred))
# R2 score 0.8823244736036472
# MAE 0.15929506744611283
# XgBoost
# step1 = ColumnTransformer(transformers=[
#     ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
# ],remainder='passthrough')

# step2 = XGBRegressor(n_estimators=45,max_depth=5,learning_rate=0.5)
#
# pipe = Pipeline([
#     ('step1',step1),
#     ('step2',step2)
# ])

# pipe.fit(X_train,y_train)
#
# y_pred = pipe.predict(X_test)
#
# print('R2 score',r2_score(y_test,y_pred))
# print('MAE',mean_absolute_error(y_test,y_pred))
# R2 score 0.8811773435850243
# MAE 0.16496203512600974
# Voting Regressor
# from sklearn.ensemble import VotingRegressor,StackingRegressor

# step1 = ColumnTransformer(transformers=[
#     ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
# ],remainder='passthrough')

#
# rf = RandomForestRegressor(n_estimators=350,random_state=3,max_samples=0.5,max_features=0.75,max_depth=15)
# gbdt = GradientBoostingRegressor(n_estimators=100,max_features=0.5)
# xgb = XGBRegressor(n_estimators=25,learning_rate=0.3,max_depth=5)
# et = ExtraTreesRegressor(n_estimators=100,random_state=3,max_samples=0.5,max_features=0.75,max_depth=10)
#
# step2 = VotingRegressor([('rf', rf), ('gbdt', gbdt), ('xgb',xgb), ('et',et)],weights=[5,1,1,1])
#
# pipe = Pipeline([
#     ('step1',step1),
#     ('step2',step2)
# ])
#
# pipe.fit(X_train,y_train)
#
# y_pred = pipe.predict(X_test)
#
# print('R2 score',r2_score(y_test,y_pred))
# print('MAE',mean_absolute_error(y_test,y_pred))
# R2 score 0.8901036732986811
# MAE 0.15847265699907628
# Stacking
# from sklearn.ensemble import VotingRegressor,StackingRegressor
#
# step1 = ColumnTransformer(transformers=[
#     ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
# ],remainder='passthrough')


# estimators = [
#     ('rf', RandomForestRegressor(n_estimators=350,random_state=3,max_samples=0.5,max_features=0.75,max_depth=15)),
#     ('gbdt',GradientBoostingRegressor(n_estimators=100,max_features=0.5)),
#     ('xgb', XGBRegressor(n_estimators=25,learning_rate=0.3,max_depth=5))
# ]
#
# step2 = StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=100))
#
# pipe = Pipeline([
#     ('step1',step1),
#     ('step2',step2)
# ])
#
# pipe.fit(X_train,y_train)
#
# y_pred = pipe.predict(X_test)
#
# print('R2 score',r2_score(y_test,y_pred))
# print('MAE',mean_absolute_error(y_test,y_pred))
# R2 score 0.8816958647512341
# MAE 0.1663048975120589
# Exporting the Model


# import pickle
#
# pickle.dump(df,open('df.pkl','wb'))
# pickle.dump(pipe,open('pipe.pkl','wb'))
