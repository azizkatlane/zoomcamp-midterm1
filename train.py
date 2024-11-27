import pickle
import boto3
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


AWS_ACCESS_KEY = 'ASIA6PTO7RRASTETAD52'
AWS_SECRET_KEY = 'NIxOYULnSkSxMHLSIr9j4zWiPutnPYqyJgOcPRrk'

s3=boto3.client('s3' ,
        region_name='us-east-1',
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY
    )


bucket_name = 'studentsdataml'
model_path = 'models/model_xgboost.pkl'

df=pd.read_csv('dataset/online_course_engagement_data.csv')

df.columns = df.columns.str.lower()

device={
    0:'desktop',
    1:'laptop'
}
df.devicetype = df.devicetype.map(device)

categorical = list(df.columns[df.dtypes=='object'])
for c in categorical:
    df[c] = df[c].str.lower()

numerical = ['timespentoncourse','numberofvideoswatched','numberofquizzestaken','quizscores','completionrate']

df_full_train , df_test = train_test_split(df , test_size=0.2 , random_state=1)


y_full_train = df_full_train.coursecompletion.values
y_test = df_test.coursecompletion.values

del df_full_train['coursecompletion']
del df_test['coursecompletion']

xgb_params = {
        'eta' : 0.1 ,
        'max_depth' : 10,
        'min_child_weight': 3,

        'objective' : 'binary:logistic',
        'nthreads': 8,
        'eval_metric' : 'auc',

        'seed': 42,
        'verbosity': 1
    }

def train(df_full_train , y_full_train, params):
    ss=StandardScaler()
    X_train_num=ss.fit_transform(df_full_train[numerical])

    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    X_train_cat = ohe.fit_transform(df_full_train[categorical].values)

    X_train = np.column_stack([X_train_cat,X_train_num])
    
    dtrain=xgb.DMatrix(X_train,label=y_full_train)
   

    model=xgb.train(params,dtrain,num_boost_round=200)

    return ss, ohe , model


def predict(model ,ss , ohe ):
    df_test_num = ss.transform(df_test[numerical])
    df_test_cat = ohe.transform(df_test[categorical].values)
    X_test = np.column_stack([df_test_cat,df_test_num])
    dtest = xgb.DMatrix(X_test)
    y_pred = model.predict(dtest)
    return y_pred


ss,ohe,model = train(df_full_train , y_full_train , xgb_params)

y_pred = predict(model , ss , ohe)

auc=roc_auc_score(y_test,y_pred)

print(f'auc={auc}')

with open('model_xgboost.bin' , 'wb') as f:
    pickle.dump((ss,ohe,model),f)

response = s3.upload_file('model_xgboost.bin' , bucket_name ,'model_xgboost.bin' )

print(f'{response}')
print(f'model saved to s3://{bucket_name}/{model_path}')
print('model is saved to model.bin')