import pickle
import numpy as np
from flask import Flask
from flask import request
from flask import jsonify
import xgboost as xgb

with open('model.bin' ,'rb') as f_in:
    ss , ohe , model = pickle.load(f_in)

numerical = ['timespentoncourse', 'numberofvideoswatched', 'numberofquizzestaken', 'quizscores', 'completionrate']
categorical = ['coursecategory', 'devicetype']

def predict_single(data, ss,ohe, model):
    X_num = ss.transform([[data[col] for col in numerical]])
    X_cat = ohe.transform([[data[col].lower() for col in categorical]])
    X = np.column_stack([X_cat,X_num])
    dX = xgb.DMatrix(X)
    y_pred = model.predict(dX)
    return y_pred

app=Flask('course')

@app.route('/predict',methods=['POST'])
def predict():
    student = request.get_json()
    prediction = predict_single(student,ss,ohe,model)
    course_completion = prediction > 0.5

    result={
        'course_completion_porb' : float(prediction),
        'course_completion' : bool(course_completion)
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True , host='0.0.0.0',port=4545)