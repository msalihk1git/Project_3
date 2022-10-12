#import relevant libraries for flask, html rendering and loading the ML model
from flask import Flask,request, url_for, redirect, render_template
import pickle
import pandas as pd
import joblib

app = Flask(__name__)


#model = pickle.load(open("model.pkl","rb"))
model = joblib.load("model.pkl")
#scale = pickle.load(open("scale.pkl","rb"))
scale = joblib.load("scale.pkl")
le_dept=joblib.load('le_dept.pkl')
le_edu=joblib.load('le_edu.pkl')
le_gen=joblib.load('le_gen.pkl')
le_rech=joblib.load('le_rech.pkl')
le_reg=joblib.load('le_reg.pkl')

# region_label = joblib.load("region_label.pkl")

@app.route("/")
def landingPage():
    return render_template('index.html')

@app.route("/predict",methods=['POST'])
def predict():
    
    Department = request.form['1']
    Department = le_dept.transform([Department])[0]
    region = request.form['2']
    region = le_reg.transform([region])[0]
    # print(Region)
    # Region = region_label.tranform(Region)
    Education = request.form['3']
    Education = le_edu.transform([Education])[0]
    Gender = request.form['4']
    Gender = le_gen.transform([Gender])[0]
    recruitment_channel = request.form['5']
    recruitment_channel = le_rech.transform([recruitment_channel])[0]
    no_of_trainings = request.form['6']
    age = request.form['7']
    previous_year_rating = request.form['8']
    length_of_service = request.form['9']
    KPIs_met_above_80_percent = request.form['10']
    awards_won = request.form['11']
    avg_training_score = request.form['12']

    # return "true"

    rowDF= pd.DataFrame([pd.Series([Department,region,Education,Gender,recruitment_channel,no_of_trainings,age,previous_year_rating,length_of_service,KPIs_met_above_80_percent,awards_won,avg_training_score])])
    rowDF_new = pd.DataFrame(scale.transform(rowDF))

    print(rowDF_new)

    #  model prediction 
    prediction= model.predict_proba(rowDF_new)
    print(f"The  Predicted values is :{prediction[0][1]}")

    if prediction[0][1] >= 0.5:
        valPred = round(prediction[0][1],3)
        print(f"The Round val {valPred*100}%")
        return render_template('result.html',pred=f'Probability of  being Promoted is {valPred*100}%.')
    else:
        valPred = round(prediction[0][0],3)

        return render_template('result.html',pred=f'probability of being  not Promoted is {valPred*100}%.')

if __name__ == '__main__':
    app.run(debug=True)
    
