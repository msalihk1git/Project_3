#import relevant libraries for flask,html rendering and loading the ML model
from flask import Flask,request,url_for,render_template
import pickle
import pandas as pd
import joblib
app = Flask(__name__)
# model=pickle.load(open("model.pkl","rb"))
model=joblib.load(open("model.pkl","rb"))
# scale=pickle.load(open("scale.pkl","rb"))
scale=joblib.load(open("scale.pkl","rb"))


@app.route("/")
def landingPage():
    return render_template("index.html") 

@app.route("/predict",methods=["POST"])
def predict():
    department = request.form['1']
    region = request.form['2']
    education = request.form['3']
    gender = request.form['4']
    recruitment_channel = request.form['5']
    no_of_trainings = request.form['6']
    age = request.form['7']
    previous_year_rating = request.form['8']
    length_of_service = request.form['9']
    KPIs_met= request.form['10']
    awards_won = request.form['11']
    avg_training_score = request.form['12']
    rowDf=pd.DataFrame([pd.Series([department,region,education,gender,recruitment_channel,no_of_trainings,age,previous_year_rating,length_of_service,KPIs_met,awards_won,avg_training_score])])
    rowDf_new=pd.DataFrame(scale.transform(rowDf))
    
    print(rowDf_new)

#  model prediction 
    prediction= model.predict_proba(rowDf_new)
    print(f"The  Predicted values is :{prediction[0][1]}")

    if prediction[0][1] >= 0.5:
        valPred = round(prediction[0][1],3)
        print(f"The Round val {valPred*100}%")
        return render_template('result.html',pred=f'You have a chance of having diabetes.\n\nProbability of you being a diabetic is {valPred*100}%.\n\nAdvice : Exercise Regularly')
    else:
        valPred = round(prediction[0][0],3)
        return render_template('result.html',pred=f'Congratulations!!!, You are in a Safe Zone.\n\n Probability of you being a non-diabetic is {valPred*100}%.\n\n Advice : Exercise Regularly and maintain like this..!')
if __name__ == "__main__":
    app.run(debug=True)
          