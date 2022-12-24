from flask import Flask,render_template,request,jsonify
# import numpy
import joblib
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
import re
import pickle
import tensorflow
import webbrowser
class_list=["Bad","Good"]
with open('./Models/Tockenizer.pickle',"rb") as f:
    Vectorizer=pickle.load(f)
with open('./Models/word_lemitizer.pickle',"rb") as f:
    word_lemitizer=pickle.load(f)
app=Flask(__name__)
# word_lemitizer=WordNetLemmatizer()
Regular_expression_definition_for_html_tags=re.compile('<.*?>')
Regular_expression_definition_for_digits=re.compile('\d+\s|\s\d+|\s\d+\s')
# english_stop_words=stopwords.words('english')
with open('./Models/english_stop_words.pickle',"rb") as f:
    english_stop_words=pickle.load(f)

def preprocessing_of_sentence(text):
    word_to_be_handled=[
    "not",
    "no",
    "very"
    ]
    text=Regular_expression_definition_for_html_tags.sub(r" ",text)
    text=Regular_expression_definition_for_digits.sub(r" ",text)
    punctuations = [".",",","!","?","'",'"',":",";","*","-","/","+","%","$","#","@","(",")","[","]","{","}"]
    for i in punctuations:
        text = text.replace(i," ")
    text=text.lower().split()
    text=[word for word in text if len(word)>1 and word not in english_stop_words or word in word_to_be_handled]
    text=[word_lemitizer.lemmatize(word) for word in text]
    # print(text)
    Vector=Vectorizer.transform([" ".join(text)])
    return Vector
def getSentimentFromNeuranNetwork(value):
    if value>0.6:
        return "Good"
    elif value<0.3:
        return "Bad"
    else:
        return "Neutral"


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/getresult',methods=['POST','GET'])
def getdiseaseresult():
    print("getresult is requested")
    if request.method=='POST':
        print("Post Method")
        print(request.json['review'])
        if request.json['review']=="":
            return jsonify({"data":"Error occured"})
        else:
            input_=preprocessing_of_sentence(request.json['review'])
            # print(input_)
            # svmModel=joblib.load("./Models/svmModel.joblib")
            # logisticModel=joblib.load("./Models/logisticModel.joblib")
            # naive_bayesModel=joblib.load("./Models/naive_bayesModel.joblib")
            # randomForestClassifierModel=joblib.load("./Models/randomForestClassifierModel.joblib")
            NeuralNetworkModel1=tensorflow.keras.models.load_model('./Models/NeuralNetworkModel1.h5')
            # prediction=class_list[svmModel.predict(input_)[0]]
            # prediction=class_list[logisticModel.predict(input_)[0]]
            # prediction=class_list[naive_bayesModel.predict(input_)[0]]
            # prediction=class_list[randomForestClassifierModel.predict(input_)[0]]
            print(NeuralNetworkModel1.predict(input_.toarray())[0][0])
            prediction=getSentimentFromNeuranNetwork(NeuralNetworkModel1.predict(input_.toarray())[0][0])

            if prediction=="Good":
                return jsonify({"data":"Good"})
            elif prediction=="Bad":
                return jsonify({"data":"Bad"})
            else:
                return jsonify({"data":"Neutral"})
    else:
        return jsonify({"data":"Error occured"})


if __name__=='__main__':
    # webbrowser.open("http://127.0.0.1:5000/")
    app.run(debug=True)