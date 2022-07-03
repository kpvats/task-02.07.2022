import numpy
import pickle
from flask import Flask, render_template,request,jsonify
import sklearn


app = Flask(__name__)

@app.route('/')
def Home():
    return render_template('index.html')


@app.route('/predict',methods= ['POST','GET'])
def results():
    x1 = float(request.form['fixed acidity'])
    x2 = float(request.form['volatile acidity'])
    x3 = float(request.form['citric acid'])
    x4 = float(request.form['residual sugar'])
    x5 = float(request.form['chlorides'])
    x6 = float(request.form['free sulfur dioxide'])
    x7 = float(request.form['density'])
    x8 = float(request.form['pH'])
    x9 = float(request.form['sulphates'])
    x10=float(request.form['alcohol'])


    x = numpy.array([[x1,x2,x3,x4,x5,x6,x7,x8,x9,x10]])

    sc = pickle.load(open('F_scale.pkl','rb'))
    x_std = sc.transform(x)

    model = pickle.load(open('logistic.pkl','rb'))
    y_prediction=model.predict(x_std)

    return jsonify({'Model Prediction': float(y_prediction)})
if __name__=='__main__':
    app.run(debug=True,port=1010)



