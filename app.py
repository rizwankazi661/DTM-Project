from flask import Flask,render_template,request
import pickle
import numpy as np

app = Flask(__name__)                       # Used to initialize app

model = pickle.load(open('model.pkl','rb'))

@app.route("/")                             # For making multiple pages
def hello_world():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET'])
def predict():

    presentPrice = int(request.form['Present_Price'])
    carAge = int(request.form['Car_Age'])
    fuelType = int(request.form['Fuel_Type'])
    sellerType = int(request.form['Seller_Type'])
    transmissionType = int(request.form['Transmission_Type'])

    features =[[presentPrice,carAge,fuelType,sellerType,transmissionType]]
    price = model.predict(features)
    output = "{:.2f}".format(price[0])

    return render_template('index.html',price='Price : {} Lakhs'.format(output))

if __name__ == "__main__":
    app.run(debug=True)                     # if any error then it will be visible in the browser itself
    # app.run(debug=True,port = 8000)       # If you want to change the port

    # predict()
