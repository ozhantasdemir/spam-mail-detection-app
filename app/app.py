from flask import Flask, render_template, request, jsonify
from utilities import predict_mail, model_final


app = Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict():

    if request.method == 'POST':

        message = request.form['message']
        data = [message]
        result = predict_mail(model_final, data)

    return render_template('index.html', prediction= result[0])

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
