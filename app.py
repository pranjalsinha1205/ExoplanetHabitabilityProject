# app.py

from flask import Flask, render_template, request
from utils.predictor import predict_similarity

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_data = {key: float(request.form[key]) for key in request.form}
        similarity, habitable = predict_similarity(input_data)

        return render_template('result.html',
                               similarity=similarity,
                               habitable=habitable)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
