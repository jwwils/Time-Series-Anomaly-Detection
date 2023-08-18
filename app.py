from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from werkzeug.utils import secure_filename
import plotly.express as px
from plotly.offline import plot

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Read the data to get column names
        data = pd.read_csv(filepath)
        columns = data.columns.tolist()
        return render_template('select_columns.html', columns=columns, filename=filename)
    else:
        return redirect(request.url)

@app.route('/process', methods=['POST'])
def process_file():
    time_column = request.form.get('time_column')
    value_column = request.form.get('value_column')
    filename = request.form.get('filename')
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Read the data using the user-selected columns
    data = pd.read_csv(filepath, usecols=[time_column, value_column])
    data.set_index(time_column, inplace=True)
    data.columns = ['Value']
    data.fillna(method='ffill', inplace=True)

    # Feature Engineering
    # Rolling window features
    data['rolling_mean'] = data['Value'].rolling(window=3).mean()
    data['rolling_std'] = data['Value'].rolling(window=3).std()

    # Remove NaN after feature engineering
    data.dropna(inplace=True)

    # Detect anomalies using Isolation Forest
    model_if = IsolationForest(contamination=0.05)
    data['anomaly_if'] = model_if.fit_predict(data[['Value', 'rolling_mean', 'rolling_std']])
    anomalies_if = data[data['anomaly_if'] == -1]

    # Detect anomalies using One-Class SVM
    model_svm = OneClassSVM(nu=0.05)
    data['anomaly_svm'] = model_svm.fit_predict(data[['Value']])
    anomalies_svm = data[data['anomaly_svm'] == -1]

    # Exporting results
    export_filename = "processed_" + filename
    data.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], export_filename))

    original_plot = plot(px.line(data, x=data.index, y='Value', title='Original Data'), output_type='div')

    return render_template('results.html', 
                           original_plot=original_plot,
                           plot_if=plot(px.scatter(data, x=data.index, y='Value', color='anomaly_if', title='Isolation Forest'), output_type='div'),
                           plot_svm=plot(px.scatter(data, x=data.index, y='Value', color='anomaly_svm', title='One-Class SVM'), output_type='div'),
                           anomalies_if=anomalies_if,
                           anomalies_svm=anomalies_svm,
                           export_filename=export_filename)

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
