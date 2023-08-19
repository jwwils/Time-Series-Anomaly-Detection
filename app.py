from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
import os
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from werkzeug.utils import secure_filename
import plotly.express as px
from plotly.offline import plot
from flask_limiter.util import get_remote_address
from flask_limiter import Limiter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np


app = Flask(__name__)
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["5 per minute"]
)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
EXPORT_FOLDER = 'exports'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['EXPORT_FOLDER'] = EXPORT_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(EXPORT_FOLDER):
    os.makedirs(EXPORT_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def plot_original_data(data):
    fig = px.line(data, x=data.index, y='Value', title='Original Data')
    return plot(fig, output_type='div')

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
@limiter.limit("5 per minute")
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
@limiter.limit("5 per minute")
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

    # K-means Clustering for Anomaly Detection
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data[['Value']].values.reshape(-1, 1))
    
    # Define the number of clusters
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(data_normalized)

    # Calculate distances to cluster centers
    distances = np.linalg.norm(data_normalized - kmeans.cluster_centers_[clusters], axis=1)

    # Determine threshold and identify anomalies
    threshold = np.percentile(distances, 95)
    data['anomaly_kmeans'] = (distances > threshold).astype(int)
    anomalies_kmeans = data[data['anomaly_kmeans'] == 1]

    fig_kmeans = px.scatter(data, x=data.index, y='Value', color='anomaly_kmeans', title='K-means Clustering Anomalies')
    plot_kmeans = plot(fig_kmeans, output_type='div')

    # Feature Engineering for Isolation Forest and SVM
    data['rolling_mean'] = data['Value'].rolling(window=5).mean()
    data['rolling_std'] = data['Value'].rolling(window=5).std()
    data['z_score'] = (data['Value'] - data['rolling_mean']) / data['rolling_std']

    # Removing NaN values introduced by rolling computations
    data = data.dropna()

    # Anomaly detection with Isolation Forest
    model_if = IsolationForest(contamination=0.05)
    data['anomaly_if'] = model_if.fit_predict(data[['Value', 'rolling_mean', 'rolling_std', 'z_score']])
    anomalies_if = data[data['anomaly_if'] == -1]

    fig_if = px.scatter(data, x=data.index, y='Value', color='anomaly_if', title='Isolation Forest Anomalies')
    plot_if = plot(fig_if, output_type='div')

    # Anomaly detection with One-Class SVM
    model_svm = OneClassSVM(nu=0.05)
    data['anomaly_svm'] = model_svm.fit_predict(data[['Value']])
    anomalies_svm = data[data['anomaly_svm'] == -1]

    fig_svm = px.scatter(data, x=data.index, y='Value', color='anomaly_svm', title='One-Class SVM Anomalies')
    plot_svm = plot(fig_svm, output_type='div')

    # Plot original data
    original_plot = plot_original_data(data)

    # Exporting the processed data
    export_filename = f"processed_{filename}"
    export_filepath = os.path.join(app.config['EXPORT_FOLDER'], export_filename)
    data.to_csv(export_filepath)

    # Descriptive Statistics
    count = data['Value'].count()
    mean = data['Value'].mean()
    median = data['Value'].median()
    std = data['Value'].std()
    min_val = data['Value'].min()
    max_val = data['Value'].max()
    q25 = data['Value'].quantile(0.25)
    q75 = data['Value'].quantile(0.75)

    # Anomaly Statistics
    total_anomalies_iso = (data['anomaly_if'] == -1).sum()
    percentage_anomalies_iso = (total_anomalies_iso / count) * 100

    total_anomalies_svm = (data['anomaly_svm'] == -1).sum()
    percentage_anomalies_svm = (total_anomalies_svm / count) * 100

    total_anomalies_kmeans = data['anomaly_kmeans'].sum()
    percentage_anomalies_kmeans = (total_anomalies_kmeans / count) * 100

    # Return results with additional statistics
    return render_template('results.html', 
                           anomalies_kmeans=anomalies_kmeans,
                           original_plot=original_plot,
                           plot_if=plot_if,
                           plot_svm=plot_svm,
                           anomalies_if=anomalies_if,
                           anomalies_svm=anomalies_svm,
                           export_filename=export_filename,
                           count=count,
                           mean=mean,
                           median=median,
                           std=std,
                           min_val=min_val,
                           max_val=max_val,
                           q25=q25,
                           q75=q75,
                           total_anomalies_iso=total_anomalies_iso,
                           percentage_anomalies_iso=percentage_anomalies_iso,
                           total_anomalies_svm=total_anomalies_svm,
                           percentage_anomalies_svm=percentage_anomalies_svm,
                           total_anomalies_kmeans=total_anomalies_kmeans,
                           percentage_anomalies_kmeans=percentage_anomalies_kmeans)

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['EXPORT_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
