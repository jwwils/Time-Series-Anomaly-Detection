from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
import os
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from werkzeug.utils import secure_filename
import plotly.express as px
from plotly.offline import plot
from flask_limiter.util import get_remote_address
from flask_limiter import Limiter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from hmmlearn.hmm import GaussianHMM
from statsmodels.tsa.holtwinters import ExponentialSmoothing




app = Flask(__name__)
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["5 per minute"]
)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'json'}
EXPORT_FOLDER = 'exports'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['EXPORT_FOLDER'] = EXPORT_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(EXPORT_FOLDER):
    os.makedirs(EXPORT_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def read_uploaded_file(filepath):
    if filepath.endswith('.csv'):
        data = pd.read_csv(filepath)
    elif filepath.endswith('.xlsx') or filepath.endswith('.xls'):
        data = pd.read_excel(filepath)
    elif filepath.endswith('.json'):
        data = pd.read_json(filepath)
    else:
        raise ValueError("Unsupported file type")
    return data

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

        # Read the data to get column names using the new function
        data = read_uploaded_file(filepath)
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
    data = read_uploaded_file(filepath)[[time_column, value_column]]
    data.set_index(time_column, inplace=True)
    data.columns = ['Value']
    data.fillna(method='ffill', inplace=True)

    def plot_anomalies(data, anomalies, method_name):
        """Function to create plots with anomalies highlighted"""
        if isinstance(anomalies, np.ndarray):
            anomalies = pd.DataFrame(anomalies, columns=['Value'])
        fig = px.line(data, x=data.index, y='Value', title=f'Anomalies Detected using {method_name}')
        if not anomalies.empty:
            fig.add_scatter(x=anomalies.index, y=anomalies['Value'], mode='markers', marker=dict(color="red", size=5), name='Anomaly')
        return plot(fig, output_type='div')

    def train_HMM(series, n_components=4, n_iter=1000):
        model = GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=n_iter)
        model.fit(series.reshape(-1, 1))
        return model
    
    def detect_anomalies_HMM(model, values, indices, threshold=-10):
        log_likelihood = np.array([model.score(np.array([val]).reshape(-1, 1)) for val in values])
        anomaly_indices = np.where(log_likelihood < threshold)[0]
        anomalies = pd.DataFrame({'Value': values[anomaly_indices]}, index=indices[anomaly_indices])
        return anomalies


    def train_holt_winters(series, seasonal_periods, trend='add', seasonal='add'):
        model = ExponentialSmoothing(series, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods)
        hw_model = model.fit()
        return hw_model
    
    def forecast_holt_winters(hw_model, steps=10):
        return hw_model.forecast(steps=steps)





    # K-means Clustering for Anomaly Detection
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data[['Value']].values.reshape(-1, 1))
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(data_normalized)
    distances = np.linalg.norm(data_normalized - kmeans.cluster_centers_[clusters], axis=1)
    threshold = np.percentile(distances, 95)
    data['anomaly_kmeans'] = (distances > threshold).astype(int)
    anomalies_kmeans = data[data['anomaly_kmeans'] == 1]
    plot_kmeans = plot_anomalies(data, anomalies_kmeans, 'K-means Clustering')

    # Feature Engineering for Isolation Forest and One-Class SVM
    data['rolling_mean'] = data['Value'].rolling(window=5).mean()
    data['rolling_std'] = data['Value'].rolling(window=5).std()
    data['z_score'] = (data['Value'] - data['rolling_mean']) / data['rolling_std']
    data = data.dropna()

    # Isolation Forest
    model_if = IsolationForest(contamination=0.05)
    data['anomaly_if'] = model_if.fit_predict(data[['Value', 'rolling_mean', 'rolling_std', 'z_score']])
    anomalies_if = data[data['anomaly_if'] == -1]
    plot_if = plot_anomalies(data, anomalies_if, 'Isolation Forest')

    # One-Class SVM
    model_svm = OneClassSVM(nu=0.05)
    data['anomaly_svm'] = model_svm.fit_predict(data[['Value']])
    anomalies_svm = data[data['anomaly_svm'] == -1]
    plot_svm = plot_anomalies(data, anomalies_svm, 'One-Class SVM')

    # Local Outlier Factor (LOF)
    model_lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
    data['anomaly_lof'] = model_lof.fit_predict(data[['Value']])
    anomalies_lof = data[data['anomaly_lof'] == -1]
    plot_lof = plot_anomalies(data, anomalies_lof, 'Local Outlier Factor')


    # Normalize the data and apply DBSCAN
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data[['Value']].values.reshape(-1, 1))
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    data['anomaly_dbscan'] = dbscan.fit_predict(data_normalized)
    anomalies_dbscan = data[data['anomaly_dbscan'] == -1]
    plot_dbscan = plot_anomalies(data, anomalies_dbscan, 'DBSCAN')


    # Train HMM
    hmm_model = train_HMM(data['Value'].values)

    # Detect anomalies using HMM
    anomalies_hmm = detect_anomalies_HMM(hmm_model, data['Value'].values, data['Value'].index)

    plot_hmm = plot_anomalies(data, anomalies_hmm, 'HMM')

    # The data is good due to its weights toward newer information 


    #hw_model = train_holt_winters(data['Value'], seasonal_periods=12)  Blocked off due to error out of bound can't predict based 

    #hw_forecast = pd.DataFrame(hw_forecast, columns=['Value'])

    #plot_hw = plot_anomalies(data, hw_forecast, 'HW')


    # Additional Statistical Features and Descriptive Statistics
    data['diff'] = data['Value'].diff()
    data['cumsum'] = data['Value'].cumsum()
    data['cumprod'] = (1 + data['Value']).cumprod()
    count = data['Value'].count()
    mean = data['Value'].mean()
    median = data['Value'].median()
    std = data['Value'].std()
    min_val = data['Value'].min()
    max_val = data['Value'].max()
    q25 = data['Value'].quantile(0.25)
    q75 = data['Value'].quantile(0.75)

    # Anomaly Statistics
    total_anomalies_iso = data['anomaly_if'].sum()
    percentage_anomalies_iso = (total_anomalies_iso / count) * 100

    total_anomalies_svm = data['anomaly_svm'].sum()
    percentage_anomalies_svm = (total_anomalies_svm / count) * 100

    total_anomalies_kmeans = data['anomaly_kmeans'].sum()
    percentage_anomalies_kmeans = (total_anomalies_kmeans / count) * 100

    total_anomalies_dbscan = data['anomaly_dbscan'].sum()
    percentage_anomalies_dbscan = (total_anomalies_dbscan / count) * 100

    total_anomalies_lof = data['anomaly_lof'].sum()
    percentage_anomalies_lof = (total_anomalies_lof / count) * 100

   

    # Plot original data
    original_plot = plot_original_data(data)

    def plot_anomalies(data, anomalies, column_name, title):
 
        fig = px.scatter(data, x=data.index, y='Value', color=data[column_name].apply(lambda x: 'Anomaly' if x == -1 else 'Normal'), 
                     title=title, color_discrete_map={'Anomaly':'red', 'Normal':'blue'})
        return plot(fig, output_type='div')

    def generate_isolation_forest_plot(data, anomalies):
        return plot_anomalies(data, anomalies, 'anomaly_if', 'Isolation Forest Anomalies')

    def generate_one_class_svm_plot(data, anomalies):
        return plot_anomalies(data, anomalies, 'anomaly_svm', 'One-Class SVM Anomalies')

    def generate_dbscan_plot(data, anomalies):
        return plot_anomalies(data, anomalies, 'anomaly_dbscan', 'DBSCAN Anomalies')

    def generate_kmeans_plot(data, anomalies):
        return plot_anomalies(data, anomalies, 'cluster', 'KMeans Anomalies')

    def generate_lof_plot(data, anomalies):
        return plot_anomalies(data, anomalies, 'anomaly_lof', 'Local Outlier Factor Anomalies')
    
    def generate_hmm_plot(data, anomalies):
        return plot_anomalies(data, anomalies, 'anomalies_hmm', 'HMM Anomalies')
    
    def generate_hw_plot(data, anomalies):
        return plot_anomalies(data, anomalies, 'hw_forecast', 'HW Forecast ')
    

# Returning the plotting functions for verification
    generate_isolation_forest_plot, generate_one_class_svm_plot, generate_dbscan_plot, generate_kmeans_plot, generate_lof_plot,generate_hmm_plot,generate_hw_plot
    

    # Exporting the processed data
    export_filename = f"processed_{filename}"
    export_filepath = os.path.join(app.config['EXPORT_FOLDER'], export_filename)
    data.to_csv(export_filepath)

    return render_template('results.html', 
                           anomalies_kmeans=anomalies_kmeans,
                           original_plot=original_plot,
                           anomalies_if=anomalies_if,
                           anomalies_svm=anomalies_svm,
                           anomalies_dbscan=anomalies_dbscan,
                           plot_kmeans=plot_kmeans,
                           plot_if=plot_if,
                           plot_svm=plot_svm,
                           plot_dbscan=plot_dbscan,
                           anomalies_lof=anomalies_lof,
                           plot_lof=plot_lof,
                           anomalies_hmm=anomalies_hmm,
                           plot_hmm=plot_hmm,
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
                           percentage_anomalies_kmeans=percentage_anomalies_kmeans,
                           total_anomalies_dbscan=total_anomalies_dbscan,
                           percentage_anomalies_dbscan=percentage_anomalies_dbscan,
                           percentage_anomalies_lof=percentage_anomalies_lof
                           )

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['EXPORT_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
