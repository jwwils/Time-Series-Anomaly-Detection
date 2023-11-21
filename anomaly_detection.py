import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor

def detect_anomalies_kmeans(data):
    # K-means Clustering
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(scaled_data)
    labels = kmeans.predict(scaled_data)
    data['Cluster'] = labels
    anomalies_kmeans = data[data['Cluster'] == 1]
    return anomalies_kmeans

def detect_anomalies_if(data):
    # Isolation Forest
    model_if = IsolationForest(contamination=0.05)
    model_if.fit(data)
    data['Anomaly_IF'] = model_if.predict(data)
    anomalies_if = data[data['Anomaly_IF'] == -1]
    return anomalies_if

def detect_anomalies_svm(data):
    # One-Class SVM
    model_svm = OneClassSVM(nu=0.05)
    model_svm.fit(data)
    data['Anomaly_SVM'] = model_svm.predict(data)
    anomalies_svm = data[data['Anomaly_SVM'] == -1]
    return anomalies_svm

def detect_anomalies_dbscan(data):
    # DBSCAN
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    model_dbscan = DBSCAN(eps=0.5, min_samples=5)
    model_dbscan.fit(scaled_data)
    data['Anomaly_DBSCAN'] = model_dbscan.labels_
    anomalies_dbscan = data[data['Anomaly_DBSCAN'] == -1]
    return anomalies_dbscan

def detect_anomalies_lof(data):
    # Local Outlier Factor (LOF)
    model_lof = LocalOutlierFactor(contamination=0.05)
    data['Anomaly_LOF'] = model_lof.fit_predict(data)
    anomalies_lof = data[data['Anomaly_LOF'] == -1]
    return anomalies_lof