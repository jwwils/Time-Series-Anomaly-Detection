
# Time Series Anomaly Detection Application

This Flask application is designed to detect anomalies in time series data using multiple techniques. It offers a user-friendly interface for uploading time series files, selecting desired anomaly detection methods, and viewing results.

## Description

The application supports various anomaly detection algorithms, including Isolation Forest, One-Class SVM, DBSCAN, and KMeans clustering. Users can upload their time series data, select the appropriate algorithm, and receive visual and statistical insights into potential anomalies in their data.

## Installation

To set up the project, ensure you have Python installed, and then follow these steps:

```bash
# Clone the repository
git clone https://github.com/jwwils/Time-Series-Anomaly-Detection
cd Anomaly

#Install dependencies
pip install -r requirements.txt
```

## Usage

Run the Flask application by executing:

```python
python app.py
```

Navigate to the provided local URL in your web browser. Upload your time series data file, select the desired anomaly detection technique, and analyze the results.

## Contributing

Contributions to the project are welcome. Please follow the standard fork-and-pull request workflow.

## License

This project was created under the MIT License
