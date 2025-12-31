# ECG Arrhythmia Detection Web App

This project is a Flask-based web application designed to analyze ECG (Electrocardiogram) data for arrhythmia detection. It utilizes a powerful hybrid Machine Learning pipeline combining **CNN, Bi-LSTM, and BR-SquareNet** architectures for deep feature extraction, along with a Meta Random Forest classifier for robust recording-level classification.

## Features

- **ECG Analysis**: Processes ECG recordings (in ZIP format containing `.dat`, `.hea`, `.atr` files).
- **Hybrid AI Model**:
  - **Deep Hybrid Network**: Integrates **CNN** (spatial features), **Bi-LSTM** (temporal dependencies), and **BR-SquareNet** (residual learning) for state-of-the-art heartbeat classification.
  - **Meta Random Forest**: Aggregates beat-level predictions to classify the entire recording.
- **Risk Assessment**: Categorizes results into Normal, Low Risk, High Risk, Critical, or Uncertain.
- **Key Visualizations**: Generates ECG plots with highlighted R-peaks.
- **PDF Reporting**: Automatically generates a downloadable PDF report with analysis results and plots.
- **REST API**: Provides an API endpoint for integration with other systems.

## Preprocessing

The raw ECG data undergoes rigorous preprocessing to ensure high-quality input for the model:

1.  **Filtering**:
    -   **Bandpass Filter**: 4th-order Butterworth filter (0.5–40 Hz) to remove muscle noise and baseline wander.
    -   **High-pass Filter**: 2nd-order Butterworth filter (0.5 Hz) for additional stability.
2.  **Normalization**: Z-score normalization `(x - mean) / std` to standardize signal amplitude.
3.  **Beat Segmentation**:
    -   R-peaks are detected (or read from annotations).
    -   Beats are extracted with a fixed window of **280 samples** (90 before R-peak, 190 after).
4.  **AAMI Mapping**: MIT-BIH annotations are mapped to 5 standard classes:
    -   **N**: Normal
    -   **S**: Supraventricular
    -   **V**: Ventricular
    -   **F**: Fusion
    -   **Q**: Unknown/Paced

## Evaluation Metrics

The model (`Train/hybrid.py`) is evaluated using comprehensive metrics to ensure reliability:

-   **Accuracy**: Overall correctness of predictions.
-   **F1-Score**: Macro and Weighted averages to account for class imbalance.
-   **Confusion Matrix**: Visualizes misclassifications (Counts & Normalized).
-   **ROC Curves & AUC**: One-vs-Rest ROC curves with Area Under the Curve (AUC) scores for each class.
-   **Training Curves**: Accuracy and Loss plotted over epochs to monitor convergence and overfitting.

## Tech Stack

- **Backend**: Flask (Python)
- **Data Processing**: NumPy, SciPy, WFDB, NeuroKit2
- **Machine Learning**: TensorFlow (Keras), Scikit-Learn (Joblib for Random Forest)
- **Visualization**: Matplotlib
- **Reporting**: ReportLab

## Project Structure

```
├── app.py                  # Main Flask application entry point
├── model.py                # Core inference logic (beat extraction, feature engineering, prediction)
├── model_pipeline.py       # Standalone pipeline script (similar to model.py)
├── utils/
│   └── pdf_report.py       # Utility for generating PDF reports
├── templates/
│   └── index.html          # Frontend HTML template
├── static/                 # Static assets (CSS, JS)
├── uploads/                # Directory for uploaded ECG files
├── results/                # Directory for generated plots and reports
│   ├── plots/
│   └── reports/
├── models/                 # Pre-trained models directory
│   └── best_final_hybrid.h5
├── meta_rf/                # Directory for Meta Random Forest model
│   └── meta_rf.pkl
└── requirements.txt        # Python dependencies
```

## Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create a Virtual Environment (Optional but Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

    *If `requirements.txt` is missing, manually install:*
    ```bash
    pip install flask numpy scipy wfdb tensorflow matplotlib joblib neurokit2 reportlab
    ```

4.  **Ensure Model Files are Present:**
    - `models/best_final_hybrid.h5`
    - `meta_rf/meta_rf.pkl`

## Usage

1.  **Run the Application:**
    ```bash
    python app.py
    ```

2.  **Access the Web Interface:**
    Open your browser and navigate to `http://127.0.0.1:5000`.

3.  **Analyze an ECG Record:**
    - Prepare a ZIP file containing the MIT-BIH style ECG record files (e.g., `100.dat`, `100.hea`, and optionally `100.atr`).
    - Upload the ZIP file via the web interface.
    - View the prediction, confidence score, risk level, and ECG plot.
    - Download the PDF report.

## API Endpoints

### `POST /analyze`
Uploads and analyzes an ECG record ZIP file.

- **RequestBody**: `multipart/form-data` with `file` (ZIP archive).
- **Response**: JSON object with:
    - `prediction`: Class label (e.g., Normal, Atrial Fibrillation).
    - `confidence`: Confidence score.
    - `risk`: Risk level.
    - `plot`: Filename of the generated ECG plot.
    - `pdf`: Filename of the generated PDF report.

## Classes & Risk Levels

The system classifies recordings into the following categories:
- **N**: Normal (Green)
- **S**: Supraventricular Ectopic Beat -> Low Risk (Orange)
- **V**: Ventricular Ectopic Beat -> High Risk (Red)
- **F**: Fusion Beat -> Critical (Dark Red)
- **Q**: Unknown Beat -> Uncertain (Gray)

## Models

- **Deep Hybrid Model**: Combines **CNN**, **Bi-LSTM**, and **BR-SquareNet** for comprehensive ECG signal analysis.
- **Meta Random Forest**: specific model trained on aggregated features of recordings.
