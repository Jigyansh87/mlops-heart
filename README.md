# MLOps Heart Disease Prediction

A machine learning project for predicting heart disease using the UCI Heart Disease dataset, with MLOps best practices including containerization and Kubernetes deployment.

## Project Structure

```
mlops-heart/
├── data/
│   ├── download_data.py      # Script to download and prepare dataset
│   └── heart_cleaned.csv     # Cleaned heart disease dataset
├── k8s/
│   └── deployment.yaml       # Kubernetes deployment configuration
├── tests/
│   └── test_data.py         # Data validation tests
└── README.md
```

## Prerequisites

- Python 3.8 or higher
- Docker (for containerization)
- Kubernetes/kubectl (for deployment)
- pip (Python package manager)

## Setup Instructions

### 1. Clone the Repository

```bash
cd mlops-heart
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install pandas pytest
```

### 4. Download Dataset

Run the data download script to fetch the UCI Heart Disease dataset:

```bash
python data/download_data.py
```

This will:
- Download the Cleveland Heart Disease dataset from UCI ML Repository
- Process and clean the data
- Convert target variable to binary (0: no disease, 1: disease)
- Save to `data/heart.csv`

### 5. Verify Data

Run tests to ensure data integrity:

```bash
pytest tests/test_data.py -v
```

The tests verify:
- Dataset file exists
- Target column is present
- No missing values in the dataset

## Dataset Information

The heart disease dataset contains 14 attributes:

- **age**: Age in years
- **sex**: Sex (1 = male; 0 = female)
- **cp**: Chest pain type (1-4)
- **trestbps**: Resting blood pressure (mm Hg)
- **chol**: Serum cholesterol (mg/dl)
- **fbs**: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
- **restecg**: Resting electrocardiographic results (0-2)
- **thalach**: Maximum heart rate achieved
- **exang**: Exercise induced angina (1 = yes; 0 = no)
- **oldpeak**: ST depression induced by exercise
- **slope**: Slope of peak exercise ST segment (1-3)
- **ca**: Number of major vessels colored by fluoroscopy (0-3)
- **thal**: Thalassemia (3 = normal; 6 = fixed defect; 7 = reversible defect)
- **target**: Heart disease diagnosis (0 = no disease; 1 = disease)

## Docker Deployment

### Build Docker Image

```bash
docker build -t heart-api:latest .
```

### Run Docker Container

```bash
docker run -p 8000:8000 heart-api:latest
```

## Kubernetes Deployment

### Deploy to Kubernetes

```bash
kubectl apply -f k8s/deployment.yaml
```

### Verify Deployment

```bash
kubectl get deployments
kubectl get pods
```

### Access the Service

```bash
kubectl port-forward deployment/heart-api-deployment 8000:8000
```

The API will be accessible at `http://localhost:8000`

### Scale the Deployment

```bash
kubectl scale deployment heart-api-deployment --replicas=3
```

### Delete Deployment

```bash
kubectl delete -f k8s/deployment.yaml
```

## Development Workflow

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_data.py -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

### Data Validation

The project includes automated data validation tests in [tests/test_data.py](tests/test_data.py):

1. **File Existence**: Ensures dataset file exists at expected location
2. **Column Validation**: Verifies target column is present
3. **Data Quality**: Checks for missing values

## Project Files

- **[data/download_data.py](data/download_data.py)**: Downloads and preprocesses the heart disease dataset
- **[data/heart_cleaned.csv](data/heart_cleaned.csv)**: Cleaned dataset with 303 records
- **[k8s/deployment.yaml](k8s/deployment.yaml)**: Kubernetes deployment configuration
- **[tests/test_data.py](tests/test_data.py)**: Data validation test suite

## Troubleshooting

### Dataset Download Issues

If download fails, ensure you have internet connectivity and try:
```bash
python data/download_data.py
```

### Kubernetes Pod Not Starting

Check pod logs:
```bash
kubectl logs deployment/heart-api-deployment
```

### Docker Image Issues

Rebuild with no cache:
```bash
docker build --no-cache -t heart-api:latest .
```
## References

- [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease)
- Dataset Citation: Janosi, Andras, et al. "Heart Disease Data Set." UCI Machine Learning Repository (1988)
