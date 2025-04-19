# Medical Scan Analysis API

A FastAPI-based application for medical image analysis using deep learning models. The application supports detection of multiple medical conditions through various scanning modalities.

## Supported Conditions

- Brain Tumor Detection
- Tuberculosis Detection
- Lung Cancer Detection
- COVID-19 Detection
- Pneumonia Detection
- Kidney Disease Detection
- Knee Osteoporosis Detection
- Diabetic Retinopathy Detection

## Prerequisites

- Python 3.9+
- Kaggle account and API credentials
- Git

## Installation

1. Clone the repository

```bash
git clone https://github.com/yourusername/Medical-Scan-App.git
cd Medical-Scan-App
```

2. Create a virtual environment

```bash
python -m venv venv
# For Windows
.\venv\Scripts\activate
# For Linux/Mac
source venv/bin/activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the Application

1. Start the FastAPI server:

```bash
uvicorn main:app --reload
```

2. Access the API documentation:

- OpenAPI documentation: http://localhost:8000/docs

## API Endpoints

Each medical condition has its own endpoint for predictions:

- `/Brain-Tumor/predict` - Brain tumor detection
- `/Tuberculosis/predict` - TB detection
- `/Lung-Cancer/predict` - Lung cancer detection
- `/Covid/predict` - COVID-19 detection
- `/Pneumonia/predict` - Pneumonia detection
- `/Kidnee/kidney/predict` - Kidney disease detection
- `/Kidnee/knee/predict` - Knee disease detection
- `/Diabetic-Retinopathy/predict` - Diabetic retinopathy detection

## Models

The application uses pre-trained models hosted on Kaggle:

- Brain Tumor: ResNet model
- Tuberculosis: ResNet model
- Lung Cancer: ResNet model
- COVID-19: Custom CNN
- Pneumonia: Inception model
- Kidney Disease: ResNet50
- Diabetic Retinopathy: ResNet50

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Models trained using Kaggle datasets
- FastAPI framework for API development
- TensorFlow for deep learning models
