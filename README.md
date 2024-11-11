# Synthetic Data Generator App

[![Python](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io)
[![YData](https://img.shields.io/badge/YData--Synthetic-1.0-brightgreen.svg)](https://github.com/ydataai/ydata-synthetic)


A Streamlit web application for generating synthetic data using the YData Synthetic library. This tool allows users to create synthetic datasets that mimic their real-world data while preserving privacy and statistical properties.

## 🚀 Features

- Upload and process tabular classification datasets
- Configure GAN model parameters through an intuitive UI
- Generate synthetic data using CGAN or WGAN-GP models
- Visual comparison between real and synthetic data distributions
- Download synthetic datasets in CSV format
- Interactive parameter selection with real-time updates

## 📋 Prerequisites

```bash
pandas
matplotlib
numpy
seaborn
streamlit
ydata-synthetic==1.4.0
```

## 🛠️ Installation

1. Clone this repository:
```bash
git clone https://github.com/rajeshai/ydata-synthetic-streamlit.git
cd ydata-synthetic-streamlit
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## 💻 Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided local URL (typically `http://localhost:8501`)

3. Follow these steps in the application:
   - Upload your preprocessed tabular classification dataset (CSV format)
   - Select the GAN model (CGAN or WGAN-GP)
   - Choose numerical and categorical columns
   - Configure model parameters:
     - Noise dimension
     - Layer dimension
     - Batch size
     - Sample interval
     - Number of epochs
     - Learning rate
     - Beta coefficients
   - Specify the number of synthetic samples to generate
   - Click the training button to start the process
   - Download the generated synthetic dataset

## 🔧 Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| Model | Choose between CGAN or WGAN-GP | CGAN |
| Noise Dimension | Input noise dimension for GAN | 128 |
| Layer Dimension | Dimension of network layers | 128 |
| Batch Size | Number of samples per training batch | 500 |
| Sample Interval | Interval for sampling during training | 100 |
| Epochs | Number of training epochs | 2 |
| Learning Rate | Model learning rate (x1e-3) | 0.05 |
| Beta Coefficients | Adam optimizer beta parameters | β1=0.5, β2=0.9 |

## 📊 Output

The application provides:
- Visual comparisons between real and synthetic data distributions
- Downloadable synthetic dataset in CSV format
- Training progress indicators
- Success notifications with celebratory balloons! 🎈

## ⚠️ Notes

- For demonstration purposes, it's recommended to start with a small number of epochs
- The application requires preprocessed data with no missing values
- Training time depends on dataset size and selected parameters
- The app uses caching to optimize performance


## 🙏 Acknowledgments

- [YData Synthetic Library](https://github.com/ydataai/ydata-synthetic) for the synthetic data generation capabilities
- Streamlit for the wonderful web application framework
