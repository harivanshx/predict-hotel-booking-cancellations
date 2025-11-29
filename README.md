# Hotel Booking Cancellation Prediction Pipeline

[![ML Pipeline](https://github.com/YOUR_USERNAME/hotel-booking-prediction/actions/workflows/ml_pipeline.yml/badge.svg)](https://github.com/YOUR_USERNAME/hotel-booking-prediction/actions/workflows/ml_pipeline.yml)
[![Deploy Streamlit](https://github.com/YOUR_USERNAME/hotel-booking-prediction/actions/workflows/deploy_streamlit.yml/badge.svg)](https://github.com/YOUR_USERNAME/hotel-booking-prediction/actions/workflows/deploy_streamlit.yml)

A production-ready machine learning pipeline for predicting hotel booking cancellations with an interactive Streamlit dashboard.

## 🌐 Live Demo

**Dashboard**: [https://your-app.streamlit.app](https://share.streamlit.io) *(Update after deployment)*

## 🎯 Features

- 🤖 **ML Pipeline**: Train 3 models (Logistic Regression, Random Forest, XGBoost)
- 📊 **Interactive Dashboard**: Explore data and visualize predictions
- 🔄 **CI/CD**: Automated testing and deployment with GitHub Actions
- 📈 **12 Engineered Features**: Advanced feature engineering
- ⚖️ **Class Imbalance Handling**: SMOTE implementation
- 🎨 **Beautiful Visualizations**: Confusion matrix, ROC curves, feature importance

## 🚀 Quick Start

### Local Development

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/hotel-booking-prediction.git
cd hotel-booking-prediction

# Install dependencies
pip install -r requirements.txt

# Generate sample data
python generate_sample_data.py --samples 10000

# Run ML pipeline
python main.py --mode full --quick

# Launch dashboard
streamlit run app.py
```

### Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment instructions.

## 📁 Project Structure

```
pipeline/
├── app.py                  # Streamlit dashboard
├── main.py                 # ML pipeline orchestrator
├── generate_sample_data.py # Sample data generator
├── src/                    # ML modules
│   ├── config.py
│   ├── data_loader.py
│   ├── eda.py
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── train.py
│   └── evaluate.py
├── tests/                  # Unit tests
├── .github/workflows/      # CI/CD pipelines
├── models/                 # Trained models
└── logs/                   # Logs and visualizations
```

## 📊 Dashboard Features

- **Overview**: Dataset statistics and visualizations
- **Data Explorer**: Interactive filtering and data export
- **Model Performance**: Metrics, confusion matrix, ROC curves
- **Predictions**: Make predictions for new bookings
- **Feature Importance**: Understand what drives cancellations

## 🎓 Assignment Compliance

✅ All 12 requirements fulfilled:
1. Exploratory Data Analysis
2. Data Cleaning
3. Feature Engineering (12 features)
4. Outlier Detection & Treatment
5. Categorical Encoding
6. Class Imbalance Handling (SMOTE)
7. Model Training (3 models)
8. Hyperparameter Tuning
9. Model Evaluation
10. Modular Pipeline
11. CI/CD Integration
12. Model Deployment

## 📚 Documentation

- [START_HERE.md](START_HERE.md) - Quick start guide
- [DEPLOYMENT.md](DEPLOYMENT.md) - Deployment instructions
- [STREAMLIT_GUIDE.md](STREAMLIT_GUIDE.md) - Dashboard guide
- [HOW_TO_RUN.md](HOW_TO_RUN.md) - Detailed usage

## 🔧 Technologies

- **ML**: scikit-learn, XGBoost, imbalanced-learn
- **Dashboard**: Streamlit, Plotly
- **CI/CD**: GitHub Actions
- **Testing**: pytest
- **Visualization**: Matplotlib, Seaborn

## 📈 Model Performance

- **Best Model**: Random Forest
- **ROC-AUC**: 0.75-0.90 (depends on data size)
- **F1-Score**: 0.65-0.85

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest tests/ -v`
5. Submit a pull request

## 📄 License

This project is provided as-is for educational and commercial use.

## 🙏 Acknowledgments

- Dataset: Hotel booking dataset
- Libraries: scikit-learn, XGBoost, Streamlit, Plotly

---

**Built with ❤️ for robust ML pipeline development**
