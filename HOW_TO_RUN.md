# How to Run This Project - Step by Step Guide

## ⚠️ Important: Installation Issue Fix

If you encounter errors with `pip install -r requirements.txt`, follow these alternative steps:

## 🚀 Quick Start (Recommended)

### Step 1: Install Core Dependencies Only

Instead of installing all packages at once, install them individually:

```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn joblib pytest
```

### Step 2: Generate Sample Data

```bash
python generate_sample_data.py --samples 10000
```

This creates a file: `data/raw/hotel_bookings.csv`

### Step 3: Run the Pipeline

```bash
# Quick mode (fastest, recommended for first run)
python main.py --mode full --quick

# OR full mode with all features
python main.py --mode full
```

## 📋 Detailed Steps

### Option A: Quick Test Run (5-10 minutes)

```bash
# 1. Generate sample data
python generate_sample_data.py --samples 5000

# 2. Run in quick mode (skips EDA and hyperparameter tuning)
python main.py --mode full --quick
```

**What you'll get:**
- Trained models in `models/` folder
- Logs in `logs/pipeline.log`
- Evaluation report in `logs/model_evaluation.txt`

### Option B: Full Pipeline (15-30 minutes)

```bash
# 1. Generate sample data
python generate_sample_data.py --samples 10000

# 2. Run complete pipeline
python main.py --mode full
```

**What you'll get:**
- All EDA visualizations in `logs/eda_plots/`
- Hyperparameter-tuned models
- Complete evaluation with all metrics
- Feature importance analysis

### Option C: Use Your Own Data

```bash
# 1. Place your hotel_bookings.csv file in data/raw/

# 2. Run the pipeline
python main.py --mode full
```

## 🔧 If You Get Errors

### Error: "No module named 'pandas'" or similar

**Solution:**
```bash
# Install missing package individually
pip install pandas
# or
pip install scikit-learn
# or
pip install xgboost
```

### Error: "Data file not found"

**Solution:**
```bash
# Generate sample data first
python generate_sample_data.py
```

### Error: "logging has no attribute config"

**Solution:** This is already fixed in the code. Make sure you're using the latest version of main.py.

### Error: Building wheel for pandas failed

**Solution:** Use pre-built wheels:
```bash
pip install --upgrade pip
pip install pandas --only-binary :all:
```

Or skip pandas version pinning:
```bash
pip install pandas  # installs latest compatible version
```

## 📊 What Happens When You Run

1. **Data Loading**: Loads and validates your data
2. **EDA** (if not in quick mode): Creates visualizations
3. **Preprocessing**: Cleans data, handles missing values, treats outliers
4. **Feature Engineering**: Creates 12 new features
5. **Training**: Trains 3 models (Logistic Regression, Random Forest, XGBoost)
6. **Tuning** (if not in quick mode): Optimizes hyperparameters
7. **Evaluation**: Generates metrics and visualizations
8. **Saving**: Saves best model to `models/best_model.pkl`

## 📁 Output Files

After running, check these locations:

```
pipeline/
├── models/
│   ├── best_model.pkl          ← Your trained model
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   └── xgboost.pkl
├── logs/
│   ├── pipeline.log            ← Execution log
│   ├── model_evaluation.txt    ← Performance metrics
│   └── eda_plots/              ← Visualizations
│       ├── target_distribution.png
│       ├── confusion_matrix_*.png
│       ├── roc_curve_*.png
│       └── feature_importance_*.png
└── data/
    └── raw/
        └── hotel_bookings.csv  ← Your data
```

## 🎯 Expected Output

You should see output like:

```
================================================================================
STEP 1: DATA LOADING AND VALIDATION
================================================================================
Loading data from data\raw\hotel_bookings.csv
Data loaded successfully. Shape: (10000, 21)
...

================================================================================
STEP 5: MODEL TRAINING
================================================================================
Training Logistic Regression
  Training accuracy: 0.7845
  Test accuracy: 0.7723
...

✓ Best model: xgboost_tuned (ROC-AUC: 0.8234)
```

## 🧪 Test the Installation

Quick test to verify everything works:

```bash
# Test 1: Generate small sample
python generate_sample_data.py --samples 100

# Test 2: Run quick pipeline
python main.py --mode full --quick
```

If this completes without errors, your setup is working!

## 💡 Pro Tips

1. **Start Small**: Use `--samples 1000` for very fast testing
2. **Use Quick Mode**: Add `--quick` flag to skip EDA and tuning
3. **Check Logs**: Always review `logs/pipeline.log` for details
4. **Monitor Progress**: The pipeline prints progress to console

## 🆘 Still Having Issues?

1. **Check Python Version**: Requires Python 3.8+
   ```bash
   python --version
   ```

2. **Update pip**:
   ```bash
   python -m pip install --upgrade pip
   ```

3. **Install packages one by one**:
   ```bash
   pip install pandas
   pip install numpy
   pip install scikit-learn
   pip install xgboost
   pip install matplotlib
   pip install seaborn
   ```

4. **Check the log file**: `logs/pipeline.log` has detailed error messages

## ✅ Success Checklist

- [ ] Dependencies installed (at minimum: pandas, numpy, scikit-learn, xgboost)
- [ ] Sample data generated (`data/raw/hotel_bookings.csv` exists)
- [ ] Pipeline runs without errors
- [ ] Model file created (`models/best_model.pkl` exists)
- [ ] Evaluation report created (`logs/model_evaluation.txt` exists)

Once all items are checked, your pipeline is ready to use! 🎉
