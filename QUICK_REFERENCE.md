# Quick Reference Guide

## 🚀 Quick Start Commands

### First Time Setup
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate sample data
python generate_sample_data.py --samples 10000

# 3. Run the pipeline
python main.py --mode full
```

### Common Commands

```bash
# Full pipeline with hyperparameter tuning
python main.py --mode full

# Quick mode (faster, no EDA or tuning)
python main.py --mode full --quick

# Training only
python main.py --mode train

# Skip hyperparameter tuning
python main.py --mode full --no-tuning

# Use custom data
python main.py --mode full --data-path path/to/data.csv

# Generate different sample sizes
python generate_sample_data.py --samples 5000
python generate_sample_data.py --samples 20000
```

### Running Tests
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# Run specific test
pytest tests/test_pipeline.py::TestDataLoader -v
```

## 📂 Important Files

### Input
- `data/raw/hotel_bookings.csv` - Your dataset goes here

### Output
- `models/best_model.pkl` - Best trained model
- `logs/pipeline.log` - Execution log
- `logs/model_evaluation.txt` - Performance report
- `logs/eda_plots/` - All visualizations

### Configuration
- `src/config.py` - Modify parameters here

## 🔧 Configuration Quick Edits

### Change Test Size
Edit `src/config.py`:
```python
TEST_SIZE = 0.3  # Change from 0.2 to 0.3 (30% test set)
```

### Change Number of Models to Tune
Edit `src/config.py`:
```python
TUNING_CONFIG = {
    'n_iter': 50,  # Increase from 20 to 50 for more thorough search
}
```

### Change SMOTE Parameters
Edit `src/config.py`:
```python
SMOTE_CONFIG = {
    'sampling_strategy': 0.8,  # Don't fully balance, just reduce imbalance
    'k_neighbors': 3,  # Use fewer neighbors
}
```

## 🐛 Troubleshooting

### Error: "Data file not found"
```bash
# Solution: Generate sample data
python generate_sample_data.py
```

### Error: "Module not found"
```bash
# Solution: Install dependencies
pip install -r requirements.txt
```

### Pipeline runs but no plots
```bash
# Check if matplotlib backend is set
# Add to your script:
import matplotlib
matplotlib.use('Agg')
```

### Out of memory during training
```bash
# Solution: Use quick mode or reduce data size
python main.py --mode full --quick
# Or generate smaller dataset
python generate_sample_data.py --samples 5000
```

## 📊 Understanding Output

### Model Evaluation Metrics

- **Accuracy**: Overall correctness (aim for >75%)
- **Precision**: When model says "will cancel", how often is it right? (aim for >70%)
- **Recall**: Of all actual cancellations, how many did we catch? (aim for >70%)
- **F1-Score**: Balance between precision and recall (aim for >0.70)
- **ROC-AUC**: Overall discriminative ability (aim for >0.75)

### Feature Importance

Check `logs/eda_plots/feature_importance_*.png` to see which features matter most.

Common important features:
- `lead_time` - How far in advance booking was made
- `deposit_type` - Type of deposit
- `previous_cancellations` - Customer history
- `adr` - Price
- `total_of_special_requests` - Customer engagement

## 🔄 Retraining the Model

### When to Retrain
- Monthly (recommended)
- When cancellation patterns change
- After major business changes
- When performance degrades

### How to Retrain
```bash
# 1. Add new data to data/raw/hotel_bookings.csv
# 2. Run pipeline
python main.py --mode train
# 3. Evaluate new model
python main.py --mode evaluate
```

## 🚀 Deployment Checklist

- [ ] Train model on full dataset
- [ ] Review evaluation metrics (>0.75 ROC-AUC)
- [ ] Test predictions on sample bookings
- [ ] Set up model serving (API/batch)
- [ ] Implement monitoring
- [ ] Schedule retraining (monthly/quarterly)
- [ ] Document business rules
- [ ] Train operations team

## 📞 Getting Help

1. Check `logs/pipeline.log` for detailed execution logs
2. Review `logs/model_evaluation.txt` for performance details
3. Examine visualizations in `logs/eda_plots/`
4. Run tests: `pytest tests/ -v`
5. Check README.md for detailed documentation

## 💡 Tips

- **Start small**: Use `--quick` mode for initial testing
- **Monitor logs**: Always check `logs/pipeline.log` after runs
- **Validate data**: Ensure your CSV has all required columns
- **Tune iteratively**: Start with default params, then tune
- **Version models**: Save models with timestamps for tracking
- **A/B test**: Compare model versions in production

## 🎯 Expected Runtime

- **Quick mode**: 2-5 minutes (10k samples)
- **Full mode (no tuning)**: 5-10 minutes
- **Full mode (with tuning)**: 15-30 minutes
- **Large dataset (100k+)**: 30-60 minutes

Times vary based on hardware and dataset size.
