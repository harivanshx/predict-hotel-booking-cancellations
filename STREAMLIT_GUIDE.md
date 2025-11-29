# 🎨 Streamlit Dashboard Guide

## 🚀 Quick Start

### 1. Install Streamlit
```bash
pip install streamlit plotly
```

### 2. Run the Pipeline First
```bash
# Generate data (if not done already)
python generate_sample_data.py --samples 10000

# Train the model
python main.py --mode full --quick
```

### 3. Launch the Dashboard
```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## 📊 Dashboard Features

### Tab 1: Overview
- **Dataset Statistics**: Total bookings, cancellation rate, features count
- **Visualizations**: Cancellation distribution, hotel type breakdown
- **Key Metrics**: Lead time and ADR statistics

### Tab 2: Data Explorer
- **Interactive Filters**: Filter by hotel type, status, market segment
- **Data Table**: View and explore filtered data
- **Download**: Export filtered data as CSV
- **Visualizations**: Box plots for lead time and ADR by cancellation status

### Tab 3: Model Performance
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Visualizations**: Confusion matrix, ROC curve, Precision-Recall curve
- **Full Report**: Complete model evaluation details

### Tab 4: Make Predictions
- **Interactive Form**: Enter booking details
- **Risk Assessment**: Get cancellation risk prediction
- **Recommendations**: Business recommendations based on risk level

### Tab 5: Feature Importance
- **Feature Analysis**: See which features matter most
- **Business Insights**: Understand what drives cancellations

## 🎯 Usage Examples

### Explore Your Data
1. Go to **Data Explorer** tab
2. Use filters to segment your data
3. Download filtered results for further analysis

### Check Model Performance
1. Go to **Model Performance** tab
2. Review metrics and visualizations
3. Expand full report for detailed analysis

### Make Predictions
1. Go to **Make Predictions** tab
2. Enter booking details
3. Get instant risk assessment

## 🔧 Customization

### Change Dashboard Theme
Edit `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
```

### Add New Visualizations
Edit `app.py` and add your custom plots using Plotly or Matplotlib.

## 💡 Tips

1. **Refresh Data**: Click the refresh button in the top-right to reload data
2. **Full Screen**: Click the expand icon on any chart for full-screen view
3. **Download Charts**: Hover over charts to see download options
4. **Responsive**: Dashboard works on mobile and tablet devices

## 🐛 Troubleshooting

### Dashboard won't start
```bash
# Make sure streamlit is installed
pip install streamlit plotly

# Try running with verbose mode
streamlit run app.py --logger.level=debug
```

### No data showing
```bash
# Generate sample data first
python generate_sample_data.py --samples 10000
```

### Model not found
```bash
# Train the model first
python main.py --mode full --quick
```

### Port already in use
```bash
# Use a different port
streamlit run app.py --server.port 8502
```

## 📱 Deployment

### Deploy to Streamlit Cloud (Free)
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy!

### Local Network Access
```bash
# Allow access from other devices on your network
streamlit run app.py --server.address 0.0.0.0
```

## 🎨 Screenshots

The dashboard includes:
- 📊 Interactive charts and graphs
- 🎯 Real-time predictions
- 📈 Model performance metrics
- 🔍 Data filtering and exploration
- 📥 Data export functionality

---

**Enjoy exploring your hotel booking data!** 🏨✨
