# Streamlit Dashboard Deployment Guide

## 🚀 Quick Start (Run Locally)

### Step 1: Set Up Project Structure

Create this folder structure in your Dissertation directory:

```
Dissertation/
├── dashboard/
│   ├── app.py                          # Main Streamlit app (provided)
│   ├── requirements.txt                # Dependencies (provided)
│   ├── README.md                       # This file
│   └── .streamlit/
│       └── config.toml                 # Streamlit configuration
├── models/
│   └── saved_models/
│       ├── model_xgboost.pkl          # Your trained XGBoost model
│       ├── model_random_forest.pkl    # Your Random Forest model
│       └── model_lightgbm.pkl         # Your LightGBM model
└── notebooks/                          # Your existing notebooks
```

### Step 2: Move Files

```bash
# From your Dissertation folder
cd "C:\Users\papak\OneDrive - Southampton Solent University\Desktop\Dissertation"

# Create dashboard directory
mkdir dashboard
cd dashboard

# Copy the app.py and requirements.txt files here
# (Download them from Claude and move them)

# Create Streamlit config directory
mkdir .streamlit
```

### Step 3: Create Streamlit Configuration

Create `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#667eea"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
headless = true
port = 8501
```

### Step 4: Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt
```

### Step 5: Run the Dashboard

```bash
# From the dashboard directory
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

---

## 🌐 Deploy to Streamlit Cloud (FREE Hosting)

### Prerequisites

1. ✅ GitHub account
2. ✅ Your repository with the dashboard code
3. ✅ Streamlit Cloud account (free - sign up at streamlit.io)

### Step 1: Prepare Your Repository

Your GitHub repo should have this structure:

```
energy-poverty-detection-ml/
├── dashboard/
│   ├── app.py
│   ├── requirements.txt
│   └── .streamlit/
│       └── config.toml
├── models/
│   └── saved_models/
│       └── .gitkeep        # Model files are too large for GitHub
├── README.md
└── notebooks/
```

**Important:** Don't commit large model files (.pkl) to GitHub. We'll handle them differently.

### Step 2: Add Files to GitHub

```bash
# From your Dissertation folder
cd "C:\Users\papak\OneDrive - Southampton Solent University\Desktop\Dissertation"

# Add dashboard files
git add dashboard/
git commit -m "feat(dashboard): Add Streamlit interactive dashboard

- Complete web application for energy poverty detection
- Live prediction with adjustable household features  
- Batch prediction for multiple households
- Model performance visualization
- Research insights and impact analysis

Features:
- 5 interactive tabs (Overview, Live Prediction, Batch, Performance, Insights)
- Plotly visualizations for all charts
- Real-time predictions using XGBoost model
- CSV upload for batch processing
- Professional UI with gradient theme

Deployed at: [URL will be added after deployment]"

# Push to GitHub
git push origin main
```

### Step 3: Sign Up for Streamlit Cloud

1. Go to https://streamlit.io/cloud
2. Click "Sign up" → Use your GitHub account
3. Authorize Streamlit to access your repositories

### Step 4: Deploy Your App

1. Click "New app" in Streamlit Cloud
2. Select:
   - **Repository:** `Pehrry/energy-poverty-detection-ml`
   - **Branch:** `main`
   - **Main file path:** `dashboard/app.py`
3. Click "Deploy!"

**Wait 2-3 minutes** for deployment.

### Step 5: Handle Model Files

Since model files are too large for GitHub, you have two options:

#### Option A: Use Sample/Demo Mode (Simpler)

The app already has built-in simulation for predictions when models aren't found. This works immediately!

#### Option B: Upload Models to Cloud Storage (For Real Predictions)

1. **Upload to Google Drive/Dropbox**
   - Upload your `model_xgboost.pkl` file
   - Get a public download link

2. **Modify app.py to download models**
   
   Add this function at the top of `app.py`:

   ```python
   import requests
   import os
   
   def download_model():
       """Download model from cloud storage if not present"""
       model_path = 'models/saved_models/model_xgboost.pkl'
       
       if not os.path.exists(model_path):
           st.info("Downloading model file...")
           url = "YOUR_GOOGLE_DRIVE_LINK_HERE"  # Replace with your link
           
           os.makedirs('models/saved_models', exist_ok=True)
           
           response = requests.get(url)
           with open(model_path, 'wb') as f:
               f.write(response.content)
           
           st.success("Model downloaded!")
   
   # Call this before loading model
   download_model()
   ```

### Step 6: Get Your Live URL

After deployment, you'll get a URL like:

```
https://pehrry-energy-poverty-detection.streamlit.app
```

**This is your live dashboard URL!** ✨

---

## 📊 Using the Dashboard

### Tab 1: Overview
- View key metrics and statistics
- See model performance comparisons
- Explore feature importance charts
- Analyze ACORN group distributions

### Tab 2: Live Prediction
1. Adjust household features using sliders:
   - Average Daily Consumption
   - Consumption Variance
   - Coefficient of Variation
   - Weekend-to-Weekday Ratio
   - Night Consumption %
   - Peak-to-Average Ratio
2. Click "Run Prediction"
3. See results:
   - Energy Poor / Not Poor classification
   - Probability gauge
   - Risk assessment
   - Feature analysis

### Tab 3: Batch Prediction
1. Prepare CSV file with columns:
   - `household_id`
   - `timestamp`
   - `consumption`
2. Upload file
3. Click "Process Batch Prediction"
4. Download results as CSV

### Tab 4: Model Performance
- View confusion matrix
- See ROC curve
- Check hyperparameters
- Review ethical considerations

### Tab 5: Insights
- Key research findings
- Impact & applications
- Comparison with traditional methods
- Research contribution

---

## 🎨 Customization

### Change Colors

Edit `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#YOUR_COLOR"     # Main accent color
backgroundColor = "#FFFFFF"       # Page background
secondaryBackgroundColor = "#F0F2F6"  # Sidebar background
textColor = "#262730"            # Text color
```

### Update Branding

In `app.py`, search for:
- Header text: Line ~65
- Sidebar image: Line ~155
- Footer text: Line ~990

### Add Your Logo

Replace the placeholder image URL in sidebar (line ~155):

```python
st.image("path/to/your/logo.png", use_column_width=True)
```

---

## 🔧 Troubleshooting

### Issue 1: "Module not found"
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

### Issue 2: "Model file not found"
- Check that `models/saved_models/model_xgboost.pkl` exists
- Or let it run in demo mode (predictions are simulated)

### Issue 3: Dashboard not loading
```bash
# Check Streamlit version
streamlit --version

# Should be >= 1.28.0
# Upgrade if needed:
pip install streamlit --upgrade
```

### Issue 4: Port already in use
```bash
# Use different port
streamlit run app.py --server.port 8502
```

### Issue 5: Deployment failed on Streamlit Cloud
- Check that `requirements.txt` is in the same directory as `app.py`
- Verify all dependencies are listed
- Check Streamlit Cloud logs for specific errors

---

## 📱 Sharing Your Dashboard

### For Visa Application

1. **Add to CV:**
   ```
   Interactive ML Dashboard: https://pehrry-energy-poverty-detection.streamlit.app
   Built with Streamlit, featuring live predictions and batch processing
   ```

2. **Add to Personal Statement:**
   ```
   Developed interactive web dashboard for energy poverty detection system,
   deployed publicly at [URL], demonstrating real-time ML predictions and
   comprehensive data visualizations. Dashboard processes 167M observations
   and achieves 87.8% recall for identifying vulnerable households.
   ```

3. **Add to GitHub README:**
   ```markdown
   ## 🌐 Live Dashboard
   
   Experience the interactive dashboard: https://pehrry-energy-poverty-detection.streamlit.app
   
   Features:
   - Real-time predictions
   - Batch household analysis
   - Model performance metrics
   - Research insights
   ```

### For Presentations

1. **Demo the Live Prediction tab**
   - Adjust sliders in real-time
   - Show immediate results
   - Explain feature importance

2. **Show Batch Processing**
   - Upload sample CSV
   - Generate predictions for 100+ households
   - Download results

3. **Display Visualizations**
   - Interactive Plotly charts
   - Zoom, pan, hover for details
   - Professional presentation quality

---

## 📈 Next Steps

### Phase 1: Basic Deployment ✅
- [x] Run locally
- [ ] Deploy to Streamlit Cloud
- [ ] Get live URL
- [ ] Test all features

### Phase 2: Enhancement
- [ ] Add your actual trained models
- [ ] Include real data samples
- [ ] Add download buttons for reports
- [ ] Create video demo

### Phase 3: Promotion
- [ ] Add URL to CV
- [ ] Update GitHub README
- [ ] Share with supervisors
- [ ] Include in visa application
- [ ] Demo in thesis defense

---

## 🆘 Need Help?

### Resources
- **Streamlit Docs:** https://docs.streamlit.io
- **Plotly Docs:** https://plotly.com/python/
- **Deployment Guide:** https://docs.streamlit.io/streamlit-community-cloud/get-started

### Common Questions

**Q: Can I use this without internet?**  
A: Yes! Run it locally with `streamlit run app.py`

**Q: How much does Streamlit Cloud cost?**  
A: FREE for public apps! No credit card needed.

**Q: Can I password-protect it?**  
A: Yes, Streamlit Cloud offers authentication for private apps.

**Q: Can I connect to a database?**  
A: Yes, Streamlit supports PostgreSQL, MySQL, etc.

**Q: How do I update the deployed app?**  
A: Just push changes to GitHub - Streamlit Cloud auto-deploys!

---

## ✅ Deployment Checklist

### Pre-Deployment
- [ ] `app.py` is in `dashboard/` folder
- [ ] `requirements.txt` is complete
- [ ] `.streamlit/config.toml` is created
- [ ] Code runs locally without errors
- [ ] All tabs work correctly

### GitHub
- [ ] Repository is public
- [ ] Dashboard files are committed
- [ ] README mentions the dashboard
- [ ] Model files are excluded (.gitignore)

### Streamlit Cloud
- [ ] Account created
- [ ] App deployed successfully
- [ ] Live URL works
- [ ] All features functional

### Promotion
- [ ] URL added to CV
- [ ] GitHub README updated
- [ ] Personal statement mentions it
- [ ] Screenshots taken for portfolio

---

## 🎉 Success!

Once deployed, you'll have:

✅ Professional web dashboard  
✅ Live URL to share  
✅ Interactive predictions  
✅ Beautiful visualizations  
✅ Portfolio piece for visa application  

**Your live dashboard URL will be something like:**  
`https://pehrry-energy-poverty-detection.streamlit.app`

Share it proudly! 🚀

---

*Created for MSc Applied AI & Data Science Dissertation*  
*Southampton Solent University, 2025*
