# Quick Start Guide - Streamlit Dashboard

## 🚀 Get Your Dashboard Running in 5 Minutes!

### Step 1: Create Dashboard Folder (30 seconds)

```bash
cd "C:\Users\papak\OneDrive - Southampton Solent University\Desktop\Dissertation"
mkdir dashboard
cd dashboard
```

### Step 2: Add Files (1 minute)

Download these 3 files from Claude and put them in the `dashboard/` folder:

1. ✅ `app.py` - Main application
2. ✅ `requirements.txt` - Dependencies
3. ✅ `DASHBOARD_DEPLOYMENT_GUIDE.md` - Full instructions

### Step 3: Install (2 minutes)

```bash
pip install streamlit pandas numpy plotly scikit-learn xgboost
```

### Step 4: Run! (1 second)

```bash
streamlit run app.py
```

**Done!** Your browser will open at `http://localhost:8501`

---

## 🎯 What You Get

✅ **5 Interactive Tabs:**
1. Overview - Key metrics and charts
2. Live Prediction - Adjust features, get instant results
3. Batch Prediction - Upload CSV, analyze multiple households
4. Model Performance - Confusion matrix, ROC curve, metrics
5. Insights - Research findings and impact

✅ **Professional Features:**
- Beautiful gradient theme
- Interactive Plotly charts
- Real-time predictions
- CSV upload/download
- Mobile responsive

---

## 📸 Quick Demo

### Try Live Prediction:
1. Go to "Live Prediction" tab
2. Adjust sliders:
   - Set Variance to 3.5 (high)
   - Set Weekend Ratio to 1.3 (high)
   - Set Night % to 35 (high)
3. Click "Run Prediction"
4. See: **Energy Poor** result! 🔴

### Try Different Values:
1. Set Variance to 1.5 (low)
2. Set Weekend Ratio to 0.95 (balanced)
3. Set Night % to 18 (low)
4. Click "Run Prediction"
5. See: **Not Energy Poor** result! ✅

---

## 🌐 Deploy to Internet (FREE)

Want a live URL to share? Follow these steps:

### 1. Add to GitHub (5 minutes)

```bash
# From Dissertation folder
git add dashboard/
git commit -m "feat: Add Streamlit dashboard"
git push origin main
```

### 2. Deploy on Streamlit Cloud (3 minutes)

1. Go to https://streamlit.io/cloud
2. Sign in with GitHub
3. Click "New app"
4. Select:
   - Repository: `Pehrry/energy-poverty-detection-ml`
   - Branch: `main`
   - File: `dashboard/app.py`
5. Click "Deploy"

### 3. Get Your URL! ✨

You'll get: `https://pehrry-energy-poverty-detection.streamlit.app`

**Share this URL with:**
- Visa reviewers
- Supervisors
- Potential employers
- On your CV

---

## 💡 Pro Tips

### Customize Colors

Edit `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#667eea"  # Change this to your color!
```

### Add Your Logo

In `app.py`, line 155, change:

```python
st.image("path/to/your/logo.png")
```

### Use Real Model

Put your `model_xgboost.pkl` in:
```
models/saved_models/model_xgboost.pkl
```

Then the predictions will use your actual trained model!

---

## 🆘 Troubleshooting

### "streamlit: command not found"
```bash
pip install streamlit
```

### "Port 8501 already in use"
```bash
streamlit run app.py --server.port 8502
```

### "Module 'plotly' not found"
```bash
pip install plotly
```

### App won't start
```bash
# Reinstall everything
pip install -r requirements.txt --force-reinstall
```

---

## ✅ Success Checklist

After running `streamlit run app.py`, you should see:

- [ ] Browser opens automatically
- [ ] Purple gradient header loads
- [ ] 5 tabs appear at top
- [ ] Overview tab shows 4 metric cards
- [ ] Charts render correctly
- [ ] Live Prediction tab has sliders
- [ ] Clicking "Run Prediction" works

**All checked?** Congrats! Your dashboard is working! 🎉

---

## 📱 For Your Visa Application

### On CV:
```
Interactive ML Dashboard
https://[your-url].streamlit.app
• Real-time energy poverty predictions
• Processes 167M smart meter observations
• 87.8% recall for vulnerable household detection
```

### In Personal Statement:
```
Developed and deployed interactive web dashboard demonstrating
ML model for energy poverty detection. Dashboard features real-time
predictions, batch processing, and comprehensive visualizations,
publicly accessible at [URL].
```

---

## 🎓 Next Steps

1. ✅ Run locally (you just did this!)
2. ⬜ Test all features
3. ⬜ Deploy to Streamlit Cloud
4. ⬜ Add URL to CV
5. ⬜ Share with supervisors
6. ⬜ Include in visa application
7. ⬜ Demo in thesis defense

---

**Questions? Check the full guide:** `DASHBOARD_DEPLOYMENT_GUIDE.md`

**Ready to deploy to the internet?** See Step 2 in the full guide!

---

*You've got this! 🚀*
