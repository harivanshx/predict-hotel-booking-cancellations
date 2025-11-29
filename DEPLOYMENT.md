# 🚀 Deployment Guide - Streamlit Cloud with CI/CD

## 📋 Overview

This guide will help you deploy your Hotel Booking Cancellation Prediction Dashboard to **Streamlit Cloud** (free!) with automated CI/CD using GitHub Actions.

## 🎯 Deployment Options

### Option 1: Streamlit Cloud (Recommended - FREE!)
- ✅ Free hosting
- ✅ Automatic updates from GitHub
- ✅ HTTPS included
- ✅ Easy sharing with URL

### Option 2: Other Platforms
- Heroku
- AWS EC2
- Google Cloud Run
- Azure App Service

---

## 🚀 Deploy to Streamlit Cloud (Step-by-Step)

### Step 1: Prepare Your Repository

1. **Initialize Git** (if not done):
```bash
cd c:\Users\hariv\OneDrive\Desktop\pipeline
git init
git add .
git commit -m "Initial commit: Hotel booking prediction pipeline with Streamlit dashboard"
```

2. **Create GitHub Repository**:
   - Go to [github.com](https://github.com)
   - Click "New Repository"
   - Name it: `hotel-booking-prediction`
   - Don't initialize with README (you already have files)
   - Click "Create repository"

3. **Push to GitHub**:
```bash
git remote add origin https://github.com/YOUR_USERNAME/hotel-booking-prediction.git
git branch -M main
git push -u origin main
```

### Step 2: Deploy to Streamlit Cloud

1. **Go to Streamlit Cloud**:
   - Visit: [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub

2. **Create New App**:
   - Click "New app"
   - Select your repository: `hotel-booking-prediction`
   - Main file path: `app.py`
   - Click "Deploy!"

3. **Wait for Deployment** (2-3 minutes):
   - Streamlit will install dependencies
   - Run your app
   - Generate a public URL

4. **Get Your URL**:
   - You'll get a URL like: `https://your-app.streamlit.app`
   - Share this with anyone!

---

## 🔄 CI/CD Pipeline

Your GitHub Actions workflow (`.github/workflows/deploy_streamlit.yml`) automatically:

1. ✅ **Tests ML Pipeline** on every push
2. ✅ **Validates Streamlit App** 
3. ✅ **Runs Unit Tests**
4. ✅ **Creates Artifacts**
5. ✅ **Triggers Streamlit Cloud** to redeploy

### How It Works

```
Push to GitHub → GitHub Actions runs tests → Tests pass → Streamlit Cloud auto-deploys
```

### View CI/CD Status

1. Go to your GitHub repository
2. Click "Actions" tab
3. See all workflow runs and their status

---

## 📝 Important Files for Deployment

### `.streamlit/config.toml` ✅ Created
- Theme settings
- Server configuration
- Browser settings

### `requirements.txt` ✅ Updated
- All dependencies listed
- Streamlit and Plotly included

### `.github/workflows/deploy_streamlit.yml` ✅ Created
- Automated testing
- Deployment validation

---

## 🔧 Configuration

### Custom Domain (Optional)

Streamlit Cloud allows custom domains:
1. Go to app settings
2. Click "Custom domain"
3. Add your domain (e.g., `hotel-analytics.yourdomain.com`)

### Secrets Management

For sensitive data (API keys, passwords):

1. In Streamlit Cloud:
   - Go to app settings
   - Click "Secrets"
   - Add secrets in TOML format:
   ```toml
   [secrets]
   api_key = "your-api-key"
   database_url = "your-db-url"
   ```

2. Access in code:
   ```python
   import streamlit as st
   api_key = st.secrets["api_key"]
   ```

---

## 🎨 Customization

### Change Theme

Edit `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#FF4B4B"  # Red
backgroundColor = "#0E1117"  # Dark
secondaryBackgroundColor = "#262730"
textColor = "#FAFAFA"
font = "sans serif"
```

### Add Authentication

```python
import streamlit as st

def check_password():
    def password_entered():
        if st.session_state["password"] == "your_password":
            st.session_state["password_correct"] = True
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("😕 Password incorrect")
        return False
    else:
        return True

if check_password():
    # Your app code here
    st.write("Welcome!")
```

---

## 📊 Monitoring

### View Logs

In Streamlit Cloud:
1. Go to your app
2. Click "Manage app"
3. View logs in real-time

### Analytics

Streamlit Cloud provides:
- Visitor count
- Usage statistics
- Error tracking

---

## 🐛 Troubleshooting

### App Won't Deploy

**Check:**
1. `requirements.txt` has all dependencies
2. `app.py` is in root directory
3. No syntax errors in code

**Fix:**
```bash
# Test locally first
streamlit run app.py

# If it works locally, it should work on cloud
```

### Missing Data

**Solution:** Generate sample data in app:
```python
# Add to app.py
if not config.RAW_DATA_FILE.exists():
    st.warning("Generating sample data...")
    import subprocess
    subprocess.run(["python", "generate_sample_data.py", "--samples", "10000"])
    st.rerun()
```

### Slow Loading

**Optimize:**
1. Use `@st.cache_data` for data loading
2. Use `@st.cache_resource` for models
3. Reduce sample data size for demo

---

## 🔒 Security Best Practices

1. **Don't commit secrets** to GitHub
2. **Use environment variables** for sensitive data
3. **Enable authentication** if needed
4. **Keep dependencies updated**

---

## 📱 Sharing Your App

Once deployed, share your app:

```
🌐 Live Dashboard: https://your-app.streamlit.app
📊 GitHub Repo: https://github.com/YOUR_USERNAME/hotel-booking-prediction
📝 Documentation: See README.md
```

### Embed in Website

```html
<iframe src="https://your-app.streamlit.app" width="100%" height="800px"></iframe>
```

---

## 🎯 Quick Deployment Checklist

- [ ] Code pushed to GitHub
- [ ] `requirements.txt` includes all dependencies
- [ ] `.streamlit/config.toml` configured
- [ ] GitHub Actions workflow added
- [ ] Streamlit Cloud account created
- [ ] App deployed and tested
- [ ] URL shared with stakeholders

---

## 💡 Pro Tips

1. **Use Branches**: Deploy `main` to production, `develop` for testing
2. **Monitor Usage**: Check Streamlit Cloud analytics regularly
3. **Update Regularly**: Keep dependencies and data fresh
4. **Get Feedback**: Share with users and iterate
5. **Document**: Keep README updated with deployment URL

---

## 🆘 Need Help?

- **Streamlit Docs**: [docs.streamlit.io](https://docs.streamlit.io)
- **Community Forum**: [discuss.streamlit.io](https://discuss.streamlit.io)
- **GitHub Issues**: Report bugs in your repo

---

## ✨ You're Ready to Deploy!

**Quick Start:**
```bash
# 1. Push to GitHub
git add .
git commit -m "Ready for deployment"
git push

# 2. Go to share.streamlit.io
# 3. Connect your repo
# 4. Deploy!
```

**Your dashboard will be live in minutes!** 🚀
