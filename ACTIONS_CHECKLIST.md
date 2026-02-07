# 🔍 GitHub Actions Checklist - What to Look For

## Step 1: Go to Actions Tab
**URL:** https://github.com/areeslindley/projects-website/actions

## Step 2: Find Your Workflow
Look for: **"Deploy Jupyter Book to Pages"**

## Step 3: Check the Status Icon

### ✅ Green Checkmark = SUCCESS
- Deployment completed successfully
- **If site still 404s:** Wait 2-3 minutes, clear cache, or check Pages settings

### ❌ Red X = FAILED
- **Click on it** to see what went wrong
- Look at the error message in the logs
- Common issues listed below

### 🟡 Yellow Circle = RUNNING
- Wait for it to complete (usually 2-5 minutes)
- Refresh the page to see updates

### ⚪ Gray Circle = QUEUED
- Waiting to start (usually starts within 1 minute)

## Step 4: If Failed - Check Each Step

Click on the failed run, then click the **"deploy"** job. You'll see steps like:

### 1. Checkout
- Should complete in seconds
- **Error?** Repository access issue

### 2. Set up Python
- Should install Python 3.9
- **Error?** Rare, but check Python version

### 3. Install dependencies
- **⚠️ MOST COMMON FAILURE POINT**
- Look for: `ERROR: Could not find a version that satisfies the requirement...`
- **Solution:** Package version conflict or missing package
- **Check:** The error will say which package failed

### 4. Build Jupyter Book
- **⚠️ SECOND MOST COMMON FAILURE POINT**
- Look for: `Exception occurred:` or `Error:`
- Common errors:
  - `FileNotFoundError: data/titanic.csv` → Missing data file
  - `SyntaxError` → Error in notebook
  - `ModuleNotFoundError` → Missing import

### 5. Setup Pages
- Should complete quickly
- **Error?** GitHub Pages configuration issue

### 6. Upload artifact
- Should upload files
- **Error?** Build didn't create `_build/html` directory

### 7. Deploy to GitHub Pages
- Final deployment step
- **Error?** Permission or configuration issue

## Step 5: Copy the Error Message

If you see an error, copy the **exact error message** and the **step name** where it failed. This will help diagnose the issue.

## Common Error Examples

### Example 1: Missing Package
```
ERROR: Could not find a version that satisfies the requirement ydata-profiling>=4.0.0
```
**Fix:** Package might not exist or version is wrong. Check package name.

### Example 2: Missing Data File
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/titanic.csv'
```
**Fix:** Data file not committed to git. Need to add it.

### Example 3: Build Error
```
Exception occurred: ... in file projects/titanic/01_exploration.ipynb
```
**Fix:** Error in notebook. Check the notebook syntax.

## Quick Test: Run Locally

To test if the build works on your machine:
```bash
cd /Users/alunrees/Documents/Code/alun-projects-website
pip install -r requirements.txt
jupyter-book build .
```

If this fails locally, it will fail in GitHub Actions too.
