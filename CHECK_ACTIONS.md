# What to Look For in GitHub Actions Tab

## Step-by-Step Guide

### 1. Navigate to Actions
- Go to: https://github.com/areeslindley/projects-website/actions
- You should see a list of workflow runs

### 2. Check the Latest Run
Look for the most recent "Deploy Jupyter Book to Pages" workflow run.

**Status Indicators:**
- 🟡 **Yellow circle** = Running (wait for it to complete)
- ✅ **Green checkmark** = Success (deployment should work)
- ❌ **Red X** = Failed (click to see errors)
- ⚪ **Gray circle** = Queued (waiting to start)

### 3. If You See a Red X (Failed)
Click on the failed run, then click on the "deploy" job. Look for:

**Common Error Messages:**

1. **"ModuleNotFoundError" or "No module named..."**
   - Missing dependency in `requirements.txt`
   - Solution: Add missing package to requirements.txt

2. **"FileNotFoundError: data/titanic.csv"**
   - Missing data file
   - Solution: Ensure data files are committed to git

3. **"SyntaxError" or "IndentationError"**
   - Python syntax error in notebooks
   - Solution: Fix the syntax error

4. **"Permission denied" or "403"**
   - GitHub Pages permissions issue
   - Solution: Check repository settings → Pages → ensure "GitHub Actions" is selected

5. **"No such file or directory: _build/html"**
   - Build failed before creating output
   - Solution: Check the "Build Jupyter Book" step for errors

### 4. Check Each Step
Expand each step in the workflow to see detailed logs:

- **Checkout** - Should complete quickly
- **Set up Python** - Should install Python 3.9
- **Install dependencies** - Should install packages from requirements.txt
- **Build Jupyter Book** - This is the critical step - look for errors here
- **Setup Pages** - Should configure GitHub Pages
- **Upload artifact** - Should upload the built files
- **Deploy to GitHub Pages** - Should deploy successfully

### 5. What Success Looks Like
If everything works, you should see:
```
✓ Checkout
✓ Set up Python
✓ Install dependencies
✓ Build Jupyter Book (with "build succeeded" message)
✓ Setup Pages
✓ Upload artifact
✓ Deploy to GitHub Pages
```

### 6. If Build Succeeds But Site Still 404s
- Wait 2-3 minutes after deployment completes
- Check Settings → Pages → should show "Your site is live at..."
- Try clearing browser cache
- Check if the URL is correct: `https://areeslindley.github.io/projects-website/`

## Quick Diagnostic Commands

If you want to test locally first:
```bash
cd /Users/alunrees/Documents/Code/alun-projects-website
jupyter-book build .
ls -la _build/html/*.html
```

This will show if the build works locally and what files are generated.
