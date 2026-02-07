# Diagnosing 404 Error on GitHub Pages

## Step-by-Step Diagnosis

### 1. Check GitHub Actions Status
Go to: https://github.com/areeslindley/projects-website/actions

**Look for:**
- Most recent "Deploy Jupyter Book to Pages" workflow
- Status icon: ✅ (green) = Success, ❌ (red) = Failed, 🟡 (yellow) = Running

**If you see a ✅ (green checkmark):**
- Click on it
- Check the "deploy" job
- Look at the "Deploy to GitHub Pages" step
- Should show: "Deployment successful" or similar

**If you see a ❌ (red X):**
- Click on it to see the error
- Check which step failed
- Copy the error message

### 2. Check GitHub Pages Settings
Go to: https://github.com/areeslindley/projects-website/settings/pages

**Verify:**
- **Source:** Should be "GitHub Actions" (NOT "Deploy from a branch")
- **Custom domain:** Should be empty (unless you set one up)
- **Status:** Should say "Your site is live at https://areeslindley.github.io/projects-website/"

**If Source is NOT "GitHub Actions":**
1. Change it to "GitHub Actions"
2. Save
3. Wait 1-2 minutes
4. Check the site again

### 3. Check the Deployment
Go to: https://github.com/areeslindley/projects-website/deployments

**Look for:**
- Most recent deployment
- Status: Should be "Active" (green)
- Environment: Should be "github-pages"

**If deployment shows as failed:**
- Click on it to see details
- Check the error message

### 4. Test the URL Directly
Try these URLs:
- https://areeslindley.github.io/projects-website/
- https://areeslindley.github.io/projects-website/intro.html
- https://areeslindley.github.io/projects-website/index.html

**If intro.html works but root doesn't:**
- This suggests the index.html redirect isn't working
- May need to wait a few more minutes for propagation

### 5. Check Browser Cache
- Try incognito/private window
- Clear browser cache
- Try a different browser
- Try from a different network

### 6. Common Issues

#### Issue: Workflow Succeeds But Site Still 404s
**Possible causes:**
- GitHub Pages source not set to "GitHub Actions"
- Deployment hasn't propagated yet (wait 2-3 minutes)
- Browser cache issue

#### Issue: "Deploy to GitHub Pages" Step Fails
**Check for:**
- Permission errors
- Missing `_build/html` directory
- Artifact upload failed

#### Issue: Build Succeeds But No Files Uploaded
**Check:**
- "Upload artifact" step - should show files uploaded
- Look for: "Uploading artifact..." and file count

### 7. Force a New Deployment
If nothing works, trigger a manual rebuild:

1. Go to Actions tab
2. Click "Deploy Jupyter Book to Pages"
3. Click "Run workflow" (top right)
4. Select "main" branch
5. Click "Run workflow"
6. Wait for it to complete
7. Check the site again

## What to Share for Help

If still having issues, share:
1. Actions tab screenshot or status (✅/❌/🟡)
2. Pages settings screenshot (showing Source)
3. Any error messages from failed steps
4. Which URLs you've tried
