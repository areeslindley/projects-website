# Troubleshooting GitHub Pages 404 Error

## Quick Checks

1. **Check GitHub Actions Status**
   - Go to: https://github.com/areeslindley/projects-website/actions
   - Look for the "Deploy Jupyter Book to Pages" workflow
   - Ensure it completed successfully (green checkmark)
   - If it failed, check the error logs

2. **Verify GitHub Pages Settings**
   - Go to: Settings → Pages
   - Source should be: **"GitHub Actions"** (not "Deploy from a branch")
   - If you see "Your site is live at...", note the URL

3. **Wait for Propagation**
   - After a successful deployment, wait 1-2 minutes
   - GitHub Pages can take a few minutes to update
   - Try clearing your browser cache or using incognito mode

## Common Issues

### Issue: Workflow Not Running
**Solution:** 
- Make sure you've pushed the code to the `main` branch
- Check that `.github/workflows/deploy.yml` exists in your repository
- Try manually triggering: Actions tab → "Deploy Jupyter Book to Pages" → "Run workflow"

### Issue: Build Fails
**Check:**
- Go to Actions → Latest workflow run → Check the error
- Common causes:
  - Missing dependencies in `requirements.txt`
  - Syntax errors in notebooks
  - Missing data files

### Issue: 404 on Root URL
**Solution:**
- The site should work at: `https://areeslindley.github.io/projects-website/`
- Try: `https://areeslindley.github.io/projects-website/intro.html`
- The `index.html` should redirect to `intro.html` automatically

### Issue: Images Not Loading
**Solution:**
- Verify images are committed: `git ls-files projects/titanic/images/`
- Check image paths in notebooks (should be relative: `images/filename.png`)
- Rebuild locally to test: `jupyter-book build .`

## Verify Configuration

Your `_config.yml` should have:
```yaml
html_baseurl: "https://areeslindley.github.io/projects-website/"
repository_url: https://github.com/areeslindley/projects-website
```

## Force Rebuild

If the site isn't updating:

1. Make a small change (add a space to README.md)
2. Commit and push:
   ```bash
   git add README.md
   git commit -m "Trigger rebuild"
   git push origin main
   ```
3. Wait for the workflow to complete
4. Check the site again

## Still Not Working?

1. Check the Actions tab for any error messages
2. Verify the workflow file exists: `.github/workflows/deploy.yml`
3. Ensure GitHub Pages is enabled in repository settings
4. Try accessing the site in an incognito/private window
