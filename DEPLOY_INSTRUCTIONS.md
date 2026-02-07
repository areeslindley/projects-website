# 🚀 Deployment Instructions for GitHub Pages

Your Jupyter Book portfolio is ready to deploy! Follow these steps:

## Step 1: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `alun-projects-website` (or your preferred name)
3. Description: "Data Science Portfolio - Jupyter Book"
4. Set to **Public** (required for free GitHub Pages)
5. **Do NOT** initialize with README, .gitignore, or license (we already have these)
6. Click "Create repository"

## Step 2: Push Your Code

Run these commands in your terminal:

```bash
cd /Users/alunrees/Documents/Code/alun-projects-website

# Add the remote repository (replace with your actual GitHub username if different)
git remote add origin https://github.com/alunrees/alun-projects-website.git

# Push to GitHub
git push -u origin main
```

If you get an error about the remote already existing:
```bash
git remote set-url origin https://github.com/alunrees/alun-projects-website.git
git push -u origin main
```

## Step 3: Enable GitHub Pages

1. Go to your repository on GitHub: `https://github.com/alunrees/alun-projects-website`
2. Click **Settings** (top menu)
3. Scroll down to **Pages** (left sidebar)
4. Under **Source**, select **"GitHub Actions"**
5. Save the settings

## Step 4: Wait for Deployment

- GitHub Actions will automatically build and deploy your site
- Go to **Actions** tab to see the deployment progress
- First deployment usually takes 2-3 minutes
- Your site will be live at: `https://alunrees.github.io/alun-projects-website/`

## Automatic Updates

Every time you push to the `main` branch, GitHub Actions will:
1. Automatically rebuild the book
2. Deploy the updated site
3. Update GitHub Pages

## Troubleshooting

### Build Fails in GitHub Actions
- Check the **Actions** tab for error messages
- Common issues:
  - Missing dependencies in `requirements.txt`
  - Syntax errors in notebooks
  - Missing data files

### Pages Not Showing
- Wait 1-2 minutes after deployment completes
- Clear browser cache
- Check that Pages source is set to "GitHub Actions" (not "Deploy from branch")

### Images Not Loading
- Verify image files are committed to git
- Check image paths in notebooks (should be relative: `images/filename.png`)

## Custom Domain (Optional)

To use a custom domain like `alunrees.com`:

1. Create `_static/CNAME` file with your domain:
   ```
   alunrees.com
   ```

2. Configure DNS as per GitHub Pages documentation

3. Commit and push:
   ```bash
   git add _static/CNAME
   git commit -m "Add custom domain"
   git push
   ```

## Manual Deployment (Alternative)

If you prefer manual deployment without GitHub Actions:

```bash
# Install ghp-import
pip install ghp-import

# Build the book
jupyter-book build .

# Deploy
ghp-import -n -p -f _build/html
```

---

**Your portfolio will be live at:** `https://alunrees.github.io/alun-projects-website/`

Enjoy your professional data science portfolio! 🎉
