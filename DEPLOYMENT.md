# Deployment Guide for GitHub Pages

This Jupyter Book portfolio is configured for automatic deployment to GitHub Pages using GitHub Actions.

## Quick Start

1. **Create a GitHub repository** (if you haven't already):
   ```bash
   # On GitHub, create a new repository named "alun-projects-website"
   # Or use a different name and update _config.yml accordingly
   ```

2. **Push your code to GitHub**:
   ```bash
   git remote add origin https://github.com/alunrees/alun-projects-website.git
   git branch -M main
   git push -u origin main
   ```

3. **Enable GitHub Pages**:
   - Go to your repository on GitHub
   - Navigate to Settings → Pages
   - Under "Source", select "GitHub Actions"
   - The workflow will automatically deploy on every push to main/master

## Manual Deployment (Alternative)

If you prefer manual deployment using `ghp-import`:

```bash
# Install ghp-import
pip install ghp-import

# Build the book
jupyter-book build .

# Deploy to GitHub Pages
ghp-import -n -p -f _build/html
```

## Repository Configuration

The repository URL in `_config.yml` is set to:
- `https://github.com/alunrees/alun-projects-website`

If your repository has a different name or username, update the `repository_url` in `_config.yml`.

## GitHub Actions Workflow

The workflow (`.github/workflows/deploy.yml`) will:
1. Install Python and dependencies
2. Build the Jupyter Book
3. Deploy to GitHub Pages automatically

No additional configuration needed - it uses the `GITHUB_TOKEN` automatically provided by GitHub Actions.

## Troubleshooting

- **Build fails**: Check that all dependencies in `requirements.txt` are installable
- **Pages not updating**: Ensure GitHub Pages is set to use "GitHub Actions" as the source
- **Images not showing**: Verify that image files are committed to git and paths are correct

## Custom Domain (Optional)

To use a custom domain:
1. Add a `CNAME` file in `_static/` with your domain name
2. Configure DNS settings as per GitHub Pages documentation
