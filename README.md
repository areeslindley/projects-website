# Data Science Portfolio

A professional data science portfolio built with Jupyter Book, showcasing projects in machine learning, statistical modeling, and data analysis.

## Overview

This portfolio demonstrates technical expertise in data science while maintaining accessibility and visual polish. Each project follows a narrative structure that guides readers through the analytical process from exploration to conclusions.

## Structure

```
data-science-portfolio/
├── _config.yml                 # Jupyter Book configuration
├── _toc.yml                    # Table of contents structure
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── intro.md                    # Landing page
├── about.md                    # About me page
├── projects/
│   ├── titanic/               # Titanic survival analysis project
│   └── [future_projects]/     # Additional projects
├── _static/
│   ├── css/
│   │   └── custom.css         # Custom styling
│   └── images/
│       └── profile.jpg        # Profile image
└── _build/                     # Generated HTML (git-ignored)
```

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Build the book:**
   ```bash
   jupyter-book build .
   ```

3. **View locally:**
   ```bash
   jupyter-book serve _build/html
   ```

## Deployment

The book can be deployed to GitHub Pages using:

```bash
ghp-import -n -p -f _build/html
```

Or configure GitHub Actions for automatic builds on push.

## Projects

### Titanic Survival Analysis
A comprehensive classification project exploring survival patterns on the RMS Titanic, covering data exploration, cleaning, modeling, and interpretation.

*More projects coming soon...*

## License

This portfolio is open-source. Code and analyses are available for educational purposes.

## Contact

For questions or collaboration opportunities, please open an issue on GitHub or contact me directly.
