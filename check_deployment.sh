#!/bin/bash
echo "🔍 GitHub Pages Deployment Checker"
echo "=================================="
echo ""
echo "1. Check Actions: https://github.com/areeslindley/projects-website/actions"
echo "2. Check Pages Settings: https://github.com/areeslindley/projects-website/settings/pages"
echo "3. Check Deployments: https://github.com/areeslindley/projects-website/deployments"
echo ""
echo "Test URLs:"
echo "- https://areeslindley.github.io/projects-website/"
echo "- https://areeslindley.github.io/projects-website/intro.html"
echo ""
echo "Local build test:"
if [ -d "_build/html" ]; then
    echo "✓ _build/html exists"
    echo "  Files: $(ls -1 _build/html/*.html 2>/dev/null | wc -l | xargs) HTML files"
    if [ -f "_build/html/index.html" ]; then
        echo "✓ index.html exists"
    else
        echo "✗ index.html missing"
    fi
    if [ -f "_build/html/intro.html" ]; then
        echo "✓ intro.html exists"
    else
        echo "✗ intro.html missing"
    fi
else
    echo "✗ _build/html does not exist (run: jupyter-book build .)"
fi
