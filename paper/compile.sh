#!/bin/bash
# Compile workshop paper

set -e  # Exit on error

cd "$(dirname "$0")"

echo "=========================================="
echo "Compiling Workshop Paper"
echo "=========================================="
echo ""

# Check if pdflatex is installed
if ! command -v pdflatex &> /dev/null; then
    echo "❌ Error: pdflatex not found"
    echo "Please install LaTeX (MacTeX or TeX Live)"
    echo ""
    echo "On macOS:"
    echo "  brew install --cask mactex-no-gui"
    echo ""
    exit 1
fi

# Check if bibtex is installed
if ! command -v bibtex &> /dev/null; then
    echo "❌ Error: bibtex not found"
    echo "Please install BibTeX (usually comes with LaTeX)"
    exit 1
fi

echo "✓ LaTeX tools found"
echo ""

# Clean previous builds
echo "Cleaning previous builds..."
rm -f workshop_paper.aux workshop_paper.bbl workshop_paper.blg workshop_paper.log workshop_paper.out workshop_paper.pdf
echo "✓ Clean complete"
echo ""

# First pass
echo "Running pdflatex (first pass)..."
pdflatex -interaction=nonstopmode workshop_paper.tex > /dev/null 2>&1 || {
    echo "❌ First pdflatex pass failed. Check workshop_paper.log for errors."
    exit 1
}
echo "✓ First pass complete"

# Run bibtex
echo "Running bibtex..."
bibtex workshop_paper > /dev/null 2>&1 || {
    echo "❌ BibTeX failed. Check workshop_paper.blg for errors."
    exit 1
}
echo "✓ BibTeX complete"

# Second pass
echo "Running pdflatex (second pass)..."
pdflatex -interaction=nonstopmode workshop_paper.tex > /dev/null 2>&1 || {
    echo "❌ Second pdflatex pass failed. Check workshop_paper.log for errors."
    exit 1
}
echo "✓ Second pass complete"

# Third pass (for references)
echo "Running pdflatex (third pass)..."
pdflatex -interaction=nonstopmode workshop_paper.tex > /dev/null 2>&1 || {
    echo "❌ Third pdflatex pass failed. Check workshop_paper.log for errors."
    exit 1
}
echo "✓ Third pass complete"

echo ""
echo "=========================================="
echo "✅ COMPILATION SUCCESSFUL!"
echo "=========================================="
echo ""
echo "Output: workshop_paper.pdf"
echo ""

# Show PDF info
if [ -f workshop_paper.pdf ]; then
    size=$(du -h workshop_paper.pdf | cut -f1)
    pages=$(pdfinfo workshop_paper.pdf 2>/dev/null | grep "Pages:" | awk '{print $2}' || echo "?")
    echo "PDF Info:"
    echo "  Size: $size"
    echo "  Pages: $pages"
    echo ""
fi

echo "Next steps:"
echo "  1. Review: open workshop_paper.pdf"
echo "  2. Submit to NeurIPS/ICML workshop"
echo "  3. Post on arXiv"
echo ""
