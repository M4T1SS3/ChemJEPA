#!/usr/bin/env python3
"""
Convert PNG figures to PDF for LaTeX paper.
"""

from pathlib import Path
from PIL import Image

def png_to_pdf(png_path: Path, pdf_path: Path):
    """Convert PNG to PDF."""
    img = Image.open(png_path)
    # Convert RGBA to RGB if needed
    if img.mode == 'RGBA':
        rgb_img = Image.new('RGB', img.size, (255, 255, 255))
        rgb_img.paste(img, mask=img.split()[3])
        img = rgb_img
    img.save(pdf_path, 'PDF', resolution=300.0)
    print(f"✓ Converted: {png_path.name} → {pdf_path.name}")

def main():
    project_root = Path(__file__).parent.parent

    # Source and destination directories
    src_dir = project_root / 'results' / 'figures'
    dst_dir = project_root / 'paper' / 'figures'
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Convert main figure
    png_to_pdf(
        src_dir / 'sample_efficiency.png',
        dst_dir / 'sample_efficiency.pdf'
    )

    print("\n✅ All figures converted to PDF!")
    print(f"Output directory: {dst_dir}")

if __name__ == '__main__':
    main()
