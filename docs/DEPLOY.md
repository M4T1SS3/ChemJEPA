# Deploying Research Paper to GitHub Pages

## 🚀 Quick Deploy (5 minutes)

### Step 1: Commit the docs folder
```bash
git add docs/ paper/ results/
git commit -m "add research paper website and results"
git push origin main
```

### Step 2: Enable GitHub Pages
1. Go to your GitHub repo: `https://github.com/yourusername/ChemWorld`
2. Click **Settings** → **Pages** (left sidebar)
3. Under "Source", select:
   - Branch: `main`
   - Folder: `/docs`
4. Click **Save**

### Step 3: Wait 2-3 minutes
GitHub will build your site automatically. You'll get a URL like:
```
https://yourusername.github.io/ChemWorld
```

## 📝 What You Get

**Live Website Features:**
- ✅ Beautiful, clean academic design
- ✅ **43× Speedup** prominently displayed
- ✅ Interactive figures (hover, click)
- ✅ Full paper content (intro, method, results)
- ✅ Code snippets with syntax highlighting
- ✅ Download links for data/code
- ✅ Citation (ready to copy-paste)
- ✅ Mobile-friendly responsive design

**SEO & Sharing:**
- ✅ Optimized for Google Scholar
- ✅ Twitter card preview
- ✅ LinkedIn preview
- ✅ Professional appearance

## 🎯 After Deployment

### 1. Share Everywhere
Tweet/post:
```
🚀 New research: 43× speedup in molecular optimization!

We use counterfactual planning in latent chemical space to
dramatically reduce expensive oracle queries in drug discovery.

📊 Results: Same quality, 43× fewer DFT simulations
🔗 Paper: https://yourusername.github.io/ChemWorld
💻 Code: https://github.com/yourusername/ChemWorld

#MachineLearning #DrugDiscovery #AI
```

### 2. Add to README
Update your main README.md:
```markdown
## 📄 Research Paper

**"Counterfactual Planning in Latent Chemical Space"**

**TL;DR:** We achieve a **43× speedup** in molecular optimization with no quality loss.

🔗 **[Read the paper](https://yourusername.github.io/ChemWorld)**
```

### 3. Get a DOI (Optional but Recommended)

**Option A: Zenodo (Easiest)**
1. Go to https://zenodo.org
2. Connect your GitHub repo
3. Create a release on GitHub
4. Zenodo automatically creates a DOI
5. Add DOI badge to README

**Option B: arXiv**
1. Submit to https://arxiv.org
2. Choose category: cs.LG (Machine Learning) or physics.chem-ph
3. Get arXiv ID (e.g., 2501.12345)
4. Update website with arXiv link

## 🔧 Customization

### Update GitHub Links
Edit `docs/index.html` line 104:
```html
<a href="https://github.com/YOURUSERNAME/ChemWorld" class="link-button github">
```

### Add Your Name
Edit `docs/index.html` line 126:
```html
<div class="authors">Your Name</div>
<div class="affiliation">Your Institution</div>
```

### Add Google Analytics (Optional)
Add before `</head>` in `docs/index.html`:
```html
<!-- Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-XXXXXXXXXX"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'G-XXXXXXXXXX');
</script>
```

## 📊 Track Impact

After deployment, you can track:
- **GitHub stars/forks** - Interest from community
- **Website traffic** - Views per day
- **Citations** - Via Google Scholar
- **Social shares** - Twitter, LinkedIn, Reddit

## 🎓 Submit to Conferences (Still Possible!)

**Important:** GitHub Pages doesn't count as "formal publication"

You can still submit to:
- NeurIPS/ICML workshops (use LaTeX version in `paper/`)
- arXiv (reference GitHub Pages as supplementary material)
- Main conferences (if you extend with OMol25)

## ✨ Pro Tips

1. **Add a demo video** - Record yourself running the code, upload to YouTube, embed in website
2. **Interactive plots** - Use Plotly.js for interactive figures
3. **Live notebook** - Add Google Colab link for people to try it
4. **Blog post** - Write a less technical summary on Medium/Substack
5. **HN/Reddit** - Share on Hacker News, r/MachineLearning

## 🐛 Troubleshooting

**Site not loading?**
- Check GitHub Pages settings (Settings → Pages)
- Ensure `/docs` folder is pushed to main branch
- Wait 5 minutes, clear browser cache

**Figures not showing?**
- Check image paths are relative (`../results/figures/...`)
- Ensure PNG files are committed to git

**Styling broken?**
- Check CSS is embedded in `index.html`
- Open browser console for errors (F12)

## 📧 Get Feedback

Share with:
- Professors/advisors
- r/MachineLearning subreddit
- Twitter ML community (#MachineLearning)
- Hacker News (https://news.ycombinator.com)
- LinkedIn

---

## Example Deployment Commands

```bash
# Full workflow
git add .
git commit -m "publish research paper: 43× speedup in molecular optimization"
git push origin main

# Wait 2-3 minutes, then visit:
# https://yourusername.github.io/ChemWorld

# Share the link!
```

**Ready to go live?** Just push and enable GitHub Pages! 🚀
