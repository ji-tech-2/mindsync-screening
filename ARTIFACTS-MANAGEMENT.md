# Artifacts Management - Important Notes

## 📌 Local Files (DO NOT OVERWRITE)

### `healthy_cluster_avg.csv`

This file is **maintained locally** and should **NOT** be overwritten by W&B downloads.

**Why?**
- Contains domain-specific baseline data
- Manually curated or calculated from production data
- May differ from training data

**Protection**:
- ✅ `download_artifacts.py` - Skips copying this file if it exists locally
- ✅ `flaskr/model.py` - Backs up before W&B download, then restores

## ⬇️ Downloaded from W&B

These files are automatically downloaded and **can be overwritten**:

1. **model.pkl** - Trained model
2. **preprocessor.pkl** - Data preprocessing pipeline
3. **model_coefficients.csv** - Model coefficients for interpretability
4. **feature_importance.csv** - Feature importance scores

## 🔄 Download Behavior

### Script: `download_artifacts.py`

```python
# Files to preserve (won't overwrite if exist locally)
preserve_files = ["healthy_cluster_avg.csv"]

# Download flow:
1. Download all files from W&B to temp location
2. Copy to artifacts/ folder
3. Skip files in preserve_files if they already exist
4. Result: healthy_cluster_avg.csv preserved ✅
```

### Function: `flaskr/model.py::download_artifacts_from_wandb()`

```python
# Protection flow:
1. Check if healthy_cluster_avg.csv exists locally
2. If exists, create backup: .csv.backup
3. Download all files from W&B
4. Restore backup over downloaded version
5. Result: local healthy_cluster_avg.csv preserved ✅
```

## 📋 Verification

After download, verify files:

```bash
cd mindsync-model-flask

# Check which files updated
ls -la artifacts/

# Verify healthy_cluster_avg.csv not changed
git status artifacts/healthy_cluster_avg.csv
# Should show "nothing to commit" if preserved correctly
```

## 🔧 Manual Update of healthy_cluster_avg.csv

If you need to update this file:

```bash
# 1. Backup current version
cp artifacts/healthy_cluster_avg.csv artifacts/healthy_cluster_avg.csv.old

# 2. Update the file (manual edit or script)
# ... your updates ...

# 3. Verify changes
python -c "import pandas as pd; print(pd.read_csv('artifacts/healthy_cluster_avg.csv').head())"

# 4. Commit to git
git add artifacts/healthy_cluster_avg.csv
git commit -m "feat: update healthy_cluster_avg baseline"
git push origin main
```

## ⚙️ Adding More Protected Files

To protect additional files from W&B overwrite:

### In `download_artifacts.py`:

```python
# Line ~60
preserve_files = [
    "healthy_cluster_avg.csv",
    "your_custom_file.csv",  # Add here
]
```

### In `flaskr/model.py`:

```python
# In download_artifacts_from_wandb() function
# Add similar backup/restore logic for new files
```

## 🚨 Troubleshooting

### healthy_cluster_avg.csv was overwritten

```bash
# 1. Check git history
git log artifacts/healthy_cluster_avg.csv

# 2. Restore from git
git checkout HEAD~1 -- artifacts/healthy_cluster_avg.csv

# 3. Verify protection code exists
grep -n "preserve_files" download_artifacts.py
grep -n "healthy_cluster_path" flaskr/model.py
```

### Want to use W&B version instead

If you want to use the version from W&B instead of local:

```bash
# Temporarily remove local file
mv artifacts/healthy_cluster_avg.csv artifacts/healthy_cluster_avg.csv.local

# Download from W&B
python download_artifacts.py

# Now W&B version is used
# To revert: mv artifacts/healthy_cluster_avg.csv.local artifacts/healthy_cluster_avg.csv
```

## 📝 Best Practices

1. **Never manually upload** `healthy_cluster_avg.csv` to W&B artifacts
2. **Always commit** changes to this file in git
3. **Document updates** in commit messages
4. **Test locally** before deploying changes
5. **Monitor logs** during deployment to verify file preserved

---

**Last Updated**: February 12, 2026
