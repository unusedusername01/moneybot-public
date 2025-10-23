# Conda Deployment Setup

## Overview

The deployment workflow has been updated to use your conda environment (`moneybot`) instead of pip-based installation. This ensures consistency between your local development environment and the deployed application.

## Files Added

- `environment.yml` - Exported conda environment from your local `moneybot` environment
- `environment.yml.b64` - Base64-encoded version for GitHub secrets (optional)

## GitHub Secrets Configuration

### Required Secrets

The following secrets should be configured in your GitHub repository:

1. **ENVIRONMENT_YML_B64** (Optional, recommended for private repos)
   - Base64-encoded version of `environment.yml`
   - Used as fallback if `environment.yml` is not committed to the repo
   - Can be generated from `environment.yml.b64` file

2. **ENV_CI_DEPLOY_B64** (Existing)
   - Base64-encoded `.env.ci.deploy` file
   
3. **CONFIG_CI_DEPLOY_YML_B64** (Existing)
   - Base64-encoded `config/ci.deploy.yml` file

### Adding the ENVIRONMENT_YML_B64 Secret

If you want to keep `environment.yml` private (not committed to repo):

1. Open `environment.yml.b64` file
2. Copy the entire Base64 string
3. Go to GitHub repo → Settings → Secrets and variables → Actions
4. Click "New repository secret"
5. Name: `ENVIRONMENT_YML_B64`
6. Value: Paste the Base64 string
7. Click "Add secret"

## Workflow Changes

### Conda Environment Setup

The workflow now:

1. Checks out the repository
2. Attempts to restore `environment.yml` from Base64 secret if not present
3. Uses `conda-incubator/setup-miniconda@v3` action to set up conda
4. Creates/updates the `moneybot` environment from `environment.yml`
5. Detects GPU availability and installs appropriate PyTorch wheels

### Shell Configuration

For Linux/macOS steps that need conda:
- Uses `shell: bash -el {0}` to ensure conda environment is activated

For Windows steps:
- Uses `shell: pwsh` and explicitly activates conda when needed

## Local Updates

To update the environment after installing new packages:

```bash
# Update environment.yml
conda env export -n moneybot > environment.yml

# Regenerate Base64 (if using secret)
powershell -Command "[Convert]::ToBase64String([System.IO.File]::ReadAllBytes('environment.yml'))" > environment.yml.b64

# Commit changes (if environment.yml is tracked)
git add environment.yml
git commit -m "Update conda environment"
git push
```

## Benefits

1. **Consistency**: Same environment locally and in deployment
2. **Reproducibility**: Exact versions pinned in environment.yml
3. **GPU Handling**: Automatic detection and appropriate PyTorch installation
4. **Platform-Aware**: Works on Windows, Linux, and macOS self-hosted runners
5. **Fallback Support**: Can restore environment from secrets if not in repo

## Service Management

The workflow maintains the same service management strategy:

- **Linux**: Attempts systemd service, falls back to nohup
- **Windows**: Attempts Windows Service, falls back to Start-Process
- Health checks verify the application is running

## Troubleshooting

### Environment not found
- Ensure `environment.yml` is in repo OR `ENVIRONMENT_YML_B64` secret is set
- Check workflow logs for "Restore environment.yml" step

### Conda activation issues
- Verify `shell: bash -el {0}` is used for bash steps
- Check that conda is available on self-hosted runner

### Package conflicts
- Update environment locally: `conda env update --file environment.yml --prune`
- Export updated environment: `conda env export -n moneybot > environment.yml`
