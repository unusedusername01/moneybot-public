# Deployment Guide

This guide documents how to deploy MoneyBot using the provided GitHub Actions workflow (`.github/workflows/deploy.yml`) and how to run the application as a persistent service on Linux or Windows self-hosted runners.

## Deployment Overview

1. **Runner Preparation**
   - Register a self-hosted runner labeled `self-hosted` (optionally add OS-specific labels like `Linux`, `Windows`).
   - Install Python 3.11 and system build tools.
   - Ensure GPU drivers (CUDA) are installed if you plan to leverage GPU inference.
   - Pre-create a systemd service (Linux) or Windows service (NSSM) named `moneybot` if you want managed lifecycles.

2. **Secrets Management**
   - Encode environment and preset files:
     - `.env.ci.deploy` → `ENV_CI_DEPLOY_B64`
     - `config/ci.deploy.yml` → `CONFIG_CI_DEPLOY_YML_B64`
   - Store both as repository secrets.

3. **Trigger Workflow**
   - Use GitHub UI → Actions → Deploy → “Run workflow”.
   - Optional input `restart_only=true` skips dependency installation and simply restarts the service.

## Workflow Breakdown

`deploy.yml` carries out the following stages:

1. **Checkout** – Pulls repository code onto the runner.
2. **Python Setup** – Ensures Python 3.11 is available and caches `pip` downloads.
3. **Config Restoration** – Materializes `.env.ci.deploy` and `config/ci.deploy.yml` from secrets if missing.
4. **Dependency Installation** – Installs `requirements.txt`. When CUDA is detected (via `nvidia-smi` on Linux or `Get-CimInstance` on Windows), attempts to install GPU-enabled PyTorch wheels. Falls back gracefully if unavailable.
5. **Service Management**:
   - Linux: Stops existing `moneybot` systemd service, reloads daemon, enables, and starts. Falls back to `nohup` background process if the service is absent.
  - Windows: Stops/starts `moneybot` Windows service if present; otherwise launches a background `python -m src.langgraph_workflow.app` process.
6. **Health Check** – Polls `http://127.0.0.1:8080/` to verify the service responds. Adjust port as needed via presets.

## Linux Service Setup

### systemd Unit Example (`/etc/systemd/system/moneybot.service`)

```
[Unit]
Description=MoneyBot FastAPI Service
After=network.target

[Service]
WorkingDirectory=/opt/moneybot
EnvironmentFile=/opt/moneybot/.env.ci.deploy
ExecStart=/usr/bin/python -m src.langgraph_workflow.app --preset ci.deploy --host 0.0.0.0 --port 8080
Restart=on-failure
User=moneybot

[Install]
WantedBy=multi-user.target
```

Reload daemon and start service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable moneybot
sudo systemctl start moneybot
```

## Windows Service Setup

Use [NSSM](https://nssm.cc/) or built-in `sc.exe` commands.

### NSSM Example

```
nssm install moneybot "C:\Python311\python.exe" "-m" "src.langgraph_workflow.app" "--preset" "ci.deploy" "--host" "0.0.0.0" "--port" "8080"
```

Add Environment tab entries pointing to `C:\path\to\.env.ci.deploy` or embed variables via `setx`.

Start the service:

```
nssm start moneybot
```

## Ports & Networking

- Default deploy preset listens on `0.0.0.0:8080`.
- Use reverse proxies (nginx, IIS) or load balancers to expose HTTPS.
- Ensure firewall rules allow inbound traffic on chosen port.

## GPU Considerations

- Deploy preset assumes GPU availability but gracefully downgrades when `nvidia-smi` is absent.
- When using Windows, check for CUDA via PowerShell snippet in workflow; update it if your GPU tooling differs.
- Monitor GPU memory usage to prevent OOM issues during heavy workloads.

## Log Management

- systemd captures stdout/stderr; inspect via `journalctl -u moneybot`.
- Windows: `moneybot.out`/`moneybot.err` files are created when falling back to background process.
- Consider shipping logs to centralized services (CloudWatch, ELK) for production use.

## Rolling Deployments

1. Run workflow with `restart_only=false` (default) to fetch code, install deps, restart service.
2. Verify health check and manual smoke tests.
3. If issues occur, roll back by redeploying a previous commit or using your infrastructure’s rollback mechanism.

## Manual Deployment Without GitHub Actions

1. Pull latest code onto server.
2. Create/refresh `.env` and preset files.
3. Install dependencies as in workflow.
4. Start service using systemd, NSSM, or manual `nohup`/`Start-Process` command.

## Post-Deployment Validation

- `curl http://127.0.0.1:8080/` → expect `{"status": "running"}` or `404` for non-root endpoints.
- Connect WebSocket from staging webapp to ensure interactive loop works.
- Run lightweight integration tests against deployed instance when feasible.

## Related Documentation

- [Testing & Coverage](testing_and_coverage.md)
- [Architecture Deep Dive](architecture.md)
- [Developer Guide](developer_guide.md)
- [Backend API Reference](backend_api_reference.md)

Keep this guide synchronized with workflow updates and infrastructure changes. Document any self-hosted runner prerequisites so new operators can reproduce the environment.
