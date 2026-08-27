<#
    Vision Training Platform launcher.

        .\start.ps1                 local only, backend + Vite dev server
        .\start.ps1 -Network        reachable from other machines on the LAN
        .\start.ps1 -Backend        backend only
        .\start.ps1 -Frontend       frontend only
        .\start.ps1 -Install        install dependencies first
        .\start.ps1 -Update         pull the latest code, rebuild, then start
        .\start.ps1 -Firewall       add the Windows Firewall rules (needs admin)

    -Network builds the frontend and serves it from the backend on a single
    port. That is deliberately different from the local dev setup: one port is
    far easier to open through a firewall, needs no Node process, and keeps the
    UI and the API same-origin so the login cookie works with no extra config.
#>
param(
    [switch]$Backend,
    [switch]$Frontend,
    [switch]$Install,
    [switch]$Network,
    [switch]$Firewall,
    [switch]$Update
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path

$backendPort = if ($env:BACKEND_PORT) { $env:BACKEND_PORT } else { '64031' }
$frontendPort = if ($env:FRONTEND_PORT) { $env:FRONTEND_PORT } else { '64030' }

function Test-Command($name) {
    $null -ne (Get-Command $name -ErrorAction SilentlyContinue)
}

function Get-LocalIPv4 {
    Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
        Where-Object {
            $_.IPAddress -notlike '127.*' -and
            $_.IPAddress -notlike '169.254.*' -and
            $_.InterfaceAlias -notlike '*Loopback*'
        } |
        Select-Object -ExpandProperty IPAddress
}

function Test-IsAdmin {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    (New-Object Security.Principal.WindowsPrincipal $identity).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Add-FirewallRules {
    <#  Windows blocks inbound connections on a Public network profile, which
        is what an office Ethernet usually is. Without a rule the port is
        simply unreachable and the browser reports a refused connection. #>
    if (-not (Test-IsAdmin)) {
        Write-Host 'Adding firewall rules needs an elevated PowerShell.' -ForegroundColor Yellow
        Write-Host 'Right-click PowerShell, "Run as administrator", then:' -ForegroundColor Yellow
        Write-Host "    cd '$root'; .\start.ps1 -Firewall" -ForegroundColor Cyan
        return $false
    }

    foreach ($port in @($backendPort, $frontendPort)) {
        $name = "Vision Training $port"
        Get-NetFirewallRule -DisplayName $name -ErrorAction SilentlyContinue |
            Remove-NetFirewallRule -ErrorAction SilentlyContinue
        New-NetFirewallRule -DisplayName $name `
            -Direction Inbound -Action Allow -Protocol TCP -LocalPort $port `
            -Profile Any | Out-Null
        Write-Host "  allowed inbound TCP $port" -ForegroundColor Green
    }
    return $true
}

if ($Firewall) {
    Write-Host 'Configuring Windows Firewall...' -ForegroundColor Cyan
    if (Add-FirewallRules) { Write-Host 'Done.' -ForegroundColor Green }
    return
}

if (-not (Test-Command 'python')) { throw 'python was not found on PATH.' }
if (-not (Test-Command 'npm')) { throw 'npm was not found on PATH.' }

function Stop-RunningServer($port) {
    <#  A server already bound to the port keeps serving the old code: the new
        process cannot bind, and requests for endpoints the update added come
        back as 404 from the one still running. That is the easiest way to
        conclude an update did not work when it did. #>
    $conns = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    foreach ($conn in $conns) {
        Write-Host "  stopping the server already on port $port (pid $($conn.OwningProcess))" -ForegroundColor DarkGray
        Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
    }
    if ($conns) { Start-Sleep -Seconds 2 }
}

if ($Update) {
    if (-not (Test-Command 'git') -or -not (Test-Path (Join-Path $root '.git'))) {
        Write-Host 'This is not a git checkout, so there is nothing to pull.' -ForegroundColor Red
        Write-Host 'Download the latest copy instead, or clone the repository.' -ForegroundColor Red
        exit 1
    }

    Set-Location $root
    Stop-RunningServer $backendPort
    Stop-RunningServer $frontendPort

    $before = (git rev-parse HEAD)

    # Annotations are tracked on purpose - they are work that cannot be
    # regenerated - so boxes drawn since the last update appear as local
    # changes and would block the pull. They are set aside and put back rather
    # than being lost or being allowed to stop the update.
    $stashed = $false
    if (git status --porcelain -- data) {
        Write-Host '  setting your data aside while the code updates' -ForegroundColor DarkGray
        git stash push --quiet --include-untracked --message 'start.ps1 -Update' -- data
        if ($LASTEXITCODE -eq 0) { $stashed = $true }
    }

    Write-Host 'Fetching...' -ForegroundColor Cyan
    # git reports "Already up to date." itself, so nothing here repeats it.
    # Its output is captured so a failure can be shown in full, and the status
    # is read from $LASTEXITCODE: sending a native command through a pipeline
    # makes $? unreliable in Windows PowerShell.
    $pullOutput = git pull --ff-only 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host ($pullOutput | Out-String).Trim() -ForegroundColor DarkGray
        Write-Host ''
        Write-Host 'The pull did not go through cleanly.' -ForegroundColor Red
        Write-Host 'Usually that means this checkout has local commits. Look at:' -ForegroundColor Red
        Write-Host '    git status'
        Write-Host '    git log --oneline -3'
        if ($stashed) { git stash pop --quiet; Write-Host 'Your data was put back.' }
        exit 1
    }

    if ($stashed) {
        git stash pop --quiet
        if ($LASTEXITCODE -eq 0) {
            Write-Host '  your data is back' -ForegroundColor DarkGray
        } else {
            Write-Host ''
            Write-Host 'Your data could not be put back automatically; it is safe in' -ForegroundColor Red
            Write-Host 'the stash. Recover it with:  git stash pop' -ForegroundColor Red
            exit 1
        }
    }

    $after = (git rev-parse HEAD)
    if ($before -ne $after) {
        Write-Host 'Updated:' -ForegroundColor Green
        git --no-pager log --oneline "$before..$after" | ForEach-Object { Write-Host "  $_" }

        # Only reinstall when the lists actually changed; pip and npm are slow
        # enough that doing it every time would discourage updating at all.
        $changed = git diff --name-only $before $after
        if ($changed -match 'backend/requirements.txt') {
            Write-Host 'Python dependencies changed; installing...' -ForegroundColor Cyan
            python -m pip install -q -r (Join-Path $root 'backend\requirements.txt')
        }
        if ($changed -match 'frontend/package.json') {
            Write-Host 'Frontend dependencies changed; installing...' -ForegroundColor Cyan
            Push-Location (Join-Path $root 'frontend'); npm install --silent; Pop-Location
        }
    }

    # The backend serves frontend/dist, not frontend/src, so without this the
    # browser keeps showing the previous version of the interface.
    Write-Host 'Building the interface...' -ForegroundColor Cyan
    Push-Location (Join-Path $root 'frontend'); npm run build; Pop-Location

    Write-Host ''
    Write-Host 'Update finished. Starting...' -ForegroundColor Green
    Write-Host ''
}

if ($Install) {
    Write-Host 'Installing Python dependencies...' -ForegroundColor Cyan
    python -m pip install -r (Join-Path $root 'backend\requirements.txt')

    Write-Host 'Installing frontend dependencies...' -ForegroundColor Cyan
    Push-Location (Join-Path $root 'frontend')
    npm install
    Pop-Location
}

# ── Network mode: one port, built UI served by the backend ──────────────────
if ($Network) {
    Write-Host 'Building the frontend...' -ForegroundColor Cyan
    Push-Location (Join-Path $root 'frontend')
    npm run build
    Pop-Location

    $env:BACKEND_HOST = '0.0.0.0'
    $addresses = @(Get-LocalIPv4)

    Write-Host ''
    Write-Host '========================================================' -ForegroundColor Yellow
    Write-Host ' This will be reachable from other machines' -ForegroundColor Yellow
    Write-Host '========================================================' -ForegroundColor Yellow
    Write-Host ' Change the admin password in Settings before leaving it open.'
    Write-Host ' The default admin/admin123 lets anyone on this network in.'
    Write-Host ''

    if (-not (Test-IsAdmin)) {
        $rule = Get-NetFirewallRule -DisplayName "Vision Training $backendPort" -ErrorAction SilentlyContinue
        if (-not $rule) {
            Write-Host ' Windows Firewall has no rule for this port yet, so other' -ForegroundColor Yellow
            Write-Host ' machines will still be refused. In an admin PowerShell run:' -ForegroundColor Yellow
            Write-Host "     cd '$root'; .\start.ps1 -Firewall" -ForegroundColor Cyan
            Write-Host ''
        }
    } else {
        Add-FirewallRules | Out-Null
    }

    Write-Host ' Open from another machine:' -ForegroundColor Green
    foreach ($ip in $addresses) { Write-Host "     http://${ip}:$backendPort" -ForegroundColor Cyan }
    Write-Host "     http://localhost:$backendPort  (this machine)" -ForegroundColor DarkGray
    Write-Host ''

    Set-Location $root
    python backend/app.py
    return
}

# ── Local development: backend on loopback, Vite in front ───────────────────
$runBackend = $Backend -or (-not $Backend -and -not $Frontend)
$runFrontend = $Frontend -or (-not $Backend -and -not $Frontend)

if ($runBackend) {
    Write-Host "Starting backend on http://127.0.0.1:$backendPort" -ForegroundColor Green
    Start-Process powershell -ArgumentList @(
        '-NoExit', '-Command',
        "Set-Location '$root'; python backend/app.py"
    )
}

if ($runFrontend) {
    if ($runBackend) { Start-Sleep -Seconds 2 }
    Write-Host "Starting frontend on http://localhost:$frontendPort" -ForegroundColor Green
    Start-Process powershell -ArgumentList @(
        '-NoExit', '-Command',
        "Set-Location '$root\frontend'; npm run dev"
    )
}

Write-Host ''
Write-Host "Open http://localhost:$frontendPort" -ForegroundColor Cyan
foreach ($ip in Get-LocalIPv4) {
    Write-Host "  or http://${ip}:$frontendPort from another machine" -ForegroundColor DarkGray
}
Write-Host 'If another machine cannot connect, the firewall rule is missing:' -ForegroundColor DarkGray
Write-Host "    .\start.ps1 -Firewall   (in an admin PowerShell)" -ForegroundColor DarkGray
Write-Host 'Close the spawned windows to stop the services.' -ForegroundColor DarkGray
