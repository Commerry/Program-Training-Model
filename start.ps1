<#
    Vision Training Platform launcher.

        .\start.ps1                 local only, backend + Vite dev server
        .\start.ps1 -Network        reachable from other machines on the LAN
        .\start.ps1 -Backend        backend only
        .\start.ps1 -Frontend       frontend only
        .\start.ps1 -Install        install dependencies first
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
    [switch]$Firewall
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
