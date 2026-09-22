<#
    Update the code and leave everything else alone.

        .\update.ps1                 fetch the latest and replace the code
        .\update.ps1 -Zip D:\x.zip   use a zip already downloaded
        .\update.ps1 -WhatIf         say what would change, change nothing

    This exists because the machines it runs on cannot use git. Their networks
    block the host GitHub serves release files from, so Git itself will not
    install, while a browser and PowerShell can still fetch the repository as a
    zip. Updating then meant extracting that zip over the installation, which
    works and is frightening: the archive also carries a data/ directory, and
    the folder it is being extracted over holds projects, annotations, trained
    weights and the database.

    So this copies the code and nothing else. data/, .env, and anything else
    that is not on the list below is never read, written or deleted, whatever
    the archive contains.
#>
param(
    [string]$Zip,
    [switch]$WhatIf
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$source = 'https://github.com/Commerry/Program-Training-Model/archive/refs/heads/main.zip'

# What an update is allowed to replace. Everything else in the installation is
# either the operator's (data, .env) or not ours to touch.
$codePaths = @(
    'backend',
    'frontend/dist',
    'frontend/src',
    'frontend/index.html',
    'frontend/package.json',
    'frontend/package-lock.json',
    'frontend/vite.config.js',
    'frontend/vitest.config.js',
    'start.ps1',
    'start.sh',
    'update.ps1',
    'README.md',
    '.env.example',
    '.gitignore'
)

function Say($text, $colour = 'Gray') { Write-Host $text -ForegroundColor $colour }

Say ''
Say 'Updating the code. data\ and .env are not touched.' Cyan
Say ''

# ── get the archive ─────────────────────────────────────────────────────────
$temp = Join-Path ([System.IO.Path]::GetTempPath()) ("vision-update-" + [guid]::NewGuid().ToString('N').Substring(0, 8))
New-Item -ItemType Directory -Path $temp | Out-Null

try {
    if ($Zip) {
        if (-not (Test-Path $Zip)) { throw "No such file: $Zip" }
        $archive = $Zip
        Say "Using $archive"
    } else {
        $archive = Join-Path $temp 'main.zip'
        # Windows PowerShell 5.1 still defaults to TLS 1.0, which GitHub closes
        # the connection on -- reported as "the underlying connection was
        # closed", which sounds like a firewall and is not one.
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Say "Downloading $source"
        Invoke-WebRequest $source -OutFile $archive -UseBasicParsing
        Say ("  {0:N1} MB" -f ((Get-Item $archive).Length / 1MB))
    }

    $unpacked = Join-Path $temp 'unpacked'
    Expand-Archive -Path $archive -DestinationPath $unpacked -Force

    # A zip of a repository holds one folder with everything inside it.
    $inner = Get-ChildItem $unpacked -Directory | Select-Object -First 1
    if (-not $inner) { throw 'That archive does not contain a checkout.' }
    if (-not (Test-Path (Join-Path $inner.FullName 'backend'))) {
        throw 'That archive has no backend\ in it; it is not this project.'
    }

    # ── replace the code, path by path ──────────────────────────────────────
    $changed = @()
    foreach ($relative in $codePaths) {
        $from = Join-Path $inner.FullName ($relative -replace '/', '\')
        $to = Join-Path $root ($relative -replace '/', '\')
        if (-not (Test-Path $from)) { continue }

        if ($WhatIf) {
            $changed += $relative
            continue
        }

        if ((Get-Item $from) -is [System.IO.DirectoryInfo]) {
            # Removed first so a file that no longer exists upstream goes away
            # rather than lingering. Safe because every path here is code: the
            # build in frontend\dist, for instance, names its files by a hash
            # of their contents, so stale ones from an older build accumulate
            # forever otherwise -- and a stale index.html beside fresh scripts
            # is a blank page.
            if (Test-Path $to) { Remove-Item $to -Recurse -Force }
            Copy-Item $from $to -Recurse -Force
        } else {
            Copy-Item $from $to -Force
        }
        $changed += $relative
    }

    Say ''
    if ($WhatIf) {
        Say 'Would replace:' Yellow
    } else {
        Say 'Replaced:' Green
    }
    foreach ($item in $changed) { Say "    $item" }

    Say ''
    Say 'Left alone:' DarkGray
    foreach ($kept in @('data\  (projects, annotations, weights, database)',
                        '.env   (password, ports)',
                        'frontend\node_modules')) {
        Say "    $kept" DarkGray
    }

    if (-not $WhatIf) {
        # Files that came out of a zip downloaded from the internet are marked,
        # and PowerShell's execution policy refuses marked scripts.
        Get-ChildItem $root -Recurse -File -ErrorAction SilentlyContinue |
            Unblock-File -ErrorAction SilentlyContinue

        Say ''
        Say 'Checking the interface that was shipped...' Cyan
        & python (Join-Path $root 'backend\tests\test_built_interface.py')

        Say ''
        Say 'Done. Start it with:' Green
        Say '    .\start.ps1 -Network' Cyan
    }
}
finally {
    Remove-Item $temp -Recurse -Force -ErrorAction SilentlyContinue
}
