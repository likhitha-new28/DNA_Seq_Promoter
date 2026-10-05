param(
    [switch]$SkipBrowser,
    [switch]$PrepareOnly,
    [switch]$Retrain,
    [switch]$RefreshDependencies,
    [ValidateRange(1024, 65535)][int]$Port = 8501
)

$ErrorActionPreference = "Stop"
$ProjectDirectory = $PSScriptRoot
$EnvironmentDirectory = Join-Path $ProjectDirectory ".venv"
$PythonExecutable = Join-Path $EnvironmentDirectory "Scripts\python.exe"
$DependencyStamp = Join-Path $EnvironmentDirectory "dna-dependencies.sha256"
$env:TF_CPP_MIN_LOG_LEVEL = '2'

Set-Location -LiteralPath $ProjectDirectory

if (-not $PrepareOnly) {
    $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $Port)
    try { $listener.Start() }
    catch { throw "Port $Port is in use. Close the earlier app, or launch with -Port 8502." }
    finally { $listener.Stop() }
}

if (-not (Test-Path -LiteralPath $PythonExecutable)) {
    Write-Host "Creating the local Python environment..." -ForegroundColor Cyan
    python -m venv $EnvironmentDirectory
    if ($LASTEXITCODE -ne 0) { throw "Could not create the Python environment." }
}

$DependencyHash = ((Get-FileHash -Algorithm SHA256 -LiteralPath 'requirements.txt').Hash +
    (Get-FileHash -Algorithm SHA256 -LiteralPath 'pyproject.toml').Hash)
$InstalledHash = if (Test-Path -LiteralPath $DependencyStamp) {
    (Get-Content -Raw -LiteralPath $DependencyStamp).Trim()
} else { '' }
if ($RefreshDependencies -or $InstalledHash -ne $DependencyHash) {
    Write-Host "Installing project dependencies (internet required for first setup)..." -ForegroundColor Cyan
    & $PythonExecutable -m pip install --disable-pip-version-check -e .
    if ($LASTEXITCODE -ne 0) {
        throw "Dependency setup failed. Use a Python version with a compatible TensorFlow wheel (3.11 recommended)."
    }
    Set-Content -LiteralPath $DependencyStamp -Value $DependencyHash -Encoding ASCII
} else {
    Write-Host "Using the installed environment; no dependency download needed." -ForegroundColor Cyan
}

Write-Host "Checking the demo model..." -ForegroundColor Cyan
$PrepareArguments = @('-m', 'dna_classifier.cli', 'prepare-demo')
if ($Retrain) { $PrepareArguments += '--force' }
& $PythonExecutable @PrepareArguments
if ($LASTEXITCODE -ne 0) { throw "Model preparation failed. See the error above; the app was not started." }
if ($PrepareOnly) { Write-Host "Demo prepared. Run this launcher again to open the app." -ForegroundColor Green; exit 0 }

$LocalUrl = "http://127.0.0.1:$Port"
Write-Host "Starting the app at $LocalUrl. Keep this window open; Ctrl+C stops it." -ForegroundColor Green
$Headless = if ($SkipBrowser) { 'true' } else { 'false' }

& $PythonExecutable -m streamlit run src/dna_classifier/webapp.py `
    --server.address 127.0.0.1 `
    --server.port $Port `
    --server.headless $Headless `
    --browser.serverAddress 127.0.0.1 `
    --browser.gatherUsageStats false
exit $LASTEXITCODE
