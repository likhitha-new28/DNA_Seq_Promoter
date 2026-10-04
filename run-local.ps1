param(
    [switch]$SkipBrowser
)

$ErrorActionPreference = "Stop"
$ProjectDirectory = $PSScriptRoot
$EnvironmentDirectory = Join-Path $ProjectDirectory ".venv"
$PythonExecutable = Join-Path $EnvironmentDirectory "Scripts\python.exe"
$ModelFile = Join-Path $ProjectDirectory "artifacts\best_model.keras"
$SettingsFile = Join-Path $ProjectDirectory "artifacts\settings.json"

Set-Location -LiteralPath $ProjectDirectory

if (-not (Test-Path -LiteralPath $PythonExecutable)) {
    Write-Host "Creating the local Python environment..." -ForegroundColor Cyan
    python -m venv $EnvironmentDirectory
    if ($LASTEXITCODE -ne 0) { throw "Could not create the Python environment." }
}

Write-Host "Installing or updating project dependencies..." -ForegroundColor Cyan
& $PythonExecutable -m pip install --disable-pip-version-check --quiet --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Could not update pip." }
& $PythonExecutable -m pip install --disable-pip-version-check --quiet -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Could not install dependencies." }
& $PythonExecutable -m pip install --disable-pip-version-check --quiet -e .
if ($LASTEXITCODE -ne 0) { throw "Could not install the project." }

if ((-not (Test-Path -LiteralPath $ModelFile)) -or (-not (Test-Path -LiteralPath $SettingsFile))) {
    Write-Host "Preparing the demo model (first launch only)..." -ForegroundColor Cyan
    & $PythonExecutable -m dna_classifier.cli demo-data `
        --output data/raw/demo_sequences.csv `
        --samples-per-class 120
    if ($LASTEXITCODE -ne 0) { throw "Could not create demo data." }
    & $PythonExecutable -m dna_classifier.cli train `
        --data data/raw/demo_sequences.csv `
        --epochs 8
    if ($LASTEXITCODE -ne 0) { throw "Could not train the demo model." }
}

$LocalUrl = "http://localhost:8501"
Write-Host "Local deployment is ready: $LocalUrl" -ForegroundColor Green
if (-not $SkipBrowser) {
    Start-Process $LocalUrl
}

& $PythonExecutable -m streamlit run src/dna_classifier/webapp.py `
    --server.address localhost `
    --server.port 8501 `
    --browser.gatherUsageStats false
