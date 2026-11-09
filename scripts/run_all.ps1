# Flujo completo de He(RL)l Kitchen, desde los tests hasta los replays.
#
#   .\scripts\run_all.ps1                  Todo: unas 11 horas en CPU, por el entrenamiento
#   .\scripts\run_all.ps1 -OnlyAnalysis    Solo tablas, figuras y replays, con las corridas ya guardadas
#
# Si se interrumpe, se puede volver a correr: las corridas terminadas no se repiten.
# Sin acentos a proposito: PowerShell 5.1 lee los .ps1 sin BOM con la codificacion del sistema.

param(
    [switch]$OnlyAnalysis,
    [int]$Jobs = 4
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$python = ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "No existe $python. Primero crea el entorno virtual (ver README, Instalacion)."
}

function Invoke-Step([string]$Title, [string[]]$Arguments) {
    Write-Host ""
    Write-Host "=== $Title" -ForegroundColor Cyan
    & $python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Fallo: $Title" }
}

if (-not $OnlyAnalysis) {
    Invoke-Step "Tests" @("-m", "pytest", "-q")
    Invoke-Step "Calibracion (Paso 0)" @("-m", "experiments.calibrate")
    Invoke-Step "Baselines" @("-m", "experiments.evaluate", "--episodes", "100")
    Invoke-Step "Tuning y grilla de 40 corridas" @(
        "-m", "experiments.run_experiments", "--stage", "all",
        "--timesteps", "1000000", "--tune-timesteps", "400000", "--jobs", "$Jobs")
}

Invoke-Step "Analisis: tablas y figuras" @("-m", "experiments.analyze")
Invoke-Step "Tabla y figuras del informe" @("-m", "experiments.report_assets")
Invoke-Step "Replays" @("-m", "experiments.export_replays")

Write-Host ""
Write-Host "Listo. Para ver a los agentes:" -ForegroundColor Green
Write-Host "  .venv\Scripts\python.exe play_sim.py --rate 3 --agent results\replays\dqn_medium.jsonl `"DQN (N=6)`" --agent results\replays\ppo_medium.jsonl `"PPO (N=2)`" --agent results\replays\fifo_medium.jsonl `"FIFO`""
