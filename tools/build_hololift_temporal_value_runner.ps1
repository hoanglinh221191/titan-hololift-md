param(
    [string]$Compiler = 'g++',
    [string]$Output = 'build\hololift_temporal_value_runner.exe',
    [string[]]$ExtraCompilerFlags = @()
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$outputPath = Join-Path $repo $Output
$outputDirectory = Split-Path -Parent $outputPath
New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null

$sources = @(
    (Join-Path $repo 'tools\hololift_temporal_value_runner.cpp'),
    (Join-Path $repo 'src\hololift\identity.cpp'),
    (Join-Path $repo 'src\hololift\frame_binding.cpp'),
    (Join-Path $repo 'src\hololift\lattice_transport.cpp'),
    (Join-Path $repo 'src\hololift\phase1.cpp'),
    (Join-Path $repo 'src\hololift\spatial_lift.cpp'),
    (Join-Path $repo 'src\hololift\validation.cpp'),
    (Join-Path $repo 'src\hololift\vibe_adapter.cpp'),
    (Join-Path $repo 'src\hololift\vibe_contract.cpp'),
    (Join-Path $repo 'src\hololift\vibe_importer.cpp'),
    (Join-Path $repo 'src\pbctopo\observation_schema.cpp'),
    (Join-Path $repo 'src\pbctopo\spatial_lift.cpp'),
    (Join-Path $repo 'src\pbctopo\lattice_math.cpp'),
    (Join-Path $repo 'src\pbctopo\temporal_path.cpp'),
    (Join-Path $repo 'src\gmxtraj\xdrfile.cpp'),
    (Join-Path $repo 'src\gmxtraj\xdrfile_xtc.cpp')
)

# Compiler warnings arrive on stderr; Windows PowerShell 5.1 turns those into terminating errors under 'Stop', so
# the exit code, not stderr, decides success.
$previousPreference = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
& $Compiler -std=c++23 -O2 -pthread -Wall -Wextra -Wpedantic -Wconversion @ExtraCompilerFlags `
    "-I$(Join-Path $repo 'src')" `
    "-I$(Join-Path $repo 'src\gmxtraj\include')" `
    @sources -o $outputPath 2>&1 | ForEach-Object { "$_" }
$compileExit = $LASTEXITCODE
$ErrorActionPreference = $previousPreference
if ($compileExit -ne 0) {
    throw "Failed to compile HoloLift temporal-value runner: $compileExit"
}
Write-Output $outputPath
