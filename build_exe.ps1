param(
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

if ($Clean) {
    Remove-Item -LiteralPath (Join-Path $Root "build") -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath (Join-Path $Root "dist") -Recurse -Force -ErrorAction SilentlyContinue
}

python -m PyInstaller --clean --noconfirm .\sdiu_command_finder.spec

Write-Host ""
Write-Host "Built executable:"
Write-Host "  $Root\dist\sdiu-command-finder.exe"
