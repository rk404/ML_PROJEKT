# Skrypt inicjalizacyjny środowiska Python
# Automatycznie ustawia kodowanie UTF-8

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"

Write-Host " Kodowanie UTF-8 aktywowane" -ForegroundColor Green
Write-Host " Środowisko .venv gotowe do pracy" -ForegroundColor Green
