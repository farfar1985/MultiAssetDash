$dataDir = "C:\Users\William Dennis\projects\nexus\data"
Get-ChildItem $dataDir -Directory | ForEach-Object {
    $hw = Join-Path $_.FullName "horizons_wide"
    if (Test-Path $hw) {
        $count = (Get-ChildItem $hw -Filter "*.joblib" -ErrorAction SilentlyContinue).Count
        Write-Output "$($_.Name): $count horizons"
    }
}
