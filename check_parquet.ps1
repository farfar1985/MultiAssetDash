$dataDir = "C:\Users\William Dennis\projects\nexus\data"
Get-ChildItem $dataDir -Directory | ForEach-Object {
    $parquet = Join-Path $_.FullName "training_data.parquet"
    $master = Join-Path $_.FullName "master_dataset.parquet"
    $has_data = (Test-Path $parquet) -or (Test-Path $master)
    if ($has_data) {
        Write-Output $_.Name
    }
}
