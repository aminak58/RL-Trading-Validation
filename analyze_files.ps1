# Get all files
$files = Get-ChildItem "C:\FreqtradeProjects\RL-Trading-Validation\user_data\data\binance\futures" -Recurse -File | Where-Object {$_.Extension -eq ".feather" -or $_.Extension -eq ".json"}

# Calculate total size
$totalSize = ($files | Measure-Object -Property Length -Sum).Sum
$totalSizeMB = [math]::Round($totalSize/1MB,2)
Write-Host "Total size of all data files: $totalSizeMB MB" -ForegroundColor Green
Write-Host ""

# Group by base name (without extension) to find duplicates
$groups = $files | Group-Object {$_.BaseName}

Write-Host "Duplicate analysis:" -ForegroundColor Yellow
$duplicateFound = $false

foreach ($group in $groups) {
    if ($group.Count -gt 1) {
        $duplicateFound = $true
        Write-Host "Duplicate files found: $($group.Name)" -ForegroundColor Red
        $group.Group | Select-Object Name, @{Name="Size(MB)";Expression={[math]::Round($_.Length/1MB,2)}} | Format-Table -AutoSize
        Write-Host ""
    }
}

if (-not $duplicateFound) {
    Write-Host "No duplicate files found." -ForegroundColor Green
}

# Summary by coin
Write-Host "Summary by coin:" -ForegroundColor Yellow
$coinGroups = $files | Group-Object {$_.Name.Split("_")[0]}
foreach ($coin in $coinGroups) {
    $coinSize = ($coin.Group | Measure-Object -Property Length -Sum).Sum
    $coinSizeMB = [math]::Round($coinSize/1MB,2)
    Write-Host "$($coin.Name): $coinSizeMB MB ($($coin.Count) files)"
}