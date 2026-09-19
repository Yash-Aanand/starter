param([Parameter(Mandatory=$true)][string]$RunId)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$env:DRYFT_API = 'https://htn.dryft.ai'
$tokenPath = Join-Path $repoRoot '.env'
if (-not $env:DRYFT_TOKEN -and (Test-Path -LiteralPath $tokenPath)) {
    $line = Get-Content -LiteralPath $tokenPath | Where-Object { $_ -like 'DRYFT_TOKEN=*' } | Select-Object -First 1
    if ($line) { $env:DRYFT_TOKEN = $line.Substring(12) }
}
if (-not $env:DRYFT_TOKEN) { throw 'Configure DRYFT_TOKEN before inspecting a run.' }
$run = Invoke-RestMethod -Uri "$env:DRYFT_API/api/v1/runs/$RunId" -Headers @{ Authorization = "Bearer $env:DRYFT_TOKEN" }
$resultsDir = Join-Path $PSScriptRoot 'results'
New-Item -ItemType Directory -Path $resultsDir -Force | Out-Null
$run | ConvertTo-Json -Depth 50 | Set-Content -LiteralPath (Join-Path $resultsDir "$RunId.json") -Encoding UTF8
$detail = $run.run
[pscustomobject]@{
    id = $detail.id
    state = $detail.state
    score = $detail.result.score
    ranked = $detail.result.ranked
    reason = $detail.result.rankingReason
    error = $detail.errorMessage
    failure = $detail.result.failureMessage
} | Format-List
foreach ($shape in $detail.result.shapes) {
    $m = $shape.modelMetrics
    $ttft = if ($m.referenceTtftMs) { [math]::Round($m.ttftMs / $m.referenceTtftMs, 3) } else { $null }
    $tpot = if ($m.referenceTpotMs) { [math]::Round($m.tpotMs / $m.referenceTpotMs, 3) } else { $null }
    [pscustomobject]@{
        workload = $shape.id
        status = $shape.caseStatus
        tps = $shape.tokensPerSecond
        ttftRatio = $ttft
        tpotRatio = $tpot
        message = $shape.caseMessage
    }
}
