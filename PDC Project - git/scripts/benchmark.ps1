# ---------------------------------------------------------------------------
# benchmark.ps1 -- Phase 2 OpenMP performance sweep.
#
# Runs the OpenMP pipeline across thread counts and dataset sizes, repeating
# each configuration several times, and records EVERY run's per-stage and total
# timings to results/benchmark_openmp.csv. It then prints a speedup/efficiency
# summary using the best (minimum) total time per configuration -- the standard
# way to report parallel timings, since the minimum is least polluted by
# OS scheduling jitter and background load.
#
# Speedup  S(p) = T(1) / T(p)        (how many times faster with p threads)
# Efficiency E(p) = S(p) / p          (fraction of ideal linear speedup)
#
# Usage:  .\scripts\benchmark.ps1
# ---------------------------------------------------------------------------
$ErrorActionPreference = "Stop"

$datasets   = @("10k", "50k", "full")   # 1k is too small to time meaningfully
$threadList = @(1, 2, 4, 8, 12)         # 12 = all logical cores on this machine
$minWords   = 5
$reps       = 5
$exe        = "./build/omp_pipeline.exe"
$csv        = "results/benchmark_openmp.csv"

if (-not (Test-Path $exe)) { throw "Build first: .\build.ps1 (missing $exe)" }
New-Item -ItemType Directory -Force -Path "results" | Out-Null

# Fresh detailed CSV (per-run, per-stage). Distinct from the in-binary CSV.
"dataset,dataset_size,threads,rep,load_ms,clean_ms,filter_ms,dedup_ms,tokenize_ms,total_ms" |
    Out-File -FilePath $csv -Encoding utf8

function Get-Stage([string[]]$out, [string]$name) {
    foreach ($line in $out) {
        if ($line -match ("^\s*{0}\s*:\s*([0-9.]+)" -f [regex]::Escape($name))) { return [double]$Matches[1] }
    }
    return [double]::NaN
}
function Get-Size([string[]]$out) {
    foreach ($line in $out) { if ($line -match 'Total documents\s*:\s*(\d+)') { return [long]$Matches[1] } }
    return 0
}

# best[ds][threads] = minimum total_ms observed
$best = @{}

foreach ($ds in $datasets) {
    $in  = "data/agnews/agnews_$ds.txt"
    $best[$ds] = @{}
    Write-Host "`n==== dataset $ds ====" -ForegroundColor Cyan
    foreach ($t in $threadList) {
        $minTotal = [double]::PositiveInfinity
        for ($r = 1; $r -le $reps; $r++) {
            $out  = & $exe $in "results/_bench_omp.txt" $minWords $t
            $size = Get-Size $out
            $load = Get-Stage $out "load"
            $cln  = Get-Stage $out "clean"
            $flt  = Get-Stage $out "filter"
            $ddp  = Get-Stage $out "dedup"
            $tok  = Get-Stage $out "tokenize"
            $tot  = Get-Stage $out "TOTAL"
            "{0},{1},{2},{3},{4},{5},{6},{7},{8},{9}" -f $ds,$size,$t,$r,$load,$cln,$flt,$ddp,$tok,$tot |
                Out-File -FilePath $csv -Append -Encoding utf8
            if ($tot -lt $minTotal) { $minTotal = $tot }
        }
        $best[$ds][$t] = $minTotal
        Write-Host ("  {0,2} threads: best total = {1,9:N3} ms" -f $t, $minTotal)
    }
}

# ----- speedup / efficiency summary (relative to 1 thread per dataset) -----
Write-Host "`n================ SPEEDUP (S = T1/Tp) ================" -ForegroundColor Yellow
$header = "dataset".PadRight(8) + ($threadList | ForEach-Object { ("p={0}" -f $_).PadLeft(10) }) -join ""
Write-Host ("dataset ".PadRight(10) + (($threadList | ForEach-Object { ("p={0}" -f $_).PadLeft(10) }) -join ""))
foreach ($ds in $datasets) {
    $t1 = $best[$ds][1]
    $row = $ds.PadRight(10)
    foreach ($t in $threadList) { $row += ("{0:N2}x" -f ($t1 / $best[$ds][$t])).PadLeft(10) }
    Write-Host $row
}
Write-Host "`n============== EFFICIENCY (E = S/p) ==============" -ForegroundColor Yellow
Write-Host ("dataset ".PadRight(10) + (($threadList | ForEach-Object { ("p={0}" -f $_).PadLeft(10) }) -join ""))
foreach ($ds in $datasets) {
    $t1 = $best[$ds][1]
    $row = $ds.PadRight(10)
    foreach ($t in $threadList) { $row += ("{0:P0}" -f (($t1 / $best[$ds][$t]) / $t)).PadLeft(10) }
    Write-Host $row
}
Write-Host "`nDetailed per-run timings written to $csv" -ForegroundColor Green
