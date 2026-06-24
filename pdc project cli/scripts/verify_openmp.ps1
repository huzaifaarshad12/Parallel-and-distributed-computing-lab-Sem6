# ---------------------------------------------------------------------------
# verify_openmp.ps1 -- proves the OpenMP build matches the sequential baseline.
#
# For every dataset size, it:
#   1. produces the sequential reference output,
#   2. runs the OpenMP build at 1, 2, 4 and 8 threads,
#   3. asserts the cleaned output file is BYTE-IDENTICAL to the reference
#      (SHA-256 hash compare),
#   4. asserts the five correctness counters match.
# Exit code is 0 only if every check passes.
# ---------------------------------------------------------------------------
$ErrorActionPreference = "Stop"

$datasets   = @("1k", "10k", "50k", "full")
$threadList = @(1, 2, 4, 8)
$minWords   = 5
$tmp        = "results/_verify"
New-Item -ItemType Directory -Force -Path $tmp | Out-Null

# Pull the 5 counters out of a pipeline run's stdout.
function Get-Counters([string[]]$out) {
    $h = @{}
    foreach ($line in $out) {
        if ($line -match 'Total documents\s*:\s*(\d+)')      { $h.total = [long]$Matches[1] }
        if ($line -match 'Removed \(low quality\):\s*(\d+)')  { $h.removed = [long]$Matches[1] }
        if ($line -match 'Duplicates removed\s*:\s*(\d+)')    { $h.dups = [long]$Matches[1] }
        if ($line -match 'Kept documents\s*:\s*(\d+)')        { $h.kept = [long]$Matches[1] }
        if ($line -match 'Total tokens\s*:\s*(\d+)')          { $h.tokens = [long]$Matches[1] }
    }
    return $h
}

$allOk = $true
foreach ($ds in $datasets) {
    $in     = "data/agnews/agnews_$ds.txt"
    $refOut = "$tmp/seq_$ds.txt"
    Write-Host "`n==== dataset $ds ====" -ForegroundColor Cyan

    $seqOut = & ./build/seq_pipeline.exe $in $refOut $minWords
    $seqC   = Get-Counters $seqOut
    $refHash = (Get-FileHash $refOut -Algorithm SHA256).Hash
    Write-Host ("  seq: total={0} removed={1} dups={2} kept={3} tokens={4}" -f $seqC.total,$seqC.removed,$seqC.dups,$seqC.kept,$seqC.tokens)

    foreach ($t in $threadList) {
        $ompOut  = "$tmp/omp_${ds}_${t}t.txt"
        $out     = & ./build/omp_pipeline.exe $in $ompOut $minWords $t
        $c       = Get-Counters $out
        $hash    = (Get-FileHash $ompOut -Algorithm SHA256).Hash

        $countersMatch = ($c.total -eq $seqC.total) -and ($c.removed -eq $seqC.removed) -and
                         ($c.dups -eq $seqC.dups) -and ($c.kept -eq $seqC.kept) -and
                         ($c.tokens -eq $seqC.tokens)
        $bytesMatch = ($hash -eq $refHash)

        if ($countersMatch -and $bytesMatch) {
            Write-Host ("  OK   {0,2} threads: output byte-identical + counters match" -f $t) -ForegroundColor Green
        } else {
            $allOk = $false
            Write-Host ("  FAIL {0,2} threads: countersMatch={1} bytesMatch={2}" -f $t,$countersMatch,$bytesMatch) -ForegroundColor Red
        }
    }
}

Write-Host ""
if ($allOk) {
    Write-Host "ALL CHECKS PASSED: OpenMP == Sequential on every dataset/thread count." -ForegroundColor Green
    exit 0
} else {
    Write-Host "VERIFICATION FAILED." -ForegroundColor Red
    exit 1
}
