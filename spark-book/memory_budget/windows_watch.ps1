# The WSL supervisor supplies $stopFile. This process reads host counters only.
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object Text.UTF8Encoding($false)
while (-not (Test-Path -LiteralPath $stopFile)) {
    $timer = [Diagnostics.Stopwatch]::StartNew()
    try {
        $m = Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory -OperationTimeoutSec 4
        if ($null -eq $m.AvailableBytes -or $null -eq $m.CommittedBytes -or $null -eq $m.CommitLimit) {
            throw "Windows memory counters were incomplete"
        }
        $sample = [ordered]@{
            utc = [DateTime]::UtcNow.ToString('o')
            available_bytes = [long]$m.AvailableBytes
            committed_bytes = [long]$m.CommittedBytes
            commit_limit_bytes = [long]$m.CommitLimit
            query_seconds = $timer.Elapsed.TotalSeconds
        }
        [Console]::Out.WriteLine(($sample | ConvertTo-Json -Compress))
        [Console]::Out.Flush()
    } catch {
        [Console]::Error.WriteLine($_.Exception.Message)
        exit 3
    }
    Start-Sleep -Seconds 1
}
