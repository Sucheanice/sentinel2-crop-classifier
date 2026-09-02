# monitor_downloads.ps1 — 每15分钟检查下载状态
# 用法: powershell -File monitor_downloads.ps1

$BASE = "E:\工作相关\2026年\0624 待测试数据"
$LOG_FILE = Join-Path $BASE "download_monitor.log"

function Write-Log($msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "$ts | $msg"
    Write-Host $line
    Add-Content -Path $LOG_FILE -Value $line
}

function Check-S1Status($dir, $label) {
    if (-not (Test-Path $dir)) {
        Write-Log "[$label] 目录不存在"
        return
    }
    $files = Get-ChildItem -Path $dir -Recurse -File -ErrorAction SilentlyContinue
    $tif_count = ($files | Where-Object { $_.Name -match '\.tif$' -and $_.Name -notmatch '\.tmp$' }).Count
    $tmp_count = ($files | Where-Object { $_.Name -match '\.tmp$' }).Count
    $total_mb = [math]::Round(($files | Measure-Object Length -Sum).Sum / 1MB, 1)
    
    $scene_dirs = Get-ChildItem -Path $dir -Directory -ErrorAction SilentlyContinue
    $complete_scenes = @()
    $partial_scenes = @()
    foreach ($sd in $scene_dirs) {
        $bands = Get-ChildItem -Path $sd.FullName -File -ErrorAction SilentlyContinue
        $done = ($bands | Where-Object { $_.Name -match '\.tif$' -and $_.Name -notmatch '\.tmp$' }).Count
        $tmp = ($bands | Where-Object { $_.Name -match '\.tmp$' }).Count
        if ($tmp -gt 0) {
            $tmp_sizes = ($bands | Where-Object { $_.Name -match '\.tmp$' } | ForEach-Object { [math]::Round($_.Length/1MB,1) })
            $partial_scenes += "$($sd.Name) (done:$done, downloading:$tmp $($tmp_sizes -join ',')MB)"
        } elseif ($done -gt 0) {
            $complete_scenes += $sd.Name
        }
    }
    
    Write-Log "[$label] 已完成: $tif_count .tif, 下载中: $tmp_count .tmp, 共 $total_mb MB"
    if ($complete_scenes) { Write-Log "  完整场景: $($complete_scenes -join ', ')" }
    if ($partial_scenes) { Write-Log "  下载中: $($partial_scenes -join '; ')" }
}

function Check-S2Status($dir, $label) {
    if (-not (Test-Path $dir)) {
        Write-Log "[$label] 目录不存在"
        return
    }
    $files = Get-ChildItem -Path $dir -Recurse -File -ErrorAction SilentlyContinue
    $tif_count = ($files | Where-Object { $_.Name -match '\.tif$' -and $_.Name -notmatch '\.tmp$' }).Count
    $tmp_count = ($files | Where-Object { $_.Name -match '\.tmp$' }).Count
    $total_mb = [math]::Round(($files | Measure-Object Length -Sum).Sum / 1MB, 1)
    
    $scene_dirs = Get-ChildItem -Path $dir -Directory -ErrorAction SilentlyContinue
    $scene_info = @()
    foreach ($sd in $scene_dirs) {
        $bands = Get-ChildItem -Path $sd.FullName -File -ErrorAction SilentlyContinue
        $done = ($bands | Where-Object { $_.Name -match '\.tif$' -and $_.Name -notmatch '\.tmp$' }).Count
        $tmp = ($bands | Where-Object { $_.Name -match '\.tmp$' }).Count
        $scene_info += "$($sd.Name) ($done/10 bands)"
    }
    
    Write-Log "[$label] 已完成: $tif_count .tif, 下载中: $tmp_count .tmp, 共 $total_mb MB"
    foreach ($si in $scene_info) { Write-Log "  $si" }
}

function Check-Processes {
    $procs = Get-Process python* -ErrorAction SilentlyContinue
    if ($procs) {
        $info = ($procs | ForEach-Object { "PID:$($_.Id)($([math]::Round($_.CPU/60,1))min)" }) -join ", "
        Write-Log "[进程] Python: $info"
    } else {
        Write-Log "[进程] 无 Python 进程运行 (下载可能已完成)"
    }
}

# === 主循环 ===
Write-Log "===== 下载监控启动 ====="

while ($true) {
    Write-Log "------------------------------"
    Check-Processes
    Check-S1Status (Join-Path $BASE "小春_s1_48RWU") "遂宁S1"
    Check-S2Status (Join-Path $BASE "江油_s2") "江油S2"
    Check-S1Status (Join-Path $BASE "江油_s1") "江油S1"
    
    # 检查是否全部完成
    $procs = Get-Process python* -ErrorAction SilentlyContinue
    if (-not $procs) {
        Write-Log "所有 Python 进程已结束, 下载可能已完成!"
        Write-Log "请检查以上状态确认。"
        break
    }
    
    Write-Log "下次检查: $(Get-Date -Format 'HH:mm:ss').AddMinutes(15)"
    Start-Sleep -Seconds 900  # 15分钟
}
