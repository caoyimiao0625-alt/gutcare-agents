# GutCare 智能体服务 · 一键启动（Windows / PowerShell）
#
# 用法：
#   .\run.ps1                        # mock 模式，8100 端口，自动打开控制台
#   .\run.ps1 -Mode replay           # 读录制结果（断网演示）
#   .\run.ps1 -Mode live -Port 8200  # 真实模型（需先设置 GCA_API_KEY）
#   .\run.ps1 -NoBrowser             # 不自动开浏览器（服务器上跑）
#
param(
  [ValidateSet("mock", "replay", "live")]
  [string]$Mode = "mock",
  [int]$Port = 8100,
  [string]$FixtureRoot = "",
  [string]$Token = "",
  [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

# 1) 找 Python
$python = $null
foreach ($candidate in @("python", "py")) {
  $cmd = Get-Command $candidate -ErrorAction SilentlyContinue
  if ($cmd) { $python = $cmd.Source; break }
}
if (-not $python) {
  Write-Host "未找到 Python 3.12+，请先安装或把 python 加入 PATH。" -ForegroundColor Red
  exit 1
}

# 2) 环境变量（不写死密钥）
if ($FixtureRoot) { $env:GCA_FIXTURE_ROOT = $FixtureRoot }
elseif (-not $env:GCA_FIXTURE_ROOT) {
  $guess = Join-Path (Split-Path $PSScriptRoot -Parent) "testdata_v4"
  if (Test-Path $guess) { $env:GCA_FIXTURE_ROOT = $guess }
}
$env:GCA_MODE = $Mode
$env:GCA_PORT = "$Port"
$env:PYTHONIOENCODING = "utf-8"
if ($Token) { $env:GCA_INTERNAL_TOKEN = $Token }

# 3) live 模式自检
if ($Mode -eq "live" -and -not $env:GCA_API_KEY) {
  Write-Host "live 模式需要 GCA_API_KEY（只放环境变量，别写进文件）。" -ForegroundColor Yellow
  $secure = Read-Host "请输入 API Key（回车跳过，改用 mock）" -AsSecureString
  $plain = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
    [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure))
  if ([string]::IsNullOrWhiteSpace($plain)) {
    Write-Host "已改为 mock 模式。" -ForegroundColor Yellow
    $env:GCA_MODE = "mock"
  } else {
    $env:GCA_API_KEY = $plain
  }
}

Write-Host "启动 GutCare Agents ..." -ForegroundColor Cyan
Write-Host "  模式：$env:GCA_MODE"
Write-Host "  端口：$Port"
Write-Host "  测试包：$env:GCA_FIXTURE_ROOT"
Write-Host "  控制台：http://127.0.0.1:$Port/" -ForegroundColor Green

if (-not $NoBrowser) {
  Start-Job -ScriptBlock {
    param($url)
    Start-Sleep -Seconds 2
    Start-Process $url
  } -ArgumentList "http://127.0.0.1:$Port/" | Out-Null
}

& $python -m gutcare_agents serve --port $Port
