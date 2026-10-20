# F08 联调启动入口（Windows PowerShell / PowerShell 7）。
# 在仓库根目录运行，示例见 docs/F08同伴联调配置.md。
# 不从 .env 读密钥；缺少令牌时隐藏输入，也不会在输出中显示。
param(
  [Parameter(Mandatory = $true)]
  [string]$BackendUrl,
  [string]$AgentHost = "127.0.0.1",
  [ValidateRange(1, 65535)]
  [int]$Port = 8100,
  [ValidateSet("mock", "live")]
  [string]$Mode = "mock",
  [string]$Python = "python",
  [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

function Read-Secret([string]$Prompt) {
  $secure = Read-Host $Prompt -AsSecureString
  $ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
  try {
    return [Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr)
  } finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr)
    $secure.Dispose()
  }
}

$parsedUrl = $null
if (-not [Uri]::TryCreate($BackendUrl, [UriKind]::Absolute, [ref]$parsedUrl) -or
    $parsedUrl.Scheme -notin @("http", "https") -or
    $parsedUrl.UserInfo -or $parsedUrl.Query -or $parsedUrl.Fragment -or
    $parsedUrl.AbsolutePath -ne "/") {
  throw "BackendUrl 必须是可访问的 http(s) 服务根地址，不得包含账号、查询参数或片段。"
}
$backendBaseUrl = $BackendUrl.TrimEnd("/")

$pythonCommand = Get-Command $Python -ErrorAction SilentlyContinue
if (-not $pythonCommand) { throw "找不到 Python：$Python（需要 Python 3.12+）" }
& $Python -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)'
if ($LASTEXITCODE -ne 0) { throw "需要 Python 3.12+，请用 -Python 指定可用的解释器。" }

$names = @(
  "GCA_MODE", "GCA_HOST", "GCA_PORT", "GCA_INTERNAL_TOKEN",
  "GCA_BACKEND_BASE_URL", "GCA_BACKEND_SERVICE_TOKEN", "GCA_API_KEY",
  "GCA_BASE_URL", "GCA_TEXT_MODEL", "GCA_VISION_MODEL"
)
$previous = @{}
foreach ($name in $names) {
  $previous[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
}

try {
  $inboundToken = $previous["GCA_INTERNAL_TOKEN"]
  if ([string]::IsNullOrWhiteSpace($inboundToken)) {
    $inboundToken = Read-Secret "输入后端 Worker 调用 Agent 时使用的令牌（GCA_INTERNAL_TOKEN）"
  }
  $backendToken = $previous["GCA_BACKEND_SERVICE_TOKEN"]
  if ([string]::IsNullOrWhiteSpace($backendToken)) {
    $backendToken = Read-Secret "输入 Agent 访问后端内部 API 的令牌（GCA_BACKEND_SERVICE_TOKEN）"
  }
  if ([string]::IsNullOrWhiteSpace($inboundToken) -or
      [string]::IsNullOrWhiteSpace($backendToken)) {
    throw "F08 联调必须配置两侧服务令牌；它们可按实际后端配置使用同一个值。"
  }

  [Environment]::SetEnvironmentVariable("GCA_MODE", $Mode, "Process")
  [Environment]::SetEnvironmentVariable("GCA_HOST", $AgentHost, "Process")
  [Environment]::SetEnvironmentVariable("GCA_PORT", [string]$Port, "Process")
  [Environment]::SetEnvironmentVariable("GCA_INTERNAL_TOKEN", $inboundToken, "Process")
  [Environment]::SetEnvironmentVariable("GCA_BACKEND_BASE_URL", $backendBaseUrl, "Process")
  [Environment]::SetEnvironmentVariable("GCA_BACKEND_SERVICE_TOKEN", $backendToken, "Process")

  if ($Mode -eq "live") {
    $modelKey = $previous["GCA_API_KEY"]
    if ([string]::IsNullOrWhiteSpace($modelKey)) {
      $modelKey = Read-Secret "输入模型服务 API Key（GCA_API_KEY）"
    }
    if ([string]::IsNullOrWhiteSpace($modelKey)) { throw "live 模式必须配置模型 API Key。" }
    [Environment]::SetEnvironmentVariable("GCA_API_KEY", $modelKey, "Process")
    if (-not $previous["GCA_BASE_URL"]) {
      [Environment]::SetEnvironmentVariable("GCA_BASE_URL", "https://api.siliconflow.cn/v1", "Process")
      if (-not $previous["GCA_TEXT_MODEL"]) {
        [Environment]::SetEnvironmentVariable("GCA_TEXT_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507", "Process")
      }
      if (-not $previous["GCA_VISION_MODEL"]) {
        [Environment]::SetEnvironmentVariable("GCA_VISION_MODEL", "Qwen/Qwen3-VL-30B-A3B-Instruct", "Process")
      }
    } elseif (-not $previous["GCA_TEXT_MODEL"] -or -not $previous["GCA_VISION_MODEL"]) {
      throw "自定义 GCA_BASE_URL 时还须设置 GCA_TEXT_MODEL 和 GCA_VISION_MODEL。"
    }
  }

  Write-Host "F08 Agent 配置就绪：模式=$Mode，后端=$backendBaseUrl，监听=$AgentHost`:$Port；两侧服务令牌已设置（不显示）。"
  if ($CheckOnly) { return }
  & $Python -m gutcare_agents serve --host $AgentHost --port $Port
  if ($LASTEXITCODE -ne 0) { throw "Agent 进程异常退出，退出码：$LASTEXITCODE" }
} finally {
  foreach ($name in $names) {
    [Environment]::SetEnvironmentVariable($name, $previous[$name], "Process")
  }
}
