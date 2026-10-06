$ErrorActionPreference = 'Stop'
$proofRoot = $PSScriptRoot
$compilerPath = 'C:\Users\josep\.elan\toolchains\leanprover--lean4---v4.33.0\bin\lean.exe'
$cachedPackageRoot = 'C:\Users\josep\lean\risley\.lake\packages'
$libraryPaths = Get-ChildItem -LiteralPath $cachedPackageRoot -Directory | ForEach-Object {
    $candidate = Join-Path $_.FullName '.lake\build\lib\lean'
    if (Test-Path -LiteralPath $candidate) { $candidate }
}
$env:LEAN_PATH = $libraryPaths -join ';'
$proofSource = Join-Path $proofRoot 'CriticalMargin.lean'
$proofObject = Join-Path $proofRoot 'CriticalMargin.olean'
$proofLog = Join-Path $proofRoot 'critical-margin-compiler-output.txt'
$sourceHashBefore = (Get-FileHash -LiteralPath $proofSource -Algorithm SHA256).Hash
$compilerVersion = & $compilerPath --version
$compilerOutput = & $compilerPath -o $proofObject $proofSource 2>&1
$compilerExit = $LASTEXITCODE
$sourceHashAfter = (Get-FileHash -LiteralPath $proofSource -Algorithm SHA256).Hash
if ($sourceHashBefore -ne $sourceHashAfter) {
    throw 'Proof source changed during compilation; result is not attributed to a single source.'
}
$objectHash = if ($compilerExit -eq 0) {
    (Get-FileHash -LiteralPath $proofObject -Algorithm SHA256).Hash
} else { 'No successful object hash claimed' }
$lines = @(
    $compilerVersion,
    "Compiler: $compilerPath",
    "Source: $proofSource",
    "Output: $proofObject",
    "Source SHA256 (unchanged during compile): $sourceHashAfter",
    "Object SHA256: $objectHash",
    'Command: lean.exe -o CriticalMargin.olean CriticalMargin.lean',
    'Imports: pre-existing cached mathlib/package .olean files, read-only',
    "Exit code: $compilerExit",
    'Output below combines compiler stdout and stderr in order.',
    '',
    $compilerOutput
)
[IO.File]::WriteAllLines($proofLog, [string[]]$lines, [Text.UTF8Encoding]::new($false))
$lines
exit $compilerExit
