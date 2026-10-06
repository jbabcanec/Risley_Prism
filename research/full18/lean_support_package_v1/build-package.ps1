$ErrorActionPreference = 'Stop'
$researchRoot = Split-Path -Parent $PSScriptRoot
$formalRoot = Join-Path $researchRoot 'formalization'
$environmentRoot = 'C:\Users\josep\lean\risley'
$payloadRoot = Join-Path $PSScriptRoot 'payload'
$zipPath = Join-Path $researchRoot 'risley_lean_support_sources_v1.zip'
$textPath = Join-Path $researchRoot 'risley_lean_support_sources.txt'
$utf8 = [System.Text.UTF8Encoding]::new($false,$true)
foreach ($target in @($payloadRoot,$zipPath,$textPath)) {
    if (Test-Path -LiteralPath $target) { throw "Refusing to overwrite existing packaging target: $target" }
}
function Hash-Bytes([byte[]]$bytes) {
    $sha = [Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($sha.ComputeHash($bytes))).Replace('-','').ToLowerInvariant() }
    finally { $sha.Dispose() }
}
function Hash-File([string]$path) { return (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() }
$before = @(Get-ChildItem -LiteralPath $formalRoot -File | Sort-Object Name | ForEach-Object {
    [ordered]@{name=$_.Name; bytes=$_.Length; sha256=(Hash-File $_.FullName)}
})
$sourceSpec = @(
    @{name='RisleySupport.lean'; sha256='402d3f1555dabe0639f54d8af43a770f30ed0a134ac6308846f4444e74d573e8'; namespace='RisleySupport'; count=11; verification='verification.json'; log='compiler-output.txt'},
    @{name='CriticalMargin.lean'; sha256='d711b8b89cc89c9afc86c6426c0136a03c2295d598774f2b58c5133ebc49fdb7'; namespace='RisleyCriticalMargin'; count=8; verification='critical-margin-verification.json'; log='critical-margin-compiler-output.txt'}
)
$theorems = @()
foreach ($spec in $sourceSpec) {
    $sourcePath = Join-Path $formalRoot $spec.name
    if ((Hash-File $sourcePath) -cne $spec.sha256) { throw "Unexpected source hash: $($spec.name)" }
    $sourceText = $utf8.GetString([IO.File]::ReadAllBytes($sourcePath))
    $names = @([regex]::Matches($sourceText,'(?m)^theorem\s+([A-Za-z_][A-Za-z0-9_]*)') | ForEach-Object { $_.Groups[1].Value })
    if ($names.Count -ne $spec.count) { throw "Unexpected declaration count: $($spec.name)" }
    $verification = Get-Content -LiteralPath (Join-Path $formalRoot $spec.verification) -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($verification.status -ne 'passed' -or $verification.compiler_exit_code -ne 0 -or $verification.theorem_count -ne $spec.count) { throw 'Recorded verification metadata mismatch.' }
    foreach ($file in $verification.files) {
        if ($file.file -notlike '*.olean') {
            if ((Hash-File (Join-Path $formalRoot $file.file)) -cne $file.sha256) { throw "Recorded evidence hash mismatch: $($file.file)" }
        }
    }
    $logText = $utf8.GetString([IO.File]::ReadAllBytes((Join-Path $formalRoot $spec.log)))
    foreach ($name in $names) {
        $qualified = $spec.namespace + '.' + $name
        $expectedLine = "'$qualified' depends on axioms: [propext, Classical.choice, Quot.sound]"
        if (-not $logText.Contains($expectedLine)) { throw "Missing exact recorded axiom inventory: $qualified" }
        $theorems += [ordered]@{name=$qualified; source=('formalization/'+$spec.name); existing_compiler_log=('formalization/'+$spec.log); recorded_axioms=@('propext','Classical.choice','Quot.sound')}
    }
}
$formalFiles = @('RisleySupport.lean','CriticalMargin.lean','README.md','critical-margin-README.md','verification.json','critical-margin-verification.json','independent_audit.md','critical_margin_independent_audit.md','compiler-output.txt','critical-margin-compiler-output.txt','check.ps1','check-critical-margin.ps1')
$metadataFiles = @('lakefile.toml','lean-toolchain','lake-manifest.json')
$lock = Get-Content -LiteralPath (Join-Path $environmentRoot 'lake-manifest.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$mathlib = @($lock.packages | Where-Object {$_.name -eq 'mathlib'})
if ($mathlib.Count -ne 1 -or $mathlib[0].rev -ne 'db584cd6d46c92f209a44c0f1c829460d327499d') { throw 'Dependency lock differs from recorded verification.' }
$toolchain = (Get-Content -LiteralPath (Join-Path $environmentRoot 'lean-toolchain') -Raw -Encoding UTF8).Trim()
if ($toolchain -ne 'leanprover/lean4:v4.33.0') { throw 'Toolchain differs from recorded verification.' }
[IO.Directory]::CreateDirectory((Join-Path $payloadRoot 'formalization')) | Out-Null
[IO.Directory]::CreateDirectory((Join-Path $payloadRoot 'dependency-environment')) | Out-Null
$inventory = @()
foreach ($name in $formalFiles) {
    $source = Join-Path $formalRoot $name
    $relative = 'formalization/' + $name
    $dest = Join-Path $payloadRoot $relative
    [IO.File]::Copy($source,$dest,$false)
    if ((Hash-File $dest) -cne (Hash-File $source)) { throw "Copy mismatch: $relative" }
    $inventory += [ordered]@{path=$relative; bytes=(Get-Item -LiteralPath $dest).Length; sha256=(Hash-File $dest); provenance='Unmodified existing formalization source/evidence'}
}
foreach ($name in $metadataFiles) {
    $source = Join-Path $environmentRoot $name
    $relative = 'dependency-environment/' + $name
    $dest = Join-Path $payloadRoot $relative
    [IO.File]::Copy($source,$dest,$false)
    if ((Hash-File $dest) -cne (Hash-File $source)) { throw "Copy mismatch: $relative" }
    $inventory += [ordered]@{path=$relative; bytes=(Get-Item -LiteralPath $dest).Length; sha256=(Hash-File $dest); provenance='Unmodified metadata snapshot of the existing cached dependency environment'}
}
$readme = @'
Risley Lean supporting sources and existing verification evidence, package v1

This package preserves two previously compiled Lean sources and their successful
verification evidence. Packaging performed file reads, byte copies, inventory
checks and SHA256 comparisons only. No Lean compiler, check script, numerical
test, proof search, dependency installation or update was run to make it.

Recorded proof scope: 19 theorems, comprising 11 in RisleySupport.lean and 8 in
CriticalMargin.lean. The saved successful compiler logs report, for every theorem,
exactly [propext, Classical.choice, Quot.sound]. Existing verification records and
independent audits report no admitted goals, sorryAx dependencies or custom axioms.
These are earlier recorded results, not a new compilation or formal audit.

RisleySupport contains source-transport determinant/factorization/inverse lemmas,
a spatial recurrence identity, critical/external-flight algebraic sign lemmas,
and a square-root difference bound. The sign lemmas check the stated algebraic
derivative expressions; they do not formalize differentiation of the optical map.
CriticalMargin proves the conditional final-prism observation-margin implication
from explicit real-coordinate intersection, traversal, slope and observation
hypotheses. It does not derive all those hypotheses from the full optical model.

No full three-prism compiler, complete inverse, boundary compactification,
corrector/contraction theorem, native accuracy certificate, or concrete 200-sample
record is formalized by this package. The later chord and interval-domain work
adds no Lean coverage. Standard logical axioms are distinct from optical premises.

Layout:
- formalization/: exact source bytes, original READMEs, successful compiler logs,
  verification JSON, independent audit notes, and unexecuted original check scripts.
- dependency-environment/: exact lakefile.toml, lean-toolchain and lake-manifest.json
  snapshots from the existing environment used by the original direct-compiler
  scripts. The lock records mathlib and its transitive dependency revisions.
- MANIFEST.json: file sizes, SHA256 hashes, theorem names and recorded axiom scope.
  Its file inventory excludes MANIFEST.json itself to avoid a circular self-hash.

Frozen toolchain and dependency identity:
- Lean 4.33.0, Windows x86_64, release commit
  d8b18978322de05a8f3dba51ef03cf5461676c17.
- mathlib v4.33.0, commit db584cd6d46c92f209a44c0f1c829460d327499d.

Reproduction boundaries:
The dependency snapshot is metadata, not a bundled Lean installation or mathlib
checkout/build cache. Its default Lake target is the original environment's
Risley library, whose unrelated source is deliberately absent. This archive is
not a standalone default-Lake-build project. Restore the recorded compiler and
locked dependencies in an appropriate separate environment before a future check.
The original check.ps1 and check-critical-margin.ps1 show the exact historical
direct compiler calls and LEAN_PATH construction; their absolute local paths
must be reviewed/adapted for another machine. Neither script was run here.
The original READMEs also retain their historical workspace-relative paths.

Excluded deliberately: .olean objects, compiler/dependency binaries, build caches,
unrelated optical/project sources, and superseded failed-draft logs. Original
READMEs and verification JSON mention some excluded objects or failed drafts;
those historical references are preserved, and do not assert archive membership.
Saved logs contain the historical local compiler/workspace paths as provenance.

The companion risley_lean_support_sources.txt concatenates every package text
file with labeled byte-count/hash boundaries. Bytes between each BEGIN and END
boundary are the complete corresponding file bytes, with a separate delimiter
newline before END. The ZIP is the authoritative individually recoverable copy.

This package preserves the originals and records existing evidence. It makes no
new proof, physical validation, global uniqueness or practical recovery claim.
'@
$readmePath = Join-Path $payloadRoot 'README.txt'
[IO.File]::WriteAllText($readmePath,$readme + "`n",$utf8)
$inventory += [ordered]@{path='README.txt'; bytes=(Get-Item -LiteralPath $readmePath).Length; sha256=(Hash-File $readmePath); provenance='New packaging scope/readme; not new proof evidence'}
$manifest = [ordered]@{
    package='risley_lean_support_sources_v1'
    packaging_scope='Existing source/evidence preservation only; no compiler, tests or new formalization run'
    toolchain=$toolchain
    lean_commit='d8b18978322de05a8f3dba51ef03cf5461676c17'
    mathlib_commit=$mathlib[0].rev
    existing_theorem_count=19
    source_theorem_counts=[ordered]@{RisleySupport=11;RisleyCriticalMargin=8}
    existing_recorded_axioms=@('propext','Classical.choice','Quot.sound')
    theorem_inventory=$theorems
    file_inventory_excludes_manifest_itself=$true
    files=@($inventory | Sort-Object { $_.path })
    exclusions=@('Compiled .olean files','Compiler/dependency binaries and caches','Superseded failed-draft logs','Unrelated original project sources')
}
$manifestPath = Join-Path $payloadRoot 'MANIFEST.json'
[IO.File]::WriteAllText($manifestPath,($manifest | ConvertTo-Json -Depth 10) + "`n",$utf8)
$allFiles = @($inventory | Sort-Object { $_.path }) + @([ordered]@{path='MANIFEST.json'; bytes=(Get-Item -LiteralPath $manifestPath).Length; sha256=(Hash-File $manifestPath)})

# The only generated executable here is this packaging script; it never invokes
# the compiler, original check scripts, optical code or any mathematics test.
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zipStream = [IO.File]::Open($zipPath,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
$archive = [IO.Compression.ZipArchive]::new($zipStream,[IO.Compression.ZipArchiveMode]::Create,$false)
try {
    foreach ($row in $allFiles) {
        $bytes = [IO.File]::ReadAllBytes((Join-Path $payloadRoot $row.path))
        [void]$utf8.GetString($bytes)
        $entry = $archive.CreateEntry($row.path,[IO.Compression.CompressionLevel]::Optimal)
        $entry.LastWriteTime = [DateTimeOffset]::new(2026,10,3,0,0,0,[TimeSpan]::Zero)
        $entryStream = $entry.Open()
        try { $entryStream.Write($bytes,0,$bytes.Length) } finally { $entryStream.Dispose() }
    }
} finally { $archive.Dispose(); $zipStream.Dispose() }
$textStream = [IO.File]::Open($textPath,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
$textSections = @()
try {
    $intro = $utf8.GetBytes("RISLEY LEAN SUPPORT SOURCES v1`nComplete labeled package text. Existing evidence only; no compiler or tests rerun.`n`n")
    $textStream.Write($intro,0,$intro.Length)
    $ordered = @($allFiles | Sort-Object @{Expression={if($_.path -eq 'README.txt'){0}elseif($_.path -eq 'formalization/RisleySupport.lean'){1}elseif($_.path -eq 'formalization/CriticalMargin.lean'){2}else{3}}},@{Expression={$_.path}})
    foreach ($row in $ordered) {
        $header = $utf8.GetBytes("===== BEGIN FILE: $($row.path) | bytes=$($row.bytes) | sha256=$($row.sha256) =====`n")
        $textStream.Write($header,0,$header.Length)
        $start = $textStream.Position
        $bytes = [IO.File]::ReadAllBytes((Join-Path $payloadRoot $row.path))
        $textStream.Write($bytes,0,$bytes.Length)
        $textSections += [ordered]@{path=$row.path;offset=$start;bytes=$bytes.Length;sha256=$row.sha256}
        $footer = $utf8.GetBytes("`n===== END FILE: $($row.path) =====`n`n")
        $textStream.Write($footer,0,$footer.Length)
    }
} finally { $textStream.Dispose() }
$readArchive = [IO.Compression.ZipFile]::OpenRead($zipPath)
try {
    if ($readArchive.Entries.Count -ne $allFiles.Count) { throw 'ZIP entry count mismatch.' }
    foreach ($row in $allFiles) {
        $entry = $readArchive.GetEntry($row.path)
        if ($null -eq $entry -or $entry.Length -ne $row.bytes) { throw "ZIP inventory mismatch: $($row.path)" }
        $stream = $entry.Open()
        $memory = [IO.MemoryStream]::new()
        try { $stream.CopyTo($memory); $bytes=$memory.ToArray() } finally {$stream.Dispose();$memory.Dispose()}
        if ((Hash-Bytes $bytes) -cne $row.sha256) { throw "ZIP payload mismatch: $($row.path)" }
    }
} finally {$readArchive.Dispose()}
$textBytes = [IO.File]::ReadAllBytes($textPath)
[void]$utf8.GetString($textBytes)
foreach ($section in $textSections) {
    $segment = [byte[]]::new($section.bytes)
    [Array]::Copy($textBytes,[long]$section.offset,$segment,[long]0,[long]$section.bytes)
    if ((Hash-Bytes $segment) -cne $section.sha256) { throw "Combined text segment mismatch: $($section.path)" }
}
foreach ($row in $before) {
    $original = Join-Path $formalRoot $row.name
    if ((Hash-File $original) -cne $row.sha256 -or (Get-Item -LiteralPath $original).Length -ne $row.bytes) { throw "Original changed: $($row.name)" }
}
$result = [ordered]@{
    status='packaged and byte-verified; no compilation or mathematical execution'
    payload_file_count=$allFiles.Count
    manifest_inventory_count=$inventory.Count
    theorem_count=19
    verified_original_formalization_files=$before.Count
    originals_unchanged=$true
    zip=[ordered]@{path=$zipPath;bytes=(Get-Item -LiteralPath $zipPath).Length;sha256=(Hash-File $zipPath);verified_entries=$allFiles.Count}
    combined_text=[ordered]@{path=$textPath;bytes=$textBytes.Length;sha256=(Hash-Bytes $textBytes);verified_full_file_segments=$textSections.Count}
    source_hashes=@($sourceSpec | ForEach-Object { [ordered]@{file=$_.name;sha256=$_.sha256;theorems=$_.count} })
    inventory=$allFiles
    combined_text_sections=$textSections
}
[IO.File]::WriteAllText((Join-Path $PSScriptRoot 'packaging-verification.json'),($result | ConvertTo-Json -Depth 10) + "`n",$utf8)
[pscustomobject]$result | Select-Object status,payload_file_count,manifest_inventory_count,theorem_count,verified_original_formalization_files,originals_unchanged,zip,combined_text | ConvertTo-Json -Depth 6
