# Native Windows build with MSVC (Visual Studio 2022 or later, C++ workload).
#
#   powershell -ExecutionPolicy Bypass -File engine\build.ps1            build everything
#   powershell -ExecutionPolicy Bypass -File engine\build.ps1 -Test      build, then run the fast tests
#   powershell -ExecutionPolicy Bypass -File engine\build.ps1 -Only poker2_train
#
# Outputs go to engine\build-win\. Linux/WSL builds use the Makefile instead.
param(
  [switch]$Test,
  [string[]]$Only = @()
)
$ErrorActionPreference = 'Stop'
$engine = $PSScriptRoot
$out = Join-Path $engine 'build-win'
New-Item -ItemType Directory -Force -Path (Join-Path $out 'obj') | Out-Null

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw 'Visual Studio with the C++ x64 tools was not found' }
$vcvars = Join-Path $vs 'VC\Auxiliary\Build\vcvars64.bat'

# /arch:AVX2 runs on any recent x86-64; the evaluator and k-means vectorize with it.
$flags = '/nologo /std:c++20 /O2 /Oi /EHsc /W3 /permissive- /utf-8 /arch:AVX2 /DNOMINMAX /D_CRT_SECURE_NO_WARNINGS /I src'
$targets = [ordered]@{
  'poker2_tests'       = (Get-ChildItem (Join-Path $engine 'tests\*.cpp') | ForEach-Object { "tests\$($_.Name)" }) -join ' '
  'poker2_solve'       = 'tools\solve.cpp'
  'poker2_tree'        = 'tools\tree_size.cpp'
  'poker2_bench'       = 'tools\bench.cpp'
  'poker2_abstraction' = 'tools\abstraction.cpp'
  'poker2_train'       = 'tools\train.cpp'
}
foreach ($name in $targets.Keys) {
  if ($Only.Count -and $Only -notcontains $name) { continue }
  $objdir = Join-Path $out "obj\$name\"
  New-Item -ItemType Directory -Force -Path $objdir | Out-Null
  if ("$objdir$out" -match ' ') { throw 'build paths must not contain spaces' }
  # Unquoted on purpose: /Fo"dir\" would let the trailing backslash escape the quote.
  $cmd = "`"$vcvars`" >nul && cd /d `"$engine`" && cl $flags $($targets[$name]) /Fo$objdir /Fe$out\$name.exe"
  Write-Output "== $name"
  cmd /c $cmd
  if ($LASTEXITCODE -ne 0) { throw "build failed: $name" }
}
if ($Test) {
  & (Join-Path $out 'poker2_tests.exe')
  if ($LASTEXITCODE -ne 0) { throw 'tests failed' }
}
