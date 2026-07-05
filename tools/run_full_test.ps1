# SPDX-License-Identifier: Apache-2.0
#
# Penguin-VL-ncnn — production test driver (Windows / PowerShell).
#
# Runs the deliverable test suite in layered phases and prints a pass/fail
# summary with a non-zero exit code on any failure, so it can gate CI or a
# release. Phases requiring the weighted model are skipped unless you pass
# -Model (and -ReferenceModel for the PyTorch equality check).
#
# Works with Windows PowerShell 5.1 (powershell.exe) and PowerShell 7 (pwsh).
#
# Examples:
#   # Toolchain-level suite only (no multi-GB model): build + unit + self-tests
#   powershell -File tools/run_full_test.ps1 -NcnnDir "$PWD/ncnn-install/lib/cmake/ncnn"
#
#   # Full production run incl. real image->scene inference + PyTorch equality
#   powershell -File tools/run_full_test.ps1 `
#       -NcnnDir "$PWD/ncnn-install/lib/cmake/ncnn" `
#       -Model .\penguin-vl-2b-ncnn `
#       -ReferenceModel .\Penguin-VL-2B `
#       -Prompt "请描述这张图片"

[CmdletBinding()]
param(
    [string]$NcnnDir = "$PWD/ncnn-install/lib/cmake/ncnn",
    [string]$BuildDir = "build",
    [string]$Model = "",
    [string]$ReferenceModel = "",
    [string]$Prompt = "Describe this image in detail.",
    [string]$AssetsDir = "assets",
    [int]$Threads = 8,
    [int]$Jobs = 4,
    [switch]$SkipBuild,
    [switch]$SkipSelftest
)

# Native tools (cmake, pnnx, hf, python) legitimately write progress/warnings to
# stderr; do NOT let that abort the driver. We gate every phase on explicit exit
# codes and output matching instead.
$ErrorActionPreference = "Continue"
$script:Results = New-Object System.Collections.Generic.List[object]

function Record([string]$Name, [bool]$Ok, [string]$Detail = "") {
    $script:Results.Add([pscustomobject]@{ Phase = $Name; Pass = $Ok; Detail = $Detail })
    $tag = if ($Ok) { "PASS" } else { "FAIL" }
    Write-Host ("[{0}] {1}  {2}" -f $tag, $Name, $Detail)
}

function Have([string]$cmd) {
    return [bool](Get-Command $cmd -ErrorAction SilentlyContinue)
}

# Run a native command, capturing stdout+stderr into a string via a temp file and
# returning the exit code through [ref]. Using a file (not `2>&1 | Out-String`)
# avoids a Windows PowerShell deadlock when a child (e.g. pnnx) floods stderr.
function Invoke-Native([string]$File, [string[]]$CmdArgs, [ref]$Out) {
    $tmp = [System.IO.Path]::GetTempFileName()
    # `*>` sends every stream to a file; unlike `2>&1 | Out-String` this cannot
    # deadlock when a child (pnnx) floods stderr. Splatting @CmdArgs preserves
    # argument boundaries. ($Args would collide with the automatic variable.)
    & $File @CmdArgs *> $tmp
    $rc = $LASTEXITCODE
    $Out.Value = (Get-Content -Raw $tmp -ErrorAction SilentlyContinue)
    Remove-Item $tmp -ErrorAction SilentlyContinue
    return $rc
}

function Summary {
    Write-Host ""
    Write-Host "==================== SUMMARY ===================="
    $script:Results | Format-Table -AutoSize Phase, Pass, Detail | Out-String | Write-Host
    $failed = @($script:Results | Where-Object { -not $_.Pass -and $_.Detail -notmatch "skipped" })
    if ($failed.Count -gt 0) {
        Write-Host ("RESULT: FAIL ({0} failing)" -f $failed.Count) -ForegroundColor Red
        exit 1
    }
    Write-Host "RESULT: PASS" -ForegroundColor Green
    exit 0
}

Write-Host "==================================================================="
Write-Host " Penguin-VL-ncnn production test driver"
Write-Host "==================================================================="

# ---- Phase 1: build -------------------------------------------------------
if (-not $SkipBuild) {
    cmake -S . -B $BuildDir -DCMAKE_BUILD_TYPE=Release -Dncnn_DIR="$NcnnDir" | Out-Host
    $cfgOk = ($LASTEXITCODE -eq 0)
    if ($cfgOk) {
        cmake --build $BuildDir --config Release -j $Jobs | Out-Host
        $cfgOk = ($LASTEXITCODE -eq 0)
    }
    Record "1. Build (CMake + MSVC, Release)" $cfgOk
    if (-not $cfgOk) { Summary }
} else {
    Record "1. Build" $true "skipped (-SkipBuild)"
}

# ---- Phase 2: unit tests --------------------------------------------------
ctest --test-dir $BuildDir --output-on-failure -C Release | Out-Host
Record "2. Unit tests (ctest)" ($LASTEXITCODE -eq 0)

# ---- Phase 3: weight-free contract self-tests -----------------------------
if (-not $SkipSelftest) {
    if (Have "python") {
        $stDir = "pvl_selftest"

        $o = ""
        $rc = Invoke-Native "python" @("tools/selftest_export.py", "--out", $stDir, "--layers", "2", "--n", "100") ([ref]$o)
        $exportOk = ($rc -eq 0) -and (Test-Path "$stDir/vision_encoder.ncnn.param")
        Record "3a. pnnx export (real dims, random weights)" $exportOk

        if ($exportOk) {
            cmake --build $BuildDir --target vision_selftest --config Release | Out-Host
            $vs = ""
            [void](Invoke-Native ".\$BuildDir\Release\vision_selftest.exe" @($stDir) ([ref]$vs))
            Write-Host $vs
            Record "3b. Vision pipeline self-test" ($vs -match "SELFTEST PASS")

            $o2 = ""
            $rc2 = Invoke-Native "python" @("tools/rename_decoder_blobs.py", "--param", "$stDir/decoder.ncnn.param", "--layers", "2") ([ref]$o2)
            $names = (Select-String -Path "$stDir/decoder.ncnn.param" -Pattern "out_cache_k0|out_cache_v0" -ErrorAction SilentlyContinue)
            Record "3c. Decoder KV-cache blob rename" (($rc2 -eq 0) -and ($null -ne $names))
        }

        # tokenizer bit-exactness (needs transformers + net for ~10MB tokenizer)
        $tokDir = "pvl_tok"
        $ot = ""
        $rct = Invoke-Native "python" @("tools/extract_tokenizer.py", "--model", "tencent/Penguin-VL-2B", "--out", $tokDir) ([ref]$ot)
        if (($rct -eq 0) -and (Test-Path "$tokDir/vocab.txt")) {
            cmake --build $BuildDir --target tok_selftest --config Release | Out-Host
            $ts = ""
            [void](Invoke-Native ".\$BuildDir\Release\tok_selftest.exe" @($tokDir) ([ref]$ts))
            Write-Host $ts
            Record "3d. Tokenizer == HuggingFace (bit-exact)" ($ts -match "TOKENIZER SELFTEST PASS")
        } else {
            Record "3d. Tokenizer == HuggingFace" $false "extract_tokenizer failed (network/transformers?)"
        }
    } else {
        Record "3. Contract self-tests" $false "python not found on PATH"
    }
} else {
    Record "3. Contract self-tests" $true "skipped (-SkipSelftest)"
}

# ---- Phase 4: real image -> scene inference -------------------------------
if ($Model -ne "") {
    $exe = ".\$BuildDir\Release\penguinvl.exe"
    if (-not (Test-Path $exe)) {
        Record "4. Real inference" $false "$exe not found (build first)"
    } elseif (-not (Test-Path $Model)) {
        Record "4. Real inference" $false "model dir not found: $Model"
    } else {
        $imgs = @(Get-ChildItem -Path $AssetsDir -Include *.jpg, *.jpeg, *.png, *.bmp -File -Recurse -ErrorAction SilentlyContinue)
        if ($imgs.Count -eq 0) {
            Record "4. Real inference" $false "no images under $AssetsDir"
        } else {
            foreach ($img in $imgs) {
                $out = ""
                [void](Invoke-Native $exe @("--model", $Model, "--image", $img.FullName, "--prompt", $Prompt, "--threads", "$Threads") ([ref]$out))
                Write-Host "----- $($img.Name) -----"
                Write-Host $out
                Record "4. Inference: $($img.Name)" ($out.Trim().Length -gt 0) "non-empty output"
            }
        }
    }
} else {
    Record "4. Real image->scene inference" $true "skipped (no -Model given)"
}

# ---- Phase 5: PyTorch text equality (golden diff) -------------------------
if ($Model -ne "" -and $ReferenceModel -ne "") {
    $imgs = @(Get-ChildItem -Path $AssetsDir -Include *.jpg, *.jpeg, *.png, *.bmp -File -Recurse -ErrorAction SilentlyContinue)
    if ($imgs.Count -gt 0 -and (Have "python")) {
        $img = $imgs[0]
        $og = ""
        [void](Invoke-Native "python" @("tools/compare_reference.py", "--model", $ReferenceModel, "--image", $img.FullName, "--prompt", $Prompt, "--out", "golden") ([ref]$og))
        $golden = Get-Content -Raw "golden/output.txt" -ErrorAction SilentlyContinue
        $ncnn = ""
        [void](Invoke-Native ".\$BuildDir\Release\penguinvl.exe" @("--model", $Model, "--image", $img.FullName, "--prompt", $Prompt, "--threads", "$Threads") ([ref]$ncnn))
        $match = ($null -ne $golden) -and ($ncnn.Trim() -eq $golden.Trim())
        Record "5. ncnn text == PyTorch (greedy, $($img.Name))" $match
    } else {
        Record "5. ncnn text == PyTorch" $false "need images + python"
    }
} else {
    Record "5. ncnn text == PyTorch (golden)" $true "skipped (no -ReferenceModel)"
}

Summary
