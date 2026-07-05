# SPDX-License-Identifier: Apache-2.0
#
# Penguin-VL-ncnn — production test driver (Windows / PowerShell).
#
# Runs the deliverable test suite in layered phases and prints a pass/fail
# summary with a non-zero exit code on any failure, so it can gate CI or a
# release. Phases requiring the weighted model are skipped unless you pass
# -Model (and -ReferenceModel for the PyTorch equality check).
#
# Examples:
#   # Toolchain-level suite only (no multi-GB model): build + unit + self-tests
#   pwsh tools/run_full_test.ps1 -NcnnDir "$PWD/ncnn-install/lib/cmake/ncnn"
#
#   # Full production run incl. real image->scene inference + PyTorch equality
#   pwsh tools/run_full_test.ps1 `
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

$ErrorActionPreference = "Stop"
$script:Results = New-Object System.Collections.Generic.List[object]

function Record([string]$Name, [bool]$Ok, [string]$Detail = "") {
    $script:Results.Add([pscustomobject]@{ Phase = $Name; Pass = $Ok; Detail = $Detail })
    $tag = if ($Ok) { "PASS" } else { "FAIL" }
    Write-Host ("[{0}] {1}  {2}" -f $tag, $Name, $Detail)
}

function Have([string]$cmd) {
    return [bool](Get-Command $cmd -ErrorAction SilentlyContinue)
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
}

Write-Host "==================================================================="
Write-Host " Penguin-VL-ncnn production test driver"
Write-Host "==================================================================="

# ---- Phase 1: build -------------------------------------------------------
if (-not $SkipBuild) {
    try {
        cmake -S . -B $BuildDir -DCMAKE_BUILD_TYPE=Release -Dncnn_DIR="$NcnnDir"
        if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }
        cmake --build $BuildDir --config Release -j $Jobs
        if ($LASTEXITCODE -ne 0) { throw "cmake build failed" }
        Record "1. Build (CMake + MSVC, Release)" $true
    } catch {
        Record "1. Build (CMake + MSVC, Release)" $false $_.Exception.Message
        Summary; exit 1
    }
} else {
    Record "1. Build" $true "skipped (-SkipBuild)"
}

# ---- Phase 2: unit tests --------------------------------------------------
try {
    ctest --test-dir $BuildDir --output-on-failure -C Release
    Record "2. Unit tests (ctest)" ($LASTEXITCODE -eq 0)
} catch {
    Record "2. Unit tests (ctest)" $false $_.Exception.Message
}

# ---- Phase 3: weight-free contract self-tests -----------------------------
if (-not $SkipSelftest) {
    if (Have "python") {
        $stDir = "pvl_selftest"
        try {
            python tools/selftest_export.py --out $stDir --layers 2 --n 100
            $exportOk = ($LASTEXITCODE -eq 0)
            Record "3a. pnnx export (real dims, random weights)" $exportOk

            if ($exportOk) {
                cmake --build $BuildDir --target vision_selftest --config Release
                $vs = & ".\$BuildDir\Release\vision_selftest.exe" $stDir 2>&1 | Out-String
                Write-Host $vs
                Record "3b. Vision pipeline self-test" ($vs -match "SELFTEST PASS")

                python tools/rename_decoder_blobs.py --param "$stDir/decoder.ncnn.param" --layers 2
                Record "3c. Decoder KV-cache blob rename" ($LASTEXITCODE -eq 0)
            }
        } catch {
            Record "3. Contract self-tests" $false $_.Exception.Message
        }

        # tokenizer bit-exactness (needs transformers + net for ~10MB tokenizer)
        try {
            $tokDir = "pvl_tok"
            python tools/extract_tokenizer.py --model tencent/Penguin-VL-2B --out $tokDir
            if ($LASTEXITCODE -eq 0) {
                cmake --build $BuildDir --target tok_selftest --config Release
                $ts = & ".\$BuildDir\Release\tok_selftest.exe" $tokDir 2>&1 | Out-String
                Write-Host $ts
                Record "3d. Tokenizer == HuggingFace (bit-exact)" ($ts -match "TOKENIZER SELFTEST PASS")
            } else {
                Record "3d. Tokenizer == HuggingFace" $false "extract_tokenizer failed (net/transformers?)"
            }
        } catch {
            Record "3d. Tokenizer == HuggingFace" $false $_.Exception.Message
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
        $imgs = @(Get-ChildItem -Path $AssetsDir -Include *.jpg,*.jpeg,*.png,*.bmp -File -Recurse -ErrorAction SilentlyContinue)
        if ($imgs.Count -eq 0) {
            Record "4. Real inference" $false "no images under $AssetsDir"
        } else {
            foreach ($img in $imgs) {
                try {
                    $out = & $exe --model $Model --image $img.FullName --prompt $Prompt --threads $Threads 2>&1 | Out-String
                    Write-Host "----- $($img.Name) -----"
                    Write-Host $out
                    Record "4. Inference: $($img.Name)" ($out.Trim().Length -gt 0) "non-empty output"
                } catch {
                    Record "4. Inference: $($img.Name)" $false $_.Exception.Message
                }
            }
        }
    }
} else {
    Record "4. Real image->scene inference" $true "skipped (no -Model given)"
}

# ---- Phase 5: PyTorch text equality (golden diff) -------------------------
if ($Model -ne "" -and $ReferenceModel -ne "") {
    $imgs = @(Get-ChildItem -Path $AssetsDir -Include *.jpg,*.jpeg,*.png,*.bmp -File -Recurse -ErrorAction SilentlyContinue)
    if ($imgs.Count -gt 0 -and (Have "python")) {
        $img = $imgs[0]
        try {
            python tools/compare_reference.py --model $ReferenceModel --image $img.FullName --prompt $Prompt --out golden
            $golden = Get-Content -Raw "golden/output.txt" -ErrorAction SilentlyContinue
            $ncnn = & ".\$BuildDir\Release\penguinvl.exe" --model $Model --image $img.FullName --prompt $Prompt --threads $Threads 2>&1 | Out-String
            $match = ($null -ne $golden) -and ($ncnn.Trim() -eq $golden.Trim())
            Record "5. ncnn text == PyTorch (greedy, $($img.Name))" $match
        } catch {
            Record "5. ncnn text == PyTorch" $false $_.Exception.Message
        }
    } else {
        Record "5. ncnn text == PyTorch" $false "need images + python"
    }
} else {
    Record "5. ncnn text == PyTorch (golden)" $true "skipped (no -ReferenceModel)"
}

Summary
