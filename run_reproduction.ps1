# ==============================================================================
# PURE Reproduction Pipeline Automation Script (SciERC Benchmark)
# Paper: A Frustratingly Easy Approach for Entity and Relation Extraction (NAACL 2021)
# ==============================================================================

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host " [PURE] SciERC End-to-End Pipeline Reproduction Script" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan

# 1. Detect Python Executable
$PythonCmd = "python"
if (Test-Path ".\venv\Scripts\python.exe") {
    $PythonCmd = ".\venv\Scripts\python.exe"
    Write-Host "[Env] Using Python virtual environment: $PythonCmd" -ForegroundColor Green
} else {
    Write-Host "[Env] Virtual environment not found, using system default python" -ForegroundColor Yellow
}

# 2. Check Models Directory
if (-not (Test-Path ".\scierc_models\ent-scib-ctx0") -or -not (Test-Path ".\scierc_models\rel-scib-ctx0")) {
    Write-Host "[Pre-check] SciERC model weights not found in scierc_models/." -ForegroundColor Yellow
    Write-Host "[Pre-check] Downloading models via download_models.py..." -ForegroundColor Yellow
    & $PythonCmd download_models.py
} else {
    Write-Host "[Pre-check] Pretrained model checkpoints verified in scierc_models/." -ForegroundColor Green
}

# 3. Stage 1: Entity Prediction
Write-Host "`n------------------------------------------------------------------" -ForegroundColor Magenta
Write-Host "▶ Step 1/3: Running Entity Span Prediction (run_entity.py)..." -ForegroundColor Magenta
Write-Host "------------------------------------------------------------------" -ForegroundColor Magenta

& $PythonCmd run_entity.py `
    --do_eval `
    --eval_test `
    --context_window 0 `
    --task scierc `
    --data_dir scierc_data/processed_data/json/ `
    --model allenai/scibert_scivocab_uncased `
    --output_dir scierc_models/ent-scib-ctx0/

if ($LASTEXITCODE -ne 0) {
    Write-Host "[Error] Entity prediction failed with exit code $LASTEXITCODE." -ForegroundColor Red
    exit $LASTEXITCODE
}
Write-Host "[Success] Entity predictions saved to scierc_models/ent-scib-ctx0/ent_pred_test.json" -ForegroundColor Green

# 4. Stage 2: Relation Extraction
Write-Host "`n------------------------------------------------------------------" -ForegroundColor Magenta
Write-Host "▶ Step 2/3: Running Relation Extraction with Typed Markers (run_relation.py)..." -ForegroundColor Magenta
Write-Host "------------------------------------------------------------------" -ForegroundColor Magenta

& $PythonCmd run_relation.py `
    --task scierc `
    --do_eval `
    --eval_test `
    --model allenai/scibert_scivocab_uncased `
    --do_lower_case `
    --context_window 0 `
    --max_seq_length 128 `
    --entity_output_dir scierc_models/ent-scib-ctx0/ `
    --output_dir scierc_models/rel-scib-ctx0/

if ($LASTEXITCODE -ne 0) {
    Write-Host "[Error] Relation prediction failed with exit code $LASTEXITCODE." -ForegroundColor Red
    exit $LASTEXITCODE
}
Write-Host "[Success] Relation predictions saved to scierc_models/rel-scib-ctx0/predictions.json" -ForegroundColor Green

# 5. Stage 3: End-to-End Metric Evaluation
Write-Host "`n------------------------------------------------------------------" -ForegroundColor Magenta
Write-Host "▶ Step 3/3: Running End-to-End Metric Evaluation (run_eval.py)..." -ForegroundColor Magenta
Write-Host "------------------------------------------------------------------" -ForegroundColor Magenta

& $PythonCmd run_eval.py --prediction_file scierc_models/rel-scib-ctx0/predictions.json

if ($LASTEXITCODE -ne 0) {
    Write-Host "[Error] Evaluation script failed with exit code $LASTEXITCODE." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "`n==================================================================" -ForegroundColor Cyan
Write-Host " [PURE] SciERC Reproduction Completed Successfully!" -ForegroundColor Cyan
Write-Host " Expected Benchmark Metrics (Paper Table 1):" -ForegroundColor Cyan
Write-Host "   - NER F1        : ~66.6%  (Boundary + Type)" -ForegroundColor Green
Write-Host "   - REL F1        : ~48.2%  (Boundary Matching)" -ForegroundColor Green
Write-Host "   - REL Strict F1 : ~35.6%  (Strict Matching)" -ForegroundColor Green
Write-Host "==================================================================" -ForegroundColor Cyan
