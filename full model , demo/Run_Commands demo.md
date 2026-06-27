# Guardian Eye Demo Run Commands

Use these commands in **PowerShell**.

---

## 0) One-time PowerShell fix for this session

Run this first if PowerShell blocks `npm`, `venv`, or scripts:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

---

## 1) Start the Backend

Open a new PowerShell window:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
cd "D:\sara\graduation\Demo Implementation\Backend_Ashour\BackEnd"
.\.venv\Scripts\Activate.ps1
$env:GUARDIAN_MOCK="0"
$env:GUARDIAN_V9_CKPT="D:\sara\graduation\Demo Implementation\models\v9\RLVS_best.pt"
$env:GUARDIAN_VIDEOMAE_CKPT="D:\sara\graduation\Demo Implementation\models\videomae\RLVS_videomae_best.pt"
$env:GUARDIAN_NARRATION_ENABLED="1"
$env:GUARDIAN_NARRATION_VLM_ENABLED="1"
$env:GUARDIAN_NARRATION_MODE="vlm_llm"
$env:GUARDIAN_VLM_MODEL_ID="D:\sara\graduation\Demo Implementation\models\Qwen2.5-VL-3B-Instruct"
$env:GUARDIAN_NARRATION_VLM_4BIT="1"
$env:GUARDIAN_NARRATION_VLM_GPU_MAX_MEMORY="5GiB"
$env:GUARDIAN_NARRATION_VLM_CPU_MAX_MEMORY="10GiB"
$env:GUARDIAN_LLM_MODEL_ID="D:\sara\graduation\Demo Implementation\models\qwen2.5-1.5b-instruct"
$env:GUARDIAN_LEGAL_LLM_ENABLED="1"
$env:GUARDIAN_LEGAL_LLM_MODEL_ID="D:\sara\graduation\Demo Implementation\models\qwen2.5-1.5b-instruct"
$env:GUARDIAN_TRANSLATION_QWEN_MODEL_PATH="D:\sara\graduation\Demo Implementation\models\qwen2.5-1.5b-instruct"
$env:GUARDIAN_MODEL_LOCAL_ONLY="1"
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
$env:GUARDIAN_TRANSLATION_DEVICE="cuda"
$env:GUARDIAN_TRANSLATION_DTYPE="bfloat16"
$env:GUARDIAN_TRANSLATION_4BIT="0"
$env:GUARDIAN_TRANSLATION_MAX_NEW_TOKENS="256"
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

The configuration above is the recommended safe CUDA mode for the RTX 5060 8GB on Windows.
It avoids the native bitsandbytes 4-bit path.

### Qwen translation CPU mode

```powershell
$env:GUARDIAN_TRANSLATION_QWEN_MODEL_PATH="D:\sara\graduation\Demo Implementation\models\qwen2.5-1.5b-instruct"
$env:GUARDIAN_MODEL_LOCAL_ONLY="1"
$env:GUARDIAN_TRANSLATION_DEVICE="cpu"
$env:GUARDIAN_TRANSLATION_DTYPE="float32"
$env:GUARDIAN_TRANSLATION_4BIT="0"
$env:GUARDIAN_TRANSLATION_MAX_NEW_TOKENS="256"
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Arabic narration uses the local Qwen checkpoint above. TranslateGemma is disabled and
is not loaded at runtime. No Hugging Face downloads are allowed at runtime.

Backend URL:

```text
http://127.0.0.1:8000
```

Useful backend checks:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

If the backend uses `/docs`, open:

```text
http://127.0.0.1:8000/docs
```

---

## 2) Start the Frontend

Open another new PowerShell window:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
cd "D:\sara\graduation\Demo Implementation\FrontEnd"
npm run dev
```

If `npm run dev` is blocked, use:

```powershell
npx.cmd vite --host 127.0.0.1 --port 5173
```

Frontend URL:

```text
http://127.0.0.1:5173
```

---

## 3) Full Demo Startup Order

1. Start backend first.
2. Confirm backend health works.
3. Start frontend.
4. Open the frontend URL.
5. Upload/select video and run prediction.

---

## 4) Stop the Demo

In each running terminal:

```powershell
Ctrl + C
```

Then confirm with:

```powershell
Y
```

---

## 5) Common Fixes

### Path with spaces issue

Always use quotes:

```powershell
cd "D:\sara\graduation\Demo Implementation\FrontEnd"
cd "D:\sara\graduation\Demo Implementation\Backend_Ashour\BackEnd"
```

### npm script blocked

```powershell
Set-ExecutionPolicy -Scope Process Bypass
npm.cmd run dev
```

### Virtual environment activation blocked

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
```

### Make sure real backend is used, not mock mode

```powershell
$env:GUARDIAN_MOCK="0"
```

### Use mock mode only if backend/model is unavailable

```powershell
$env:GUARDIAN_MOCK="1"
```

---

## 6) Optional: Install dependencies if needed

Frontend:

```powershell
cd "D:\sara\graduation\Demo Implementation\FrontEnd"
npm install
```

Backend:

```powershell
cd "D:\sara\graduation\Demo Implementation\Backend_Ashour\BackEnd"
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
