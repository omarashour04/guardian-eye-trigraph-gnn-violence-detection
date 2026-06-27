# Guardian Eye Demo Run Guide

This frontend is integrated with the FastAPI backend using mock backend outputs. The full demo flow is wired end to end: upload, prediction display, overlay playback, narrative, incident history review, and Ask.

## Backend Setup

Backend folder:

```powershell
D:\sara\graduation\Demo Implementation\BackEnd
```

Create and activate the virtual environment:

```powershell
cd "D:\sara\graduation\Demo Implementation\BackEnd"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use a process-scoped bypass:

```powershell
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
.\.venv\Scripts\Activate.ps1
```

Install backend dependencies:

```powershell
pip install -r requirements.txt
```

## Run Backend In Mock Mode

```powershell
cd "D:\sara\graduation\Demo Implementation\BackEnd"
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
.\.venv\Scripts\Activate.ps1
$env:GUARDIAN_MOCK = "1"
uvicorn main:app --reload
```

Backend URLs:

```text
API:  http://127.0.0.1:8000
Docs: http://127.0.0.1:8000/docs
Health: http://127.0.0.1:8000/health
```

## Frontend Setup

Frontend folder:

```powershell
D:\sara\graduation\Demo Implementation\FrontEnd
```

Install frontend dependencies:

```powershell
cd "D:\sara\graduation\Demo Implementation\FrontEnd"
npm install
```

Use backend mode in `.env`:

```text
VITE_API_MODE=backend
VITE_API_BASE_URL=http://127.0.0.1:8000
```

## Run Frontend In Backend Mode

```powershell
cd "D:\sara\graduation\Demo Implementation\FrontEnd"
npm run dev
```

Open the Vite URL shown in the terminal, usually:

```text
http://127.0.0.1:5173
```

## Demo Flow

1. Start the backend in mock mode.
2. Start the frontend in backend mode.
3. Upload a sample video, for example:

```text
D:\sara\graduation\Demo Implementation\BackEnd\cam1_clip_14.mp4
```

4. Confirm the verdict and confidence update.
5. Confirm the gate bars and peak window update.
6. Play the overlay video in the Overlay Video panel.
7. Read the generated incident narrative.
8. Review latest incidents in Incident History.
9. Click an older history card to reload that incident's verdict, confidence, narrative, and overlay/original video.
10. Ask questions in the Ask Guardian Eye panel.
11. Switch between English and Arabic from the language selector and verify labels plus backend narrative/Ask language behavior.

## What Is Still Mock

- Model prediction output is mocked by the backend when `GUARDIAN_MOCK=1`.
- The RAG/Ask response is still mock-style backend output.
- The frontend is integrated with backend endpoints, but the AI intelligence is not production model inference yet.

## What Jana/Team Should Replace Later

- Replace `model_service.py` with the real trained model inference pipeline.
- Replace or extend `explanation_service.py` with the final explanation logic.
- Replace the mock RAG/Ask service with the real retrieval and answer-generation service.
- Keep the existing API contracts stable if possible:
  - `POST /predict`
  - `POST /overlay`
  - `POST /explain`
  - `GET /history`
  - `GET /incident/{incident_id}`
  - `POST /ask`
  - `GET /health`

## Integration Status

Frontend/backend integration is complete using mock backend outputs. The frontend is ready for Jana/team to swap in real model and RAG services behind the existing FastAPI contract.
