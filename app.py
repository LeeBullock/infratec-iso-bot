from fastapi import FastAPI

app = FastAPI(title="INFRATEC ISO Coach API")

@app.get("/health")
def health():
    return {"ok": True}
