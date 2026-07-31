import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.models.database import init_db, SessionLocal, TrafficFlow
from app.api.routes import router as api_router
from app.api.websocket import ws_manager, websocket_endpoint
from app.services.classifier import classifier_service
from app.utils.preprocessing import SyntheticDataGenerator

# Background flow simulation task reference
sim_task = None

async def live_traffic_simulator():
    """Simulates real-time network traffic flow captures and broadcasts via WebSocket."""
    print("[+] Live Traffic Simulator active. Streaming real-time flow classifications...")
    while True:
        await asyncio.sleep(4.0)
        if ws_manager.active_connections:
            synth = SyntheticDataGenerator.generate_flow()
            res = classifier_service.classify_flow(synth)
            
            payload = {
                "type": "NEW_FLOW",
                "flow": {
                    "flow_id": res["flow_id"],
                    "src_ip": synth["src_ip"],
                    "src_port": synth["src_port"],
                    "dst_ip": synth["dst_ip"],
                    "dst_port": synth["dst_port"],
                    "protocol": synth["protocol"],
                    "duration": synth["duration"],
                    "total_packets": synth["total_packets"],
                    "total_bytes": synth["total_bytes"],
                    "is_vpn": synth["is_vpn"],
                    "predicted_category": res["predicted_category"],
                    "confidence": res["confidence"],
                    "uncertainty_score": res["uncertainty_score"],
                    "is_unknown": res["is_unknown"],
                    "timestamp": res["explanation"].get("target_node_index", 0)
                }
            }
            await ws_manager.broadcast(payload)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global sim_task
    print("[*] Initializing Database & Loading Machine Learning Models...")
    init_db()
    
    # Auto-seed DB if empty
    db = SessionLocal()
    try:
        if db.query(TrafficFlow).count() == 0:
            from app.api.routes import _seed_db_with_synthetic_data
            _seed_db_with_synthetic_data(db, count=50)
    finally:
        db.close()

    # Start live traffic simulator task
    sim_task = asyncio.create_task(live_traffic_simulator())
    
    yield
    
    # Graceful Shutdown
    print("[*] Shutting down application services...")
    if sim_task:
        sim_task.cancel()

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Fine-Grained Encrypted Traffic Classification using PyTorch Geometric Graph Neural Networks (GraphSAGE & GAT)",
    lifespan=lifespan
)

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount REST API Routes
app.include_router(api_router, prefix=settings.API_PREFIX)

# Mount WebSocket Route
@app.websocket("/api/ws")
async def ws_route(websocket: WebSocket):
    await websocket_endpoint(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
