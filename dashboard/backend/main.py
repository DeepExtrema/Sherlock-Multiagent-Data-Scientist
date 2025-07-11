from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from confluent_kafka import Consumer
import asyncio
import json
import threading
from typing import Dict, List, Optional
import logging
from bson import ObjectId

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Deepline Observability Dashboard")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB connection
db = AsyncIOMotorClient("mongodb://localhost:27017").deepline

# Store active WebSocket connections
active_connections: List[WebSocket] = []

def serialize_doc(doc):
    """Convert MongoDB document to JSON serializable format"""
    if doc is None:
        return None
    if isinstance(doc, list):
        return [serialize_doc(item) for item in doc]
    if isinstance(doc, dict):
        result = {}
        for key, value in doc.items():
            if isinstance(value, ObjectId):
                result[key] = str(value)
            elif isinstance(value, (dict, list)):
                result[key] = serialize_doc(value)
            else:
                result[key] = value
        return result
    return doc

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "running", "message": "Deepline Observability Dashboard"}

@app.get("/runs/{run_id}")
async def get_run(run_id: str):
    """Get a specific run with its tasks"""
    try:
        run = await db.runs.find_one({"run_id": run_id})
        tasks = await db.tasks.find({"run_id": run_id}).to_list(100)
        return {
            "run": serialize_doc(run), 
            "tasks": serialize_doc(tasks)
        }
    except Exception as e:
        logger.error(f"Error fetching run {run_id}: {e}")
        return {"error": str(e)}

@app.get("/runs")
async def get_runs(limit: int = 10):
    """Get recent runs"""
    try:
        runs = await db.runs.find().sort([("_id", -1)]).limit(limit).to_list(limit)
        return {"runs": serialize_doc(runs)}
    except Exception as e:
        logger.error(f"Error fetching runs: {e}")
        return {"error": str(e)}

@app.get("/tasks/{task_id}")
async def get_task(task_id: str):
    """Get a specific task"""
    try:
        task = await db.tasks.find_one({"task_id": task_id})
        return {"task": serialize_doc(task)}
    except Exception as e:
        logger.error(f"Error fetching task {task_id}: {e}")
        return {"error": str(e)}

@app.websocket("/ws/events")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for live event streaming"""
    await websocket.accept()
    active_connections.append(websocket)
    logger.info(f"Client connected. Total connections: {len(active_connections)}")
    
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except Exception as e:
        logger.info(f"WebSocket disconnected: {e}")
    finally:
        active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total connections: {len(active_connections)}")

async def broadcast_event(event: dict):
    """Broadcast event to all connected WebSocket clients"""
    if active_connections:
        disconnected = []
        for connection in active_connections:
            try:
                await connection.send_text(json.dumps(event))
            except Exception as e:
                logger.error(f"Error sending to WebSocket: {e}")
                disconnected.append(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            active_connections.remove(connection)

def kafka_consumer():
    """Kafka consumer running in a separate thread"""
    try:
        consumer = Consumer({
            'bootstrap.servers': 'localhost:9092',
            'group.id': 'dashboard',
            'auto.offset.reset': 'latest'
        })
        consumer.subscribe(['task.events'])
        logger.info("Kafka consumer started")
        
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                logger.error(f"Kafka error: {msg.error()}")
                continue
            
            try:
                event = json.loads(msg.value().decode('utf-8'))
                # Schedule the broadcast in the main event loop
                asyncio.run_coroutine_threadsafe(broadcast_event(event), asyncio.get_event_loop())
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing Kafka message: {e}")
            except Exception as e:
                logger.error(f"Error processing Kafka message: {e}")
                
    except Exception as e:
        logger.error(f"Kafka consumer error: {e}")

@app.on_event("startup")
async def startup_event():
    """Start background tasks"""
    logger.info("Starting Deepline Observability Dashboard")
    
    # Start Kafka consumer in a separate thread
    kafka_thread = threading.Thread(target=kafka_consumer, daemon=True)
    kafka_thread.start()
    logger.info("Kafka consumer thread started")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True) 