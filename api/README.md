# Resonant Model API

Production REST API for the world's first resonant neural model repository. This API enables inference and serving for models built on phase dynamics (AFRB + PVM + PLM) as an alternative to traditional attention mechanisms.

## Overview

This API provides:
- **Text Generation**: Autoregressive generation with phase-aware sampling
- **Content Retrieval**: Needle-in-haystack pattern matching via phase coherence
- **Model Introspection**: Access to phase metrics and memory diagnostics
- **Production Ready**: Async FastAPI, CORS, error handling, structured logging

## Architecture

```
Resonant Model Stack:
┌─────────────────────────────────────┐
│  FastAPI Server (api/server.py)    │
├─────────────────────────────────────┤
│  Inference Engine (api/inference.py)│
├─────────────────────────────────────┤
│  Base Model + AFRB Adapters         │
│  ├─ AFRB Layer 1 (ω₁, γ₁, φ₁)      │
│  ├─ AFRB Layer 2 (ω₂, γ₂, φ₂)      │
│  └─ ... (N layers)                  │
├─────────────────────────────────────┤
│  Phase Memory (PVM + PLM)           │
│  ├─ Phase-Vector Memory (O(d))     │
│  └─ Phase-Lattice Memory (grid)    │
└─────────────────────────────────────┘
```

**Key Components:**
- **AFRB**: Adaptive Frequency-Resonant Blocks replace attention
- **PVM**: Phase-Vector Memory for temporal coherence
- **PLM**: Phase-Lattice Memory for spatial binding

## Installation

### 1. Install Dependencies

```bash
# Option A: Install with API extras (recommended)
pip install -e ".[api]"

# Option B: Install manually
pip install fastapi uvicorn[standard] pydantic
```

### 2. Set Environment Variables

```bash
# Required: Path to trained model checkpoint
export MODEL_CHECKPOINT=checkpoints/model_best.pt

# Optional: Configuration
export MODEL_NAME=EleutherAI/pythia-160m  # Base model
export DEVICE=cuda                         # cuda, cpu, or mps
export API_HOST=0.0.0.0                   # Server host
export API_PORT=8000                      # Server port
export CORS_ORIGINS=*                     # CORS allowed origins
```

### 3. Start Server

```bash
# Development mode (auto-reload)
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload

# Production mode (single worker)
uvicorn api.server:app --host 0.0.0.0 --port 8000

# Production with Gunicorn (multiple workers)
gunicorn api.server:app \
  -w 4 \
  -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120
```

## API Endpoints

### 1. Health Check

**GET** `/v1/health`

Check server health and model readiness.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "uptime_seconds": 3600.5,
  "version": "1.0.0",
  "timestamp": "2025-11-14T12:34:56.789Z"
}
```

**Example:**
```bash
curl http://localhost:8000/v1/health
```

### 2. Model Information

**GET** `/v1/model/info`

Get model metadata and configuration.

**Response:**
```json
{
  "model_name": "EleutherAI/pythia-160m",
  "num_afrb_layers": 8,
  "hidden_size": 768,
  "vocab_size": 50304,
  "max_context_length": 2048,
  "phase_features": {
    "adaptive_omega": true,
    "learnable_gamma": true,
    "omega_base": 6.0,
    "phi_base": 0.0
  },
  "memory_features": {
    "pvm_enabled": true,
    "plm_enabled": false
  },
  "device": "cuda"
}
```

**Example:**
```bash
curl http://localhost:8000/v1/model/info
```

### 3. Text Generation

**POST** `/v1/generate`

Generate text continuation with phase-aware sampling.

**Request Body:**
```json
{
  "prompt": "Once upon a time in a neural network",
  "max_tokens": 100,
  "temperature": 0.8,
  "top_k": 50,
  "top_p": 0.95,
  "repetition_penalty": 1.1,
  "stop_sequences": ["\n\n", "END"],
  "seed": 42,
  "return_phase_metrics": true,
  "return_memory_stats": true
}
```

**Parameters:**
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt` | string | **required** | Input text to continue |
| `max_tokens` | int | 100 | Maximum tokens to generate (1-2048) |
| `temperature` | float | 0.7 | Sampling temperature (0.0-2.0) |
| `top_k` | int | 50 | Top-K sampling (0 = disabled) |
| `top_p` | float | 0.95 | Nucleus sampling threshold (0.0-1.0) |
| `repetition_penalty` | float | 1.0 | Penalty for token repetition (1.0-2.0) |
| `stop_sequences` | array | null | Stop at these strings |
| `seed` | int | null | Random seed for reproducibility |
| `return_phase_metrics` | bool | false | Include phase coherence diagnostics |
| `return_memory_stats` | bool | false | Include PVM/PLM memory diagnostics |

**Response:**
```json
{
  "text": ", there lived a transformer who dreamed of resonance...",
  "prompt": "Once upon a time in a neural network",
  "tokens_generated": 87,
  "finish_reason": "max_tokens",
  "timing_ms": 342.5,
  "phase_metrics": {
    "phase_coherence": 0.76,
    "gamma_mean": 0.23,
    "omega_values": [6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7],
    "phase_offsets": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
  },
  "memory_stats": {
    "pvm_mem_norm": 0.45,
    "pvm_gate_strength": 0.12,
    "plm_coherence": null,
    "total_memory_kb": 24.5
  },
  "model_info": {
    "model_name": "EleutherAI/pythia-160m",
    "num_afrb_layers": 8,
    "hidden_size": 768
  }
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The theory of resonant neural networks",
    "max_tokens": 50,
    "temperature": 0.7,
    "return_phase_metrics": true
  }'
```

### 4. Content Retrieval

**POST** `/v1/retrieve`

Needle-in-haystack retrieval via phase-based content addressing.

**Request Body:**
```json
{
  "context": "...long text... The secret code is X7Z9 ...more text...",
  "needle": "secret code is X7Z9",
  "query": null,
  "retrieval_mode": "pvm",
  "top_k": 5,
  "return_positions": true,
  "return_similarity_scores": true
}
```

**Parameters:**
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `context` | string | **required** | Long text containing the needle |
| `needle` | string | **required** | Pattern to retrieve |
| `query` | string | null | Alternative query (defaults to needle) |
| `retrieval_mode` | string | "pvm" | Retrieval method: "pvm", "plm", or "hybrid" |
| `top_k` | int | 5 | Number of top matches to return |
| `return_positions` | bool | true | Include token position indices |
| `return_similarity_scores` | bool | true | Include cosine similarity scores |

**Response:**
```json
{
  "matches": [
    {
      "text": "secret code is X7Z9",
      "position": 1247,
      "similarity": 0.94,
      "confidence": 0.94
    },
    {
      "text": "code for security",
      "position": 523,
      "similarity": 0.67,
      "confidence": 0.67
    }
  ],
  "needle": "secret code is X7Z9",
  "retrieval_mode": "pvm",
  "timing_ms": 45.2
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/v1/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "context": "Random text... The key insight is resonance... More text...",
    "needle": "key insight",
    "retrieval_mode": "pvm",
    "top_k": 3
  }'
```

## Python Client Example

```python
import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"

# 1. Check health
response = requests.get(f"{BASE_URL}/v1/health")
print("Health:", response.json())

# 2. Get model info
response = requests.get(f"{BASE_URL}/v1/model/info")
print("Model:", response.json()["model_name"])

# 3. Generate text
response = requests.post(
    f"{BASE_URL}/v1/generate",
    json={
        "prompt": "Resonant neural networks use phase dynamics to",
        "max_tokens": 100,
        "temperature": 0.8,
        "return_phase_metrics": True
    }
)
result = response.json()
print("Generated:", result["text"])
print("Coherence:", result["phase_metrics"]["phase_coherence"])

# 4. Retrieve pattern
response = requests.post(
    f"{BASE_URL}/v1/retrieve",
    json={
        "context": "Long context... The answer is 42 ...more text...",
        "needle": "answer is 42",
        "retrieval_mode": "pvm",
        "top_k": 3
    }
)
result = response.json()
print("Top match:", result["matches"][0]["text"])
```

For a complete client implementation, see `examples/api_client.py`.

## Error Handling

All errors return a structured JSON response:

```json
{
  "error": "validation_error",
  "message": "max_tokens must be between 1 and 2048",
  "details": {
    "field": "max_tokens",
    "value": 5000
  },
  "timestamp": "2025-11-14T12:34:56.789Z"
}
```

**Error Types:**
- `validation_error`: Invalid request parameters (HTTP 400)
- `model_error`: Model inference failure (HTTP 500)
- `resource_error`: Out of memory or compute (HTTP 503)
- `internal_error`: Unexpected server error (HTTP 500)

## Performance Tuning

### Memory Management

```bash
# Reduce memory with smaller batch size
export MAX_BATCH_SIZE=1

# Use mixed precision (bfloat16 on CUDA)
# Automatically enabled for CUDA devices in inference.py

# Clear GPU cache between requests (handled automatically)
```

### Concurrency

```bash
# Single worker (low concurrency, low memory)
uvicorn api.server:app --host 0.0.0.0 --port 8000

# Multiple workers (high concurrency, high memory)
gunicorn api.server:app \
  -w 4 \
  -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

**Note:** Each worker loads a full model copy. For 4 workers with a 160M model:
- Memory: ~4 x 600MB = 2.4GB VRAM
- Throughput: ~4x single-worker

### Rate Limiting

Implement rate limiting via reverse proxy (nginx, Caddy):

```nginx
# nginx configuration
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;

server {
    location /v1/ {
        limit_req zone=api_limit burst=20 nodelay;
        proxy_pass http://localhost:8000;
    }
}
```

## Monitoring

### Health Checks

```bash
# Kubernetes readiness probe
livenessProbe:
  httpGet:
    path: /v1/health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10
```

### Metrics

The API logs all requests with timing:
```
2025-11-14 12:34:56 - INFO - POST /v1/generate - Status: 200 - Duration: 342.50ms
```

For production monitoring, integrate with:
- **Prometheus**: Add `prometheus-fastapi-instrumentator`
- **Datadog**: Use `ddtrace` for APM
- **CloudWatch**: Send logs via AWS Lambda

## Security Best Practices

### 1. Authentication

The API does not include built-in authentication. Add via middleware:

```python
from fastapi import Security, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

@app.middleware("http")
async def verify_token(request: Request, call_next):
    if request.url.path.startswith("/v1/"):
        auth = request.headers.get("Authorization")
        if not auth or not verify_api_key(auth):
            return JSONResponse(
                status_code=401,
                content={"error": "unauthorized", "message": "Invalid API key"}
            )
    return await call_next(request)
```

### 2. CORS Configuration

```bash
# Restrict origins in production
export CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
```

### 3. Input Validation

All inputs are validated via Pydantic schemas:
- Prompt length: 1-8192 characters
- max_tokens: 1-2048
- temperature: 0.0-2.0

### 4. Rate Limiting

Use reverse proxy (nginx, Caddy) for rate limiting:
- Per-IP limits: 10 req/s
- Burst allowance: 20 requests
- Global limit: 1000 req/s

## Deployment

### Docker

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install fastapi uvicorn[standard] pydantic

# Copy code
COPY . .

# Set environment
ENV MODEL_CHECKPOINT=/models/model_best.pt
ENV DEVICE=cuda
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

# Expose port
EXPOSE 8000

# Run server
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t resonant-api .
docker run -p 8000:8000 -v /path/to/models:/models --gpus all resonant-api
```

### Kubernetes

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: resonant-api
spec:
  replicas: 2
  selector:
    matchLabels:
      app: resonant-api
  template:
    metadata:
      labels:
        app: resonant-api
    spec:
      containers:
      - name: api
        image: resonant-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_CHECKPOINT
          value: /models/model_best.pt
        - name: DEVICE
          value: cuda
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: 4Gi
        livenessProbe:
          httpGet:
            path: /v1/health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        volumeMounts:
        - name: models
          mountPath: /models
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: resonant-api
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8000
  selector:
    app: resonant-api
```

Deploy:
```bash
kubectl apply -f deployment.yaml
```

## Troubleshooting

### Model Not Loading

```
Error: Model initialization failed
```

**Solution:**
1. Check `MODEL_CHECKPOINT` path exists
2. Verify checkpoint format matches training output
3. Ensure CUDA is available if `DEVICE=cuda`

```bash
# Test checkpoint loading
python -c "import torch; print(torch.load('checkpoints/model.pt').keys())"
```

### Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solution:**
1. Reduce `max_tokens` in requests
2. Use single worker: `uvicorn` instead of `gunicorn`
3. Enable CPU offloading (not yet implemented)

### Slow Inference

```
Generation taking >5 seconds per request
```

**Solution:**
1. Enable GPU: `export DEVICE=cuda`
2. Use bfloat16 (automatic on CUDA)
3. Reduce `max_tokens` or `top_k`

## Advanced Usage

### Custom Model Loading

```python
from api.inference import ResonantModelInference

# Load with custom config
model = ResonantModelInference(
    checkpoint_path="checkpoints/custom_model.pt",
    model_name="EleutherAI/pythia-410m",  # Larger base model
    device="cuda",
    dtype=torch.bfloat16,
    max_batch_size=4
)

# Generate with custom parameters
output = model.generate(
    prompt="Custom prompt",
    max_tokens=200,
    temperature=0.9,
    return_phase_metrics=True
)
```

### Phase Metrics Analysis

```python
# Collect phase dynamics during generation
output = model.generate(
    prompt="Test prompt",
    return_phase_metrics=True
)

metrics = output.phase_metrics
print(f"Phase coherence: {metrics['phase_coherence']:.3f}")
print(f"Resonance depth: {metrics['gamma_mean']:.3f}")
print(f"Frequencies: {metrics['omega_values']}")
```

## Contributing

See `CONTRIBUTING.md` for development guidelines.

## License

MIT License - see `LICENSE` file.

## Citation

If you use this API in research, please cite:

```bibtex
@software{attention_free_phase_blocks_api_2025,
  title = {Attention-Free Phase Blocks API: Production Inference for Phase-Based Neural Networks},
  author = {Damjan Žakelj},
  year = {2025},
  url = {https://github.com/Freeky7819/attention-free-phase-blocks}
}
```

## Support

- **Documentation**: `/docs` (Swagger UI) or `/redoc` (ReDoc)
- **Issues**: GitHub Issues
- **Email**: support@yourproject.com

---

**Built with phase dynamics. Powered by resonance.**
