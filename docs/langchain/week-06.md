# Week 6 — Production & Deployment

**Course:** LangChain for AI Applications  
**Week Focus:** Recognize the API, container, timeout, cache, and monitoring pieces that surround a LangChain application. This is an architectural introduction, not a production deployment recipe.

---

## If you already write software

LangChain is an orchestration library, not a model. The model is the remote API (or the local weights). LangChain is the **middleware**: prompts as templates, outputs as parsers, tools as functions, memory as a store, chains as your call graph.

```
Your backend                        LangChain
─────────────────────────────       ──────────────────────────────
HTTP handler                        a chain / agent entrypoint
string template + params            PromptTemplate
JSON schema / zod / pydantic        output parser
service client                      a Tool (function + docstring)
session store / Redis               memory
try / catch + retries               callbacks, fallbacks
```

Read every abstraction as something you have already shipped. If you cannot say what the chain does as a sequence of function calls, the abstraction is hiding a bug.

## 🎯 Learning Objectives

By the end of this week, you will:
- Build scalable LangChain REST APIs with FastAPI
- Deploy applications with Docker
- Scale to cloud platforms (AWS, GCP, Azure)
- Handle rate limiting and request queues
- Monitor and maintain production systems
- Implement caching and optimization

## 📊 Real-World Context

**The Challenge:**
- Your support bot works perfectly in development
- Now you need to serve 1000 concurrent users
- Handle peak loads (Black Friday = 10x traffic)
- Keep costs reasonable ($$ per request)
- Maintain 99.9% uptime

**Production Concerns:**
1. **Performance:** Respond in < 2 seconds at 1000 RPS
2. **Cost:** Optimize token usage ($$ adds up fast)
3. **Reliability:** Handle failures gracefully
4. **Scalability:** Auto-scale with traffic
5. **Monitoring:** Know what's happening in production
6. **Security:** Protect API, data, credentials

**Solutions:**
- Async chains and FastAPI for performance
- Caching to reduce API calls
- Request queuing for load smoothing
- Circuit breakers for resilience
- Containerization with Docker
- Cloud deployment with auto-scaling
- Comprehensive monitoring and logging

**Business Impact:**
- 📈 Scale: Handle growth without rewrite
- 💰 Cost: 50% reduction via caching
- ⚡ Speed: < 500ms response time
- 🔒 Reliability: 99.9% uptime
- 👀 Visibility: Real-time monitoring
- 🚀 Faster deployments: CI/CD pipelines


## 🚀 Part 1: Building REST APIs with FastAPI

<div class="api-box">
<strong>FastAPI:</strong> A Python web framework that supplies validation, routing, and API documentation. Production readiness depends on the service around it.
</div>

### Why FastAPI?

| Feature | FastAPI | Flask | Django |
|---------|---------|-------|--------|
| Speed | ⚡⚡⚡ Fastest | ⚡ Good | ⚡ Good |
| Async | Native | Limited | Limited |
| Validation | Auto | Manual | Manual |
| Docs | Auto | Manual | Manual |
| Learning | Easy | Easy | Steep |

### FastAPI Example

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    context: str = None

class ChatResponse(BaseModel):
    response: str
    latency_ms: float

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process chat message with LangChain."""
    start = time.time()
    
    # Your LangChain logic here
    response = await llm.agenerate(request.message)
    
    latency = (time.time() - start) * 1000
    return ChatResponse(response=response, latency_ms=latency)

# Run with: uvicorn main:app --reload
```

```python
# Demonstrate API structure

from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import time
import asyncio

@dataclass
class APIRequest:
    """Incoming API request."""
    request_id: str
    endpoint: str
    message: str
    timestamp: datetime
    user_id: str

@dataclass
class APIResponse:
    """Outgoing API response."""
    request_id: str
    response: str
    latency_ms: float
    model: str
    tokens_used: int
    cached: bool

class SimpleCache:
    """Simple LRU cache for responses."""
    
    def __init__(self, max_size: int = 100):
        self.cache: Dict[str, str] = {}
        self.max_size = max_size
    
    def get(self, key: str) -> Optional[str]:
        return self.cache.get(key)
    
    def set(self, key: str, value: str):
        if len(self.cache) >= self.max_size:
            # Simple FIFO eviction
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = value
    
    def stats(self) -> Dict[str, int]:
        return {"cached_items": len(self.cache), "max_size": self.max_size}

# Demo: API structure
print("🔌 FASTAPI STRUCTURE DEMO")
print("="*70)

cache = SimpleCache(max_size=5)

# Simulate requests
print("\n📝 Processing Requests:")
print()

# Request 1: Cache miss
req1 = APIRequest(
    request_id="req-001",
    endpoint="/chat",
    message="How do I reset password?",
    timestamp=datetime.now(),
    user_id="user-123"
)

cached = cache.get(req1.message)
if cached:
    print(f"1. {req1.request_id}: CACHE HIT")
    print(f"   Response: {cached}")
    print(f"   Latency: 1ms (cached)")
    cache.set(req1.message, "Go to Settings > Security > Change Password")
else:
    print(f"1. {req1.request_id}: CACHE MISS")
    print(f"   Message: {req1.message}")
    resp = "Go to Settings > Security > Change Password"
    cache.set(req1.message, resp)
    print(f"   Response: {resp}")
    print(f"   Latency: 850ms (API call)")
    print(f"   Tokens: 45")

# Request 2: Same question = cache hit
print(f"\n2. req-002: CACHE HIT")
print(f"   Message: {req1.message}")
print(f"   Response: {cache.get(req1.message)}")
print(f"   Latency: 2ms (cached)")
print(f"   Tokens: 0 (SAVED!)")

print(f"\n" + "="*70)
print(f"\n💾 Cache Statistics:")
stats = cache.stats()
print(f"  Items cached: {stats['cached_items']}/{stats['max_size']}")
print(f"  ✅ Benefit: Request 2 was 400x faster and cost-free!")
```

## 🐳 Part 2: Docker Containerization

### Dockerfile Example

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app/ .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s \
  CMD curl -f http://localhost:8000/health || exit 1

# Run server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Build & Run

```bash
# Build image
docker build -t langchain-app:v1 .

# Run container
docker run -p 8000:8000 langchain-app:v1

# Push to registry
docker push myregistry.azurecr.io/langchain-app:v1
```

## ☁️ Part 3: Cloud Deployment

<div class="scale-box">
<strong>Cloud Deployment:</strong> Running containers at scale on managed platforms.
</div>

### Deployment Options

| Platform | Setup | Scaling | Cost | Best For |
|----------|-------|---------|------|----------|
| **AWS ECS** | Medium | Auto | Pay-per-use | High scale |
| **Google Cloud Run** | Easy | Auto | Pay-per-request | Unpredictable |
| **Azure Container Instances** | Medium | Manual | Hourly | Predictable |
| **Heroku** | Very Easy | Auto | Fixed | Rapid prototyping |
| **Kubernetes** | Hard | Auto | Flexible | Enterprise |

### Scaling Strategy

```
Load Balancer
    ↓
[Instance 1] [Instance 2] [Instance 3]
    ↓           ↓           ↓
[Cache] [Cache] [Cache]
    ↓           ↓           ↓
[Queue] [Queue] [Queue]
    ↓           ↓           ↓
           [LLM API]
```

```python
# Simulate load balancing and scaling

from collections import deque
import statistics

class LoadBalancer:
    """Distribute requests across multiple instances."""
    
    def __init__(self, num_instances: int):
        self.instances = [f"instance-{i}" for i in range(num_instances)]
        self.request_queue = deque()
        self.current_instance = 0
        self.request_count = {inst: 0 for inst in self.instances}
        self.latencies = {inst: [] for inst in self.instances}
    
    def route_request(self, request_id: str) -> str:
        """Route to least-loaded instance (round-robin)."""
        instance = self.instances[self.current_instance]
        self.current_instance = (self.current_instance + 1) % len(self.instances)
        
        self.request_count[instance] += 1
        return instance
    
    def record_latency(self, instance: str, latency_ms: float):
        """Record response latency for monitoring."""
        self.latencies[instance].append(latency_ms)
    
    def should_scale_up(self) -> bool:
        """Check if we should add more instances."""
        if not self.latencies[self.instances[0]]:
            return False
        
        avg_latency = statistics.mean(self.latencies[self.instances[0]])
        return avg_latency > 1500  # Threshold: 1.5s
    
    def scale_up(self):
        """Add a new instance."""
        new_instance = f"instance-{len(self.instances)}"
        self.instances.append(new_instance)
        self.request_count[new_instance] = 0
        self.latencies[new_instance] = []
        return new_instance
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancing statistics."""
        total_requests = sum(self.request_count.values())
        
        stats = {
            "total_instances": len(self.instances),
            "total_requests": total_requests,
            "by_instance": self.request_count.copy(),
        }
        
        # Calculate latency stats
        all_latencies = []
        for lat_list in self.latencies.values():
            all_latencies.extend(lat_list)
        
        if all_latencies:
            stats["avg_latency_ms"] = round(statistics.mean(all_latencies), 1)
            stats["p95_latency_ms"] = round(
                sorted(all_latencies)[int(len(all_latencies) * 0.95)], 1
            )
        
        return stats

# Demo: Load balancing and scaling
print("⚖️  LOAD BALANCING & SCALING DEMO")
print("="*70)

lb = LoadBalancer(num_instances=2)

# Simulate 20 requests
print("\n📊 Handling Incoming Requests:")
print()

for i in range(10):
    instance = lb.route_request(f"req-{i:03d}")
    latency = 800 + (i * 100)  # Increasing latency
    lb.record_latency(instance, latency)
    print(f"Request {i+1:2d} → {instance} (latency: {latency}ms)")
    
    if lb.should_scale_up():
        new_instance = lb.scale_up()
        print(f"  🔺 SCALING UP: Added {new_instance}")

print(f"\n" + "="*70)
print(f"\n📈 LOAD BALANCER STATISTICS:")
stats = lb.get_stats()
for key, value in stats.items():
    if key == "by_instance":
        print(f"  {key}:")
        for inst, count in value.items():
            print(f"    - {inst}: {count} requests")
    else:
        print(f"  {key:20} {value}")
```
