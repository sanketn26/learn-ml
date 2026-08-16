# Week 4 — Production & Scaling

**Course:** CrewAI for Multi-Agent Systems  
**Week Focus:** Understand what reliability, queues, and observability would require around a crew. This is an operational map, not a deployment recipe.

---

## If you already write software

CrewAI is **jobs + role descriptions + a shared ticket queue**. An “agent” is a worker with a system prompt (its job description), tools (its IAM permissions), and a task (a ticket). A “crew” is the team you put on the sprint.

```
Job description          agent role + backstory + goal
IAM policy               tools the agent is allowed to call
Ticket                   Task(description, expected_output, agent=)
Sprint team              Crew(agents, tasks, process=sequential|hierarchical)
Standup notes            the shared result / memory the next task reads
```

This is useful when the work actually needs different roles (researcher vs writer vs reviewer). It is ceremony when one function would do. Staff a crew the way you staff a project: few roles, clear tickets, an output contract. “Add more agents” is not an architecture.

## 🎯 Learning Objectives

By the end of this week, you will:
- Design production-grade agent systems
- Implement robust error handling and recovery
- Deploy agents as APIs and services
- Monitor agent performance and health
- Scale for high throughput and concurrency
- Handle failures gracefully
- Build a real-world production system

## 📊 Real-World Context

**The Challenge:** Taking agents from notebook to production:
- Agents can fail (LLM timeouts, rate limits, API errors)
- Need to handle 10,000+ requests/day
- Must maintain 99.9% uptime
- Need to debug failures quickly
- Scale as demand grows

**Production Requirements:**
1. **Reliability**: Retry failed tasks, circuit breakers
2. **Scalability**: Queue-based architecture, load balancing
3. **Observability**: Logging, metrics, tracing
4. **Security**: API authentication, rate limiting, input validation
5. **Performance**: Caching, async processing, optimization

**Business Impact:**
- ⏱️ Achieve 99.9% uptime SLA (vs 70% in development)
- 📈 Scale from 100 → 10,000 requests/day
- 💰 Reduce operational costs by 40% (efficient resource use)
- 🚨 MTTR (Mean Time to Resolution): 30 mins → 5 mins
- 📊 Gain visibility into system behavior

**Deployment Targets:**
- Cloud: AWS, GCP, Azure
- Container: Docker + Kubernetes
- Serverless: AWS Lambda, Google Cloud Functions
- On-premises: Private data centers


## 🔍 Part 1: Production Architecture

### Development vs Production

**Development (Week 1-3):**
```
Notebook → Agent → Print output
Simple, fast iteration
```

**Production:**
```
Client API Request
        ↓
   API Gateway (auth, rate limit, validate)
        ↓
   Task Queue (Redis/RabbitMQ)
        ↓
Worker Pool (10+ agents running)
        ↓
  Database (store results)
        ↓
   Monitoring (Prometheus, CloudWatch)
        ↓
Client Gets Result (via webhook or polling)
```

### Key Production Concerns

| Concern | Development | Production |
|---------|-------------|------------|
| **Error Handling** | Print error ❌ | Retry, fallback, escalate ✅ |
| **Concurrency** | 1 request at a time | 100+ concurrent requests |
| **Latency** | 30 seconds ok | 5 seconds max |
| **Availability** | 50% uptime ok | 99.9% uptime required |
| **Monitoring** | None | Comprehensive logging |
| **Cost** | Doesn't matter | Every $$ counts |
| **Security** | Ignore | Critical |
| **Data** | In-memory | Persistent, replicated |

## 📚 Part 2: Error Handling & Reliability

<div class="reliability-box">
<strong>🔄 Reliability Patterns:</strong>
<ul>
<li><strong>Retry:</strong> Try again (with exponential backoff)</li>
<li><strong>Fallback:</strong> Use alternative agent/approach</li>
<li><strong>Circuit Breaker:</strong> Stop calling failing service</li>
<li><strong>Timeout:</strong> Don't wait forever</li>
<li><strong>Bulkhead:</strong> Isolate failures</li>
</ul>
</div>

### Retry Strategy

```python
# Exponential backoff with jitter
attempt 1: wait 1s (fail)
attempt 2: wait 2s (fail)
attempt 3: wait 4s (fail)
attempt 4: wait 8s (success!) ✓

# Add randomness to avoid thundering herd
wait 1s + random(0-1s)
wait 2s + random(0-2s)
wait 4s + random(0-4s)
```

```python
import time
import random
from typing import Callable, Any
from functools import wraps

def retry_with_backoff(max_attempts: int = 3, base_wait: float = 1.0):
    """Decorator for retry logic with exponential backoff."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt < max_attempts:
                        # Exponential backoff with jitter
                        wait_time = base_wait * (2 ** (attempt - 1))
                        jitter = random.uniform(0, wait_time * 0.1)
                        total_wait = wait_time + jitter
                        
                        print(f"  ⚠️ Attempt {attempt} failed: {str(e)[:50]}...")
                        print(f"     Retrying in {total_wait:.1f}s...")
                        time.sleep(total_wait)
                    else:
                        print(f"  ❌ All {max_attempts} attempts failed")
            
            raise last_exception
        
        return wrapper
    return decorator

# Example: Unreliable API call
call_count = 0

@retry_with_backoff(max_attempts=3, base_wait=0.5)
def flaky_api_call():
    global call_count
    call_count += 1
    
    # Fails first 2 times, succeeds on 3rd
    if call_count < 3:
        raise Exception(f"API temporary unavailable (attempt {call_count})")
    
    return {"status": "success", "data": "Important data"}

print("Testing retry with backoff:")
print()

result = flaky_api_call()
print(f"\n✅ Success! Result: {result}")
```

## 🚀 Part 3: Scaling Agents

<div class="scaling-box">
<strong>Scaling Strategies:</strong>
<ul>
<li><strong>Vertical:</strong> Bigger machines (limit: cost and physics)</li>
<li><strong>Horizontal:</strong> More machines (load balance requests)</li>
<li><strong>Async:</strong> Queue + workers (handle bursts)</li>
<li><strong>Caching:</strong> Avoid recomputing results</li>
<li><strong>Batching:</strong> Process multiple requests together</li>
</ul>
</div>

### Queue-Based Architecture

```
Client 1 ─→ ┐
Client 2 ─→ ├→ API Gateway ─→ Task Queue ─→ Worker Pool ─→ Database
Client 3 ─→ └→ (validates)     (Redis)       (10+ procs)    (results)

Without queue: "I have 3 requests, but only 1 agent. Wait in line!"
With queue: "I have 3 requests. Distribute to 3 different agents. Done!"
```

```python
# Simple in-memory task queue example
from datetime import datetime
from enum import Enum
from typing import Optional
import uuid

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class Task:
    def __init__(self, agent_role: str, input_data: dict):
        self.id = str(uuid.uuid4())[:8]
        self.agent_role = agent_role
        self.input_data = input_data
        self.status = TaskStatus.PENDING
        self.result = None
        self.error = None
        self.created_at = datetime.now()
        self.completed_at: Optional[datetime] = None
    
    def duration_seconds(self) -> float:
        end = self.completed_at or datetime.now()
        return (end - self.created_at).total_seconds()

class TaskQueue:
    def __init__(self):
        self.tasks: dict[str, Task] = {}
        self.pending_queue: list[Task] = []
    
    def enqueue(self, task: Task) -> str:
        """Add task to queue."""
        self.tasks[task.id] = task
        self.pending_queue.append(task)
        return task.id
    
    def dequeue(self) -> Optional[Task]:
        """Get next task from queue."""
        if self.pending_queue:
            task = self.pending_queue.pop(0)
            task.status = TaskStatus.RUNNING
            return task
        return None
    
    def mark_completed(self, task_id: str, result: dict):
        """Mark task as completed."""
        task = self.tasks[task_id]
        task.status = TaskStatus.COMPLETED
        task.result = result
        task.completed_at = datetime.now()
    
    def stats(self) -> dict:
        """Get queue statistics."""
        statuses = {"pending": 0, "running": 0, "completed": 0, "failed": 0}
        for task in self.tasks.values():
            statuses[task.status.value] += 1
        
        return {
            "total_tasks": len(self.tasks),
            "pending_in_queue": len(self.pending_queue),
            "status_breakdown": statuses,
            "average_duration_s": sum(
                t.duration_seconds() for t in self.tasks.values()
            ) / max(len(self.tasks), 1)
        }

# Simulate queue with multiple workers
queue = TaskQueue()

# Enqueue multiple tasks
print("📊 Simulating Production Task Queue")
print("="*70)
print()

# 10 incoming requests
print("➕ Enqueuing 10 tasks...")
for i in range(10):
    task = Task(
        agent_role="ContentWriter",
        input_data={"topic": f"topic_{i}", "priority": "high" if i % 2 == 0 else "normal"}
    )
    queue.enqueue(task)

print(f"   Queue stats: {queue.stats()}")
print()

# Simulate 3 workers processing tasks
print("👷 Processing with 3 workers...")
workers_completed = 0

for worker_id in range(1, 4):  # 3 workers
    print(f"\nWorker {worker_id}:")
    while True:
        task = queue.dequeue()
        if not task:
            print("  ✓ Queue empty, waiting for more work...")
            break
        
        # Simulate work
        print(f"  Processing task {task.id}...")
        time.sleep(0.1)  # Simulate work
        
        # Mark completed
        queue.mark_completed(task.id, {"status": "written"})
        workers_completed += 1
        print(f"  ✓ Completed (duration: {task.duration_seconds():.2f}s)")

print()
print("📈 Final Queue Stats:")
stats = queue.stats()
print(f"   Total tasks: {stats['total_tasks']}")
print(f"   Status breakdown: {stats['status_breakdown']}")
print(f"   Avg duration: {stats['average_duration_s']:.2f}s")
```

## 📊 Part 4: Monitoring & Observability

<div class="production-box">
<strong>What to Monitor:</strong>
<ul>
<li><strong>Availability:</strong> Is the service up? (uptime %)</li>
<li><strong>Latency:</strong> How fast? (p50, p95, p99 milliseconds)</li>
<li><strong>Throughput:</strong> How many requests/second?</li>
<li><strong>Errors:</strong> What failed and why? (error rate %)</li>
<li><strong>Cost:</strong> How much did this request cost? (per request)</li>
</ul>
</div>

### Key Metrics

**SLO (Service Level Objective):**
- Availability: 99.9% ("three nines")
- Latency: p95 < 500ms
- Error rate: < 0.1%

**Example Dashboard:**
```
Uptime: 99.95% ✓ (target: 99.9%)
Latency (p95): 450ms ✓ (target: 500ms)
Error Rate: 0.08% ✓ (target: 0.1%)
Throughput: 1,200 req/s ✓ (capacity: 2,000)
Cost: $8.50/1000 req ($0.0085 per request)
```
