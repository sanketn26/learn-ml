# Exercises — Week 4 — Production & Scaling

Do these after reading [Week 4 — Production & Scaling](../week-04.md).

## ✍️ Hands-On Exercises

!!! example "Exercise"

    **🎯 Exercise 1: Build Production Wrapper**

    Wrap an agent with production features:

    - Input validation

    - Retry logic (exponential backoff)

    - Timeout handling

    - Error logging

    - Result caching (avoid recomputation)



```python
# Your implementation here!
print("Your production wrapper here!")
```

!!! example "Exercise"

    **🎯 Exercise 2: Task Queue System**

    Build a queue-based system:

    - Task queue (store pending tasks)

    - Worker pool (multiple agents working in parallel)

    - Load balancing (distribute work evenly)

    - Status tracking (query task status)



```python
# Your implementation here!
print("Your task queue system here!")
```

## 📝 Week 4 Project: Production Agent System

**Build a local operational sketch and identify what a real deployment would still require.**

### Requirements:

**1. Reliability:**
- Retry logic with exponential backoff
- Timeout handling (max 30 seconds per task)
- Graceful error handling
- Circuit breaker for failing services

**2. Scalability:**
- Task queue architecture
- Worker pool (5+ workers)
- Load balancing
- Concurrency support

**3. Observability:**
- Comprehensive logging
- Metrics (latency, throughput, errors)
- Health checks
- Dashboard/reporting

**4. Deployment:**
- Docker container
- Environment configuration
- Start/stop scripts
- Deployment docs

### Test Scenarios:
1. **Normal load**: 100 requests/second
2. **Peak load**: 500 requests/second
3. **Failures**: 10% of requests timeout
4. **Recovery**: System recovers when failures stop

### Success Criteria:
- ✅ Handle 500 concurrent requests
- ✅ 99% success rate (even with 10% failures)
- ✅ p95 latency < 5 seconds
- ✅ Detailed logging and metrics
- ✅ Graceful degradation under load

```python
# Week 4 Project Starter

# TODO: Build production wrapper with error handling
# TODO: Implement task queue + worker pool
# TODO: Add monitoring and metrics
# TODO: Create API endpoint
# TODO: Write deployment docs
# TODO: Test with load generator

print("🎯 Your production agent system here!")
```

## 🎓 Key Takeaways

**What you learned this week:**

✅ **Production Architecture:**
- API Gateway, Task Queue, Workers, Database
- Horizontal scaling via worker pools
- Async processing with queues

✅ **Reliability:**
- Retry logic with exponential backoff
- Circuit breakers for failing services
- Timeout handling
- Graceful degradation

✅ **Observability:**
- Comprehensive logging
- Key metrics (latency, throughput, errors)
- SLOs and alerting
- Health checks

✅ **Deployment:**
- Containerization (Docker)
- Configuration management
- Scaling strategies
- Cost optimization

## 📚 Additional Resources

- [Kubernetes for Deployment](https://kubernetes.io/)
- [Site Reliability Engineering (SRE) Book](https://sre.google/)
- [AWS Best Practices](https://aws.amazon.com/architecture/well-architected/)
- [Prometheus Monitoring](https://prometheus.io/)

---

You should now understand CrewAI's core abstractions, be able to build a bounded crew, and know what to investigate next in the official documentation. Operating a production agent system remains a separate engineering project.
