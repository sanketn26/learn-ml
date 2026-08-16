# Exercises — Week 6 — Production & Deployment

Do these after reading [Week 6 — Production & Deployment](../week-06.md).

## ✍️ Hands-On Exercises

!!! example "Exercise"

    **🎯 Exercise 1: Build FastAPI Server**

    Create a small API that demonstrates the production-facing shape:

    - Define request/response models

    - Implement async handlers

    - Add error handling and validation



```python
# Exercise 1: Your FastAPI server here!
print("Your production FastAPI server implementation!")
```

!!! example "Exercise"

    **🎯 Exercise 2: Create Dockerfile & Deploy**

    Containerize and deploy:

    - Write Dockerfile with best practices

    - Build and test locally

    - Push to Docker registry



```python
# Exercise 2: Your Docker deployment here!
print("Your Dockerfile and deployment script!")
```

## 📝 Week 6 Project: Production Deployment

**Sketch and run the operational shell around a small LangChain application.** Local execution is enough; cloud deployment is an optional extension.

### Requirements:

**1. FastAPI Server:**
- `/chat` endpoint (POST)
- `/health` endpoint for monitoring
- Input validation with Pydantic
- Async request handling
- Error handling & retries

**2. Caching:**
- LRU cache for common queries
- Reduce API calls by 50%+
- Track cache hit rate

**3. Docker Setup:**
- Optimized Dockerfile
- Multi-stage builds
- Health checks
- Environment variables

**4. Load Testing:**
- Test with 100+ concurrent users
- Measure response times
- Identify bottlenecks

**5. Monitoring:**
- Request/response logging
- Performance metrics
- Error tracking
- Uptime monitoring

### Deliverables:
- main.py (FastAPI app)
- requirements.txt (dependencies)
- Dockerfile (containerization)
- docker-compose.yml (local testing)
- load_test.py (performance testing)
- deployment_guide.md (cloud deployment)
- monitoring_dashboard.md (production metrics)

```python
# Week 6 Project Starter

# TODO: Build FastAPI server with async handlers
# TODO: Implement request caching
# TODO: Create Docker setup
# TODO: Write load testing script
# TODO: Set up monitoring
# TODO: Deploy to cloud platform
# TODO: Document deployment process

print("🎯 Your local operational sketch here!")
```

## 🎓 Key Takeaways

**What you learned this week:**

✅ **REST APIs:**
- FastAPI for high-performance servers
- Async request handling
- Automatic validation & documentation

✅ **Containerization:**
- Docker for reproducible deployments
- Multi-stage builds for optimization
- Health checks for reliability

✅ **Cloud Deployment:**
- Scaling strategies
- Load balancing
- Auto-scaling policies

✅ **Production Operations:**
- Monitoring and logging
- Performance optimization
- Cost management
- Continuous deployment

## Capstone: prove the introductory mental model

**You have now seen the main LangChain application path:**

- ✅ Week 1-2: Fundamentals & memory
- ✅ Week 3: Agents & tools
- ✅ Week 4: RAG & embeddings
- ✅ Week 5: Evaluation & debugging
- ✅ Week 6: Production & deployment

**Build your final capstone project:**
- A small LLM application with explicit contracts, evaluation, and operational boundaries
- Tested, evaluated, and monitored
- Deployed and scaling in the cloud

---

You should now be able to build and inspect a small LangChain application, recognize its major failure boundaries, and continue independently with the official documentation. Production readiness still depends on your system, traffic, security, evaluation, and operations.
