# Week 4 — RAG & Embeddings

**Course:** LangChain for AI Applications  
**Week Focus:** Retrieval-Augmented Generation (RAG) - ground LLMs in your own data.

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
- Understand embeddings and why they work
- Build and query vector stores efficiently
- Implement document loading and chunking strategies
- Create RAG systems that ground LLMs in real data
- Combine retrieval with generation for accurate answers
- Build a knowledge-grounded customer support system

## 📊 Real-World Context

**The Problem:** Your SaaS company has 500+ pages of documentation:
- API guides, tutorials, troubleshooting docs
- Product specs, pricing guides, FAQs
- Support runbooks, internal procedures

**Challenges:**
- LLMs hallucinate answers not in docs
- Users get outdated or incorrect information
- Takes 5-10 minutes to manually search docs
- Support team wastes time answering FAQ questions

**The Solution (RAG):**
1. **Embed** all documentation into a vector database
2. **Retrieve** the most relevant docs based on user question
3. **Generate** an answer grounded in those specific documents
4. **Cite sources** so users can verify information

**Business Impact:**
- ⏱️ Answer docs questions in <5 seconds vs 5-10 minutes manual search
- ✅ Zero hallucinations (answers grounded in real docs)
- 💰 Save $80K/year in support time
- 😊 Improve customer satisfaction (consistent, accurate answers)
- 🚀 Scale support without hiring

Companies like **GitHub Copilot Docs, Notion Help AI, Intercom** use RAG in production.


## 🔍 Part 1: Understanding Embeddings

### What Are Embeddings?

Embeddings convert text into numerical vectors that capture semantic meaning.

**Visual Example:**

```
Text: "How do I reset my password?"
      ↓
Embedding: [-0.123, 0.456, -0.789, 0.234, ...] (1536 dimensions)

Text: "I forgot my password, how to recover?"
      ↓
Embedding: [-0.125, 0.458, -0.787, 0.235, ...] (very similar!)
```

**Key Properties:**
- **Semantic similarity:** Similar texts have similar embeddings
- **Fixed dimensions:** Always same size (e.g., 1536 dims for OpenAI)
- **Dense vectors:** All numbers matter (unlike sparse one-hot encoding)
- **Normalized:** Can use similarity metrics (cosine distance)

**Why This Matters:**
- "Reset password" and "Recover account" are semantically similar
- But raw text comparison would miss this
- Embeddings capture the semantic connection

### Embedding Space Visualization

```
2D Projection of Real Embeddings:

            "Learn Python"
                   |
          "Programming tutorial"
                 /|\
                / | \
    "Java guide" | "Python basics"  → All clustered together!
                \ | /               These are semantically similar
          "Coding lesson"
                   |
            "Write code"

                   PROGRAMMING CLUSTER

------- SEMANTIC SPACE DIVIDER -------

           "Check balance"
                   |
          "Account balance query"
                 /|\
                / | \
  "View account" | "Money status"   → Different cluster!
                \ | /               Banking/Financial semantic
          "How much do I have?"
                   |
            "Account status"
```

## 📚 Part 2: Building RAG Systems

### RAG Pipeline Architecture

```
RAG SYSTEM FLOW:

OFFLINE (Setup once):
  1. Documents → Load & Split → ["chunk1", "chunk2", ...]
  2. Embed Chunks → [vector1, vector2, ...]
  3. Store in Vector DB → indexed and queryable

ONLINE (For each user query):
  1. User Question → Embed the question
  2. Semantic Search → Find K nearest documents
  3. Build Context → "Based on these docs: ..."
  4. Generate Answer → LLM answers using context
  5. Return Result → "Here's your answer (sources: docs 1, 3, 5)"
```

### Document Chunking Strategies

**Why chunking matters:**
- Can't embed entire 500-page document (too long, loses relevance)
- Need to split into meaningful segments
- Optimal chunk size: 500-2000 characters

**Chunking Strategies:**

1. **Fixed Size:** 1000 char chunks
   - ✅ Simple, predictable
   - ❌ Might split sentences

2. **Semantic:** Split at section boundaries
   - ✅ Preserves meaning
   - ❌ Variable sizes

3. **Overlap:** Chunks overlap by 20%
   - ✅ Preserves context at boundaries
   - ❌ Slightly redundant storage

**Example:**

```
DOCUMENT:
"How to reset your password. Step 1: Click Settings. Step 2: Click Security. Step 3: Click Change Password. Enter your new password twice."

CHUNKED (overlap=20%):
Chunk 1: "How to reset your password. Step 1: Click Settings. Step 2: Click Security."
Chunk 2: "Step 2: Click Security. Step 3: Click Change Password. Enter your new password twice."
```

```python
# Part 3: Building a Simple RAG System

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import numpy as np
from typing import List

# Sample documentation
DOCUMENTATION = {
    "api-auth": """# API Authentication

## Overview
All API requests require authentication using an API key.

## Getting Your API Key
1. Log in to your account
2. Go to Settings > API Keys
3. Click 'Generate New Key'
4. Copy the key (it won't be shown again!)

## Using the API Key
Include the key in the Authorization header:
```
curl -H "Authorization: Bearer YOUR_API_KEY" https://api.example.com/v1/users
```

## Key Rotation
For security, rotate your API keys every 90 days.
Old keys continue to work for 7 days after rotation.""",

    "password-reset": """# Password Reset Guide

## Forgot Your Password?
1. Click 'Forgot Password' on the login page
2. Enter your email address
3. Check your email for a reset link
4. Click the link and create a new password
5. Password must be at least 12 characters

## Locked Account
If you fail login 5 times in 30 minutes, your account is locked for 1 hour.

## Resetting Through Admin
Admins can reset user passwords from Settings > User Management.""",

    "billing-plans": """# Billing & Plans

## Available Plans
- Free: $0/month, up to 100 requests/day
- Pro: $99/month, up to 100K requests/day
- Enterprise: Custom pricing, unlimited requests

## Billing Period
- Monthly plans renew on the 1st of each month
- Annual plans get 20% discount
- Cancel anytime, no questions asked

## Invoices
Invoices are available in Settings > Billing.
Download or email invoices for accounting."""
}

# Step 1: Load and chunk documents
print("📄 Step 1: Loading and Chunking Documents")
print("="*70)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)

documents = []
for doc_name, content in DOCUMENTATION.items():
    chunks = text_splitter.split_text(content)
    for i, chunk in enumerate(chunks):
        doc = Document(
            page_content=chunk,
            metadata={
                "source": doc_name,
                "chunk_index": i
            }
        )
        documents.append(doc)

print(f"Total documents loaded: {len(DOCUMENTATION)}")
print(f"Total chunks created: {len(documents)}")
print()

# Display sample chunks
print("Sample Chunks:")
for i, doc in enumerate(documents[:3]):
    print(f"\nChunk {i+1} (from {doc.metadata['source']}):")
    print(f"  {doc.page_content[:100]}...")
```

```python
# Step 2: Simulated Embeddings and Vector Store

print("\n🔢 Step 2: Creating Embeddings")
print("="*70)

class SimpleVectorStore:
    """Simple in-memory vector store for demonstration."""
    
    def __init__(self, documents: List[Document]):
        self.documents = documents
        # In real scenarios, use OpenAI embeddings, Sentence Transformers, etc.
        self.embeddings = self._create_fake_embeddings()
    
    def _create_fake_embeddings(self):
        """Create fake embeddings for demo (hash-based, not semantic)."""
        embeddings = []
        for doc in self.documents:
            # Simple hash-based "embedding" for demo
            np.random.seed(hash(doc.page_content) % 2**32)
            embedding = np.random.randn(768)  # 768-dimensional
            embedding = embedding / np.linalg.norm(embedding)  # Normalize
            embeddings.append(embedding)
        return np.array(embeddings)
    
    def similarity_search(self, query: str, k: int = 3):
        """Find k most similar documents to query."""
        # Create query embedding
        np.random.seed(hash(query) % 2**32)
        query_embedding = np.random.randn(768)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Compute similarity scores
        scores = np.dot(self.embeddings, query_embedding)
        
        # Get top k
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            results.append({
                'document': self.documents[idx],
                'score': float(scores[idx])
            })
        
        return results

# Create vector store
vector_store = SimpleVectorStore(documents)
print(f"Vector store created with {len(documents)} documents")
print(f"Each embedding: 768 dimensions")
print(f"Total storage: {len(documents) * 768 * 4 / 1024:.1f} KB")
```

```python
# Step 3: Query the Vector Store

print("\n🔍 Step 3: Testing RAG Queries")
print("="*70)

test_queries = [
    "How do I get my API key?",
    "I forgot my password, what should I do?",
    "What's the price of the Pro plan?"
]

for query in test_queries:
    print(f"\n❓ Query: {query}")
    print("-" * 70)
    
    # Retrieve relevant documents
    results = vector_store.similarity_search(query, k=2)
    
    print("Retrieved Documents:")
    for i, result in enumerate(results, 1):
        doc = result['document']
        score = result['score']
        print(f"\n  [{i}] Source: {doc.metadata['source']} (similarity: {score:.3f})")
        print(f"      {doc.page_content[:150]}...")

print("\n✅ RAG retrieval working! These documents would be passed to the LLM.")
```

## 🏗️ Part 4: Complete RAG System

<div class="rag-box">
<strong>Building End-to-End RAG:</strong><br><br>
Combine retrieval with generation for complete answers.
</div>

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain.llms.fake import FakeListLLM

# RAG Prompt - instructs LLM to use retrieved docs
rag_prompt = ChatPromptTemplate.from_template(
    """You are a helpful customer support assistant. 

Using ONLY the following documentation, answer the user's question.
If the answer is not in the documentation, say "I don't have that information."

DOCUMENTATION:
{context}

QUESTION: {question}

ANSWER:"""
)

# Simulated LLM responses
llm = FakeListLLM(responses=[
    "To get your API key: 1) Log in to your account, 2) Go to Settings > API Keys, 3) Click 'Generate New Key', 4) Copy the key (it won't be shown again). Remember to rotate your key every 90 days for security.",
    "If you forgot your password: 1) Click 'Forgot Password' on the login page, 2) Enter your email, 3) Check your email for a reset link, 4) Click the link and create a new password. Your new password must be at least 12 characters long.",
    "Our Pro plan costs $99/month and includes up to 100K requests per day. We also offer a Free plan ($0/month with 100 requests/day) and an Enterprise plan with custom pricing and unlimited requests."
])

def rag_query(query: str):
    """Execute a RAG query."""
    # Step 1: Retrieve relevant documents
    retrieved_docs = vector_store.similarity_search(query, k=3)
    
    # Step 2: Format context from retrieved docs
    context = "\n\n".join([
        f"[From {doc['document'].metadata['source']}]\n{doc['document'].page_content}"
        for doc in retrieved_docs
    ])
    
    # Step 3: Generate answer using LLM
    answer = llm.predict(
        context=context,
        question=query
    )
    
    return {
        'query': query,
        'answer': answer,
        'sources': [doc['document'].metadata['source'] for doc in retrieved_docs],
        'retrieved_docs': retrieved_docs
    }

print("✅ RAG System Ready!")
print("\nTesting RAG queries...\n")

# Test the RAG system
test_queries = [
    "How do I get my API key?",
    "I forgot my password",
    "How much is the Pro plan?"
]

for query in test_queries:
    result = rag_query(query)
    print(f"\n{'='*70}")
    print(f"❓ Question: {result['query']}")
    print(f"\n💬 Answer:")
    print(f"   {result['answer']}")
    print(f"\n📚 Sources: {', '.join(result['sources'])}")

print(f"\n{'='*70}")
print("✅ RAG System Complete!")
```

## 🤔 Reflection Questions

**Q1: Why is RAG better than fine-tuning for knowledge updates?**
<details>
<summary>Click for answer</summary>
<strong>RAG advantages:</strong>
<ul>
<li>Update docs instantly (no retraining)</li>
<li>Always cite sources (transparency)</li>
<li>Change sources without changing model</li>
<li>Much cheaper than fine-tuning</li>
<li>Easier to debug (see which docs were retrieved)</li>
</ul>
<strong>Fine-tuning advantages:</strong>
<ul>
<li>Knowledge deeply integrated</li>
<li>Faster inference (no retrieval step)</li>
<li>Smaller model footprint</li>
</ul>
<strong>Best practice:</strong> Use RAG for frequently-updated knowledge, fine-tune for core domain knowledge.
</details>

**Q2: How do you measure RAG quality?**
<details>
<summary>Click for answer</summary>
<ol>
<li><strong>Retrieval metrics:</strong> Did we retrieve relevant documents? (Precision@K, Recall)</li>
<li><strong>Generation metrics:</strong> Is the answer good? (BLEU, ROUGE, semantic similarity)</li>
<li><strong>Human evaluation:</strong> Is it actually helpful? (manual rating 1-5)</li>
<li><strong>Hallucination rate:</strong> How often does LLM invent facts?</li>
<li><strong>Latency:</strong> Is it fast enough? (<500ms for retrieval + generation)</li>
</ol>
</details>

**Q3: When should you use different embedding models?**
<details>
<summary>Click for answer</summary>
<ul>
<li><strong>OpenAI text-embedding-3-large:</strong> Best quality, $0.13 per 1M tokens</li>
<li><strong>Cohere Embed:</strong> Fast, cost-effective</li>
<li><strong>Sentence Transformers:</strong> Free, open-source, run locally</li>
<li><strong>Domain-specific:</strong> Medical docs need medical embeddings (e.g., BioBERT)</li>
</ul>
<strong>Rule of thumb:</strong> Start with text-embedding-3-small (fast, cheap), upgrade if accuracy is insufficient.
</details>

## 📝 Week 4 Project: Knowledge Base Assistant

**Build a complete RAG-powered knowledge base assistant.**

### Requirements:

**Functionality:**
1. Load documentation (at least 5 documents, 5+ KB each)
2. Split intelligently (experiment with chunk sizes)
3. Create embeddings and vector store
4. Retrieve relevant documents for queries
5. Generate grounded answers with citations

**Quality Metrics:**
- Test with 20+ queries
- Measure retrieval accuracy (did we get relevant docs?)
- Human evaluation (are answers helpful?)
- Track hallucinations (does LLM invent facts?)

**Advanced Features (bonus):**
- Hybrid search (semantic + keyword)
- Re-ranking retrieved docs
- Query expansion (expand query before search)
- Chat history awareness

### Deliverables:
- Document loading and chunking pipeline
- Embedding creation and storage
- Similarity search implementation
- RAG generation with citations
- Quality analysis and metrics
- User interface or API endpoint

```python
# Week 4 Project Starter

# TODO: Load your documents
# TODO: Implement chunking with configurable chunk_size and overlap
# TODO: Create embeddings and vector store
# TODO: Build RAG query function
# TODO: Test with 20+ queries
# TODO: Evaluate quality (retrieval accuracy, hallucination rate)

print("🎯 Your knowledge base assistant implementation here!")
```

## 🎓 Key Takeaways

**What you learned this week:**

✅ **Embeddings:**
- Convert text to semantic vectors
- Similar texts have similar embeddings
- Enable semantic search in vector stores

✅ **Document Processing:**
- Chunking strategies and trade-offs
- Overlap importance for context
- Metadata preservation

✅ **RAG Architecture:**
- Retrieve relevant docs based on query
- Use retrieved docs as context
- Generate grounded, cited answers

✅ **Real-world applications:**
- Documentation assistants
- Customer support automation
- Knowledge base Q&A
- Internal wiki search

## 🔜 Next Week: Evaluation & Debugging

In Week 5, we'll master quality assurance:
- Measuring LLM output quality
- Debugging failures
- A/B testing different approaches
- Production monitoring

**Preview question:** How would you detect and fix hallucinations in your RAG system?

## 📚 Additional Resources

- [LangChain RAG Documentation](https://python.langchain.com/docs/use_cases/question_answering/)
- [Vector Store Comparison](https://python.langchain.com/docs/integrations/vectorstores/)
- [Embeddings Guide](https://python.langchain.com/docs/integrations/text_embedding/)
- [RAG Research Papers](https://arxiv.org/abs/2005.11401)

---

**🎉 Congratulations on completing Week 4!** You can now build production RAG systems that ground LLMs in real data. See you next week! 🚀
