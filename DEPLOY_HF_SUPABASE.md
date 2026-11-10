# Hugging Face Spaces + Supabase Deployment Guide

Deploy QueryFusion with high-performance CPU inference on **Hugging Face Spaces** (16 GB RAM, 2 vCPUs) paired with a free managed **Supabase PostgreSQL** instance (`pgvector` enabled).

---

## 1. Architecture & Cost Breakdown

| Component | Platform | Specs | Cost |
| :--- | :--- | :--- | :--- |
| **Web Service & ONNX Inference** | Hugging Face Spaces | 16 GB RAM, 2 vCPUs, Docker SDK | **$0.00 / month** |
| **Vector & Metadata Database** | Supabase | PostgreSQL 16 with `pgvector`, 500 MB storage | **$0.00 / month** |
| **Embeddings & Reranking** | In-Container ONNX | `bge-small-en-v1.5` + `bge-reranker-base` | **$0.00** |
| **Generation & Verification** | Google AI Studio | `gemini-3.5-flash-lite` + `gemini-3.5-flash` | **$0.00 (Free Tier)** |

---

## 2. Step 1: Set Up Supabase Database

1. Go to [supabase.com](https://supabase.com) and log in with your GitHub account.
2. Click **New Project**:
   * Name: `queryfusion-db` (or any name you prefer).
   * Database Password: *(Generate and copy a secure password)*.
   * Region: Choose the region geographically closest to your users.
3. Once provisioned, navigate to **Project Settings** (gear icon) -> **Database**.
4. Scroll to **Connection string** and select the **URI** tab:
   * **Direct Connection (Port 5432):**
     ```text
     postgresql://postgres.[PROJECT_REF]:[PASSWORD]@aws-0-[REGION].pooler.supabase.com:5432/postgres
     ```
     *(Use this for `DATABASE_URL_DIRECT` for Alembic migrations).*
   * **Transaction Pooler Connection (Port 6543):**
     ```text
     postgresql://postgres.[PROJECT_REF]:[PASSWORD]@aws-0-[REGION].pooler.supabase.com:6543/postgres
     ```
     *(Use this for `DATABASE_URL` for the FastAPI application).*
5. *(Optional verification)* In Supabase's **SQL Editor**, verify that `pgvector` is ready:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```
   *(Supabase enables this by default).*

---

## 3. Step 2: Create a Hugging Face Space

1. Log in to [huggingface.co](https://huggingface.co).
2. Click your profile avatar at the top right -> **New Space**.
3. Fill in the details:
   * **Space name:** `queryfusion` (or `queryfusion-rag`)
   * **License:** `mit`
   * **Space SDK:** Choose **Docker** -> **Blank**.
   * **Space Hardware:** Select **Free (2 vCPU, 16 GB RAM)**.
   * **Visibility:** Public (or Private).
4. Click **Create Space**.

---

## 4. Step 3: Configure Environment Secrets on Hugging Face

In your Space's dashboard:
1. Click **Settings** -> Scroll down to **Variables and secrets**.
2. Click **New secret** and add the following:

| Secret Name | Value | Description |
| :--- | :--- | :--- |
| `DATABASE_URL` | `postgresql+psycopg://postgres.[REF]:[PASS]...:6543/postgres` | Supabase Transaction Pooler (Port 6543) |
| `DATABASE_URL_DIRECT` | `postgresql+psycopg://postgres.[REF]:[PASS]...:5432/postgres` | Supabase Direct Connection (Port 5432) |
| `GEMINI_API_KEY` | *(Your Google AI Studio API key)* | LLM generation & claim verifier |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | Initial answer generation |
| `GEMINI_VERIFIER_MODEL` | `gemini-3.5-flash` | Strict NLI claim verification |
| `RERANKER_MODEL` | `BAAI/bge-reranker-base` | SOTA cross-encoder reranker |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Dense 384-dim embedder |
| `APP_ENV` | `production` | Production mode |

---

## 5. Step 4: Deploy Your Code

You can push your local repository directly to your Hugging Face Space:

```bash
# 1. Add your Hugging Face Space as a git remote
git remote add space https://huggingface.co/spaces/<YOUR_HF_USERNAME>/queryfusion

# 2. Push main branch
git push space main
```

*Note: Hugging Face will trigger `docker build` using your multi-stage `Dockerfile`. The models (`bge-small-en-v1.5` and `bge-reranker-base`) are pre-downloaded during the build step, eliminating cold-start delays when instances initialize.*

---

## 6. Step 5: Prevent the "Dual Sleep Deadlock" (Keep-Alive Cron)

Free tiers have sleep policies:
* **Hugging Face Spaces:** Sleep after ~48 hours of inactivity.
* **Supabase Free Tier:** Pauses after 7 days without database queries.

To keep both permanently alive and active:
1. Go to [UptimeRobot](https://uptimerobot.com) (Free account, no credit card required).
2. Click **Add New Monitor**:
   * **Monitor Type:** `HTTP(s)`
   * **Friendly Name:** `QueryFusion Keep-Alive`
   * **URL:** `https://<YOUR_HF_USERNAME>-queryfusion.hf.space/api/v1/health`
   * **Monitoring Interval:** Every `30 minutes` (or every `12 hours`).
3. Because `/api/v1/health` executes a `SELECT 1` query against Supabase, **each ping wakes or keeps alive both the web container and the PostgreSQL database simultaneously**.

---

## 7. Verifying the Deployment

1. Once the Space build completes, open your Space URL.
2. Verify system health at `/api/v1/health`:
   ```json
   {
     "status": "ok",
     "db": true,
     "environment": "production"
   }
   ```
3. In the UI, click **Ingest Demo Docs** to seed sample technical documents and test hybrid search with `bge-reranker-base` reranking and HyDE query expansion.

