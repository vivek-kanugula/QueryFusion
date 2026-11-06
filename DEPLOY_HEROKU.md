# Heroku Deployment Guide for QueryFusion

Deploy QueryFusion with 24/7 uptime using your GitHub Student Developer Pack credits ($13/month).

---

## Architecture on Heroku

* **Dyno:** Basic Dyno ($7/mo) — **Runs 24/7 with zero sleeping and zero cold starts**.
* **Database:** Heroku Postgres Essential-0 ($5/mo) — Includes native `pgvector` support, 1 GB storage, 10M rows.
* **Total Out-of-Pocket Cost:** **$0.00** ($12 covered by the $13/mo student credit).

---

## Quick Setup via Heroku Dashboard

### 1. Create App
1. Go to [Heroku Dashboard](https://dashboard.heroku.com/apps).
2. Click **New** -> **Create new app**.
3. Name your app (e.g., `queryfusion-rag` or `queryfusion-live`) and select region **United States**.

### 2. Attach Heroku Postgres
1. In your app dashboard, navigate to the **Resources** tab.
2. In the **Add-ons** search box, type `Heroku Postgres`.
3. Select **Heroku Postgres** and choose plan **Essential-0** ($5/month).
4. Click **Submit Order Form**.
   > Heroku will automatically populate the `DATABASE_URL` config var. QueryFusion automatically translates Heroku's `postgres://` connection string to `postgresql+psycopg://`.

### 3. Configure Environment Variables
1. Go to the **Settings** tab and click **Reveal Config Vars**.
2. Add your application keys:
   * `GEMINI_API_KEY`: *(Your Google AI Studio Gemini API key)*
   * `GEMINI_MODEL`: `gemini-3.5-flash-lite` *(or your preferred Gemini model)*
   * `GROQ_API_KEY`: *(Optional, for Groq LLM dispatch)*
   * `APP_ENV`: `production`

### 4. Connect GitHub & Deploy
1. Go to the **Deploy** tab.
2. Under **Deployment method**, select **GitHub**.
3. Search for your repository: `vivek-kanugula/QueryFusion` and click **Connect**.
4. Scroll down to **Manual deploy**, select branch `main`, and click **Deploy Branch**.
   *(Optional: Enable **Automatic Deploys** so every `git push` to `main` updates your live demo).*

### 5. Verify the Live Demo
1. Once deployment succeeds, click **Open app** at the top right.
2. Navigate to `/api/v1/health` to confirm database connectivity:
   ```json
   { "status": "ok", "db": true, "environment": "production" }
   ```
3. In the UI, click **Ingest Demo Docs** to seed the 3 sample documents and test Q&A search immediately.

---

## Alternative: Deploy via Heroku CLI

If you have `heroku` CLI installed:

```bash
# 1. Login to Heroku
heroku login

# 2. Create app
heroku create queryfusion-live

# 3. Add Heroku Postgres Essential-0
heroku addons:create heroku-postgresql:essential-0 -a queryfusion-live

# 4. Set environment variables
heroku config:set APP_ENV=production -a queryfusion-live
heroku config:set GEMINI_API_KEY="your-gemini-key" -a queryfusion-live
heroku config:set GEMINI_MODEL="gemini-3.5-flash-lite" -a queryfusion-live

# 5. Set dyno to Basic (never sleeps)
heroku ps:type web=basic -a queryfusion-live

# 6. Deploy
git push heroku main
```
