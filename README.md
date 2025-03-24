# YotamGPT: From-Scratch GPT-2 in Java with Live API

**YotamGPT** is a fully custom implementation of a GPT-2-like transformer model written entirely in Java, complete with Byte Pair Encoding (BPE) tokenization, multi-head attention, and a lightweight REST API for real-time text generation. It runs on Fly.io and powers a hosted frontend at [yotamtwersky.com/gpt](https://yotamtwersky.com/gpt).

---

## 🚀 What It Does

- Implements a from-scratch GPT-2-style architecture (no ML libraries)
- Includes:
  - Token + positional embeddings
  - Multi-head self-attention
  - Feed-forward network and LayerNorm
  - BPE tokenizer using encoder.json and vocab.bpe
- Supports:
  - Temperature sampling
  - Top-K filtering
  - Real-time inference through REST API (Spark Java)
- Deploys as a cloud API via Fly.io

---

## 🧠 Why Java?
This project is a deliberate deep dive into transformer internals, with the challenge of doing everything manually — from linear algebra to attention — using just Java. It also showcases backend deployment capabilities.

---

## 🗂 Project Structure
```
SimpleLLMJava/
├── src/main/java/com/example/gpt/
│   ├── GPTModel.java             # Transformer core
│   ├── BytePairEncoding.java    # Tokenizer
│   ├── GPTService.java          # Inference engine
│   ├── GPTController.java       # REST API routes
│   ├── GPTServer.java           # Entry point + CORS + server
├── gpt2_weights.json            # Model weights
├── models/gpt2/
│   ├── encoder.json
│   └── vocab.bpe
├── pom.xml                      # Maven build config
├── Dockerfile                   # Deployment config
└──fly.toml                     # Fly.io setup
```

---

## ⚙️ Getting Started

### 1. Clone & Set Up
```bash
git clone https://github.com/YOUR_USERNAME/SimpleLLMJava.git
cd SimpleLLMJava
```

### 2. Requirements
- Java 17+ (JDK 23 recommended)
- Maven

### 3. Add Model Files
Place the following in the root:
- `gpt2_weights.json`
- `models/gpt2/encoder.json`
- `models/gpt2/vocab.bpe`

### 4. Build and Run
```bash
mvn clean package
java -Xmx6g -jar target/gpt-api-1.0-SNAPSHOT-jar-with-dependencies.jar
```

Note: At least 6GB RAM is required due to model size.

---

## ☁️ Deployment

### ✅ Live Demo
API is deployed at: `https://api.yotamtwersky.com/generate`
Frontend is at: [yotamtwersky.com/gpt](https://yotamtwersky.com/gpt)

### Fly.io Steps:
```bash
fly launch
fly deploy
```
Ensure `JAVA_OPTS="-Xmx6g"` in Dockerfile and `fly.toml` has 8GB memory.

---

## 🔥 Using the API

```
POST https://api.yotamtwersky.com/generate
```

**Request Body:**
```json
{
  "prompt": "The future of AI is",
  "maxTokens": 20,
  "temperature": 0.8,
  "topK": 40
}
```

**Example cURL:**
```bash
curl -X POST https://api.yotamtwersky.com/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt":"The mitochondria is the","maxTokens":20,"temperature":0.9,"topK":40}'
```

---

## 🌐 HTML Frontend (gpt.html)
Single-page frontend with loading indicator, hosted at `/gpt`. 
Supports instant user interaction, smooth UX, and full mobile compatibility.

---

## ✨ Highlights
- ✅ No external ML libs
- ✅ Full transformer, BPE, and top-k logic written by hand
- ✅ Real backend deployed in production
- ✅ Elegant browser frontend
- ✅ CORS-enabled for real apps

---

## 👤 Author
Built by [Yotam Twersky](https://yotamtwersky.com), with the goal of understanding transformers inside out and demonstrating backend fluency.
Credit to Stanley Su, with whom I co-built the linear algebra library as the original version of this project.

---

## 🧵 TL;DR
- ⚙️ GPT-2 in Java
- 🌐 REST API w/ generation
- ☁️ Live at `api.yotamtwersky.com`
- 💻 HTML frontend at `yotamtwersky.com/gpt`

---

## 📎 License
MIT — use freely, learn deeply.
