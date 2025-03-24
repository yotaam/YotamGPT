# SimpleLLMJava: From-Scratch GPT-2 in Java with API Inference

Welcome to **SimpleLLMJava** — a from-scratch implementation of a GPT-2-like transformer in pure Java, complete with Byte Pair Encoding (BPE) tokenization, multi-head attention, and a full backend API for real-time text generation.

This project is both a technical feat and a playground: build, understand, and deploy large language models... without ever touching Python. 💡

---

## 🚀 What It Does

- Implements the full GPT-2 architecture from scratch (no ML libraries)
- Includes:
  - Token + positional embeddings
  - Multi-head attention
  - Feed-forward blocks
  - Layer normalization
  - BPE tokenizer with vocab merging
- Supports:
  - Text generation via temperature sampling
  - Top-K filtering
  - Prompt-to-text API via REST

---

## 🧠 Why Java?

Why not? This project was built to deeply understand transformers by implementing everything manually, from math to memory. Also, it's fun to flex Java for ML infrastructure.

---

## 🗂 Project Structure

```
SimpleLLMJava/
├── src/
│   ├── GPTModel.java             # Core transformer logic
│   ├── BytePairEncoding.java    # BPE tokenizer
│   ├── GPTService.java          # Wrapper for inference
│   ├── GPTController.java       # Spark API endpoints
│   ├── GPTServer.java           # Entry point
│   ├── ...                      # All other core math/model classes
├── gpt2_weights.json            # Pretrained weights
├── models/gpt2/
│   ├── encoder.json             # BPE encoder
│   └── vocab.bpe                # BPE merge rules
├── pom.xml                      # Maven config
├── target/                      # Maven output
└── start-api.sh                 # Optional helper script
```

---

## ⚙️ Getting Started

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/SimpleLLMJava.git
cd SimpleLLMJava
```

Make sure you have:
- Java JDK 17+ (you can use 23 as tested)
- Maven installed

### 2. Add GPT-2 Weights & BPE Files

Place these files in the project root:
- `gpt2_weights.json`
- `models/gpt2/encoder.json`
- `models/gpt2/vocab.bpe`

### 3. Build the Project

```bash
mvn clean package
```

### 4. Run the Server

```bash
java -Xmx8g -jar target/gpt-api-1.0-SNAPSHOT-jar-with-dependencies.jar
```
Note, setting memory to 8 GB (with -Xmx8g) is required for loading the weights.

---

## 🔥 Using the API

Once running, the API is live at:
```
POST http://localhost:4567/generate
```

**Example Request**
```json
{
  "prompt": "Once upon a time",
  "maxTokens": 30,
  "temperature": 0.8,
  "topK": 50
}
```

**Example Response**
```json
{
  "generatedText": "Once upon a time, there lived a very curious fox who...",
  "inferenceTimeSeconds": 1.42
}
```

You can test it directly via:
```bash
curl -X POST http://localhost:4567/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt":"The mitochondria is the","maxTokens":20,"temperature":0.9,"topK":40}'
```

---

## ☁️ Deployment

### 🛫 Deploy with Fly.io (Coming Soon)

You can easily deploy this project using [Fly.io](https://fly.io):

1. Install Fly CLI
```bash
brew install flyctl
```

2. Create `fly.toml` (or let Fly do it)
```bash
fly launch
```

3. Deploy:
```bash
fly deploy
```

*(We recommend setting `Xmx` to 4g or more in your `Dockerfile` or `fly.toml`)*

### 🔧 Replit / Render Setup

- Replit: Create a Java Repl, upload files, update run command to:
```bash
java -Xmx4g -jar target/gpt-api-1.0-SNAPSHOT-jar-with-dependencies.jar
```

- Render: Deploy with a `Dockerfile` (see `Dockerfile` example coming soon)

---

## ✨ Bonus Features

- Supports Top-K Sampling
- Temperature control
- Timing / inference logs
- Easily extensible for BLEU/ROUGE scoring

---

## 📖 Credits

Built by [Yotam Twersky](https://github.com/YotamTwersky) to explore transformer internals from scratch.

---

## 📎 License

MIT. Use it, break it, remix it, learn from it.

---

## 🧵 TL;DR

- ✅ Pure Java GPT-2
- ✅ Fully custom tokenizer
- ✅ REST API with generation
- ✅ Local or hosted
- ✅ Actually works

What are you waiting for? Start generating.

---

