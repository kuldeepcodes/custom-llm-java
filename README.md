# custom-llm-java

[![ci](https://github.com/kuldeepcodes/custom-llm-java/actions/workflows/ci.yml/badge.svg)](https://github.com/kuldeepcodes/custom-llm-java/actions/workflows/ci.yml)
[![Java 17](https://img.shields.io/badge/Java-17-007396)](https://adoptium.net/)
[![licence: MIT](https://img.shields.io/badge/licence-MIT-green.svg)](LICENSE)

Build a question-answering assistant over **your own documents** — one that answers with
**citations** and says "I don't know" when your documents do not cover the question.

> **New to this?** [GETTING-STARTED.md](GETTING-STARTED.md) builds the whole project from an empty
> directory and explains every design choice: chunking, embeddings, retrieval, grounding, citation
> auditing, Ollama model definitions, and fine-tuning export.

```text
$ java -jar target/customllm.jar ingest data --embedder ollama
Embedding with ollama:all-minilm ...
Indexed 6 chunks from 2 document(s)

$ java -jar target/customllm.jar ask "how many days of annual leave do employees get" --embedder ollama
Q: how many days of annual leave do employees get
A: Employees receive 27 days of paid annual leave, plus public holidays [1].

Sources: handbook.md:19

$ java -jar target/customllm.jar ask "what is the capital of France" --embedder ollama
Q: what is the capital of France
A: I don't have anything in the indexed documents that answers that.
```

That refusal is the point. A system that will not admit ignorance is worse than no system at all.

---

## Read this before you start: what "custom LLM" actually means

People asking for "a custom LLM trained on my data" usually want one of three different things.
Choosing wrong wastes time and produces unsafe systems, so this repository implements all three and
labels them honestly.

| | What it does | What it costs | When it is right |
| --- | --- | --- | --- |
| **1. RAG** (the default) | Finds relevant passages at question time and puts them in the prompt | Minutes; CPU is fine | You want factual answers about your documents, with citations |
| **2. Custom model definition** | Bakes a system prompt and parameters into a named Ollama model | Seconds | You want tone, format, persona, or refusal style |
| **3. Fine-tuning export** | Writes JSONL examples for LoRA tooling | Export is instant; real training takes hours and a GPU | You want to change behaviour in ways prompting cannot |

If your goal is "answer questions about my documents", you want option 1. Fine-tuning teaches style
and behaviour; it does not give you a reliable, citable fact table. Facts change, and re-indexing is
simpler and safer than retraining.

### The demonstration that settles it

During development, `create-model` built a real Ollama model whose system prompt described this
coffee company's knowledge base and explicitly forbade inventing sources. Asked a question that is
**nowhere** in that knowledge base, it replied:

```text
$ ollama run nimbus-bot "What is the boiling point of water?"

The information you're asking about can be found in document [2], which states: "Water has a
standard atmospheric pressure boiling point at exactly 100 degrees Celsius (°C) or 212 degrees
Fahrenheit (°F)." Therefore, the answer is that water boils at 100°C or 212°F. [2]
```

There is no document [2] containing that. The model invented the quotation and the citation. The
system prompt was stored correctly; instructions are not guarantees.

The same question through RAG:

```text
$ java -jar target/customllm.jar ask "what is the boiling point of water"
A: I don't have anything in the indexed documents that answers that.
```

That is the argument. Retrieval can refuse because it knows what it actually found. Use RAG for
facts; use custom models and fine-tuning for behaviour.

---

## Quick start

**Prerequisites:** Java 17. The repository includes the Maven Wrapper. Ollama is optional.

```powershell
git clone https://github.com/kuldeepcodes/custom-llm-java.git
cd custom-llm-java

.\mvnw.cmd clean verify          # Windows
# ./mvnw clean verify            # Linux/macOS

java -jar target\customllm.jar ingest data --embedder hashing
java -jar target\customllm.jar ask "how many days of annual leave" --embedder hashing --no-llm
```

For real semantic search and generated answers:

```bash
ollama pull all-minilm    # embedding model
ollama pull phi3          # chat model
java -jar target/customllm.jar ingest data --embedder ollama
java -jar target/customllm.jar ask "how many days of annual leave do employees get" --embedder ollama
```

Without Ollama, the project still works. `--embedder hashing` uses deterministic keyword-style
embeddings, and `--no-llm` returns the best passage verbatim with a citation. Retrieval quality is
lower, but the pipeline is real and testable.

---

## Commands

| Command | What it does |
| --- | --- |
| `ingest <path>` | Chunk, embed, and index a folder or file |
| `ask <question>` | Retrieve passages, answer, and cite sources |
| `search <question>` | Show retrieval results without generation |
| `info` | Describe the current index |
| `create-model <name>` | Write/build an Ollama Modelfile (path 2) |
| `export` | Write JSONL training data for LoRA tooling (path 3) |

Useful flags:

| Flag | Applies to | Meaning |
| --- | --- | --- |
| `--index` | all index commands | Choose the JSON index path |
| `--embedder {auto,ollama,hashing}` | ingest/ask/search | Choose embedding backend |
| `--embed-model` | Ollama embedding | Embedding model, default `all-minilm` |
| `--ollama-url` | Ollama | Base URL, default `http://localhost:11434` |
| `--chunk-size`, `--overlap` | ingest | Tune chunk packing |
| `--top-k` | ask/search | Number of retrieved passages |
| `--model` | ask | Chat model, default `phi3` |
| `--no-llm` | ask | Return best passage instead of generating |
| `--show-context` | ask | Print numbered retrieved passages |
| `--persona`, `--base`, `--write-only` | create-model | Customize Modelfile creation |
| `--out` | create-model/export | Output path |

`search` is the first command to run when an answer looks wrong:

```text
$ java -jar target/customllm.jar search "how much holiday allowance do staff get" --embedder ollama
[1] handbook.md:19  score=0.351 (vector=0.468 keyword=0.000)
    ## Time off Everyone receives 27 days of paid annual leave...
```

`keyword=0.000` means the query and passage share no words. The embedding found the paraphrase. Run
the same query with `--embedder hashing` to see the difference.

---

## How it works

```text
documents ─► chunk ─► embed ─► index
                                 │
question ─► embed ─────────────► search ─► passages ─► prompt ─► LLM ─► answer + citations
                                              │                            │
                                    below relevance floor?          citations audited
                                              └──► refuse                  │
                                                                  invalid / weak → warn
```

### Sentence-aware chunking with overlap

Chunks are packed from whole sentences up to a soft character budget. The next chunk repeats trailing
sentences from the previous one. Overlap matters because a fact on a boundary can otherwise be split
across two chunks and retrievable in neither. Each `Chunk` carries `text`, `source`, `index`, and
`startLine`, so answers can cite `handbook.md:19` rather than hand-wave at a corpus.

### Embeddings

`HashingEmbedder` is deterministic and dependency-free. It tokenizes `[a-z0-9]+`, drops stop words,
hashes each token with SHA-256, chooses a bucket and sign, accumulates, and L2-normalizes. It is not
semantic, but it makes tests and CI reliable.

`OllamaEmbedder` calls `/api/embed` with `all-minilm` by default. It normalizes returned vectors and
turns HTTP 501 into a helpful explanation that generation models such as `phi3` are not embedding
models.

### Hybrid retrieval

The score is 75% cosine similarity and 25% keyword overlap. Pure vector search can smooth away rare
literal tokens such as part numbers, error codes, or surnames. Pure keyword search misses paraphrase.
Together they cover each other's blind spots.

### Relevance floor

If the top score is below 0.25, the tool refuses before calling the LLM. Passing irrelevant context to
a model is an invitation to improvise.

### Citation auditing

The prompt tells the model to cite numbered passages, but prompts are not guarantees. The answer is
checked afterwards:

- **Invalid citations** cite passage numbers that were never supplied.
- **Weak citations** point at passages sharing little wording with the answer when another passage
  matches much better.

This catches the real small-model failure where an answer is right but the citation number is wrong.

### Embedder identity in the index

The JSON index records `embedder` and `dimensions`. Querying an index built with another model is
refused because cosine similarity across different embedding spaces is arithmetic without meaning.
The failure is otherwise silent: plausible-looking scores and terrible retrieval.

---

## Using your own documents

```powershell
java -jar target\customllm.jar ingest C:\path\to\notes --embedder ollama
java -jar target\customllm.jar ask "what did we decide about pricing" --embedder ollama --show-context
```

### Supported formats

| Extension | How it is read |
| --- | --- |
| `.md`, `.markdown`, `.txt` | One document per file |
| `.json` | A top-level array becomes **one document per element**, named `file.json#0`, `file.json#1`, ... A single wrapped array (`{"items": [...]}`) is unwrapped. Anything else is one document |
| `.jsonl`, `.ndjson` | One document per line, named `file.jsonl#1`, `#2`, ... numbered as your editor shows them |

JSON records are flattened into readable `key: value` lines rather than fed in raw, because embedding
models were trained on prose and braces carry no meaning:

```json
{"id": "TKT-1041", "customer": {"name": "Priya"}, "tags": ["hardware", "grinder"]}
```

becomes

```text
id: TKT-1041
customer.name: Priya
tags: hardware, grinder
```

Field names are kept because they are genuine context — `subject: Grinder jams` embeds better than
the bare value. Nulls are dropped, since a line reading `resolution: null` is noise that embeds.

Splitting an array into one document per element is the point of JSON support: a citation reading
`support-tickets.json#3:1` sends you to a specific record you can open and check, whereas "somewhere
in tickets.json" tells you nothing.

For PDFs or Word files, convert them first (`pandoc`, `pdftotext`) — deliberately not built in, so
the dependency list stays honest. [GETTING-STARTED.md](GETTING-STARTED.md#4-step-1-of-the-pipeline-loading-your-data)
shows how to add a format in about five lines.

Tuning tips:

- Long reference documents: try `--chunk-size 1200`.
- Dense FAQs: try `--chunk-size 400`.
- Boundary facts missing: increase `--overlap`.
- Correct passage not retrieved: raise `--top-k` and inspect `search`.
- Index mismatch errors: query with the same embedder used during ingest, or re-ingest.

---

## Tests

```powershell
.\mvnw.cmd clean verify
```

The suite has 103 deterministic tests. It covers chunking edge cases, hashing determinism across
instances, unit-length vectors, cosine identities and mismatch errors, index round-trip through disk,
embedder compatibility refusal, hybrid ranking, relevance-floor refusal, `NOT_IN_CONTEXT`, invalid
citations, weak citations using the observed Meridian case, template artifact stripping, Modelfile
contents, JSONL export validity, and end-to-end ingest/ask with a stub chat client, plus JSON
loading, flattening, JSONL line numbering, and mixed corpora.

Tests use hashing embeddings and stubs, so they need no network, no model, and no GPU.

---

## Build and CI notes

The Maven Wrapper is intentionally committed. Its properties file is pinned to Maven 3.9.16. The
`.gitattributes` file forces LF on `mvnw` and CRLF on `mvnw.cmd`; without that, Linux CI can fail with
`bad interpreter`.

CI runs on Ubuntu and Windows:

```text
./mvnw --batch-mode clean verify
java -jar target/customllm.jar ingest data --embedder hashing
java -jar target/customllm.jar info
java -jar target/customllm.jar ask "how many days of annual leave" --embedder hashing --no-llm
```

The CLI steps pass `--embedder hashing` explicitly so a runner with Ollama installed does not produce
machine-dependent indexes.

---

## The same project in other languages

- **[custom-llm-python](https://github.com/kuldeepcodes/custom-llm-python)** — Python 3.11+
- **[custom-llm-dotnet](https://github.com/kuldeepcodes/custom-llm-dotnet)** — C# / .NET
- **[custom-llm-java](https://github.com/kuldeepcodes/custom-llm-java)** — Java 17 *(this repo)*

## Licence

[MIT](LICENSE)
