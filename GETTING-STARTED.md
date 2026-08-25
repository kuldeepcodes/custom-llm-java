# Getting started: build a custom LLM assistant over your own data in Java

This guide builds the project from an empty directory. By the end you will have a Java 17 command
line tool that indexes your documents, retrieves the relevant passages for a question, asks an LLM
only when the retrieved evidence is strong enough, cites the passages it used, and refuses when your
corpus does not contain the answer.

**Time:** about ninety minutes if you read and run the examples.

---

## Contents

1. [First, decide what you actually need](#1-first-decide-what-you-actually-need)
2. [Before you start](#2-before-you-start)
3. [Create the Java project](#3-create-the-java-project)
4. [Step 1 of the pipeline: loading your data](#4-step-1-of-the-pipeline-loading-your-data)
5. [Step 2: chunking](#5-step-2-chunking)
6. [Step 3: embeddings](#6-step-3-embeddings)
7. [Step 4: the vector index and retrieval](#7-step-4-the-vector-index-and-retrieval)
8. [Step 5: grounded generation](#8-step-5-grounded-generation)
9. [Step 6: auditing citations](#9-step-6-auditing-citations)
10. [Wiring the pipeline](#10-wiring-the-pipeline)
11. [The command line](#11-the-command-line)
12. [Path 2: building a real custom model definition](#12-path-2-building-a-real-custom-model-definition)
13. [Path 3: preparing data for fine-tuning](#13-path-3-preparing-data-for-fine-tuning)
14. [Testing without a GPU](#14-testing-without-a-gpu)
15. [Making retrieval better](#15-making-retrieval-better)
16. [Java-specific gotchas](#16-java-specific-gotchas)
17. [Troubleshooting](#17-troubleshooting)
18. [Where to go next](#18-where-to-go-next)

---

## 1. First, decide what you actually need

Almost everyone who asks how to "train a custom LLM on my data" is describing one of three different
jobs. They sound similar in conversation, but they have different costs and failure modes.

| Path | What it really does | Cost | Changes facts? | Produces citations? | Best for |
| --- | --- | --- | --- | --- | --- |
| **RAG** | Stores your documents in an index, retrieves passages at question time, and puts them in the prompt | Minutes; CPU is fine | Yes, by re-indexing | Yes | Factual Q&A over documents |
| **Custom model definition** | Bakes a system prompt and parameters into a named Ollama model | Seconds | No, not reliably | No, unless retrieval supplies passages | Tone, format, persona, refusal style |
| **LoRA fine-tuning** | Updates model weights from training examples | Hours and a GPU | Not as a reliable fact table | No | Behaviour patterns prompting cannot reach |

### The experiment that decides it

This project exists because the difference is observable, not philosophical. A custom model was built
with `ollama create`. Its system prompt explicitly described the corpus and explicitly forbade
inventing sources. `ollama show` confirmed the prompt was stored. Then the model was asked a question
absent from the corpus:

```text
$ ollama run nimbus-bot "What is the boiling point of water?"

The information you're asking about can be found in document [2], which states: "Water has a
standard atmospheric pressure boiling point at exactly 100 degrees Celsius (°C) or 212 degrees
Fahrenheit (°F)." Therefore, the answer is that water boils at 100°C or 212°F. [2]
```

There was no document [2] containing that sentence. The model invented both the document and the
quotation. The instruction "never invent sources" was present, but instructions are not guarantees.

The same question through the RAG path:

```text
$ java -jar target/customllm.jar ask "what is the boiling point of water"
A: I don't have anything in the indexed documents that answers that.
```

Retrieval can refuse because it knows what it actually found. A model asked to answer from memory has
nothing to check against. The conclusion for this entire guide is therefore explicit:

- Use **RAG for facts**.
- Use **custom model definitions for behaviour**.
- Use **fine-tuning for behaviour**, not as a citable database.

If your requirement is "answer questions about my documents", build RAG first. Add the custom model
path only after retrieval is working.

---

## 2. Before you start

You need Java 17 and Maven. This repository includes the Maven Wrapper, so after bootstrapping the
wrapper you can build with `./mvnw` on Linux/macOS or `.\mvnw.cmd` on Windows.

```powershell
java -version
.\mvnw.cmd -v
```

Optional, but useful for real semantic search and generation:

```bash
ollama pull all-minilm   # small embedding model
ollama pull phi3         # chat/generation model
```

`all-minilm` and `phi3` are not interchangeable. `all-minilm` has the `embedding` capability and can
serve `/api/embed`. `phi3` is a generation model. Asking it for embeddings returns HTTP 501. The Java
client turns that raw status into a useful message telling you to pull `all-minilm`.

The project also works without Ollama. In that mode it uses a deterministic hashing embedder and, if
no chat model is reachable, returns the best retrieved passage verbatim. That is not generation, but
it is honest and testable.

---

## 3. Create the Java project

Start with a plain Maven project, not Spring Boot. A RAG demo should have as few moving parts as
possible: Java records, `java.net.http.HttpClient`, Jackson for JSON, JUnit 5 for tests, and a shade
plugin for the runnable jar.

```powershell
mkdir custom-llm-java
cd custom-llm-java
mkdir src\main\java\dev\kuldeepcodes\customllm
mkdir src\test\java\dev\kuldeepcodes\customllm
mkdir data
```

The important parts of `pom.xml` are these:

```xml
<properties>
  <maven.compiler.release>17</maven.compiler.release>
  <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
  <jackson.version>2.22.2</jackson.version>
</properties>

<dependencies>
  <dependency>
    <groupId>com.fasterxml.jackson.core</groupId>
    <artifactId>jackson-databind</artifactId>
    <version>${jackson.version}</version>
  </dependency>
  <dependency>
    <groupId>org.junit.jupiter</groupId>
    <artifactId>junit-jupiter</artifactId>
    <version>5.13.4</version>
    <scope>test</scope>
  </dependency>
  <dependency>
    <groupId>org.assertj</groupId>
    <artifactId>assertj-core</artifactId>
    <version>3.27.4</version>
    <scope>test</scope>
  </dependency>
</dependencies>
```

`maven.compiler.release` is better than separate source/target flags because it also prevents you
from accidentally using APIs that exist only in newer JDKs.

For the runnable jar, use Shade:

```xml
<build>
  <finalName>customllm</finalName>
  <plugins>
    <plugin>
      <groupId>org.apache.maven.plugins</groupId>
      <artifactId>maven-shade-plugin</artifactId>
      <version>3.6.0</version>
      <executions>
        <execution>
          <phase>package</phase>
          <goals><goal>shade</goal></goals>
          <configuration>
            <createDependencyReducedPom>false</createDependencyReducedPom>
            <transformers>
              <transformer implementation="org.apache.maven.plugins.shade.resource.ManifestResourceTransformer">
                <mainClass>dev.kuldeepcodes.customllm.Main</mainClass>
              </transformer>
            </transformers>
          </configuration>
        </execution>
      </executions>
    </plugin>
  </plugins>
</build>
```

That produces `target/customllm.jar`, so users can run one file without constructing a classpath.

Add a Maven Wrapper and then check its properties file. On PowerShell, the wrapper plugin invocation
can mis-parse `-Dmaven=3.9.16`, so verify the file by hand. It should be exactly:

```properties
wrapperVersion=3.3.4
distributionType=only-script
distributionUrl=https://repo.maven.apache.org/maven2/org/apache/maven/apache-maven/3.9.16/apache-maven-3.9.16-bin.zip
```

Also add `.gitattributes`:

```text
mvnw text eol=lf
mvnw.cmd text eol=crlf
```

This is not cosmetic. If `mvnw` gets CRLF endings, Linux CI can fail with `bad interpreter`.

---

## 4. Step 1 of the pipeline: loading your data

This is the front door of the whole system, and the part most directly about *your* data. Everything
downstream depends on it producing two things:

- **text that reads naturally**, because embedding models were trained on prose, not on
  punctuation-heavy data structures;
- **a stable name for every document**, because that name ends up in the citation a user sees and
  needs to be able to look up.

Create **`src/main/java/dev/kuldeepcodes/customllm/loaders/Loaders.java`**. Start with what counts as
a document:

```java
/** Plain text document suffixes. */
public static final Set<String> TEXT_SUFFIXES = Set.of(".txt", ".md", ".markdown");
/** JSON document suffixes. */
public static final Set<String> JSON_SUFFIXES = Set.of(".json", ".jsonl", ".ndjson");
/** Every suffix the loader accepts. */
public static final Set<String> SUPPORTED_SUFFIXES = Set.of(
    ".txt", ".md", ".markdown", ".json", ".jsonl", ".ndjson"
);
```

### Walking the corpus

```java
public static List<LoadedDocument> loadDocuments(Path source) throws IOException {
    if (!Files.exists(source)) {
        throw new FileNotFoundException("No such path: " + source);
    }

    List<Path> files;
    if (Files.isRegularFile(source)) {
        files = List.of(source);
    } else {
        try (Stream<Path> stream = Files.walk(source)) {
            files = stream.filter(Files::isRegularFile)
                .filter(path -> SUPPORTED_SUFFIXES.contains(suffix(path)))
                .sorted()
                .toList();
        }
    }

    List<LoadedDocument> documents = new ArrayList<>();
    for (Path path : files) {
        String suffix = suffix(path);
        if (!SUPPORTED_SUFFIXES.contains(suffix)) {
            continue;
        }

        String baseName = Files.isRegularFile(source)
            ? path.getFileName().toString()
            : source.relativize(path).toString();
        baseName = baseName.replace('\\', '/');

        if (JSON_SUFFIXES.contains(suffix)) {
            documents.addAll(loadJsonDocuments(path, baseName));
        } else {
            String text = readUtf8ReplacingMalformedInput(path);
            if (!text.trim().isEmpty()) {
                documents.add(new LoadedDocument(baseName, text));
            }
        }
    }
    return documents;
}
```

Four decisions, each with a reason:

**`sorted()`** — ingestion must be reproducible. Filesystem iteration order is not guaranteed, so
without sorting, two runs over the same folder could produce chunks in different orders, different
`index` values, and a diff in the index file for no reason.

**`source.relativize(path)` and `.replace('\\', '/')`** — the name goes into every citation. Storing
an absolute path would leak your home directory into the index and break the moment the folder moves.
Forcing forward slashes means an index built on Windows is byte-identical to one built on Linux.

**Replacement-character decoding** — one file with a stray byte should not abort ingestion of five
hundred others. This project uses `new String(Files.readAllBytes(path), StandardCharsets.UTF_8)`,
whose UTF-8 decoder replaces malformed input rather than throwing.

**Accept a file as well as a directory** — `java -jar target\customllm.jar ingest data\handbook.md`
is a natural thing to type, and supporting it costs one conditional.

### Why JSON is not just "read the file"

Suppose you export 500 support tickets as a JSON array. Handing the raw file to an embedder is close
to useless:

- braces, quotes and field names dominate the token stream, crowding out the actual content;
- every record has identical structure, so records look alike to a vector even when they say
  completely different things;
- a citation could only ever point at the whole file — "somewhere in `tickets.json`" is not a useful
  citation.

What you want is **one document per record**, rendered as readable lines, each individually
addressable. Two functions do that.

### Splitting a JSON file into records

```java
public static List<LoadedDocument> loadJsonDocuments(Path path, String baseName) throws IOException {
    String raw = readUtf8ReplacingMalformedInput(path);
    if (raw.trim().isEmpty()) {
        return List.of();
    }

    if (Set.of(".jsonl", ".ndjson").contains(suffix(path))) {
        return loadJsonLines(raw, baseName);
    }

    JsonNode payload;
    try {
        payload = JSON.readTree(raw);
    } catch (JsonProcessingException e) {
        throw new IllegalArgumentException(jsonError(baseName, e), e);
    }

    JsonNode records = asRecords(payload);
    if (records == null) {
        String text = flattenJson(payload);
        return text.trim().isEmpty() ? List.of() : List.of(new LoadedDocument(baseName, text));
    }

    List<LoadedDocument> documents = new ArrayList<>();
    for (int position = 0; position < records.size(); position++) {
        String text = flattenJson(records.get(position));
        if (!text.trim().isEmpty()) {
            documents.add(new LoadedDocument(baseName + "#" + position, text));
        }
    }
    return documents;
}
```

The naming convention is the important part. Record 3 of `support-tickets.json` becomes
`support-tickets.json#3`, and after chunking its citation reads `support-tickets.json#3:1`. A user
can open the file, count to the fourth element, and check the claim. That is the whole promise of a
citation, and it only works if the name is precise.

Note the error message. `JsonProcessingException` on its own tells you *what* is wrong but not *which
of your 500 files* it happened in — so the file name goes in the message.

### Deciding what the collection is

```java
private static JsonNode asRecords(JsonNode payload) {
    if (payload.isArray()) {
        return payload;
    }
    if (payload.isObject()) {
        List<JsonNode> nonEmptyArrays = new ArrayList<>();
        payload.fields().forEachRemaining(entry -> {
            JsonNode value = entry.getValue();
            if (value.isArray() && !value.isEmpty()) {
                nonEmptyArrays.add(value);
            }
        });
        if (nonEmptyArrays.size() == 1) {
            return nonEmptyArrays.get(0);
        }
    }
    return null;
}
```

A top-level array is unambiguous: it is the collection.

An object needs judgement. Export formats routinely wrap their payload — `{"items": [...]}`,
`{"results": [...]}` — and the wrapper carries no information worth indexing. So a single array
property is unwrapped.

But **two** arrays are left alone deliberately. `{"users": [...], "orders": [...]}` is not a
collection with a wrapper; it is a document containing two collections, and picking one would silently
discard the other. When the shape is ambiguous, do the safe thing and index the whole object.

### Turning a record into readable text

```java
public static String flattenJson(JsonNode value, String prefix, int depth) {
    if (depth > MAX_JSON_DEPTH) {
        return prefix.isEmpty() ? "..." : prefix + ": ...";
    }

    if (value == null || value.isNull()) {
        return "";
    }

    if (value.isObject()) {
        List<String> lines = new ArrayList<>();
        value.fields().forEachRemaining(entry -> {
            String key = prefix.isEmpty() ? entry.getKey() : prefix + "." + entry.getKey();
            String line = flattenJson(entry.getValue(), key, depth + 1);
            if (!line.isEmpty()) {
                lines.add(line);
            }
        });
        return String.join("\n", lines);
    }

    if (value.isArray()) {
        if (allScalars(value)) {
            List<String> rendered = new ArrayList<>();
            for (JsonNode item : value) {
                if (!item.isNull()) {
                    rendered.add(scalar(item));
                }
            }
            return rendered.isEmpty() ? "" : prefix + ": " + String.join(", ", rendered);
        }

        List<String> lines = new ArrayList<>();
        for (int position = 0; position < value.size(); position++) {
            String key = prefix.isEmpty() ? "[" + position + "]" : prefix + "[" + position + "]";
            String line = flattenJson(value.get(position), key, depth + 1);
            if (!line.isEmpty()) {
                lines.add(line);
            }
        }
        return String.join("\n", lines);
    }

    return prefix.isEmpty() ? scalar(value) : prefix + ": " + scalar(value);
}
```

Given this ticket:

```json
{
  "id": "TKT-1041",
  "subject": "Grinder jams on the darkest roast",
  "customer": { "name": "Priya Raman", "plan": "wholesale" },
  "status": "resolved",
  "tags": ["hardware", "grinder", "roast-profile"],
  "body": "Customer reported the EK43 jamming when grinding our darkest roast. ...",
  "resolution": "Cleaning schedule adjusted. No hardware fault found."
}
```

it produces:

```text
id: TKT-1041
subject: Grinder jams on the darkest roast
customer.name: Priya Raman
customer.plan: wholesale
status: resolved
tags: hardware, grinder, roast-profile
body: Customer reported the EK43 jamming when grinding our darkest roast. ...
resolution: Cleaning schedule adjusted. No hardware fault found.
```

Five choices worth understanding:

**Keep the field names.** They are real context. `subject: Grinder jams` embeds better than
`Grinder jams` alone, because the key tells the model what kind of thing the value is.

**Join nested keys with a dot.** `customer.name` stays traceable back to the original structure, so
someone reading a citation can find the field in the source file.

**Render scalar lists inline.** `tags: hardware, grinder` reads like prose. One line per tag would
fragment a single idea across three lines and dilute the embedding.

**Drop nulls entirely.** A line reading `resolution: null` is worse than no line: it is noise that
embeds, and a model may well read it as a meaningful value.

**Cap the depth.** A deeply nested structure would otherwise produce thousands of lines of key paths
and nothing else. Six levels is far more than any sensible record needs.

### JSONL, and why line numbers

```java
private static List<LoadedDocument> loadJsonLines(String raw, String baseName) {
    List<LoadedDocument> documents = new ArrayList<>();
    String[] lines = raw.split("\\R", -1);
    for (int number = 1; number <= lines.length; number++) {
        String line = lines[number - 1];
        if (line.trim().isEmpty()) {
            continue;
        }
        JsonNode record;
        try {
            record = JSON.readTree(line);
        } catch (JsonProcessingException e) {
            throw new IllegalArgumentException(baseName + " line " + number
                + " is not valid JSON: " + e.getOriginalMessage(), e);
        }
        String text = flattenJson(record);
        if (!text.trim().isEmpty()) {
            documents.add(new LoadedDocument(baseName + "#" + number, text));
        }
    }
    return documents;
}
```

Records are numbered from **1**, not 0, because that number matches what a text editor shows you.
`events.jsonl#42` should send you to line 42. Array elements start at 0 because that is what indexing
an array means. The convention matches the format, in both cases.

### Try it

```powershell
java -jar target\customllm.jar ingest data --embedder ollama
java -jar target\customllm.jar info
```

```text
chunks:     11
documents:  7
  handbook.md  (3 chunks)
  products.md  (3 chunks)
  support-tickets.json#0  (1 chunks)
  support-tickets.json#1  (1 chunks)
  ...
```

Markdown and JSON in one corpus, each ticket its own document. Now ask something that only the JSON
can answer:

```powershell
java -jar target\customllm.jar ask "why did the grinder jam on the dark roast" --embedder ollama
```

```text
A: A customer reported their EK43 jamming when grinding the darkest roast, caused by oil build-up on
   the burrs from a high-oil bean [1].

Sources: support-tickets.json#0:1
```

The citation points at ticket 0. You can open the file and check it.

### Adding your own format

The pattern generalises. To support PDFs, add `.pdf` to the suffix set and a branch that extracts
text — everything downstream is unchanged, because it only ever sees `(name, text)` pairs:

```java
if (".pdf".equals(suffix)) {
    String text = extractPdfText(path);
    if (!text.trim().isEmpty()) {
        documents.add(new LoadedDocument(baseName, text));
    }
}
```

That narrow interface is exactly why the loader is a separate package. CSV rows, database records and
API responses all fit the same shape: **give every record a stable name and readable text**, and the
rest of the pipeline does not need to know where it came from.

---
## 5. Step 2: chunking

Whole documents do not go into prompts. They are too long, and burying one relevant sentence in ten
pages of text degrades the answer. Documents become retrievable chunks.

Chunking has two common failure modes:

- **Too large:** retrieval returns a wall of text and the answer sentence is diluted.
- **Too small:** the fact is detached from context. "It is 27 days" is useless without the sentence
  naming annual leave.

The project uses sentence-aware packing with overlap. Sentence-aware means never cutting inside a
sentence. Overlap means repeating the last part of one chunk at the start of the next. A fact on a
boundary would otherwise be split across two chunks and retrievable in neither.

The provenance record is tiny but important:

```java
/**
 * One retrievable passage, with enough provenance to cite it.
 *
 * @param text passage text
 * @param source document name
 * @param index zero-based chunk index
 * @param startLine one-based start line
 */
public record Chunk(String text, String source, int index, int startLine) {
    /** Returns a compact citation such as {@code handbook.md:12}. */
    public String citation() {
        return source + ":" + startLine;
    }
}
```

Carry source and line numbers from the start. Retrofitting citations after retrieval is painful and
usually inaccurate.

The sentence splitter is deliberately simple:

```java
private static final Pattern SENTENCE_END = Pattern.compile("(?<=[.!?])\\s+|\\n{2,}");

public static List<String> splitSentences(String text) {
    List<String> sentences = new ArrayList<>();
    for (String part : SENTENCE_END.split(text)) {
        String trimmed = part.trim();
        if (!trimmed.isEmpty()) {
            sentences.add(trimmed);
        }
    }
    return sentences;
}
```

A full NLP sentence splitter is a large dependency for a teaching project. This rule handles normal
prose and blank-line paragraph boundaries well enough.

The packing loop is the core:

```java
for (int position = 0; position < sentences.size(); position++) {
    String sentence = sentences.get(position);
    int addition = sentence.length() + (current.isEmpty() ? 0 : 1);

    if (!current.isEmpty() && currentLength + addition > chunkSize) {
        chunks.add(build(current, source, chunks.size(), lines.get(firstSentence)));
        Carry carry = carryOver(current, position, overlap, firstSentence);
        current = new ArrayList<>(carry.sentences());
        currentLength = current.stream().mapToInt(String::length).sum()
            + Math.max(0, current.size() - 1);
        firstSentence = carry.firstSentence();
    }

    if (current.isEmpty()) {
        firstSentence = position;
    }
    current.add(sentence);
    currentLength += addition;
}
```

Notice the budget is soft. A single sentence longer than the chunk size is emitted whole. Splitting it
would violate the more important rule: never cut a sentence.

Overlap is chosen from whole trailing sentences:

```java
private static Carry carryOver(List<String> sentences, int next, int overlap, int first) {
    if (overlap == 0) {
        return new Carry(List.of(), next);
    }
    List<String> carried = new ArrayList<>();
    int size = 0;
    for (int i = sentences.size() - 1; i >= 0; i--) {
        String sentence = sentences.get(i);
        if (size + sentence.length() > overlap && !carried.isEmpty()) {
            break;
        }
        carried.add(0, sentence);
        size += sentence.length() + 1;
    }
    return new Carry(carried, first + sentences.size() - carried.size());
}
```

The returned `firstSentence` keeps citations accurate even when content is repeated.

**Try it:** temporarily add a small JUnit test or JShell snippet:

```java
String text = "First sentence here. Second sentence follows. Third one closes.";
for (Chunk chunk : Chunker.chunkText(text, "demo.md", 40, 25)) {
    System.out.println(chunk.citation() + " -> " + chunk.text());
}
```

You should see the second sentence repeated. That repetition is intentional; it protects boundary
facts.

---

## 6. Step 3: embeddings

An embedding turns text into a vector. Similar meanings should land near each other, so "holiday
allowance" can retrieve a sentence that says "annual leave". This project has two embedders behind
one interface:

```java
public interface Embedder {
    List<double[]> embed(List<String> texts);
    String name();
    int dimensions();

    default double[] embedOne(String text) {
        return embed(List.of(text)).get(0);
    }
}
```

The `name()` method is not just display text. It is written into the index so query-time code can
refuse incompatible vectors.

### The hashing embedder: no model required

The hashing embedder has no semantic understanding. It matches shared vocabulary. That sounds weak,
but it is exactly what you want for tests and CI: deterministic, no network, no model download, and
identical results across JVM runs.

Do **not** use `String.hashCode()` for this. It is stable in current Java implementations, but it is a
language-level string hash, not a deliberately specified embedding hash with enough bits for bucket
and sign selection. Use SHA-256 so the algorithm is explicit and portable.

```java
private double[] embedText(String text) {
    double[] vector = new double[dimensions];
    Matcher matcher = TOKEN.matcher(text.toLowerCase());
    MessageDigest digest = sha256();
    while (matcher.find()) {
        String token = matcher.group();
        if (STOPWORDS.contains(token)) {
            continue;
        }
        byte[] hash = digest.digest(token.getBytes(StandardCharsets.UTF_8));
        long head = ((hash[0] & 0xffL) << 24)
            | ((hash[1] & 0xffL) << 16)
            | ((hash[2] & 0xffL) << 8)
            | (hash[3] & 0xffL);
        int bucket = (int) (head % dimensions);
        vector[bucket] += (hash[4] & 1) == 1 ? 1.0 : -1.0;
    }
    return Vectors.normalize(vector);
}
```

The token pattern is `[a-z0-9]+`, and a short stop-word list removes words that add little retrieval
signal. The sign bit matters: without signed hashing, collisions only add positive weight and make
unrelated texts look more similar.

### L2 normalization

Every vector is L2-normalized:

```java
public static double[] normalize(double[] vector) {
    double norm = 0.0;
    for (double value : vector) {
        norm += value * value;
    }
    norm = Math.sqrt(norm);
    double[] copy = vector.clone();
    if (norm == 0.0) {
        return copy;
    }
    for (int i = 0; i < copy.length; i++) {
        copy[i] /= norm;
    }
    return copy;
}
```

Why normalize? Without it, longer chunks tend to have larger vectors and can win because they contain
more words, not because they are more relevant. With unit vectors, cosine similarity is comparable
across chunk sizes and effectively becomes a dot product.

Cosine itself still guards against zero vectors and length mismatches:

```java
public static double cosineSimilarity(double[] a, double[] b) {
    if (a.length != b.length) {
        throw new IllegalArgumentException("Vector length mismatch: " + a.length + " vs " + b.length + ".");
    }
    double dot = 0.0;
    double normA = 0.0;
    double normB = 0.0;
    for (int i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    if (normA == 0.0 || normB == 0.0) {
        return 0.0;
    }
    return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}
```

The length mismatch must throw. A cosine score between different dimensionalities is a bug, not a
low-confidence answer.

### The real Ollama embedder

Ollama's embedding endpoint accepts a batch:

```java
String body = JSON.writeValueAsString(Map.of("model", model, "input", texts));
HttpRequest request = HttpRequest.newBuilder(URI.create(baseUrl + "/api/embed"))
    .timeout(timeout)
    .header("Content-Type", "application/json")
    .POST(HttpRequest.BodyPublishers.ofString(body))
    .build();
HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
```

Handle the common mistake explicitly:

```java
if (response.statusCode() == 501) {
    throw new IllegalStateException(
        "Model '" + model + "' cannot produce embeddings. Generation models "
            + "such as phi3 are not embedding models. Run: ollama pull all-minilm"
    );
}
```

That branch is not decoration. The raw 501 does not tell a new user that they used a chat model where
an embedding model is required.

Availability checks hit `/api/tags`:

```java
for (JsonNode item : JSON.readTree(response.body()).path("models")) {
    String name = item.path("name").asText("");
    if (name.equals(model) || name.startsWith(model + ":")) {
        return true;
    }
}
```

The factory uses Ollama only when it is actually present:

```java
return switch (choice) {
    case "hashing" -> new HashingEmbedder();
    case "ollama" -> new OllamaEmbedder(model, baseUrl);
    case "auto" -> OllamaEmbedder.isAvailable(model, baseUrl)
        ? new OllamaEmbedder(model, baseUrl)
        : new HashingEmbedder();
    default -> throw new IllegalArgumentException("Unknown embedder '" + prefer + "'.");
};
```

**Try it:** compare paraphrase retrieval. With Ollama running, the semantic embedder should score
these as related; hashing mostly sees no shared words.

```java
Embedder semantic = new OllamaEmbedder("all-minilm", "http://localhost:11434");
Embedder hashing = new HashingEmbedder();
String a = "How much holiday allowance do staff get?";
String b = "Everyone receives 27 days of paid annual leave.";
System.out.println(Vectors.cosineSimilarity(semantic.embedOne(a), semantic.embedOne(b)));
System.out.println(Vectors.cosineSimilarity(hashing.embedOne(a), hashing.embedOne(b)));
```

That gap is what embeddings buy you.

---

## 7. Step 4: the vector index and retrieval

A production system may use a vector database. This project uses readable JSON and an exhaustive scan
on purpose. At a few thousand chunks it is fast enough, and a reader can open the index and inspect
exactly what the system knows.

The saved payload contains:

```json
{
  "version": 1,
  "embedder": "ollama:all-minilm",
  "dimensions": 384,
  "chunks": [],
  "vectors": []
}
```

Versioning lets future code refuse old formats. Storing the embedder name prevents the worst retrieval
bug: querying an index with vectors from a different model.

### Hybrid scoring

Pure vector search is weak on rare literal tokens. Embeddings smooth away part numbers, error codes,
customer names, and policy IDs. Pure keyword search has the opposite failure: it misses paraphrase.
Hybrid retrieval covers both:

```java
double vectorScore = Vectors.cosineSimilarity(queryVector, vectors.get(i));
double keywordScore = keywordOverlap(queryTokens, tokenise(chunk.text()));
double score = 0.75 * vectorScore + 0.25 * keywordScore;
results.add(new SearchResult(chunk, score, vectorScore, keywordScore));
```

Keyword overlap is deliberately simple:

```java
private static double keywordOverlap(Set<String> queryTokens, Set<String> chunkTokens) {
    if (queryTokens.isEmpty()) {
        return 0.0;
    }
    int shared = 0;
    for (String token : queryTokens) {
        if (chunkTokens.contains(token)) {
            shared++;
        }
    }
    return (double) shared / queryTokens.size();
}
```

Sorting is by score descending, then source and chunk index. The tie-breakers make results stable
across JVM runs:

```java
results.sort(Comparator.comparingDouble(SearchResult::score).reversed()
    .thenComparing(result -> result.chunk().source())
    .thenComparingInt(result -> result.chunk().index()));
```

### Refuse to mix embedders

```java
public void ensureCompatible(Embedder embedder) {
    if (!embedder.name().equals(embedderName)) {
        throw new IllegalArgumentException(
            "This index was built with '" + embedderName + "' but you are querying with '"
                + embedder.name() + "'. Vectors from different models are not comparable."
        );
    }
}
```

Cross-model cosine similarity is arithmetic without meaning. It does not crash. It simply returns
plausible-looking bad scores, which is worse. Record the embedder and fail loudly.

**Try it:** build with hashing, then query with Ollama.

```powershell
java -jar target\customllm.jar ingest data --embedder hashing
java -jar target\customllm.jar ask "annual leave" --embedder ollama
```

The tool refuses instead of giving meaningless results.

---

## 8. Step 5: grounded generation

Retrieval finds passages. Generation turns them into an answer. Grounding means the model is asked to
answer only from numbered passages and to cite those numbers.

The system prompt is intentionally strict:

```java
public static final String SYSTEM_PROMPT = """
    You answer questions using only the numbered passages provided by the user.

    Rules:
    - Use only information present in the passages. Do not add outside knowledge.
    - Mark every claim with the number of the passage it came from, like [1] or [2].
    - If the passages do not contain the answer, reply exactly: NOT_IN_CONTEXT
    - Never invent a passage number you were not given.
    - Quote figures, dates and names exactly as they appear.
    - Answer in at most four sentences.""";
```

The `NOT_IN_CONTEXT` escape hatch matters. Without an explicit refusal token, a model asked an
unanswerable question tends to produce something anyway.

The user prompt numbers passages:

```java
public static String buildPrompt(String question, List<SearchResult> results) {
    StringBuilder builder = new StringBuilder("Passages:\n\n");
    for (int i = 0; i < results.size(); i++) {
        SearchResult result = results.get(i);
        builder.append('[').append(i + 1).append("] (")
            .append(result.chunk().citation()).append(") ")
            .append(result.chunk().text()).append("\n\n");
    }
    return builder.append("Question: ").append(question).toString();
}
```

Ollama is called with temperature zero and a 4096 context window:

```java
"options", Map.of("temperature", temperature, "num_ctx", 4096)
```

For grounded QA, creativity is the failure mode. Set temperature to zero.

### Relevance floor

Before asking the model, check whether retrieval found anything strong enough:

```java
if (results.isEmpty() || results.get(0).score() < RELEVANCE_FLOOR) {
    return new Answer(REFUSAL, results, false, List.of(), List.of());
}
```

The floor is 0.25 in this project. If the best chunk is weaker than that, handing irrelevant passages
to the model invites fabrication. Refusing early is safer and easier to explain.

If there is no chat client, the system still returns useful evidence:

```java
if (chat == null) {
    return new Answer(results.get(0).chunk().text() + " [1]", results);
}
```

That is extractive mode. It is not pretending to generate; it is honest retrieval.

If the model says `NOT_IN_CONTEXT`, map it to the public refusal:

```java
String raw = stripTemplateArtifacts(chat.complete(SYSTEM_PROMPT, buildPrompt(question, results), 0.0));
if (raw.isBlank() || raw.toUpperCase().contains("NOT_IN_CONTEXT")) {
    return new Answer(REFUSAL, results, false, List.of(), List.of());
}
```

---

## 9. Step 6: auditing citations

Most RAG tutorials stop after prompting. This project checks the answer afterwards because
instructions are not guarantees. During development, phi3 cited a passage number it was never given.
In another run it answered the content correctly but attached the wrong citation number.

The `Answer` object exposes the text, retrieved results, grounded flag, and warnings:

```java
public final class Answer {
    private static final Pattern CITATION = Pattern.compile("\\[(\\d+)]");

    private final String text;
    private final List<SearchResult> results;
    private final boolean grounded;
    private final List<Integer> invalidCitations;
    private final List<Integer> weakCitations;
}
```

Citation parsing preserves first-seen order and removes duplicates:

```java
public List<Integer> citedIndices() {
    List<Integer> seen = new ArrayList<>();
    Matcher matcher = CITATION.matcher(text);
    while (matcher.find()) {
        int number = Integer.parseInt(matcher.group(1));
        if (!seen.contains(number)) {
            seen.add(number);
        }
    }
    return seen;
}
```

Invalid citations are unambiguous fabrications: the model cited a passage number it was not given.

```java
List<Integer> invalid = initial.citedIndices().stream()
    .filter(number -> number < 1 || number > results.size())
    .toList();
```

Weak citations are a heuristic for the more subtle failure: the answer is right, but the number is
wrong. The code extracts distinctive words from the answer and each passage, ignoring common words.

```java
private static Set<String> distinctiveWords(String text) {
    Set<String> words = new LinkedHashSet<>();
    Matcher matcher = WORD.matcher(text.toLowerCase());
    while (matcher.find()) {
        String word = matcher.group();
        if (word.length() > 2 && !COMMON.contains(word)) {
            words.add(word);
        }
    }
    return words;
}
```

Then it compares the cited passage with the best matching passage:

```java
for (int number : answer.citedIndices()) {
    if (number < 1 || number > overlaps.size()) {
        continue;
    }
    double cited = overlaps.get(number - 1);
    if (cited < 0.34 && best > cited + 0.15) {
        weak.add(number);
    }
}
```

The real observed case is exactly this:

```text
Passage 1: Meridian is 60 percent Brazilian Cerrado.
Passage 2: Cold brew steeps for sixteen hours.
Answer:    Meridian contains 60 percent Brazilian Cerrado beans. [2]
```

The answer words match passage 1 and barely match passage 2, so citation `[2]` is weak. The warning is
not proof, but it tells the user to inspect the context.

### Strip template artifacts

Small chat models sometimes leak prompt-template tokens into their output:

```text
A: Employees receive 27 days of paid annual leave. [/INST]
```

These tokens are never content. Remove them before citation parsing:

```java
private static final List<String> ARTIFACTS = List.of(
    "[/INST]", "[INST]", "<|end|>", "<|endoftext|>", "<|assistant|>",
    "<|user|>", "<|system|>", "</s>", "<s>"
);

public static String stripTemplateArtifacts(String text) {
    String cleaned = text;
    for (String artifact : ARTIFACTS) {
        cleaned = cleaned.replace(artifact, " ");
    }
    return cleaned.replaceAll("[ \\t]{2,}", " ").trim();
}
```

---

## 10. Wiring the pipeline

The pipeline has four jobs: load documents, chunk them, embed chunks in one batch, and save the index.

Document loading now delegates to `dev.kuldeepcodes.customllm.loaders.Loaders`, so text, Markdown, JSON,
JSONL and NDJSON all become the same narrow `(name, text)` shape before chunking:

```java
if (Files.isRegularFile(source)) {
    files = List.of(source);
} else {
    try (Stream<Path> stream = Files.walk(source)) {
        files = stream.filter(Files::isRegularFile)
            .filter(path -> SUPPORTED_SUFFIXES.contains(suffix(path)))
            .sorted()
            .toList();
    }
}
```

Supported suffixes are `.txt`, `.md`, `.markdown`, `.json`, `.jsonl`, and `.ndjson`. Blank files
and empty JSON records are skipped. Relative names are converted to forward slashes so citations are
stable on Windows and Linux.

Ingestion batches embeddings:

```java
List<double[]> vectors = embedder.embed(chunks.stream().map(Chunk::text).toList());
VectorIndex index = new VectorIndex(embedder.name(), vectors.get(0).length);
index.add(chunks, vectors);
index.save(indexPath);
```

Batching is critical. One HTTP request for 500 chunks is dramatically faster than 500 HTTP requests.
If ingest is slow, check that you did not accidentally call `embedOne` inside the chunk loop.

Retrieval enforces compatibility before search:

```java
public static List<SearchResult> retrieve(String question, VectorIndex index, Embedder embedder, int topK) {
    index.ensureCompatible(embedder);
    return index.search(question, embedder.embedOne(question), topK);
}
```

Answering is deliberately small:

```java
public static Answer ask(String question, VectorIndex index, Embedder embedder, ChatClient chat, int topK) {
    if (question == null || question.trim().isEmpty()) {
        throw new IllegalArgumentException("Ask an actual question.");
    }
    return Grounding.answerQuestion(question, retrieve(question, index, embedder, topK), chat);
}
```

Keeping orchestration thin makes each layer independently testable.

---

## 11. The command line

The CLI has six commands:

| Command | What it does |
| --- | --- |
| `ingest <path>` | Chunk, embed, and save an index |
| `ask <question>` | Retrieve, generate or extract, then cite |
| `search <question>` | Show raw retrieval results and score components |
| `info` | Print index metadata and document counts |
| `create-model <name>` | Write/build an Ollama model definition |
| `export` | Write JSONL examples for fine-tuning |

Hand-rolled parsing is enough here and avoids a CLI dependency:

```java
if (!arg.startsWith("--")) {
    positions.add(arg);
} else if (flags.contains(arg.substring(2))) {
    options.put(arg.substring(2), "true");
} else {
    options.put(arg.substring(2), args.get(++i));
}
```

`search` is the debugging command. If the right passage is missing from search results, generation is
not the problem. Fix chunking, embedding, or documents first.

```text
[1] handbook.md:19  score=0.521 (vector=0.361 keyword=1.000)
    Everyone receives 27 days of paid annual leave...
```

The CLI prints invalid citations in red and weak citations in yellow when a console is attached.
Colours are disabled under CI or redirection.

### UTF-8 output on Windows

Force UTF-8 for stdout and stderr:

```java
private static void forceUtf8() {
    System.setOut(new PrintStream(new FileOutputStream(FileDescriptor.out), true, StandardCharsets.UTF_8));
    System.setErr(new PrintStream(new FileOutputStream(FileDescriptor.err), true, StandardCharsets.UTF_8));
}
```

Windows consoles often default to a legacy code page. Documents containing em-dashes, accents, or
currency symbols can print as mojibake or fail on redirect. A document QA tool must assume user
documents contain real Unicode.

---

## 12. Path 2: building a real custom model definition

`create-model` genuinely creates a named Ollama model, but it shapes behaviour rather than storing a
reliable fact database.

The Modelfile builder lists the corpus and writes behavioural rules:

```java
String described = persona != null && !persona.isBlank()
    ? persona
    : "You are a specialist assistant for a knowledge base of " + sources.size()
        + " document(s) covering:\n" + listed.toString().stripTrailing();

return """
    # Generated by customllm. Build with:
    #   ollama create %s -f Modelfile
    FROM %s

    PARAMETER temperature %s
    PARAMETER num_ctx 4096

    SYSTEM \"\"\"%s

    Rules:
    - Answer only from context passages supplied with the question.
    - If the passages do not contain the answer, say so plainly.
    - Cite the passage number for every claim, like [1].
    - Never invent sources, section numbers or figures.\"\"\"
    """.formatted(name, baseModel, temperature, described);
```

Notice what it does not do: it does not paste the full documents into the system prompt. A prompt is
not a database. It is capped by context length, cannot cite, and produced the fabrication shown in
section 1.

Running Ollama is best-effort:

```java
try {
    Process process = new ProcessBuilder("ollama", "create", name, "-f", modelfile.toString()).start();
    if (!process.waitFor(Duration.ofMinutes(10).toSeconds(), TimeUnit.SECONDS)) {
        process.destroyForcibly();
        return new ModelCreation(modelfile, "Wrote " + modelfile + ", but `ollama create` timed out.");
    }
} catch (IOException e) {
    return new ModelCreation(modelfile, "Wrote " + modelfile + ", but the `ollama` command was not found.");
}
```

The command should not crash just because Ollama is not installed. Writing the Modelfile is still
useful.

---

## 13. Path 3: preparing data for fine-tuning

Fine-tuning export writes JSONL records in a standard chat-message shape:

```java
Map<String, Object> record = Map.of(
    "messages", List.of(
        Map.of("role", "system", "content", "You are a specialist assistant for this knowledge base."),
        Map.of("role", "user", "content", "What does " + chunk.source() + " say about this topic?"),
        Map.of("role", "assistant", "content", chunk.text())
    ),
    "source", chunk.citation()
);
writer.write(JSON.writeValueAsString(record));
```

This prepares data; it does not train. Real fine-tuning needs tools such as Unsloth, Axolotl, or
torchtune, plus a GPU and curated examples. Generated examples are scaffolding, not a finished
dataset.

Add `*.jsonl` to `.gitignore`. Training exports are generated artifacts and can contain private
source text.

---

## 14. Testing without a GPU

The hashing embedder makes the full pipeline deterministic:

```java
@Test
void hashingIsDeterministicAcrossInstances() {
    assertThat(new HashingEmbedder().embedOne("the quick brown fox"))
        .containsExactly(new HashingEmbedder().embedOne("the quick brown fox"));
}
```

Generation is tested with a stub `ChatClient`:

```java
private static final class StubChat implements ChatClient {
    private final String reply;

    @Override
    public String complete(String system, String user, double temperature) {
        return reply;
    }
}
```

That lets tests prove refusal, invalid citation detection, and weak citation detection without a
running model.

Test refusal hard:

```java
Answer answer = Grounding.answerQuestion("capital of France", results(0.05), null);
assertThat(answer.text()).isEqualTo(Grounding.REFUSAL);
assertThat(answer.grounded()).isFalse();
```

A system that cannot say "I don't know" is worse than no system.

Run:

```powershell
.\mvnw.cmd clean verify
```

The project currently has 103 deterministic tests.

---

## 15. Making retrieval better

Work in this order when answers disappoint:

1. **Run `search`.** Inspect the passages and score components. If the right passage is absent,
   generation cannot fix it.
2. **Check the embedder.** `info` shows `hashing-512` or `ollama:all-minilm`. Hashing is useful but
   not semantic.
3. **Adjust chunk size.** Long reference documents may need 1200 characters. Dense FAQs may need 400.
4. **Adjust overlap.** Increase overlap when boundary facts disappear. Too much overlap creates many
   near-duplicates.
5. **Raise `--top-k`.** More passages can help, but too many dilute the model's attention.
6. **Improve headings and documents.** Retrieval cannot find what was never written clearly.
7. **Only then change the prompt.** Prompt edits cannot recover missing evidence.

---

## 16. Java-specific gotchas

**Maven Wrapper properties.** After generating the wrapper, inspect `.mvn/wrapper/maven-wrapper.properties`.
PowerShell can mis-parse the Maven version property. The `distributionUrl` must point to Maven 3.9.16.

**Line endings.** Keep `mvnw` LF and `mvnw.cmd` CRLF with `.gitattributes`. CRLF in `mvnw` breaks Linux
CI with `bad interpreter`.

**Fat jar.** Jackson is an external dependency. Without Shade or Assembly, `java -jar` cannot find it.
Set the manifest main class and `<finalName>customllm</finalName>`.

**Stable hashing.** Do not base a persisted embedding format on ad hoc or process-randomized hashes.
This implementation uses SHA-256 and documents how bucket and sign are chosen.

**UTF-8 output.** Wrap stdout and stderr in UTF-8 `PrintStream`s. Windows console defaults are not a
safe assumption for document text.

**Jackson JSON nodes.** Use `JsonNode.numberValue().toString()` for flattened numbers rather than
converting everything through doubles, or integer-looking IDs and counts can pick up `.0` artefacts.
`JsonNode.asText()` is fine for strings, but booleans are rendered explicitly as lower-case `true` and
`false`. Jackson preserves object field order while parsing, and the loader relies on that to keep
flattened records readable and predictable.

**Readable JSON.** Jackson's default pretty printer keeps primitive arrays on one line. This project
serializes vectors as lists and configures an array indenter so the index remains readable.

---

## 17. Troubleshooting

**`Model 'phi3' cannot produce embeddings` or HTTP 501.**
You used a generation model as an embedder. Run `ollama pull all-minilm` and use `--embed-model all-minilm`.

**`Vectors from different models are not comparable`.**
The index was built with one embedder and queried with another. Re-run `ingest` or pass matching
`--embedder` and `--embed-model` flags.

**Everything refuses with `I don't have anything...`.**
Run `info` to confirm chunks exist. Then run `search` and inspect the top score. If scores are just
below 0.25, the documents may be relevant but weakly worded. If scores are near zero, retrieval did
not find evidence.

**The answer contains `[/INST]`, `<|end|>`, or similar.**
That is chat-template leakage. Add the token to `ARTIFACTS` in `Grounding` and keep stripping before
citation parsing.

**The answer is right but the citation number is wrong.**
This is known small-model behaviour. The weak-citation warning exists for it. Re-run with
`--show-context` and inspect the numbered passages.

**Non-ASCII characters are mangled on Windows.**
Make sure `Main.forceUtf8()` still wraps stdout and stderr. Do not rely on the process default
encoding.

**Ingest is slow with Ollama.**
Check that chunks are embedded in one batch: `embedder.embed(chunks.stream().map(Chunk::text).toList())`.
Per-chunk HTTP calls are the usual cause.

**`java -jar target/customllm.jar` cannot find Jackson classes.**
The jar is not shaded. Run `mvn package` with the shade plugin configured, and use `target/customllm.jar`.

**Linux CI says `bad interpreter` for `mvnw`.**
The shell script has CRLF endings. Keep `.gitattributes` and re-check out the file.

---

## 18. Where to go next

- Point the tool at your own documents: `java -jar target/customllm.jar ingest C:\notes`.
- Add document converters for PDF or Word before `loadDocuments`.
- Swap the JSON index for a vector database when exhaustive scan becomes too slow.
- Add reranking: retrieve 20 candidates, then score them with a cross-encoder.
- Try the same design in Python or .NET to see which ideas are language-independent.

The complete working source for every code fragment above is in this repository.
