package dev.kuldeepcodes.customllm.pipeline;

import com.fasterxml.jackson.databind.ObjectMapper;
import dev.kuldeepcodes.customllm.chunking.Chunk;
import dev.kuldeepcodes.customllm.chunking.Chunker;
import dev.kuldeepcodes.customllm.embeddings.Embedder;
import dev.kuldeepcodes.customllm.generate.Answer;
import dev.kuldeepcodes.customllm.generate.ChatClient;
import dev.kuldeepcodes.customllm.generate.Grounding;
import dev.kuldeepcodes.customllm.index.SearchResult;
import dev.kuldeepcodes.customllm.index.VectorIndex;
import dev.kuldeepcodes.customllm.loaders.Loaders;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.TimeUnit;

/** End-to-end RAG pipeline and custom-model/fine-tuning helpers. */
public final class Pipeline {
    /** Supported document extensions. */
    public static final Set<String> SUPPORTED_SUFFIXES = Loaders.SUPPORTED_SUFFIXES;
    /** Default index path. */
    public static final Path DEFAULT_INDEX_PATH = Path.of(".customllm", "index.json");

    private static final ObjectMapper JSON = new ObjectMapper();

    private Pipeline() {
    }

    /**
     * Loads supported documents from a file or directory.
     *
     * @param source file or directory
     * @return document records sorted by name
     * @throws IOException when reading fails
     */
    public static List<Document> loadDocuments(Path source) throws IOException {
        return Loaders.loadDocuments(source).stream()
            .map(document -> new Document(document.name(), document.text()))
            .toList();
    }

    /**
     * Chunks, embeds, and persists documents.
     *
     * @param source source file or directory
     * @param embedder embedder
     * @param indexPath output path
     * @param chunkSize chunk size
     * @param overlap overlap size
     * @return ingest report
     * @throws IOException on IO failure
     */
    public static IngestReport ingest(
        Path source,
        Embedder embedder,
        Path indexPath,
        int chunkSize,
        int overlap
    ) throws IOException {
        List<Document> documents = loadDocuments(source);
        if (documents.isEmpty()) {
            throw new IllegalArgumentException("No readable documents in " + source
                + ". Supported extensions: .json, .jsonl, .markdown, .md, .ndjson, .txt");
        }
        List<Chunk> chunks = new ArrayList<>();
        List<String> skipped = new ArrayList<>();
        for (Document document : documents) {
            List<Chunk> made = Chunker.chunkText(document.text(), document.name(), chunkSize, overlap);
            if (made.isEmpty()) {
                skipped.add(document.name());
            } else {
                chunks.addAll(made);
            }
        }
        if (chunks.isEmpty()) {
            throw new IllegalArgumentException("Every document produced zero chunks; is the corpus empty?");
        }
        List<double[]> vectors = embedder.embed(chunks.stream().map(Chunk::text).toList());
        VectorIndex index = new VectorIndex(embedder.name(), vectors.get(0).length);
        index.add(chunks, vectors);
        index.save(indexPath);
        return new IngestReport(documents.size(), chunks.size(), embedder.name(), index.dimensions(),
            indexPath, List.copyOf(skipped));
    }

    /**
     * Retrieves passages for a question.
     *
     * @param question question
     * @param index index
     * @param embedder compatible embedder
     * @param topK maximum results
     * @return search results
     */
    public static List<SearchResult> retrieve(String question, VectorIndex index, Embedder embedder, int topK) {
        index.ensureCompatible(embedder);
        return index.search(question, embedder.embedOne(question), topK);
    }

    /**
     * Retrieves and answers.
     *
     * @param question question
     * @param index index
     * @param embedder compatible embedder
     * @param chat optional chat client
     * @param topK maximum retrieval results
     * @return answer
     */
    public static Answer ask(String question, VectorIndex index, Embedder embedder, ChatClient chat, int topK) {
        if (question == null || question.trim().isEmpty()) {
            throw new IllegalArgumentException("Ask an actual question.");
        }
        return Grounding.answerQuestion(question, retrieve(question, index, embedder, topK), chat);
    }

    /**
     * Builds an Ollama Modelfile.
     *
     * @param name model name
     * @param baseModel base model
     * @param index index
     * @param persona optional persona
     * @param temperature temperature
     * @return Modelfile content
     */
    public static String buildModelfile(
        String name,
        String baseModel,
        VectorIndex index,
        String persona,
        double temperature
    ) {
        if (name == null || name.trim().isEmpty()) {
            throw new IllegalArgumentException("The model needs a name.");
        }
        List<String> sources = index.sources();
        StringBuilder listed = new StringBuilder();
        for (String source : sources.stream().limit(20).toList()) {
            listed.append("- ").append(source).append('\n');
        }
        if (sources.size() > 20) {
            listed.append("- ...and ").append(sources.size() - 20).append(" more\n");
        }
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

            SYSTEM \"%s

            Rules:
            - Answer only from context passages supplied with the question.
            - If the passages do not contain the answer, say so plainly.
            - Cite the passage number for every claim, like [1].
            - Never invent sources, section numbers or figures.\"\"\"
            """.formatted(name, baseModel, temperature, described).replace("SYSTEM \"", "SYSTEM \"\"\"");
    }

    /**
     * Writes a Modelfile and optionally runs Ollama.
     *
     * @param name model name
     * @param baseModel base model
     * @param index index
     * @param outputDir output directory
     * @param persona optional persona
     * @param runOllama whether to run Ollama
     * @return result
     * @throws IOException when writing fails
     */
    public static ModelCreation createModel(
        String name,
        String baseModel,
        VectorIndex index,
        Path outputDir,
        String persona,
        boolean runOllama
    ) throws IOException {
        Files.createDirectories(outputDir);
        Path modelfile = outputDir.resolve("Modelfile");
        Files.writeString(modelfile, buildModelfile(name, baseModel, index, persona, 0.2), StandardCharsets.UTF_8);
        if (!runOllama) {
            return new ModelCreation(modelfile,
                "Wrote " + modelfile + ". Build it with: ollama create " + name + " -f " + modelfile);
        }
        try {
            Process process = new ProcessBuilder("ollama", "create", name, "-f", modelfile.toString()).start();
            if (!process.waitFor(Duration.ofMinutes(10).toSeconds(), TimeUnit.SECONDS)) {
                process.destroyForcibly();
                return new ModelCreation(modelfile, "Wrote " + modelfile + ", but `ollama create` timed out.");
            }
            String out = new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8).trim();
            String err = new String(process.getErrorStream().readAllBytes(), StandardCharsets.UTF_8).trim();
            if (process.exitValue() != 0) {
                return new ModelCreation(modelfile, "Wrote " + modelfile + ", but `ollama create` failed: "
                    + (err.isBlank() ? out : err));
            }
            return new ModelCreation(modelfile, "Created model '" + name + "'. Try: ollama run " + name);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return new ModelCreation(modelfile, "Wrote " + modelfile + ", but `ollama create` was interrupted.");
        } catch (IOException e) {
            return new ModelCreation(modelfile, "Wrote " + modelfile + ", but the `ollama` command was not found. "
                + "Install Ollama, then run: ollama create " + name + " -f " + modelfile);
        }
    }

    /**
     * Exports JSONL training examples.
     *
     * @param index index
     * @param output output path
     * @return written records
     * @throws IOException when writing fails
     */
    public static int exportTrainingData(VectorIndex index, Path output) throws IOException {
        Path parent = output.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        int written = 0;
        try (BufferedWriter writer = Files.newBufferedWriter(output, StandardCharsets.UTF_8)) {
            for (Chunk chunk : index.chunks()) {
                Map<String, Object> record = Map.of(
                    "messages", List.of(
                        Map.of("role", "system", "content", "You are a specialist assistant for this knowledge base."),
                        Map.of("role", "user", "content", "What does " + chunk.source() + " say about this topic?"),
                        Map.of("role", "assistant", "content", chunk.text())
                    ),
                    "source", chunk.citation()
                );
                writer.write(JSON.writeValueAsString(record));
                writer.newLine();
                written++;
            }
        }
        return written;
    }

    /**
     * Loaded document.
     *
     * @param name citation-friendly name
     * @param text document text
     */
    public record Document(String name, String text) {
    }

    /**
     * Model creation status.
     *
     * @param modelfile written Modelfile
     * @param message status message
     */
    public record ModelCreation(Path modelfile, String message) {
    }
}
