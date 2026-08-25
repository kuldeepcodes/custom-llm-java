package dev.kuldeepcodes.customllm.index;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.core.util.DefaultIndenter;
import com.fasterxml.jackson.core.util.DefaultPrettyPrinter;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import dev.kuldeepcodes.customllm.chunking.Chunk;
import dev.kuldeepcodes.customllm.embeddings.Embedder;
import dev.kuldeepcodes.customllm.embeddings.Vectors;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** In-memory vector index persisted as readable JSON. */
public final class VectorIndex {
    /** On-disk index version. */
    public static final int INDEX_VERSION = 1;

    private static final double VECTOR_WEIGHT = 0.75;
    private static final double KEYWORD_WEIGHT = 0.25;
    private static final Pattern TOKEN = Pattern.compile("[a-z0-9]+");
    private static final ObjectMapper JSON = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);
    private static final DefaultPrettyPrinter PRETTY = new DefaultPrettyPrinter()
        .withArrayIndenter(DefaultIndenter.SYSTEM_LINEFEED_INSTANCE);

    private final String embedderName;
    private final int dimensions;
    private final List<Chunk> chunks = new ArrayList<>();
    private final List<double[]> vectors = new ArrayList<>();

    /**
     * Creates an empty index.
     *
     * @param embedderName embedder identity
     * @param dimensions vector dimensions
     */
    public VectorIndex(String embedderName, int dimensions) {
        this.embedderName = embedderName;
        this.dimensions = dimensions;
    }

    /** @return embedder identity */
    public String embedderName() {
        return embedderName;
    }

    /** @return vector dimensions */
    public int dimensions() {
        return dimensions;
    }

    /** @return chunk count */
    public int size() {
        return chunks.size();
    }

    /** @return chunks in insertion order */
    public List<Chunk> chunks() {
        return List.copyOf(chunks);
    }

    /** @return source names in sorted order */
    public List<String> sources() {
        TreeSet<String> sources = new TreeSet<>();
        for (Chunk chunk : chunks) {
            sources.add(chunk.source());
        }
        return List.copyOf(sources);
    }

    /**
     * Adds chunks with precomputed vectors.
     *
     * @param newChunks chunks
     * @param newVectors vectors
     */
    public void add(List<Chunk> newChunks, List<double[]> newVectors) {
        if (newChunks.size() != newVectors.size()) {
            throw new IllegalArgumentException(
                "Got " + newChunks.size() + " chunks but " + newVectors.size() + " vectors."
            );
        }
        for (int i = 0; i < newChunks.size(); i++) {
            Chunk chunk = newChunks.get(i);
            double[] vector = newVectors.get(i);
            if (vector.length != dimensions) {
                throw new IllegalArgumentException(
                    "Vector for " + chunk.citation() + " has " + vector.length
                        + " dimensions, expected " + dimensions + "."
                );
            }
            chunks.add(chunk);
            vectors.add(vector.clone());
        }
    }

    /**
     * Searches for best matches.
     *
     * @param query query text
     * @param queryVector embedded query
     * @param topK maximum results
     * @return ranked results
     */
    public List<SearchResult> search(String query, double[] queryVector, int topK) {
        if (topK <= 0) {
            throw new IllegalArgumentException("top_k must be positive.");
        }
        if (chunks.isEmpty()) {
            return List.of();
        }
        Set<String> queryTokens = tokenise(query);
        List<SearchResult> results = new ArrayList<>();
        for (int i = 0; i < chunks.size(); i++) {
            Chunk chunk = chunks.get(i);
            double vectorScore = Vectors.cosineSimilarity(queryVector, vectors.get(i));
            double keywordScore = keywordOverlap(queryTokens, tokenise(chunk.text()));
            double score = VECTOR_WEIGHT * vectorScore + KEYWORD_WEIGHT * keywordScore;
            results.add(new SearchResult(chunk, score, vectorScore, keywordScore));
        }
        results.sort(Comparator.comparingDouble(SearchResult::score).reversed()
            .thenComparing(result -> result.chunk().source())
            .thenComparingInt(result -> result.chunk().index()));
        return results.stream().limit(topK).toList();
    }

    /**
     * Saves the index.
     *
     * @param path output path
     * @throws IOException if writing fails
     */
    public void save(Path path) throws IOException {
        Path parent = path.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        List<List<Double>> serializableVectors = vectors.stream()
            .map(vector -> java.util.Arrays.stream(vector).boxed().toList())
            .toList();
        JSON.writer(PRETTY).writeValue(path.toFile(),
            new Payload(INDEX_VERSION, embedderName, dimensions, chunks, serializableVectors));
    }

    /**
     * Loads an index.
     *
     * @param path JSON path
     * @return loaded index
     * @throws IOException if reading fails
     */
    public static VectorIndex load(Path path) throws IOException {
        if (!Files.exists(path)) {
            throw new FileNotFoundException("No index at " + path + ". Run `customllm ingest` first.");
        }
        Payload payload = JSON.readValue(path.toFile(), Payload.class);
        if (payload.version() != INDEX_VERSION) {
            throw new IllegalArgumentException(
                "Index at " + path + " is version " + payload.version()
                    + ", this build expects " + INDEX_VERSION + ". Re-run `customllm ingest`."
            );
        }
        VectorIndex index = new VectorIndex(payload.embedder(), payload.dimensions());
        List<double[]> loadedVectors = payload.vectors().stream()
            .map(values -> values.stream().mapToDouble(Double::doubleValue).toArray())
            .toList();
        index.add(payload.chunks(), loadedVectors);
        return index;
    }

    /**
     * Ensures query vectors are comparable with indexed vectors.
     *
     * @param embedder query embedder
     */
    public void ensureCompatible(Embedder embedder) {
        if (!embedder.name().equals(embedderName)) {
            throw new IllegalArgumentException(
                "This index was built with '" + embedderName + "' but you are querying with '"
                    + embedder.name() + "'. Vectors from different models are not comparable. "
                    + "Re-run `customllm ingest` with the same embedder."
            );
        }
    }

    private static Set<String> tokenise(String text) {
        Set<String> tokens = new LinkedHashSet<>();
        Matcher matcher = TOKEN.matcher(text.toLowerCase());
        while (matcher.find()) {
            tokens.add(matcher.group());
        }
        return tokens;
    }

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

    private record Payload(
        int version,
        String embedder,
        int dimensions,
        List<Chunk> chunks,
        List<List<Double>> vectors
    ) {
        @JsonCreator
        private Payload(
            @JsonProperty("version") int version,
            @JsonProperty("embedder") String embedder,
            @JsonProperty("dimensions") int dimensions,
            @JsonProperty("chunks") List<Chunk> chunks,
            @JsonProperty("vectors") List<List<Double>> vectors
        ) {
            this.version = version;
            this.embedder = embedder;
            this.dimensions = dimensions;
            this.chunks = chunks == null ? List.of() : chunks;
            this.vectors = vectors == null ? List.of() : vectors;
        }
    }
}
