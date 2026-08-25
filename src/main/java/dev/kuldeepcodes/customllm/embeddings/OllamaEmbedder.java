package dev.kuldeepcodes.customllm.embeddings;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/** Embeddings from a local Ollama model such as {@code all-minilm}. */
public final class OllamaEmbedder implements Embedder {
    /** Default Ollama URL. */
    public static final String DEFAULT_OLLAMA_URL = "http://localhost:11434";
    /** Default embedding model. */
    public static final String DEFAULT_EMBED_MODEL = "all-minilm";

    private static final ObjectMapper JSON = new ObjectMapper();

    private final String model;
    private final String baseUrl;
    private final Duration timeout;
    private final HttpClient client;
    private int dimensions;

    /** Creates the default Ollama embedder. */
    public OllamaEmbedder() {
        this(DEFAULT_EMBED_MODEL, DEFAULT_OLLAMA_URL);
    }

    /**
     * Creates an Ollama embedder.
     *
     * @param model model name
     * @param baseUrl Ollama base URL
     */
    public OllamaEmbedder(String model, String baseUrl) {
        this(model, baseUrl, Duration.ofSeconds(120));
    }

    /**
     * Creates an Ollama embedder with timeout.
     *
     * @param model model name
     * @param baseUrl Ollama base URL
     * @param timeout request timeout
     */
    public OllamaEmbedder(String model, String baseUrl, Duration timeout) {
        this.model = model;
        this.baseUrl = stripSlash(baseUrl);
        this.timeout = timeout;
        this.client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(10)).build();
    }

    @Override
    public List<double[]> embed(List<String> texts) {
        if (texts.isEmpty()) {
            return List.of();
        }
        try {
            String body = JSON.writeValueAsString(Map.of("model", model, "input", texts));
            HttpRequest request = HttpRequest.newBuilder(URI.create(baseUrl + "/api/embed"))
                .timeout(timeout)
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(body))
                .build();
            HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
            if (response.statusCode() == 501) {
                throw new IllegalStateException(
                    "Model '" + model + "' cannot produce embeddings. Generation models "
                        + "such as phi3 are not embedding models. Run: ollama pull all-minilm"
                );
            }
            if (response.statusCode() >= 400) {
                throw new IllegalStateException(
                    "Ollama embedding request failed (" + response.statusCode() + "): " + response.body()
                );
            }
            return parseEmbeddings(response.body(), texts.size());
        } catch (IOException e) {
            throw new IllegalStateException("Ollama embedding request failed: " + e.getMessage(), e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Ollama embedding request was interrupted.", e);
        }
    }

    @Override
    public String name() {
        return "ollama:" + model;
    }

    @Override
    public int dimensions() {
        if (dimensions == 0) {
            dimensions = embedOne("dimension probe").length;
        }
        return dimensions;
    }

    /**
     * Checks whether Ollama is reachable and the model is present.
     *
     * @param model model name
     * @param baseUrl Ollama base URL
     * @return true when available
     */
    public static boolean isAvailable(String model, String baseUrl) {
        try {
            HttpClient client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(5)).build();
            HttpRequest request = HttpRequest.newBuilder(URI.create(stripSlash(baseUrl) + "/api/tags"))
                .timeout(Duration.ofSeconds(5))
                .GET()
                .build();
            HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
            if (response.statusCode() >= 400) {
                return false;
            }
            for (JsonNode item : JSON.readTree(response.body()).path("models")) {
                String name = item.path("name").asText("");
                if (name.equals(model) || name.startsWith(model + ":")) {
                    return true;
                }
            }
            return false;
        } catch (IOException | InterruptedException | IllegalArgumentException e) {
            if (e instanceof InterruptedException) {
                Thread.currentThread().interrupt();
            }
            return false;
        }
    }

    /**
     * Checks default model availability.
     *
     * @return true when available
     */
    public static boolean isAvailable() {
        return isAvailable(DEFAULT_EMBED_MODEL, DEFAULT_OLLAMA_URL);
    }

    private List<double[]> parseEmbeddings(String body, int expected) throws IOException {
        JsonNode embeddings = JSON.readTree(body).path("embeddings");
        if (!embeddings.isArray() || embeddings.isEmpty()) {
            throw new IllegalStateException("Ollama returned no embeddings for " + expected + " input(s).");
        }
        List<double[]> vectors = new ArrayList<>();
        for (JsonNode node : embeddings) {
            double[] vector = new double[node.size()];
            for (int i = 0; i < node.size(); i++) {
                vector[i] = node.get(i).asDouble();
            }
            vectors.add(Vectors.normalize(vector));
        }
        dimensions = vectors.get(0).length;
        return vectors;
    }

    private static String stripSlash(String value) {
        String stripped = value;
        while (stripped.endsWith("/")) {
            stripped = stripped.substring(0, stripped.length() - 1);
        }
        return stripped;
    }
}

