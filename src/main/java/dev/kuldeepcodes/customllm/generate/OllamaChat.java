package dev.kuldeepcodes.customllm.generate;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.List;
import java.util.Map;

/** Minimal client for Ollama's chat API. */
public final class OllamaChat implements ChatClient {
    /** Default chat model. */
    public static final String DEFAULT_CHAT_MODEL = "phi3";
    /** Default Ollama URL. */
    public static final String DEFAULT_OLLAMA_URL = "http://localhost:11434";

    private static final ObjectMapper JSON = new ObjectMapper();

    private final String model;
    private final String baseUrl;
    private final Duration timeout;
    private final HttpClient client;

    /** Creates a default client. */
    public OllamaChat() {
        this(DEFAULT_CHAT_MODEL, DEFAULT_OLLAMA_URL, Duration.ofSeconds(300));
    }

    /**
     * Creates a client.
     *
     * @param model chat model
     * @param baseUrl Ollama base URL
     */
    public OllamaChat(String model, String baseUrl) {
        this(model, baseUrl, Duration.ofSeconds(300));
    }

    /**
     * Creates a client with timeout.
     *
     * @param model chat model
     * @param baseUrl Ollama base URL
     * @param timeout request timeout
     */
    public OllamaChat(String model, String baseUrl, Duration timeout) {
        this.model = model;
        this.baseUrl = stripSlash(baseUrl);
        this.timeout = timeout;
        this.client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(10)).build();
    }

    @Override
    public String complete(String system, String user, double temperature) {
        try {
            Map<String, Object> payload = Map.of(
                "model", model,
                "stream", false,
                "messages", List.of(
                    Map.of("role", "system", "content", system),
                    Map.of("role", "user", "content", user)
                ),
                "options", Map.of("temperature", temperature, "num_ctx", 4096)
            );
            HttpRequest request = HttpRequest.newBuilder(URI.create(baseUrl + "/api/chat"))
                .timeout(timeout)
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(JSON.writeValueAsString(payload)))
                .build();
            HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
            if (response.statusCode() >= 400) {
                throw new IllegalStateException("Ollama chat failed (" + response.statusCode() + "): "
                    + response.body());
            }
            return JSON.readTree(response.body()).path("message").path("content").asText("").trim();
        } catch (IOException e) {
            throw new IllegalStateException("Ollama chat failed: " + e.getMessage(), e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Ollama chat was interrupted.", e);
        }
    }

    /**
     * Checks whether Ollama is reachable.
     *
     * @param baseUrl base URL
     * @return true when tags endpoint responds
     */
    public static boolean isAvailable(String baseUrl) {
        try {
            HttpClient client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(5)).build();
            HttpRequest request = HttpRequest.newBuilder(URI.create(stripSlash(baseUrl) + "/api/tags"))
                .timeout(Duration.ofSeconds(5)).GET().build();
            return client.send(request, HttpResponse.BodyHandlers.ofString()).statusCode() < 400;
        } catch (IOException | InterruptedException | IllegalArgumentException e) {
            if (e instanceof InterruptedException) {
                Thread.currentThread().interrupt();
            }
            return false;
        }
    }

    private static String stripSlash(String value) {
        String stripped = value;
        while (stripped.endsWith("/")) {
            stripped = stripped.substring(0, stripped.length() - 1);
        }
        return stripped;
    }
}
