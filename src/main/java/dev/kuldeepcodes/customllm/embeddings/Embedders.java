package dev.kuldeepcodes.customllm.embeddings;

/** Factory for embedder selection. */
public final class Embedders {
    private Embedders() {
    }

    /**
     * Creates an embedder.
     *
     * @param prefer auto, ollama, or hashing
     * @param model Ollama model
     * @param baseUrl Ollama base URL
     * @return selected embedder
     */
    public static Embedder create(String prefer, String model, String baseUrl) {
        String choice = prefer == null ? "auto" : prefer.toLowerCase();
        return switch (choice) {
            case "hashing" -> new HashingEmbedder();
            case "ollama" -> new OllamaEmbedder(model, baseUrl);
            case "auto" -> OllamaEmbedder.isAvailable(model, baseUrl)
                ? new OllamaEmbedder(model, baseUrl)
                : new HashingEmbedder();
            default -> throw new IllegalArgumentException(
                "Unknown embedder '" + prefer + "'. Use auto, ollama or hashing."
            );
        };
    }
}
