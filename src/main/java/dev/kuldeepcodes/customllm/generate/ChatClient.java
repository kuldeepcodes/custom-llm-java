package dev.kuldeepcodes.customllm.generate;

/** Minimal chat abstraction for tests and Ollama. */
@FunctionalInterface
public interface ChatClient {
    /**
     * Completes a system/user exchange.
     *
     * @param system system prompt
     * @param user user prompt
     * @param temperature sampling temperature
     * @return reply text
     */
    String complete(String system, String user, double temperature);
}

