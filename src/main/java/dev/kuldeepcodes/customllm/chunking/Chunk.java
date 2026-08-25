package dev.kuldeepcodes.customllm.chunking;

/**
 * One retrievable passage, with enough provenance to cite it.
 *
 * @param text passage text
 * @param source document name
 * @param index zero-based chunk index
 * @param startLine one-based start line
 */
public record Chunk(String text, String source, int index, int startLine) {
    /**
     * Returns a compact citation such as {@code handbook.md:12}.
     *
     * @return source and line reference
     */
    public String citation() {
        return source + ":" + startLine;
    }
}

