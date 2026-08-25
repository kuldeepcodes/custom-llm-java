package dev.kuldeepcodes.customllm.index;

import dev.kuldeepcodes.customllm.chunking.Chunk;

/**
 * A retrieved chunk and why it was retrieved.
 *
 * @param chunk retrieved chunk
 * @param score blended score
 * @param vectorScore vector component
 * @param keywordScore keyword component
 */
public record SearchResult(Chunk chunk, double score, double vectorScore, double keywordScore) {
}
