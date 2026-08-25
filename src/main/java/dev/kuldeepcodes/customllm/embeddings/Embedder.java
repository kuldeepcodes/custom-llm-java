package dev.kuldeepcodes.customllm.embeddings;

import java.util.List;

/** Turns text into fixed-length vectors. */
public interface Embedder {
    /**
     * Embeds a batch in input order.
     *
     * @param texts input texts
     * @return vectors
     */
    List<double[]> embed(List<String> texts);

    /**
     * Returns the stable embedder identity stored in the index.
     *
     * @return name
     */
    String name();

    /**
     * Returns vector dimensionality.
     *
     * @return dimensions
     */
    int dimensions();

    /**
     * Embeds a single text.
     *
     * @param text input text
     * @return vector
     */
    default double[] embedOne(String text) {
        return embed(List.of(text)).get(0);
    }
}

