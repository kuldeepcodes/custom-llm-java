package dev.kuldeepcodes.customllm.embeddings;

/** Vector math helpers. */
public final class Vectors {
    private Vectors() {
    }

    /**
     * Computes cosine similarity.
     *
     * @param a first vector
     * @param b second vector
     * @return cosine similarity, or zero for zero vectors
     */
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

    /**
     * Returns a normalized copy of a vector.
     *
     * @param vector vector to normalize
     * @return normalized vector
     */
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
}

