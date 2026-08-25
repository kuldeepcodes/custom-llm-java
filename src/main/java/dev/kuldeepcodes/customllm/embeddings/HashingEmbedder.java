package dev.kuldeepcodes.customllm.embeddings;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** Deterministic dependency-free bag-of-words embedder. */
public final class HashingEmbedder implements Embedder {
    /** Default output dimensionality. */
    public static final int DEFAULT_DIMENSIONS = 512;

    private static final Pattern TOKEN = Pattern.compile("[a-z0-9]+");
    private static final Set<String> STOPWORDS = Set.of(
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
        "have", "how", "in", "is", "it", "its", "of", "on", "or", "that", "the",
        "this", "to", "was", "what", "when", "where", "which", "who", "why", "with"
    );

    private final int dimensions;

    /** Creates an embedder with 512 dimensions. */
    public HashingEmbedder() {
        this(DEFAULT_DIMENSIONS);
    }

    /**
     * Creates an embedder.
     *
     * @param dimensions output dimensions
     */
    public HashingEmbedder(int dimensions) {
        if (dimensions <= 0) {
            throw new IllegalArgumentException("dimensions must be positive.");
        }
        this.dimensions = dimensions;
    }

    @Override
    public List<double[]> embed(List<String> texts) {
        List<double[]> result = new ArrayList<>(texts.size());
        for (String text : texts) {
            result.add(embedText(text == null ? "" : text));
        }
        return result;
    }

    @Override
    public String name() {
        return "hashing-" + dimensions;
    }

    @Override
    public int dimensions() {
        return dimensions;
    }

    private double[] embedText(String text) {
        double[] vector = new double[dimensions];
        Matcher matcher = TOKEN.matcher(text.toLowerCase());
        MessageDigest digest = sha256();
        while (matcher.find()) {
            String token = matcher.group();
            if (STOPWORDS.contains(token)) {
                continue;
            }
            byte[] hash = digest.digest(token.getBytes(StandardCharsets.UTF_8));
            long head = ((hash[0] & 0xffL) << 24)
                | ((hash[1] & 0xffL) << 16)
                | ((hash[2] & 0xffL) << 8)
                | (hash[3] & 0xffL);
            int bucket = (int) (head % dimensions);
            vector[bucket] += (hash[4] & 1) == 1 ? 1.0 : -1.0;
        }
        return Vectors.normalize(vector);
    }

    private static MessageDigest sha256() {
        try {
            return MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 is required by Java.", e);
        }
    }
}

