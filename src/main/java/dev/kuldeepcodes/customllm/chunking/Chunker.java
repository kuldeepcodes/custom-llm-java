package dev.kuldeepcodes.customllm.chunking;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;

/** Sentence-aware chunking with configurable overlap. */
public final class Chunker {
    /** Default soft character limit. */
    public static final int DEFAULT_CHUNK_SIZE = 800;
    /** Default carried overlap. */
    public static final int DEFAULT_OVERLAP = 150;

    private static final Pattern SENTENCE_END = Pattern.compile("(?<=[.!?])\\s+|\\n{2,}");

    private Chunker() {
    }

    /**
     * Splits text into sentence-ish units.
     *
     * @param text input text
     * @return non-empty sentence units
     */
    public static List<String> splitSentences(String text) {
        List<String> sentences = new ArrayList<>();
        for (String part : SENTENCE_END.split(text)) {
            String trimmed = part.trim();
            if (!trimmed.isEmpty()) {
                sentences.add(trimmed);
            }
        }
        return sentences;
    }

    /**
     * Splits text with default settings.
     *
     * @param text document text
     * @param source citation source
     * @return chunks
     */
    public static List<Chunk> chunkText(String text, String source) {
        return chunkText(text, source, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP);
    }

    /**
     * Splits text into sentence-aligned chunks with overlap.
     *
     * @param text document text
     * @param source citation source
     * @param chunkSize soft character budget
     * @param overlap carried character budget
     * @return chunks in order
     */
    public static List<Chunk> chunkText(String text, String source, int chunkSize, int overlap) {
        if (text == null || text.trim().isEmpty()) {
            return List.of();
        }
        if (chunkSize <= 0) {
            throw new IllegalArgumentException("chunk_size must be positive.");
        }
        if (overlap < 0 || overlap >= chunkSize) {
            throw new IllegalArgumentException("overlap must be >= 0 and smaller than chunk_size.");
        }

        List<String> sentences = splitSentences(text);
        if (sentences.isEmpty()) {
            return List.of();
        }
        List<Integer> lines = sentenceLineNumbers(text, sentences);
        List<Chunk> chunks = new ArrayList<>();
        List<String> current = new ArrayList<>();
        int currentLength = 0;
        int firstSentence = 0;

        for (int position = 0; position < sentences.size(); position++) {
            String sentence = sentences.get(position);
            int addition = sentence.length() + (current.isEmpty() ? 0 : 1);
            if (!current.isEmpty() && currentLength + addition > chunkSize) {
                chunks.add(build(current, source, chunks.size(), lines.get(firstSentence)));
                Carry carry = carryOver(current, position, overlap, firstSentence);
                current = new ArrayList<>(carry.sentences());
                currentLength = current.stream().mapToInt(String::length).sum()
                    + Math.max(0, current.size() - 1);
                firstSentence = carry.firstSentence();
            }
            if (current.isEmpty()) {
                firstSentence = position;
            }
            current.add(sentence);
            currentLength += addition;
        }
        if (!current.isEmpty()) {
            chunks.add(build(current, source, chunks.size(), lines.get(firstSentence)));
        }
        return chunks;
    }

    private static Chunk build(List<String> sentences, String source, int index, int startLine) {
        return new Chunk(String.join(" ", sentences), source, index, startLine);
    }

    private static Carry carryOver(List<String> sentences, int next, int overlap, int first) {
        if (overlap == 0) {
            return new Carry(List.of(), next);
        }
        List<String> carried = new ArrayList<>();
        int size = 0;
        for (int i = sentences.size() - 1; i >= 0; i--) {
            String sentence = sentences.get(i);
            if (size + sentence.length() > overlap && !carried.isEmpty()) {
                break;
            }
            carried.add(0, sentence);
            size += sentence.length() + 1;
        }
        return new Carry(carried, first + sentences.size() - carried.size());
    }

    private static List<Integer> sentenceLineNumbers(String text, List<String> sentences) {
        List<Integer> result = new ArrayList<>();
        int cursor = 0;
        for (String sentence : sentences) {
            String probe = sentence.substring(0, Math.min(sentence.length(), 40));
            int found = text.indexOf(probe, cursor);
            if (found < 0) {
                found = cursor;
            }
            result.add((int) text.substring(0, found).chars().filter(c -> c == '\n').count() + 1);
            cursor = found + 1;
        }
        return result;
    }

    private record Carry(List<String> sentences, int firstSentence) {
    }
}
