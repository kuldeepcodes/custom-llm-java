package dev.kuldeepcodes.customllm.generate;

import dev.kuldeepcodes.customllm.index.SearchResult;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** Grounded generation and citation auditing. */
public final class Grounding {
    /** Minimum top retrieval score for attempting an answer. */
    public static final double RELEVANCE_FLOOR = 0.25;
    /** Refusal text for unsupported questions. */
    public static final String REFUSAL = "I don't have anything in the indexed documents that answers that.";
    /** System prompt for grounded answers. */
    public static final String SYSTEM_PROMPT = """
        You answer questions using only the numbered passages provided by the user.

        Rules:
        - Use only information present in the passages. Do not add outside knowledge.
        - Mark every claim with the number of the passage it came from, like [1] or [2].
        - If the passages do not contain the answer, reply exactly: NOT_IN_CONTEXT
        - Never invent a passage number you were not given.
        - Quote figures, dates and names exactly as they appear.
        - Answer in at most four sentences.""";

    private static final List<String> ARTIFACTS = List.of(
        "[/INST]", "[INST]", "<|end|>", "<|endoftext|>", "<|assistant|>",
        "<|user|>", "<|system|>", "</s>", "<s>"
    );
    private static final Pattern WORD = Pattern.compile("[a-z0-9]+");
    private static final Set<String> COMMON = Set.of(
        "a", "about", "all", "also", "an", "and", "any", "are", "as", "at", "be",
        "been", "but", "by", "can", "do", "does", "for", "from", "get", "given",
        "has", "have", "how", "if", "in", "is", "it", "its", "may", "more", "must",
        "no", "not", "of", "on", "or", "our", "per", "that", "the", "their", "them",
        "then", "there", "these", "they", "this", "to", "up", "use", "was", "we",
        "what", "when", "where", "which", "who", "why", "will", "with", "you", "your"
    );

    private Grounding() {
    }

    /**
     * Builds a numbered-passage prompt.
     *
     * @param question question
     * @param results retrieved passages
     * @return prompt
     */
    public static String buildPrompt(String question, List<SearchResult> results) {
        StringBuilder builder = new StringBuilder("Passages:\n\n");
        for (int i = 0; i < results.size(); i++) {
            SearchResult result = results.get(i);
            builder.append('[').append(i + 1).append("] (")
                .append(result.chunk().citation()).append(") ")
                .append(result.chunk().text()).append("\n\n");
        }
        return builder.append("Question: ").append(question).toString();
    }

    /**
     * Produces an answer or a refusal.
     *
     * @param question question
     * @param results retrieved passages
     * @param chat chat client, or null for extractive mode
     * @return answer
     */
    public static Answer answerQuestion(String question, List<SearchResult> results, ChatClient chat) {
        return answerQuestion(question, results, chat, RELEVANCE_FLOOR);
    }

    /**
     * Produces an answer or a refusal with a custom relevance floor.
     *
     * @param question question
     * @param results retrieved passages
     * @param chat chat client, or null for extractive mode
     * @param floor relevance floor
     * @return answer
     */
    public static Answer answerQuestion(
        String question,
        List<SearchResult> results,
        ChatClient chat,
        double floor
    ) {
        if (results.isEmpty() || results.get(0).score() < floor) {
            return new Answer(REFUSAL, results, false, List.of(), List.of());
        }
        if (chat == null) {
            return new Answer(results.get(0).chunk().text() + " [1]", results);
        }
        String raw = stripTemplateArtifacts(chat.complete(SYSTEM_PROMPT, buildPrompt(question, results), 0.0));
        if (raw.isBlank() || raw.toUpperCase().contains("NOT_IN_CONTEXT")) {
            return new Answer(REFUSAL, results, false, List.of(), List.of());
        }
        Answer initial = new Answer(raw, results);
        List<Integer> invalid = initial.citedIndices().stream()
            .filter(number -> number < 1 || number > results.size())
            .toList();
        return new Answer(raw, results, true, invalid, weakCitations(initial));
    }

    /**
     * Removes leaked chat template tokens.
     *
     * @param text raw text
     * @return cleaned text
     */
    public static String stripTemplateArtifacts(String text) {
        String cleaned = text;
        for (String artifact : ARTIFACTS) {
            cleaned = cleaned.replace(artifact, " ");
        }
        return cleaned.replaceAll("[ \\t]{2,}", " ").trim();
    }

    private static List<Integer> weakCitations(Answer answer) {
        Set<String> answerWords = distinctiveWords(answer.text());
        if (answerWords.size() < 3) {
            return List.of();
        }
        List<Double> overlaps = answer.results().stream()
            .map(result -> overlap(answerWords, distinctiveWords(result.chunk().text())))
            .toList();
        if (overlaps.isEmpty()) {
            return List.of();
        }
        double best = overlaps.stream().mapToDouble(Double::doubleValue).max().orElse(0.0);
        List<Integer> weak = new ArrayList<>();
        for (int number : answer.citedIndices()) {
            if (number < 1 || number > overlaps.size()) {
                continue;
            }
            double cited = overlaps.get(number - 1);
            if (cited < 0.34 && best > cited + 0.15) {
                weak.add(number);
            }
        }
        return weak;
    }

    private static Set<String> distinctiveWords(String text) {
        Set<String> words = new LinkedHashSet<>();
        Matcher matcher = WORD.matcher(text.toLowerCase());
        while (matcher.find()) {
            String word = matcher.group();
            if (word.length() > 2 && !COMMON.contains(word)) {
                words.add(word);
            }
        }
        return words;
    }

    private static double overlap(Set<String> answerWords, Set<String> passageWords) {
        int shared = 0;
        for (String word : answerWords) {
            if (passageWords.contains(word)) {
                shared++;
            }
        }
        return answerWords.isEmpty() ? 0.0 : (double) shared / answerWords.size();
    }
}
