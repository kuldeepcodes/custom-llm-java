package dev.kuldeepcodes.customllm.generate;

import dev.kuldeepcodes.customllm.index.SearchResult;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** A generated answer plus citation audit data. */
public final class Answer {
    private static final Pattern CITATION = Pattern.compile("\\[(\\d+)]");

    private final String text;
    private final List<SearchResult> results;
    private final boolean grounded;
    private final List<Integer> invalidCitations;
    private final List<Integer> weakCitations;

    /**
     * Creates an answer.
     *
     * @param text answer text
     * @param results retrieved passages
     * @param grounded grounded flag
     * @param invalidCitations invalid citation numbers
     * @param weakCitations weak citation numbers
     */
    public Answer(
        String text,
        List<SearchResult> results,
        boolean grounded,
        List<Integer> invalidCitations,
        List<Integer> weakCitations
    ) {
        this.text = text;
        this.results = List.copyOf(results);
        this.grounded = grounded;
        this.invalidCitations = List.copyOf(invalidCitations);
        this.weakCitations = List.copyOf(weakCitations);
    }

    /**
     * Creates a grounded answer with no citation warnings.
     *
     * @param text answer text
     * @param results retrieved passages
     */
    public Answer(String text, List<SearchResult> results) {
        this(text, results, true, List.of(), List.of());
    }

    /** @return answer text */
    public String text() {
        return text;
    }

    /** @return retrieved passages */
    public List<SearchResult> results() {
        return results;
    }

    /** @return true when grounded */
    public boolean grounded() {
        return grounded;
    }

    /** @return invalid citation numbers */
    public List<Integer> invalidCitations() {
        return invalidCitations;
    }

    /** @return weak citation numbers */
    public List<Integer> weakCitations() {
        return weakCitations;
    }

    /**
     * Returns cited passage numbers, distinct and in order.
     *
     * @return 1-based citation indices
     */
    public List<Integer> citedIndices() {
        List<Integer> seen = new ArrayList<>();
        Matcher matcher = CITATION.matcher(text);
        while (matcher.find()) {
            int number = Integer.parseInt(matcher.group(1));
            if (!seen.contains(number)) {
                seen.add(number);
            }
        }
        return seen;
    }

    /**
     * Maps valid citation numbers to source citations.
     *
     * @return source citations
     */
    public List<String> citations() {
        List<String> citations = new ArrayList<>();
        for (int number : citedIndices()) {
            if (number >= 1 && number <= results.size()) {
                citations.add(results.get(number - 1).chunk().citation());
            }
        }
        return citations;
    }
}

