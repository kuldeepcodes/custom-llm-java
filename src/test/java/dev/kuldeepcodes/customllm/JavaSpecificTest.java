package dev.kuldeepcodes.customllm;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import dev.kuldeepcodes.customllm.chunking.Chunk;
import dev.kuldeepcodes.customllm.embeddings.HashingEmbedder;
import dev.kuldeepcodes.customllm.embeddings.Vectors;
import dev.kuldeepcodes.customllm.generate.Answer;
import dev.kuldeepcodes.customllm.generate.Grounding;
import dev.kuldeepcodes.customllm.index.SearchResult;
import dev.kuldeepcodes.customllm.index.VectorIndex;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class JavaSpecificTest {
    @TempDir
    Path tempDir;

    @Test
    void normalizeLeavesZeroVectorAlone() {
        assertThat(Vectors.normalize(new double[] {0.0, 0.0})).containsExactly(0.0, 0.0);
    }

    @Test
    void normalizeReturnsUnitVector() {
        assertThat(Vectors.normalize(new double[] {3.0, 4.0})).containsExactly(0.6, 0.8);
    }

    @Test
    void searchRejectsNonPositiveTopK() {
        HashingEmbedder embedder = new HashingEmbedder(8);
        VectorIndex index = new VectorIndex(embedder.name(), embedder.dimensions());
        Chunk chunk = new Chunk("text", "a.md", 0, 1);
        index.add(List.of(chunk), embedder.embed(List.of(chunk.text())));
        assertThatThrownBy(() -> index.search("text", embedder.embedOne("text"), 0))
            .hasMessageContaining("top_k");
    }

    @Test
    void loadingWrongVersionIsRefused() throws Exception {
        Path path = tempDir.resolve("bad.json");
        Files.writeString(path, """
            {"version":99,"embedder":"hashing-8","dimensions":8,"chunks":[],"vectors":[]}
            """, StandardCharsets.UTF_8);
        assertThatThrownBy(() -> VectorIndex.load(path)).hasMessageContaining("version 99");
    }

    @Test
    void answerCitationsIgnoreOutOfRangeNumbers() {
        Answer answer = new Answer("Fact [2] and invented [9].", results());
        assertThat(answer.citations()).containsExactly("b.md:2");
    }

    @Test
    void citedIndicesPreserveFirstSeenOrder() {
        Answer answer = new Answer("First [3], then [1], then [3].", results());
        assertThat(answer.citedIndices()).containsExactly(3, 1);
    }

    @Test
    void notInContextEmbeddedInVerboseReplyStillRefuses() {
        Answer answer = Grounding.answerQuestion("q", results(), (system, user, temperature) -> "Sorry, NOT_IN_CONTEXT");
        assertThat(answer.text()).isEqualTo(Grounding.REFUSAL);
        assertThat(answer.grounded()).isFalse();
    }

    @Test
    void stripTemplateArtifactsCollapsesRepeatedSpaces() {
        assertThat(Grounding.stripTemplateArtifacts("A  [/INST]  B")).isEqualTo("A B");
    }

    @Test
    void hashingNameIncludesDimensions() {
        assertThat(new HashingEmbedder(123).name()).isEqualTo("hashing-123");
    }

    @Test
    void indexSourcesAreDistinctAndSorted() {
        VectorIndex index = new VectorIndex("test", 1);
        index.add(List.of(new Chunk("a", "b.md", 0, 1), new Chunk("b", "a.md", 0, 1)),
            List.of(new double[] {1}, new double[] {1}));
        assertThat(index.sources()).containsExactly("a.md", "b.md");
    }

    private static List<SearchResult> results() {
        return List.of(
            new SearchResult(new Chunk("Alpha supported fact.", "a.md", 0, 1), 0.9, 0.9, 0),
            new SearchResult(new Chunk("Beta supported fact.", "b.md", 1, 2), 0.8, 0.8, 0),
            new SearchResult(new Chunk("Gamma supported fact.", "c.md", 2, 3), 0.7, 0.7, 0)
        );
    }
}
