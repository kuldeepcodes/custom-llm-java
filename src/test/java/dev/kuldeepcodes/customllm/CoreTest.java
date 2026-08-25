package dev.kuldeepcodes.customllm;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import dev.kuldeepcodes.customllm.chunking.Chunk;
import dev.kuldeepcodes.customllm.chunking.Chunker;
import dev.kuldeepcodes.customllm.embeddings.Embedders;
import dev.kuldeepcodes.customllm.embeddings.HashingEmbedder;
import dev.kuldeepcodes.customllm.embeddings.Vectors;
import dev.kuldeepcodes.customllm.index.SearchResult;
import dev.kuldeepcodes.customllm.index.VectorIndex;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

class CoreTest {
    @TempDir
    Path tempDir;

    @Test
    void emptyInputProducesNoChunks() {
        assertThat(Chunker.chunkText("", "empty.md")).isEmpty();
        assertThat(Chunker.chunkText("   \n  ", "blank.md")).isEmpty();
    }

    @Test
    void shortDocumentIsOneChunk() {
        List<Chunk> chunks = Chunker.chunkText("A single short sentence.", "short.md");
        assertThat(chunks).hasSize(1);
        assertThat(chunks.get(0).text()).isEqualTo("A single short sentence.");
        assertThat(chunks.get(0).source()).isEqualTo("short.md");
        assertThat(chunks.get(0).index()).isZero();
    }

    @Test
    void longDocumentSplitsIntoSeveralChunks() {
        String text = java.util.stream.IntStream.range(0, 200)
            .mapToObj(i -> "This is sentence number " + i + ".")
            .collect(java.util.stream.Collectors.joining(" "));
        List<Chunk> chunks = Chunker.chunkText(text, "long.md", 200, 50);
        assertThat(chunks).hasSizeGreaterThan(1);
        assertThat(chunks.stream().map(Chunk::index)).containsExactlyElementsOf(
            java.util.stream.IntStream.range(0, chunks.size()).boxed().toList()
        );
    }

    @Test
    void sentencesAreNeverSplitMidWay() {
        List<Chunk> chunks = Chunker.chunkText("word ".repeat(100).strip() + ".", "big.md", 50, 10);
        assertThat(chunks).hasSize(1);
        assertThat(chunks.get(0).text()).endsWith(".");
    }

    @Test
    void overlapRepeatsTrailingContent() {
        String text = java.util.stream.IntStream.range(0, 30)
            .mapToObj(i -> "Sentence " + i + " carries distinct content here.")
            .collect(java.util.stream.Collectors.joining(" "));
        List<Chunk> withOverlap = Chunker.chunkText(text, "a.md", 200, 100);
        List<Chunk> withoutOverlap = Chunker.chunkText(text, "a.md", 200, 0);
        assertThat(withOverlap.size()).isGreaterThanOrEqualTo(withoutOverlap.size());
        assertThat(withOverlap.stream().mapToInt(c -> c.text().length()).sum())
            .isGreaterThan(withoutOverlap.stream().mapToInt(c -> c.text().length()).sum());
    }

    @Test
    void citationReportsSourceAndLine() {
        List<Chunk> chunks = Chunker.chunkText("First line.\n\nSecond paragraph starts here.", "doc.md", 20, 0);
        assertThat(chunks.get(0).citation()).isEqualTo("doc.md:1");
        assertThat(chunks).allSatisfy(chunk -> assertThat(chunk.citation()).contains(":"));
    }

    @Test
    void lineNumbersIncreaseThroughDocument() {
        String text = java.util.stream.IntStream.range(0, 20)
            .mapToObj(i -> "Paragraph " + i + " with enough text to matter.")
            .collect(java.util.stream.Collectors.joining("\n\n"));
        List<Integer> lines = Chunker.chunkText(text, "doc.md", 100, 0).stream().map(Chunk::startLine).toList();
        assertThat(lines).isSorted();
        assertThat(lines.get(0)).isEqualTo(1);
    }

    @ParameterizedTest
    @CsvSource({"0,0", "-1,0", "100,-1", "100,100", "100,200"})
    void invalidChunkParametersAreRejected(int chunkSize, int overlap) {
        assertThatThrownBy(() -> Chunker.chunkText("Some text.", "doc.md", chunkSize, overlap))
            .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void splitSentencesHandlesPunctuationAndBlankLines() {
        assertThat(Chunker.splitSentences("One. Two! Three?\n\nFour."))
            .containsExactly("One.", "Two!", "Three?", "Four.");
    }

    @Test
    void hashingIsDeterministicAcrossInstances() {
        assertThat(new HashingEmbedder().embedOne("the quick brown fox"))
            .containsExactly(new HashingEmbedder().embedOne("the quick brown fox"));
    }

    @Test
    void hashingProducesRequestedDimensions() {
        assertThat(new HashingEmbedder(128).embedOne("text")).hasSize(128);
    }

    @Test
    void hashingVectorsAreUnitLength() {
        double[] vector = new HashingEmbedder().embedOne("some representative text here");
        assertThat(Math.sqrt(java.util.Arrays.stream(vector).map(v -> v * v).sum()))
            .isCloseTo(1.0, org.assertj.core.data.Offset.offset(1e-9));
    }

    @Test
    void sharedVocabularyScoresHigherThanUnrelatedText() {
        HashingEmbedder embedder = new HashingEmbedder();
        double shared = Vectors.cosineSimilarity(embedder.embedOne("annual leave policy for employees"),
            embedder.embedOne("employees receive annual leave"));
        double unrelated = Vectors.cosineSimilarity(embedder.embedOne("annual leave policy for employees"),
            embedder.embedOne("espresso grind ratio brewing"));
        assertThat(shared).isGreaterThan(unrelated);
    }

    @Test
    void emptyTextYieldsZeroVector() {
        assertThat(new HashingEmbedder().embedOne("")).containsOnly(0.0);
    }

    @Test
    void batchReturnsOneVectorPerInputInOrder() {
        HashingEmbedder embedder = new HashingEmbedder();
        List<double[]> vectors = embedder.embed(List.of("first text", "second text", "third text"));
        assertThat(vectors).hasSize(3);
        assertThat(vectors.get(0)).containsExactly(embedder.embedOne("first text"));
    }

    @Test
    void rejectsNonPositiveDimensions() {
        assertThatThrownBy(() -> new HashingEmbedder(0)).hasMessageContaining("dimensions");
    }

    @Test
    void cosineIdentitiesHold() {
        assertThat(Vectors.cosineSimilarity(new double[] {1, 2, 3}, new double[] {1, 2, 3}))
            .isCloseTo(1.0, org.assertj.core.data.Offset.offset(1e-12));
        assertThat(Vectors.cosineSimilarity(new double[] {1, 0}, new double[] {-1, 0}))
            .isCloseTo(-1.0, org.assertj.core.data.Offset.offset(1e-12));
        assertThat(Vectors.cosineSimilarity(new double[] {1, 0}, new double[] {0, 1})).isZero();
        assertThat(Vectors.cosineSimilarity(new double[] {0, 0}, new double[] {1, 1})).isZero();
    }

    @Test
    void cosineMismatchedLengthsAreRejected() {
        assertThatThrownBy(() -> Vectors.cosineSimilarity(new double[] {1}, new double[] {1, 2}))
            .hasMessageContaining("length mismatch");
    }

    @Test
    void searchRanksRelevantChunkFirst() {
        Fixture fixture = fixture();
        List<SearchResult> results = fixture.index.search("how many days of annual leave",
            fixture.embedder.embedOne("how many days of annual leave"), 4);
        assertThat(results.get(0).chunk().source()).isEqualTo("hr.md");
    }

    @Test
    void searchReturnsAtMostTopKAndScoresAreSorted() {
        Fixture fixture = fixture();
        List<SearchResult> results = fixture.index.search("espresso grams",
            fixture.embedder.embedOne("espresso grams"), 2);
        assertThat(results).hasSize(2);
        assertThat(results.stream().map(SearchResult::score)).isSortedAccordingTo(java.util.Comparator.reverseOrder());
    }

    @Test
    void emptyIndexReturnsNothing() {
        HashingEmbedder embedder = new HashingEmbedder(64);
        assertThat(new VectorIndex(embedder.name(), 64).search("anything", embedder.embedOne("anything"), 4))
            .isEmpty();
    }

    @Test
    void keywordComponentRewardsLiteralMatches() {
        Fixture fixture = fixture();
        SearchResult best = fixture.index.search("espresso", fixture.embedder.embedOne("espresso"), 3).get(0);
        assertThat(best.chunk().source()).isEqualTo("coffee.md");
        assertThat(best.keywordScore()).isGreaterThan(0.0);
    }

    @Test
    void indexRoundTripsThroughDisk() throws Exception {
        Fixture fixture = fixture();
        Path path = tempDir.resolve("index.json");
        fixture.index.save(path);
        VectorIndex loaded = VectorIndex.load(path);
        assertThat(loaded.size()).isEqualTo(fixture.index.size());
        assertThat(loaded.embedderName()).isEqualTo(fixture.index.embedderName());
        assertThat(loaded.chunks().get(0).text()).isEqualTo(fixture.index.chunks().get(0).text());
    }

    @Test
    void missingIndexExplainsIngest() {
        assertThatThrownBy(() -> VectorIndex.load(tempDir.resolve("absent.json"))).hasMessageContaining("ingest");
    }

    @Test
    void addRejectsWrongVectorShapeAndCount() {
        VectorIndex index = new VectorIndex("test", 64);
        assertThatThrownBy(() -> index.add(List.of(new Chunk("text", "a.md", 0, 1)), List.of(new double[32])))
            .hasMessageContaining("dimensions");
        assertThatThrownBy(() -> index.add(List.of(new Chunk("a", "a.md", 0, 1)),
            List.of(new double[64], new double[64]))).hasMessageContaining("chunks but");
    }

    @Test
    void ensureCompatibleRefusesMismatches() {
        assertThatThrownBy(() -> fixture().index.ensureCompatible(new HashingEmbedder(128)))
            .hasMessageContaining("not comparable");
    }

    @Test
    void factoryHandlesHashingAndUnknownChoice() {
        assertThat(Embedders.create("hashing", "all-minilm", "http://localhost:11434"))
            .isInstanceOf(HashingEmbedder.class);
        assertThatThrownBy(() -> Embedders.create("magic", "all-minilm", "http://localhost:11434"))
            .hasMessageContaining("Unknown embedder");
    }

    @Test
    void hybridTieBreaksBySourceThenIndex() {
        VectorIndex index = new VectorIndex("test", 2);
        index.add(List.of(new Chunk("same", "b.md", 2, 1), new Chunk("same", "a.md", 1, 1)),
            List.of(new double[] {1, 0}, new double[] {1, 0}));
        assertThat(index.search("", new double[] {1, 0}, 2).get(0).chunk().source()).isEqualTo("a.md");
    }

    private static Fixture fixture() {
        HashingEmbedder embedder = new HashingEmbedder(64);
        VectorIndex index = new VectorIndex(embedder.name(), embedder.dimensions());
        List<Chunk> chunks = List.of(
            new Chunk("Employees receive 27 days of paid annual leave.", "hr.md", 0, 1),
            new Chunk("Espresso uses 18 grams in and 36 grams out.", "coffee.md", 0, 1),
            new Chunk("Laptops are replaced every three years.", "it.md", 0, 1)
        );
        index.add(chunks, embedder.embed(chunks.stream().map(Chunk::text).toList()));
        return new Fixture(index, embedder);
    }

    private record Fixture(VectorIndex index, HashingEmbedder embedder) {
    }
}
