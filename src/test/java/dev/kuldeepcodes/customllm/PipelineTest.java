package dev.kuldeepcodes.customllm;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.fasterxml.jackson.databind.ObjectMapper;
import dev.kuldeepcodes.customllm.chunking.Chunk;
import dev.kuldeepcodes.customllm.embeddings.HashingEmbedder;
import dev.kuldeepcodes.customllm.generate.Answer;
import dev.kuldeepcodes.customllm.generate.ChatClient;
import dev.kuldeepcodes.customllm.generate.Grounding;
import dev.kuldeepcodes.customllm.index.SearchResult;
import dev.kuldeepcodes.customllm.index.VectorIndex;
import dev.kuldeepcodes.customllm.pipeline.IngestReport;
import dev.kuldeepcodes.customllm.pipeline.Pipeline;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

class PipelineTest {
    private static final String HANDBOOK = """
        # Handbook

        ## Time off

        Everyone receives 27 days of paid annual leave, plus public holidays.
        Unused days may be carried over up to a maximum of five days.
        """;
    private static final String PRODUCTS = """
        # Products

        ## Meridian blend

        Meridian is 60 percent Brazilian Cerrado and 30 percent Colombian Huila.
        It rests for seven days after roasting.
        """;

    @TempDir
    Path tempDir;
    Path corpus;

    @BeforeEach
    void setup() throws Exception {
        corpus = tempDir.resolve("corpus");
        Files.createDirectories(corpus);
        Files.writeString(corpus.resolve("handbook.md"), HANDBOOK, StandardCharsets.UTF_8);
        Files.writeString(corpus.resolve("products.md"), PRODUCTS, StandardCharsets.UTF_8);
    }

    @Test
    void loadDocumentsReadsSupportedFiles() throws Exception {
        assertThat(Pipeline.loadDocuments(corpus).stream().map(Pipeline.Document::name))
            .containsExactly("handbook.md", "products.md");
    }

    @Test
    void loadDocumentsAcceptsSingleFile() throws Exception {
        assertThat(Pipeline.loadDocuments(corpus.resolve("handbook.md"))).hasSize(1)
            .first().extracting(Pipeline.Document::name).isEqualTo("handbook.md");
    }

    @Test
    void loadDocumentsIgnoresUnsupportedAndBlankFiles() throws Exception {
        Files.write(corpus.resolve("image.png"), new byte[] {1, 2});
        Files.writeString(corpus.resolve("blank.md"), "  \n", StandardCharsets.UTF_8);
        assertThat(Pipeline.loadDocuments(corpus).stream().map(Pipeline.Document::name))
            .doesNotContain("image.png", "blank.md");
    }

    @Test
    void loadDocumentsReportsMissingPath() {
        assertThatThrownBy(() -> Pipeline.loadDocuments(tempDir.resolve("nowhere")))
            .isInstanceOf(java.io.FileNotFoundException.class);
    }

    @Test
    void loadDocumentsIsStableAndUsesForwardSlashNames() throws Exception {
        Files.createDirectories(corpus.resolve("nested"));
        Files.writeString(corpus.resolve("nested").resolve("note.md"), "Nested fact.", StandardCharsets.UTF_8);
        assertThat(Pipeline.loadDocuments(corpus)).isEqualTo(Pipeline.loadDocuments(corpus));
        assertThat(Pipeline.loadDocuments(corpus).stream().map(Pipeline.Document::name)).contains("nested/note.md");
    }

    @Test
    void ingestBuildsPersistsAndQueriesIndex() throws Exception {
        HashingEmbedder embedder = new HashingEmbedder(64);
        Path indexPath = tempDir.resolve("index.json");
        IngestReport report = Pipeline.ingest(corpus, embedder, indexPath, 800, 150);
        assertThat(report.documents()).isEqualTo(2);
        assertThat(report.chunks()).isPositive();
        assertThat(report.dimensions()).isEqualTo(64);
        assertThat(indexPath).exists();
        assertThat(Pipeline.retrieve("annual leave days", VectorIndex.load(indexPath), embedder, 4)
            .get(0).chunk().source()).isEqualTo("handbook.md");
    }

    @Test
    void ingestReportsEmptyCorpus() throws Exception {
        Path empty = tempDir.resolve("empty");
        Files.createDirectories(empty);
        Files.writeString(empty.resolve("notes.rst"), "unsupported", StandardCharsets.UTF_8);
        assertThatThrownBy(() -> Pipeline.ingest(empty, new HashingEmbedder(), empty.resolve("i.json"), 800, 150))
            .hasMessageContaining("No readable documents");
    }

    @Test
    void groundingRefusesWhenNothingRetrieved() {
        Answer answer = Grounding.answerQuestion("anything", List.of(), null);
        assertThat(answer.text()).isEqualTo(Grounding.REFUSAL);
        assertThat(answer.grounded()).isFalse();
    }

    @Test
    void groundingRefusesBelowFloor() {
        Answer answer = Grounding.answerQuestion("capital of France", results(0.05), null);
        assertThat(answer.text()).isEqualTo(Grounding.REFUSAL);
        assertThat(answer.grounded()).isFalse();
    }

    @Test
    void groundingRefusesNotInContext() {
        Answer answer = Grounding.answerQuestion("question", results(0.9), new StubChat("NOT_IN_CONTEXT"));
        assertThat(answer.text()).isEqualTo(Grounding.REFUSAL);
        assertThat(answer.grounded()).isFalse();
    }

    @Test
    void extractiveModeReturnsBestPassage() {
        Answer answer = Grounding.answerQuestion("question", results(0.9, 0.5), null);
        assertThat(answer.text()).contains("Passage 0").endsWith("[1]");
        assertThat(answer.grounded()).isTrue();
    }

    @Test
    void promptNumbersPassagesAndIncludesQuestion() {
        assertThat(Grounding.buildPrompt("How much leave?", results(0.9, 0.8)))
            .contains("[1]", "[2]", "Question: How much leave?");
    }

    @Test
    void answerParsesDistinctCitationsAndMapsSources() {
        Answer answer = new Answer("Leave [1], rollover [2], again [1].", results(1, 1));
        assertThat(answer.citedIndices()).containsExactly(1, 2);
        assertThat(answer.citations()).containsExactly("doc.md:1", "doc.md:2");
    }

    @Test
    void invalidCitationIsDetected() {
        Answer answer = Grounding.answerQuestion("question", results(0.9, 0.8, 0.7), new StubChat("Claim [7]."));
        assertThat(answer.invalidCitations()).containsExactly(7);
    }

    @Test
    void validCitationIsNotFlagged() {
        Answer answer = Grounding.answerQuestion("question", results(0.9, 0.8), new StubChat("Claim [1]."));
        assertThat(answer.invalidCitations()).isEmpty();
    }

    @ParameterizedTest
    @ValueSource(strings = {"[/INST]", "[INST]", "<|end|>", "</s>", "<|assistant|>", "<|user|>", "<s>"})
    void templateArtifactsAreRemoved(String artifact) {
        assertThat(Grounding.stripTemplateArtifacts("Leave is 27 days [1]. " + artifact))
            .isEqualTo("Leave is 27 days [1].");
    }

    @Test
    void weakCitationDetectionUsesObservedCase() {
        List<SearchResult> searchResults = List.of(
            new SearchResult(new Chunk("Meridian is 60 percent Brazilian Cerrado.", "p.md", 0, 1), 0.9, 0.9, 0),
            new SearchResult(new Chunk("Cold brew steeps for sixteen hours.", "p.md", 1, 20), 0.8, 0.8, 0)
        );
        Answer answer = Grounding.answerQuestion("what is in Meridian", searchResults,
            new StubChat("Meridian contains 60 percent Brazilian Cerrado beans. [2]"));
        assertThat(answer.weakCitations()).containsExactly(2);
    }

    @Test
    void wellSupportedCitationIsNotWeak() {
        List<SearchResult> searchResults = List.of(
            new SearchResult(new Chunk("Meridian is 60 percent Brazilian Cerrado.", "p.md", 0, 1), 0.9, 0.9, 0),
            new SearchResult(new Chunk("Cold brew steeps for sixteen hours.", "p.md", 1, 20), 0.8, 0.8, 0)
        );
        Answer answer = Grounding.answerQuestion("what is in Meridian", searchResults,
            new StubChat("Meridian contains 60 percent Brazilian Cerrado. [1]"));
        assertThat(answer.weakCitations()).isEmpty();
    }

    @Test
    void endToEndAskWithStubChat() throws Exception {
        HashingEmbedder embedder = new HashingEmbedder(256);
        Path indexPath = tempDir.resolve("index.json");
        Pipeline.ingest(corpus, embedder, indexPath, 800, 150);
        Answer answer = Pipeline.ask("how many days of annual leave", VectorIndex.load(indexPath), embedder,
            new StubChat("Employees get 27 days of annual leave. [1]"), 4);
        assertThat(answer.text()).contains("27 days");
        assertThat(answer.invalidCitations()).isEmpty();
    }

    @Test
    void endToEndRefusesOutOfCorpusQuestion() throws Exception {
        HashingEmbedder embedder = new HashingEmbedder(256);
        Path indexPath = tempDir.resolve("index.json");
        Pipeline.ingest(corpus, embedder, indexPath, 800, 150);
        Answer answer = Pipeline.ask("what is the atomic number of tungsten", VectorIndex.load(indexPath),
            embedder, null, 4);
        assertThat(answer.text()).isEqualTo(Grounding.REFUSAL);
        assertThat(answer.grounded()).isFalse();
    }

    @Test
    void emptyQuestionIsRejected() throws Exception {
        HashingEmbedder embedder = new HashingEmbedder(64);
        Path indexPath = tempDir.resolve("index.json");
        Pipeline.ingest(corpus, embedder, indexPath, 800, 150);
        assertThatThrownBy(() -> Pipeline.ask("   ", VectorIndex.load(indexPath), embedder, null, 4))
            .hasMessageContaining("actual question");
    }

    @Test
    void modelfileContainsExpectedContentAndPersona() {
        String content = Pipeline.buildModelfile("bot", "phi3", smallIndex(), null, 0.2);
        assertThat(content).contains("FROM phi3", "PARAMETER temperature", "SYSTEM", "handbook.md");
        assertThat(Pipeline.buildModelfile("bot", "phi3", smallIndex(), "You are Nimbus.", 0.2))
            .contains("You are Nimbus.");
    }

    @Test
    void modelfileRequiresName() {
        assertThatThrownBy(() -> Pipeline.buildModelfile("", "phi3", smallIndex(), null, 0.2))
            .hasMessageContaining("name");
    }

    @Test
    void createModelWriteOnlyWritesModelfile() throws Exception {
        Pipeline.ModelCreation result = Pipeline.createModel("bot", "phi3", smallIndex(),
            tempDir.resolve("model"), null, false);
        assertThat(result.modelfile()).exists();
        assertThat(result.message()).contains("ollama create bot");
    }

    @Test
    void trainingExportWritesValidJsonl() throws Exception {
        HashingEmbedder embedder = new HashingEmbedder(64);
        Path indexPath = tempDir.resolve("index.json");
        Pipeline.ingest(corpus, embedder, indexPath, 800, 150);
        VectorIndex index = VectorIndex.load(indexPath);
        Path output = tempDir.resolve("train.jsonl");
        int count = Pipeline.exportTrainingData(index, output);
        List<String> lines = Files.readAllLines(output);
        assertThat(count).isEqualTo(lines.size()).isEqualTo(index.size());
        assertThat(new ObjectMapper().readTree(lines.get(0)).path("source").asText()).isNotBlank();
    }

    @Test
    void retrieveEnforcesCompatibility() throws Exception {
        HashingEmbedder embedder = new HashingEmbedder(64);
        Path indexPath = tempDir.resolve("index.json");
        Pipeline.ingest(corpus, embedder, indexPath, 800, 150);
        assertThatThrownBy(() -> Pipeline.retrieve("annual leave", VectorIndex.load(indexPath),
            new HashingEmbedder(128), 4)).hasMessageContaining("not comparable");
    }

    private static List<SearchResult> results(double... scores) {
        List<SearchResult> out = new java.util.ArrayList<>();
        for (int i = 0; i < scores.length; i++) {
            out.add(new SearchResult(new Chunk("Passage " + i + " about annual leave.", "doc.md", i, i + 1),
                scores[i], scores[i], 0.0));
        }
        return out;
    }

    private static VectorIndex smallIndex() {
        HashingEmbedder embedder = new HashingEmbedder(32);
        VectorIndex index = new VectorIndex(embedder.name(), embedder.dimensions());
        List<Chunk> chunks = List.of(new Chunk("Some content.", "handbook.md", 0, 1));
        index.add(chunks, embedder.embed(chunks.stream().map(Chunk::text).toList()));
        return index;
    }

    private static final class StubChat implements ChatClient {
        private final String reply;

        StubChat(String reply) {
            this.reply = reply;
        }

        @Override
        public String complete(String system, String user, double temperature) {
            return reply;
        }
    }
}
