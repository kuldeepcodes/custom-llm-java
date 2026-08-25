package dev.kuldeepcodes.customllm;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import dev.kuldeepcodes.customllm.loaders.Loaders;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class LoaderTest {
    private static final ObjectMapper JSON = new ObjectMapper();

    @TempDir
    Path tempDir;

    @Test
    void readsMarkdownAndText() throws Exception {
        Files.writeString(tempDir.resolve("a.md"), "# Heading\n\nBody text.", StandardCharsets.UTF_8);
        Files.writeString(tempDir.resolve("b.txt"), "Plain text.", StandardCharsets.UTF_8);

        assertThat(names(Loaders.loadDocuments(tempDir))).containsExactly("a.md", "b.txt");
    }

    @Test
    void walksNestedFoldersKeepingRelativeForwardSlashNames() throws Exception {
        Path nested = tempDir.resolve("policies").resolve("hr");
        Files.createDirectories(nested);
        Files.writeString(nested.resolve("leave.md"), "Leave policy.", StandardCharsets.UTF_8);

        assertThat(names(Loaders.loadDocuments(tempDir))).containsExactly("policies/hr/leave.md");
    }

    @Test
    void singleFileIsNamedByItself() throws Exception {
        Path path = tempDir.resolve("handbook.md");
        Files.writeString(path, "Content.", StandardCharsets.UTF_8);

        assertThat(Loaders.loadDocuments(path).get(0).name()).isEqualTo("handbook.md");
    }

    @Test
    void ignoresUnsupportedExtensions() throws Exception {
        Files.writeString(tempDir.resolve("notes.md"), "Keep.", StandardCharsets.UTF_8);
        Files.write(tempDir.resolve("photo.png"), new byte[] {(byte) 0x89, 'P', 'N', 'G'});
        Files.write(tempDir.resolve("sheet.xlsx"), new byte[] {'P', 'K', 3, 4});

        assertThat(names(Loaders.loadDocuments(tempDir))).containsExactly("notes.md");
    }

    @Test
    void skipsBlankFiles() throws Exception {
        Files.writeString(tempDir.resolve("real.md"), "Content.", StandardCharsets.UTF_8);
        Files.writeString(tempDir.resolve("blank.md"), "   \n\n ", StandardCharsets.UTF_8);

        assertThat(names(Loaders.loadDocuments(tempDir))).containsExactly("real.md");
    }

    @Test
    void orderIsStable() throws Exception {
        for (String name : List.of("c.md", "a.md", "b.md")) {
            Files.writeString(tempDir.resolve(name), "Content.", StandardCharsets.UTF_8);
        }

        assertThat(Loaders.loadDocuments(tempDir)).isEqualTo(Loaders.loadDocuments(tempDir));
    }

    @Test
    void missingPathIsReported() {
        assertThatThrownBy(() -> Loaders.loadDocuments(tempDir.resolve("nowhere")))
            .isInstanceOf(java.io.FileNotFoundException.class);
    }

    @Test
    void jsonIsASupportedSuffix() {
        assertThat(Loaders.SUPPORTED_SUFFIXES).contains(".json", ".jsonl", ".md", ".txt");
    }

    @Test
    void eachArrayElementBecomesItsOwnDocument() throws Exception {
        Path path = writeJson("records.json", List.of(
            Map.of("id", "A1", "body", "First record."),
            Map.of("id", "A2", "body", "Second record."),
            Map.of("id", "A3", "body", "Third record.")
        ));

        List<Loaders.LoadedDocument> documents = Loaders.loadDocuments(path);

        assertThat(documents).hasSize(3);
        assertThat(names(documents)).containsExactly("records.json#0", "records.json#1", "records.json#2");
    }

    @Test
    void recordTextIsReadableKeyValueLines() throws Exception {
        Path path = writeJson("records.json", List.of(Map.of("title", "Refunds", "body", "Within 45 days.")));

        String text = Loaders.loadDocuments(path).get(0).text();

        assertThat(text).contains("title: Refunds", "body: Within 45 days.")
            .doesNotContain("{", "\"");
    }

    @Test
    void nestedObjectsKeepTraceableDottedPaths() throws Exception {
        Path path = writeJson("records.json", List.of(Map.of("customer", Map.of("name", "Priya", "plan", "wholesale"))));

        String text = Loaders.loadDocuments(path).get(0).text();

        assertThat(text).contains("customer.name: Priya", "customer.plan: wholesale");
    }

    @Test
    void scalarListsAreRenderedInline() throws Exception {
        Path path = writeJson("records.json", List.of(Map.of("tags", List.of("hardware", "grinder"))));

        assertThat(Loaders.loadDocuments(path).get(0).text()).contains("tags: hardware, grinder");
    }

    @Test
    void singleWrappedArrayIsUnwrapped() throws Exception {
        Path path = writeJson("export.json", Map.of("items", List.of(Map.of("body", "One."), Map.of("body", "Two."))));

        assertThat(Loaders.loadDocuments(path)).hasSize(2);
    }

    @Test
    void ambiguousTwoArrayWrapperIsLeftAlone() throws Exception {
        Path path = writeJson("two.json", Map.of(
            "users", List.of(Map.of("a", 1)),
            "orders", List.of(Map.of("b", 2))
        ));

        List<Loaders.LoadedDocument> documents = Loaders.loadDocuments(path);

        assertThat(documents).hasSize(1);
        assertThat(documents.get(0).name()).isEqualTo("two.json");
    }

    @Test
    void plainObjectBecomesOneDocument() throws Exception {
        Path path = writeJson("config.json", Map.of("name", "Nimbus", "founded", 2019));

        List<Loaders.LoadedDocument> documents = Loaders.loadDocuments(path);

        assertThat(documents).hasSize(1);
        assertThat(documents.get(0).text()).contains("name: Nimbus");
    }

    @Test
    void emptyRecordsAreSkipped() throws Exception {
        Path path = writeJson("records.json", List.of(json("{\"body\":\"Real.\"}"), json("{}"), json("{\"body\":null}")));

        assertThat(Loaders.loadDocuments(path)).hasSize(1);
    }

    @Test
    void malformedJsonNamesFileAndPosition() throws Exception {
        Path path = tempDir.resolve("broken.json");
        Files.writeString(path, "{\"unclosed\": ", StandardCharsets.UTF_8);

        assertThatThrownBy(() -> Loaders.loadDocuments(path))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("broken.json is not valid JSON");
    }

    @Test
    void jsonLinesEachLineBecomesDocumentNumberedFromOne() throws Exception {
        Path path = tempDir.resolve("events.jsonl");
        Files.writeString(path, "{\"body\": \"First.\"}\n{\"body\": \"Second.\"}\n", StandardCharsets.UTF_8);

        assertThat(names(Loaders.loadDocuments(path))).containsExactly("events.jsonl#1", "events.jsonl#2");
    }

    @Test
    void jsonLinesBlankLinesAreIgnored() throws Exception {
        Path path = tempDir.resolve("events.jsonl");
        Files.writeString(path, "{\"body\": \"First.\"}\n\n\n{\"body\": \"Second.\"}\n", StandardCharsets.UTF_8);

        assertThat(Loaders.loadDocuments(path)).hasSize(2);
    }

    @Test
    void badJsonLineReportsLineNumber() throws Exception {
        Path path = tempDir.resolve("events.jsonl");
        Files.writeString(path, "{\"body\": \"Fine.\"}\nnot json at all\n", StandardCharsets.UTF_8);

        assertThatThrownBy(() -> Loaders.loadDocuments(path)).hasMessageContaining("line 2");
    }

    @Test
    void ndjsonIsTreatedTheSame() throws Exception {
        Path path = tempDir.resolve("events.ndjson");
        Files.writeString(path, "{\"body\": \"Only.\"}\n", StandardCharsets.UTF_8);

        assertThat(Loaders.loadDocuments(path)).hasSize(1);
    }

    @Test
    void booleansRenderLowerCase() throws Exception {
        assertThat(flatten("{\"active\":true}")).isEqualTo("active: true");
        assertThat(flatten("{\"active\":false}")).isEqualTo("active: false");
    }

    @Test
    void nullsAreOmittedRatherThanWrittenAsNull() throws Exception {
        assertThat(flatten("{\"a\":null,\"b\":\"kept\"}")).isEqualTo("b: kept");
    }

    @Test
    void numbersArePreservedExactly() throws Exception {
        assertThat(flatten("{\"count\":27,\"ratio\":1.5}")).contains("count: 27", "ratio: 1.5");
    }

    @Test
    void listsOfObjectsAreIndexed() throws Exception {
        String text = flatten("{\"items\":[{\"name\":\"a\"},{\"name\":\"b\"}]}");

        assertThat(text).contains("items[0].name: a", "items[1].name: b");
    }

    @Test
    void deepNestingIsTruncatedRatherThanExploding() throws Exception {
        JsonNode deep = json("{\"v\":\"bottom\"}");
        for (int i = 0; i < 20; i++) {
            deep = JSON.valueToTree(Map.of("nest", deep));
        }

        assertThat(Loaders.flattenJson(deep)).contains("...");
    }

    @Test
    void emptyStructuresProduceNoText() throws Exception {
        assertThat(flatten("{}")).isEmpty();
        assertThat(flatten("[]")).isEmpty();
    }

    @Test
    void textAndJsonCoexistInOneFolder() throws Exception {
        Files.writeString(tempDir.resolve("handbook.md"), "# Handbook\n\nLeave policy.", StandardCharsets.UTF_8);
        writeJson("tickets.json", List.of(Map.of("body", "Ticket one."), Map.of("body", "Ticket two.")));

        assertThat(names(Loaders.loadDocuments(tempDir)))
            .containsExactly("handbook.md", "tickets.json#0", "tickets.json#1");
    }

    @Test
    void jsonDocumentsCanBeLoadedDirectly() throws Exception {
        Path path = writeJson("records.json", List.of(Map.of("body", "One.")));

        assertThat(Loaders.loadJsonDocuments(path, "records.json").get(0).name()).isEqualTo("records.json#0");
    }

    private Path writeJson(String name, Object payload) throws Exception {
        Path path = tempDir.resolve(name);
        Files.writeString(path, JSON.writeValueAsString(payload), StandardCharsets.UTF_8);
        return path;
    }

    private String flatten(String json) throws Exception {
        return Loaders.flattenJson(json(json));
    }

    private JsonNode json(String json) throws Exception {
        return JSON.readTree(json);
    }

    private List<String> names(List<Loaders.LoadedDocument> documents) {
        return documents.stream().map(Loaders.LoadedDocument::name).toList();
    }
}

