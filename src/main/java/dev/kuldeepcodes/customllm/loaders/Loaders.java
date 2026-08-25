package dev.kuldeepcodes.customllm.loaders;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.stream.Stream;

/** Reads user data into citation-friendly document records. */
public final class Loaders {
    /** Plain text document suffixes. */
    public static final Set<String> TEXT_SUFFIXES = Set.of(".txt", ".md", ".markdown");
    /** JSON document suffixes. */
    public static final Set<String> JSON_SUFFIXES = Set.of(".json", ".jsonl", ".ndjson");
    /** Every suffix the loader accepts. */
    public static final Set<String> SUPPORTED_SUFFIXES = Set.of(
        ".txt", ".md", ".markdown", ".json", ".jsonl", ".ndjson"
    );
    /** Maximum JSON nesting depth to flatten before truncating. */
    public static final int MAX_JSON_DEPTH = 6;

    private static final ObjectMapper JSON = new ObjectMapper();

    private Loaders() {
    }

    /**
     * Reads every supported document under a source file or directory.
     *
     * @param source file or directory to load
     * @return loaded document records
     * @throws IOException when files cannot be read
     */
    public static List<LoadedDocument> loadDocuments(Path source) throws IOException {
        if (!Files.exists(source)) {
            throw new FileNotFoundException("No such path: " + source);
        }

        List<Path> files;
        if (Files.isRegularFile(source)) {
            files = List.of(source);
        } else {
            try (Stream<Path> stream = Files.walk(source)) {
                files = stream.filter(Files::isRegularFile)
                    .filter(path -> SUPPORTED_SUFFIXES.contains(suffix(path)))
                    .sorted()
                    .toList();
            }
        }

        List<LoadedDocument> documents = new ArrayList<>();
        for (Path path : files) {
            String suffix = suffix(path);
            if (!SUPPORTED_SUFFIXES.contains(suffix)) {
                continue;
            }

            String baseName = Files.isRegularFile(source)
                ? path.getFileName().toString()
                : source.relativize(path).toString();
            baseName = baseName.replace('\\', '/');

            if (JSON_SUFFIXES.contains(suffix)) {
                documents.addAll(loadJsonDocuments(path, baseName));
            } else {
                String text = readUtf8ReplacingMalformedInput(path);
                if (!text.trim().isEmpty()) {
                    documents.add(new LoadedDocument(baseName, text));
                }
            }
        }
        return documents;
    }

    /**
     * Turns a JSON, JSONL, or NDJSON file into individually citable documents.
     *
     * @param path source file
     * @param baseName citation-friendly base name
     * @return loaded JSON records
     * @throws IOException when files cannot be read
     */
    public static List<LoadedDocument> loadJsonDocuments(Path path, String baseName) throws IOException {
        String raw = readUtf8ReplacingMalformedInput(path);
        if (raw.trim().isEmpty()) {
            return List.of();
        }

        if (Set.of(".jsonl", ".ndjson").contains(suffix(path))) {
            return loadJsonLines(raw, baseName);
        }

        JsonNode payload;
        try {
            payload = JSON.readTree(raw);
        } catch (JsonProcessingException e) {
            throw new IllegalArgumentException(jsonError(baseName, e), e);
        }

        JsonNode records = asRecords(payload);
        if (records == null) {
            String text = flattenJson(payload);
            return text.trim().isEmpty() ? List.of() : List.of(new LoadedDocument(baseName, text));
        }

        List<LoadedDocument> documents = new ArrayList<>();
        for (int position = 0; position < records.size(); position++) {
            String text = flattenJson(records.get(position));
            if (!text.trim().isEmpty()) {
                documents.add(new LoadedDocument(baseName + "#" + position, text));
            }
        }
        return documents;
    }

    /**
     * Renders JSON as readable {@code key: value} lines.
     *
     * @param value JSON value
     * @return flattened text
     */
    public static String flattenJson(JsonNode value) {
        return flattenJson(value, "", 0);
    }

    /**
     * Renders JSON as readable {@code key: value} lines.
     *
     * @param value JSON value
     * @param prefix key prefix
     * @param depth current depth
     * @return flattened text
     */
    public static String flattenJson(JsonNode value, String prefix, int depth) {
        if (depth > MAX_JSON_DEPTH) {
            return prefix.isEmpty() ? "..." : prefix + ": ...";
        }

        if (value == null || value.isNull()) {
            return "";
        }

        if (value.isObject()) {
            List<String> lines = new ArrayList<>();
            for (Map.Entry<String, JsonNode> entry : value.properties()) {
                String key = prefix.isEmpty() ? entry.getKey() : prefix + "." + entry.getKey();
                String line = flattenJson(entry.getValue(), key, depth + 1);
                if (!line.isEmpty()) {
                    lines.add(line);
                }
            }
            return String.join("\n", lines);
        }

        if (value.isArray()) {
            if (allScalars(value)) {
                List<String> rendered = new ArrayList<>();
                for (JsonNode item : value) {
                    if (!item.isNull()) {
                        rendered.add(scalar(item));
                    }
                }
                return rendered.isEmpty() ? "" : prefix + ": " + String.join(", ", rendered);
            }

            List<String> lines = new ArrayList<>();
            for (int position = 0; position < value.size(); position++) {
                String key = prefix.isEmpty() ? "[" + position + "]" : prefix + "[" + position + "]";
                String line = flattenJson(value.get(position), key, depth + 1);
                if (!line.isEmpty()) {
                    lines.add(line);
                }
            }
            return String.join("\n", lines);
        }

        return prefix.isEmpty() ? scalar(value) : prefix + ": " + scalar(value);
    }

    private static List<LoadedDocument> loadJsonLines(String raw, String baseName) {
        List<LoadedDocument> documents = new ArrayList<>();
        String[] lines = raw.split("\\R", -1);
        for (int number = 1; number <= lines.length; number++) {
            String line = lines[number - 1];
            if (line.trim().isEmpty()) {
                continue;
            }
            JsonNode record;
            try {
                record = JSON.readTree(line);
            } catch (JsonProcessingException e) {
                throw new IllegalArgumentException(baseName + " line " + number
                    + " is not valid JSON: " + e.getOriginalMessage(), e);
            }
            String text = flattenJson(record);
            if (!text.trim().isEmpty()) {
                documents.add(new LoadedDocument(baseName + "#" + number, text));
            }
        }
        return documents;
    }

    private static JsonNode asRecords(JsonNode payload) {
        if (payload.isArray()) {
            return payload;
        }
        if (payload.isObject()) {
            List<JsonNode> nonEmptyArrays = new ArrayList<>();
            for (Map.Entry<String, JsonNode> entry : payload.properties()) {
                JsonNode value = entry.getValue();
                if (value.isArray() && value.size() > 0) {
                    nonEmptyArrays.add(value);
                }
            }
            if (nonEmptyArrays.size() == 1) {
                return nonEmptyArrays.get(0);
            }
        }
        return null;
    }

    private static boolean allScalars(JsonNode value) {
        for (JsonNode item : value) {
            if (item.isObject() || item.isArray()) {
                return false;
            }
        }
        return true;
    }

    private static String scalar(JsonNode value) {
        if (value.isBoolean()) {
            return value.booleanValue() ? "true" : "false";
        }
        if (value.isNumber()) {
            return value.numberValue().toString();
        }
        return value.asText();
    }

    private static String jsonError(String baseName, JsonProcessingException e) {
        long line = e.getLocation() == null ? -1 : e.getLocation().getLineNr();
        long column = e.getLocation() == null ? -1 : e.getLocation().getColumnNr();
        return baseName + " is not valid JSON: " + e.getOriginalMessage()
            + " (line " + line + ", column " + column + "). Fix the file or remove it from the corpus.";
    }

    private static String readUtf8ReplacingMalformedInput(Path path) throws IOException {
        return new String(Files.readAllBytes(path), StandardCharsets.UTF_8);
    }

    private static String suffix(Path path) {
        String name = path.getFileName().toString().toLowerCase(Locale.ROOT);
        int dot = name.lastIndexOf('.');
        return dot < 0 ? "" : name.substring(dot);
    }

    /**
     * Loaded document.
     *
     * @param name citation-friendly name
     * @param text readable document text
     */
    public record LoadedDocument(String name, String text) {
    }
}
