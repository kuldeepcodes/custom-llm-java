package dev.kuldeepcodes.customllm;

import dev.kuldeepcodes.customllm.embeddings.Embedder;
import dev.kuldeepcodes.customllm.embeddings.Embedders;
import dev.kuldeepcodes.customllm.embeddings.OllamaEmbedder;
import dev.kuldeepcodes.customllm.generate.Answer;
import dev.kuldeepcodes.customllm.generate.ChatClient;
import dev.kuldeepcodes.customllm.generate.OllamaChat;
import dev.kuldeepcodes.customllm.index.SearchResult;
import dev.kuldeepcodes.customllm.index.VectorIndex;
import dev.kuldeepcodes.customllm.pipeline.IngestReport;
import dev.kuldeepcodes.customllm.pipeline.Pipeline;
import java.io.FileDescriptor;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/** CLI entry point. */
public final class Main {
    private Main() {
    }

    /**
     * Runs the command line application.
     *
     * @param args command-line arguments
     */
    public static void main(String[] args) {
        forceUtf8();
        int code;
        try {
            code = dispatch(args);
        } catch (Exception e) {
            System.err.println(Ansi.RED + "error:" + Ansi.RESET + " " + e.getMessage());
            code = 1;
        }
        if (code != 0) {
            System.exit(code);
        }
    }

    private static int dispatch(String[] args) throws Exception {
        if (args.length == 0 || "--help".equals(args[0]) || "-h".equals(args[0])) {
            System.out.println("Usage: java -jar target/customllm.jar <ingest|ask|search|info|create-model|export>");
            return args.length == 0 ? 1 : 0;
        }
        Parsed parsed = parse(List.of(args).subList(1, args.length));
        return switch (args[0]) {
            case "ingest" -> ingest(parsed);
            case "ask" -> ask(parsed);
            case "search" -> search(parsed);
            case "info" -> info(parsed);
            case "create-model" -> createModel(parsed);
            case "export" -> export(parsed);
            default -> throw new IllegalArgumentException("Unknown command: " + args[0]);
        };
    }

    private static int ingest(Parsed parsed) throws Exception {
        Embedder embedder = embedder(parsed);
        System.out.println(Ansi.DIM + "Embedding with " + embedder.name() + " ..." + Ansi.RESET);
        IngestReport report = Pipeline.ingest(Path.of(parsed.require(0, "ingest <path>")), embedder,
            indexPath(parsed), parsed.intOption("chunk-size", 800), parsed.intOption("overlap", 150));
        System.out.println(Ansi.GREEN + "Indexed" + Ansi.RESET + " " + report.chunks()
            + " chunks from " + report.documents() + " document(s)");
        System.out.println("  embedder:   " + report.embedder() + " (" + report.dimensions() + " dimensions)");
        System.out.println("  index:      " + report.indexPath());
        if (embedder.name().startsWith("hashing")) {
            System.out.println("\n" + Ansi.YELLOW + "Note:" + Ansi.RESET
                + " using keyword-style hashing embeddings. For semantic matching run: ollama pull "
                + parsed.option("embed-model", OllamaEmbedder.DEFAULT_EMBED_MODEL));
        }
        return 0;
    }

    private static int ask(Parsed parsed) throws Exception {
        Embedder embedder = embedder(parsed);
        VectorIndex index = VectorIndex.load(indexPath(parsed));
        ChatClient chat = null;
        String baseUrl = parsed.option("ollama-url", OllamaChat.DEFAULT_OLLAMA_URL);
        if (!parsed.flag("no-llm")) {
            if (OllamaChat.isAvailable(baseUrl)) {
                chat = new OllamaChat(parsed.option("model", OllamaChat.DEFAULT_CHAT_MODEL), baseUrl);
            } else {
                System.err.println(Ansi.YELLOW + "Note:" + Ansi.RESET + " no LLM reachable at "
                    + baseUrl + "; returning the best matching passage instead.");
            }
        }
        String question = String.join(" ", parsed.positions());
        Answer answer = Pipeline.ask(question, index, embedder, chat, parsed.intOption("top-k", 4));
        if (parsed.flag("show-context")) {
            System.out.println(Ansi.DIM + "Retrieved passages:" + Ansi.RESET);
            for (int i = 0; i < answer.results().size(); i++) {
                SearchResult result = answer.results().get(i);
                System.out.printf("  [%d] %s  score=%.3f%n", i + 1, result.chunk().citation(), result.score());
            }
            System.out.println();
        }
        System.out.println(Ansi.CYAN + "Q:" + Ansi.RESET + " " + question);
        System.out.println(Ansi.GREEN + "A:" + Ansi.RESET + " " + answer.text());
        if (!answer.citations().isEmpty()) {
            System.out.println("\n" + Ansi.DIM + "Sources: " + String.join(", ", answer.citations()) + Ansi.RESET);
        }
        warn(answer);
        return 0;
    }

    private static int search(Parsed parsed) throws Exception {
        Embedder embedder = embedder(parsed);
        VectorIndex index = VectorIndex.load(indexPath(parsed));
        List<SearchResult> results = Pipeline.retrieve(String.join(" ", parsed.positions()), index, embedder,
            parsed.intOption("top-k", 5));
        if (results.isEmpty()) {
            System.out.println("No matches.");
            return 0;
        }
        for (int i = 0; i < results.size(); i++) {
            SearchResult result = results.get(i);
            String snippet = result.chunk().text().replace('\n', ' ');
            if (snippet.length() > 160) {
                snippet = snippet.substring(0, 160) + "...";
            }
            System.out.printf("[%d] %s  score=%.3f (vector=%.3f keyword=%.3f)%n",
                i + 1, result.chunk().citation(), result.score(), result.vectorScore(), result.keywordScore());
            System.out.println("    " + snippet + "\n");
        }
        return 0;
    }

    private static int info(Parsed parsed) throws Exception {
        Path path = indexPath(parsed);
        VectorIndex index = VectorIndex.load(path);
        System.out.println("index:      " + path);
        System.out.println("embedder:   " + index.embedderName() + " (" + index.dimensions() + " dimensions)");
        System.out.println("chunks:     " + index.size());
        System.out.println("documents:  " + index.sources().size());
        for (String source : index.sources()) {
            long count = index.chunks().stream().filter(chunk -> chunk.source().equals(source)).count();
            System.out.println("  " + source + "  (" + count + " chunks)");
        }
        return 0;
    }

    private static int createModel(Parsed parsed) throws Exception {
        String name = parsed.require(0, "create-model <name>");
        Pipeline.ModelCreation result = Pipeline.createModel(name, parsed.option("base", OllamaChat.DEFAULT_CHAT_MODEL),
            VectorIndex.load(indexPath(parsed)), Path.of(parsed.option("out", ".customllm")),
            parsed.options().get("persona"), !parsed.flag("write-only"));
        System.out.println(Ansi.DIM + "Modelfile:" + Ansi.RESET + " " + result.modelfile());
        System.out.println(result.message());
        System.out.println("\n" + Ansi.YELLOW + "Important:" + Ansi.RESET
            + " a custom model shapes how the model behaves, not what it knows.\n"
            + "A model built this way was observed inventing a quote and citing a passage that did not exist.\n"
            + "For factual answers use `customllm ask`, which retrieves real passages and refuses.");
        return 0;
    }

    private static int export(Parsed parsed) throws Exception {
        Path output = Path.of(parsed.option("out", "training-data.jsonl"));
        int count = Pipeline.exportTrainingData(VectorIndex.load(indexPath(parsed)), output);
        System.out.println(Ansi.GREEN + "Wrote" + Ansi.RESET + " " + count + " training examples to " + output);
        System.out.println("\n" + Ansi.DIM + "This prepares data; it does not train a model." + Ansi.RESET);
        return 0;
    }

    private static Embedder embedder(Parsed parsed) {
        return Embedders.create(parsed.option("embedder", "auto"),
            parsed.option("embed-model", OllamaEmbedder.DEFAULT_EMBED_MODEL),
            parsed.option("ollama-url", OllamaEmbedder.DEFAULT_OLLAMA_URL));
    }

    private static Path indexPath(Parsed parsed) {
        return Path.of(parsed.option("index", Pipeline.DEFAULT_INDEX_PATH.toString()));
    }

    private static void warn(Answer answer) {
        if (!answer.invalidCitations().isEmpty()) {
            System.err.println("\n" + Ansi.RED + "Warning:" + Ansi.RESET + " the answer cited passage(s) "
                + answer.invalidCitations() + ", which were not provided. Treat it with suspicion.");
        }
        if (!answer.weakCitations().isEmpty()) {
            System.err.println("\n" + Ansi.YELLOW + "Note:" + Ansi.RESET + " citation(s) "
                + answer.weakCitations() + " point at passages that share little wording with the answer.");
        }
    }

    private static Parsed parse(List<String> args) {
        Map<String, String> options = new HashMap<>();
        List<String> positions = new ArrayList<>();
        Set<String> flags = Set.of("no-llm", "show-context", "write-only");
        for (int i = 0; i < args.size(); i++) {
            String arg = args.get(i);
            if (!arg.startsWith("--")) {
                positions.add(arg);
            } else if (flags.contains(arg.substring(2))) {
                options.put(arg.substring(2), "true");
            } else {
                if (i + 1 >= args.size()) {
                    throw new IllegalArgumentException("Missing value for " + arg);
                }
                options.put(arg.substring(2), args.get(++i));
            }
        }
        return new Parsed(options, positions);
    }

    private static void forceUtf8() {
        System.setOut(new PrintStream(new FileOutputStream(FileDescriptor.out), true, StandardCharsets.UTF_8));
        System.setErr(new PrintStream(new FileOutputStream(FileDescriptor.err), true, StandardCharsets.UTF_8));
    }

    private record Parsed(Map<String, String> options, List<String> positions) {
        String option(String name, String fallback) {
            return options.getOrDefault(name, fallback);
        }

        boolean flag(String name) {
            return Boolean.parseBoolean(options.getOrDefault(name, "false"));
        }

        int intOption(String name, int fallback) {
            return Integer.parseInt(option(name, Integer.toString(fallback)));
        }

        String require(int index, String usage) {
            if (positions.size() <= index) {
                throw new IllegalArgumentException("Usage: " + usage);
            }
            return positions.get(index);
        }
    }

    private static final class Ansi {
        private static final boolean ENABLED = System.console() != null;
        private static final String RESET = ENABLED ? "\033[0m" : "";
        private static final String DIM = ENABLED ? "\033[90m" : "";
        private static final String RED = ENABLED ? "\033[31m" : "";
        private static final String GREEN = ENABLED ? "\033[32m" : "";
        private static final String YELLOW = ENABLED ? "\033[33m" : "";
        private static final String CYAN = ENABLED ? "\033[36m" : "";
    }
}
