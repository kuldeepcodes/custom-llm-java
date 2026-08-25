package dev.kuldeepcodes.customllm.pipeline;

import java.nio.file.Path;
import java.util.List;

/**
 * Summary of an ingestion run.
 *
 * @param documents readable documents
 * @param chunks indexed chunks
 * @param embedder embedder identity
 * @param dimensions vector dimensions
 * @param indexPath written index path
 * @param skipped skipped document names
 */
public record IngestReport(
    int documents,
    int chunks,
    String embedder,
    int dimensions,
    Path indexPath,
    List<String> skipped
) {
}

