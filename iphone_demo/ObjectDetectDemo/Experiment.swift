import AVFoundation
import Foundation

/// A single recorded experiment run. Designed to be uploaded to the web server later,
/// so it carries enough metadata to identify the run and attach inference results.
struct Experiment: Codable, Identifiable {
    let id: UUID
    var protocolTitle: String
    var createdAt: Date
    var duration: TimeInterval
    /// File name relative to the experiments directory (videos are stored on disk, not inline).
    var videoFileName: String
    /// Placeholder for inference runs produced from this video. Populated once the
    /// server-side inference pipeline is wired up.
    var inferenceRuns: [InferenceRun]
    /// Tracks whether the video + metadata have been pushed to the web server yet.
    var isUploaded: Bool

    init(
        id: UUID = UUID(),
        protocolTitle: String,
        createdAt: Date,
        duration: TimeInterval,
        videoFileName: String,
        inferenceRuns: [InferenceRun] = [],
        isUploaded: Bool = false
    ) {
        self.id = id
        self.protocolTitle = protocolTitle
        self.createdAt = createdAt
        self.duration = duration
        self.videoFileName = videoFileName
        self.inferenceRuns = inferenceRuns
        self.isUploaded = isUploaded
    }
}

/// A single inference pass over an experiment video. Filled in by the (future) server pipeline.
struct InferenceRun: Codable, Identifiable {
    let id: UUID
    var createdAt: Date
    var summary: String

    init(id: UUID = UUID(), createdAt: Date, summary: String) {
        self.id = id
        self.createdAt = createdAt
        self.summary = summary
    }
}

/// On-device persistence for recorded experiments. Videos live as files under
/// Documents/Experiments and metadata is serialized to a single JSON manifest.
///
/// This is intentionally local-only for now. When the web server is ready, an uploader
/// can walk `experiments` (filtering on `isUploaded == false`) and POST each video +
/// metadata, then flip `isUploaded` and persist.
final class ExperimentStore {
    static let shared = ExperimentStore()

    private let fileManager = FileManager.default
    private let manifestQueue = DispatchQueue(label: "experiment.store.manifest")

    private(set) var experiments: [Experiment] = []

    private init() {
        createDirectoryIfNeeded()
        load()
    }

    /// Directory that holds the recorded video files.
    var experimentsDirectory: URL {
        let documents = fileManager.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return documents.appendingPathComponent("Experiments", isDirectory: true)
    }

    private var manifestURL: URL {
        experimentsDirectory.appendingPathComponent("experiments.json")
    }

    func videoURL(for experiment: Experiment) -> URL {
        experimentsDirectory.appendingPathComponent(experiment.videoFileName)
    }

    /// A unique destination URL to record a fresh video into.
    func newVideoURL() -> URL {
        createDirectoryIfNeeded()
        let fileName = "experiment-\(UUID().uuidString).mov"
        return experimentsDirectory.appendingPathComponent(fileName)
    }

    /// Registers a video that already lives at `videoURL` (inside the experiments dir).
    func addExperiment(videoURL: URL, protocolTitle: String, duration: TimeInterval, createdAt: Date) {
        let experiment = Experiment(
            protocolTitle: protocolTitle,
            createdAt: createdAt,
            duration: duration,
            videoFileName: videoURL.lastPathComponent
        )
        experiments.insert(experiment, at: 0)
        save()
    }

    /// Copies an external video (e.g. picked from the photo library) into the experiments
    /// directory and registers it.
    @discardableResult
    func importExternalVideo(from sourceURL: URL, protocolTitle: String) -> Experiment? {
        createDirectoryIfNeeded()
        let destination = experimentsDirectory.appendingPathComponent("experiment-\(UUID().uuidString).mov")
        do {
            try fileManager.copyItem(at: sourceURL, to: destination)
        } catch {
            return nil
        }

        let duration = AVURLAssetDurationSeconds(destination)
        let experiment = Experiment(
            protocolTitle: protocolTitle,
            createdAt: Date(),
            duration: duration,
            videoFileName: destination.lastPathComponent
        )
        experiments.insert(experiment, at: 0)
        save()
        return experiment
    }

    func deleteExperiment(_ experiment: Experiment) {
        let url = videoURL(for: experiment)
        try? fileManager.removeItem(at: url)
        experiments.removeAll { $0.id == experiment.id }
        save()
    }

    private func createDirectoryIfNeeded() {
        let directory = experimentsDirectory
        guard !fileManager.fileExists(atPath: directory.path) else { return }
        try? fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    private func load() {
        manifestQueue.sync {
            guard
                let data = try? Data(contentsOf: manifestURL),
                let decoded = try? JSONDecoder.experimentDecoder.decode([Experiment].self, from: data)
            else { return }
            experiments = decoded
        }
    }

    private func save() {
        let snapshot = experiments
        manifestQueue.async { [manifestURL] in
            guard let data = try? JSONEncoder.experimentEncoder.encode(snapshot) else { return }
            try? data.write(to: manifestURL, options: .atomic)
        }
    }
}

/// Synchronously reads a video's duration. Only called off the main thread (from the import
/// path), so blocking on the async loader here is safe.
private func AVURLAssetDurationSeconds(_ url: URL) -> TimeInterval {
    let asset = AVURLAsset(url: url)
    let semaphore = DispatchSemaphore(value: 0)
    var result: TimeInterval = 0
    Task {
        if let duration = try? await asset.load(.duration) {
            let seconds = CMTimeGetSeconds(duration)
            result = seconds.isFinite ? seconds : 0
        }
        semaphore.signal()
    }
    semaphore.wait()
    return result
}

private extension JSONEncoder {
    static var experimentEncoder: JSONEncoder {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return encoder
    }
}

private extension JSONDecoder {
    static var experimentDecoder: JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }
}
