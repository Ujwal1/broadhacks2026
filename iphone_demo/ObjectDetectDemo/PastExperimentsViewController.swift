import AVFoundation
import AVKit
import PhotosUI
import UIKit
import UniformTypeIdentifiers

/// Lists previously recorded experiments. Tapping a row plays the video. Videos can also be
/// imported from the phone's photo library. (Uploading to the web server is a future step.)
final class PastExperimentsViewController: UIViewController {
    private let store = ExperimentStore.shared
    private let tableView = UITableView(frame: .zero, style: .insetGrouped)
    private let emptyLabel = UILabel()
    private let thumbnailCache = NSCache<NSString, UIImage>()

    /// Experiments grouped by protocol, each group ordered newest-first. Sections are ordered
    /// by the recency of their newest experiment (mirrors the newest-first store ordering).
    private var sections: [(protocolTitle: String, experiments: [Experiment])] = []

    private lazy var dateFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .none
        return formatter
    }()

    private lazy var timeFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateStyle = .none
        formatter.timeStyle = .short
        return formatter
    }()

    private func rebuildSections() {
        var order: [String] = []
        var grouped: [String: [Experiment]] = [:]
        for experiment in store.experiments {
            if grouped[experiment.protocolTitle] == nil {
                order.append(experiment.protocolTitle)
            }
            grouped[experiment.protocolTitle, default: []].append(experiment)
        }
        sections = order.map { ($0, grouped[$0] ?? []) }
    }

    override func viewDidLoad() {
        super.viewDidLoad()
        title = "Past Experiments"
        view.backgroundColor = .systemBackground
        configureNavigationBar()
        configureTableView()
        configureEmptyLabel()
        rebuildSections()
        updateEmptyState()
    }

    private func configureNavigationBar() {
        navigationItem.leftBarButtonItem = UIBarButtonItem(
            barButtonSystemItem: .done,
            target: self,
            action: #selector(dismissSelf)
        )
        navigationItem.rightBarButtonItem = UIBarButtonItem(
            image: UIImage(systemName: "photo.on.rectangle"),
            style: .plain,
            target: self,
            action: #selector(importFromPhotoLibrary)
        )
        navigationItem.rightBarButtonItem?.accessibilityLabel = "Import from photo library"
    }

    private func configureTableView() {
        tableView.translatesAutoresizingMaskIntoConstraints = false
        tableView.dataSource = self
        tableView.delegate = self
        tableView.rowHeight = 84
        tableView.register(ExperimentCell.self, forCellReuseIdentifier: ExperimentCell.reuseIdentifier)
        view.addSubview(tableView)

        NSLayoutConstraint.activate([
            tableView.topAnchor.constraint(equalTo: view.topAnchor),
            tableView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            tableView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            tableView.bottomAnchor.constraint(equalTo: view.bottomAnchor)
        ])
    }

    private func configureEmptyLabel() {
        emptyLabel.translatesAutoresizingMaskIntoConstraints = false
        emptyLabel.text = "No experiments yet.\nRecord one from the camera, or import a video."
        emptyLabel.textColor = .secondaryLabel
        emptyLabel.font = .systemFont(ofSize: 15, weight: .medium)
        emptyLabel.numberOfLines = 0
        emptyLabel.textAlignment = .center
        view.addSubview(emptyLabel)

        NSLayoutConstraint.activate([
            emptyLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            emptyLabel.centerYAnchor.constraint(equalTo: view.centerYAnchor),
            emptyLabel.leadingAnchor.constraint(greaterThanOrEqualTo: view.leadingAnchor, constant: 32),
            emptyLabel.trailingAnchor.constraint(lessThanOrEqualTo: view.trailingAnchor, constant: -32)
        ])
    }

    private func updateEmptyState() {
        let isEmpty = store.experiments.isEmpty
        emptyLabel.isHidden = !isEmpty
        tableView.isHidden = isEmpty
    }

    @objc private func dismissSelf() {
        dismiss(animated: true)
    }

    @objc private func importFromPhotoLibrary() {
        var configuration = PHPickerConfiguration()
        configuration.filter = .videos
        configuration.selectionLimit = 1
        let picker = PHPickerViewController(configuration: configuration)
        picker.delegate = self
        present(picker, animated: true)
    }

    private func play(_ experiment: Experiment) {
        let url = store.videoURL(for: experiment)
        guard FileManager.default.fileExists(atPath: url.path) else {
            presentMissingFileAlert()
            return
        }
        let player = AVPlayer(url: url)
        let playerController = AVPlayerViewController()
        playerController.player = player
        present(playerController, animated: true) {
            player.play()
        }
    }

    private func presentMissingFileAlert() {
        let alert = UIAlertController(
            title: "Video unavailable",
            message: "The video file for this experiment could not be found.",
            preferredStyle: .alert
        )
        alert.addAction(UIAlertAction(title: "OK", style: .default))
        present(alert, animated: true)
    }

    /// Generates (and caches) a thumbnail for the first frame of an experiment video.
    private func loadThumbnail(for experiment: Experiment, into cell: ExperimentCell) {
        let key = experiment.id.uuidString as NSString
        if let cached = thumbnailCache.object(forKey: key) {
            cell.setThumbnail(cached)
            return
        }

        let url = store.videoURL(for: experiment)
        DispatchQueue.global(qos: .userInitiated).async { [weak self, weak cell] in
            let asset = AVURLAsset(url: url)
            let generator = AVAssetImageGenerator(asset: asset)
            generator.appliesPreferredTrackTransform = true
            let time = CMTime(seconds: 0.1, preferredTimescale: 600)
            guard let cgImage = try? generator.copyCGImage(at: time, actualTime: nil) else { return }
            let image = UIImage(cgImage: cgImage)
            self?.thumbnailCache.setObject(image, forKey: key)
            DispatchQueue.main.async {
                guard cell?.representedExperimentID == experiment.id else { return }
                cell?.setThumbnail(image)
            }
        }
    }
}

extension PastExperimentsViewController: UITableViewDataSource, UITableViewDelegate {
    func numberOfSections(in tableView: UITableView) -> Int {
        sections.count
    }

    func tableView(_ tableView: UITableView, titleForHeaderInSection section: Int) -> String? {
        sections[section].protocolTitle
    }

    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        sections[section].experiments.count
    }

    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let cell = tableView.dequeueReusableCell(
            withIdentifier: ExperimentCell.reuseIdentifier,
            for: indexPath
        ) as? ExperimentCell ?? ExperimentCell(style: .default, reuseIdentifier: ExperimentCell.reuseIdentifier)

        let experiment = sections[indexPath.section].experiments[indexPath.row]
        cell.configure(
            experiment: experiment,
            dateText: dateFormatter.string(from: experiment.createdAt),
            timeText: timeFormatter.string(from: experiment.createdAt),
            durationText: Self.formatDuration(experiment.duration)
        )
        loadThumbnail(for: experiment, into: cell)
        return cell
    }

    func tableView(_ tableView: UITableView, didSelectRowAt indexPath: IndexPath) {
        tableView.deselectRow(at: indexPath, animated: true)
        play(sections[indexPath.section].experiments[indexPath.row])
    }

    func tableView(
        _ tableView: UITableView,
        trailingSwipeActionsConfigurationForRowAt indexPath: IndexPath
    ) -> UISwipeActionsConfiguration? {
        let delete = UIContextualAction(style: .destructive, title: "Delete") { [weak self] _, _, completion in
            guard let self else { return completion(false) }
            let experiment = self.sections[indexPath.section].experiments[indexPath.row]
            self.store.deleteExperiment(experiment)
            self.rebuildSections()
            self.tableView.reloadData()
            self.updateEmptyState()
            completion(true)
        }
        return UISwipeActionsConfiguration(actions: [delete])
    }

    private static func formatDuration(_ duration: TimeInterval) -> String {
        let totalSeconds = Int(duration.rounded())
        let minutes = totalSeconds / 60
        let seconds = totalSeconds % 60
        return String(format: "%d:%02d", minutes, seconds)
    }
}

extension PastExperimentsViewController: PHPickerViewControllerDelegate {
    func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
        picker.dismiss(animated: true)
        guard let provider = results.first?.itemProvider else { return }

        let movieType = UTType.movie.identifier
        guard provider.hasItemConformingToTypeIdentifier(movieType) else { return }

        provider.loadFileRepresentation(forTypeIdentifier: movieType) { [weak self] url, _ in
            guard let self, let url else { return }
            // loadFileRepresentation hands back a temporary URL that is removed once this
            // closure returns, so copy it to our own staging file synchronously and let the
            // user pick a protocol on the main thread before filing it away.
            let ext = url.pathExtension.isEmpty ? "mov" : url.pathExtension
            let staging = FileManager.default.temporaryDirectory
                .appendingPathComponent("import-\(UUID().uuidString).\(ext)")
            do {
                try FileManager.default.copyItem(at: url, to: staging)
            } catch {
                return
            }

            DispatchQueue.main.async {
                self.promptForProtocol { protocolTitle in
                    defer { try? FileManager.default.removeItem(at: staging) }
                    guard let protocolTitle else { return }
                    guard self.store.importExternalVideo(from: staging, protocolTitle: protocolTitle) != nil else { return }
                    self.rebuildSections()
                    self.tableView.reloadData()
                    self.updateEmptyState()
                }
            }
        }
    }

    /// Asks which protocol an imported video belongs to. Passes `nil` to the completion if the
    /// user cancels. Offers every bundled protocol plus an "Uncategorized" bucket.
    private func promptForProtocol(completion: @escaping (String?) -> Void) {
        let alert = UIAlertController(
            title: "Add to protocol",
            message: "Which protocol is this video for?",
            preferredStyle: .actionSheet
        )

        for title in ProtocolLibrary.loadAll().map(\.title) {
            alert.addAction(UIAlertAction(title: title, style: .default) { _ in completion(title) })
        }
        alert.addAction(UIAlertAction(title: "Uncategorized", style: .default) { _ in completion("Uncategorized") })
        alert.addAction(UIAlertAction(title: "Cancel", style: .cancel) { _ in completion(nil) })

        // Anchor the action sheet on iPad.
        alert.popoverPresentationController?.barButtonItem = navigationItem.rightBarButtonItem
        present(alert, animated: true)
    }
}

/// Table cell showing a video thumbnail, the protocol title, and the recording date/duration.
private final class ExperimentCell: UITableViewCell {
    static let reuseIdentifier = "ExperimentCell"

    private(set) var representedExperimentID: UUID?

    private let thumbnailView = UIImageView()
    private let titleLabel = UILabel()
    private let subtitleLabel = UILabel()
    private let durationBadge = UILabel()

    override init(style: UITableViewCell.CellStyle, reuseIdentifier: String?) {
        super.init(style: style, reuseIdentifier: reuseIdentifier)
        configure()
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        configure()
    }

    private func configure() {
        accessoryType = .disclosureIndicator

        thumbnailView.translatesAutoresizingMaskIntoConstraints = false
        thumbnailView.contentMode = .scaleAspectFill
        thumbnailView.clipsToBounds = true
        thumbnailView.layer.cornerRadius = 8
        thumbnailView.backgroundColor = .secondarySystemFill
        thumbnailView.tintColor = .tertiaryLabel
        contentView.addSubview(thumbnailView)

        durationBadge.translatesAutoresizingMaskIntoConstraints = false
        durationBadge.font = .monospacedDigitSystemFont(ofSize: 11, weight: .semibold)
        durationBadge.textColor = .white
        durationBadge.backgroundColor = UIColor.black.withAlphaComponent(0.6)
        durationBadge.textAlignment = .center
        durationBadge.layer.cornerRadius = 4
        durationBadge.clipsToBounds = true
        contentView.addSubview(durationBadge)

        titleLabel.translatesAutoresizingMaskIntoConstraints = false
        titleLabel.font = .systemFont(ofSize: 16, weight: .semibold)
        titleLabel.textColor = .label
        titleLabel.numberOfLines = 1
        contentView.addSubview(titleLabel)

        subtitleLabel.translatesAutoresizingMaskIntoConstraints = false
        subtitleLabel.font = .systemFont(ofSize: 13, weight: .regular)
        subtitleLabel.textColor = .secondaryLabel
        subtitleLabel.numberOfLines = 1
        contentView.addSubview(subtitleLabel)

        NSLayoutConstraint.activate([
            thumbnailView.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 4),
            thumbnailView.centerYAnchor.constraint(equalTo: contentView.centerYAnchor),
            thumbnailView.widthAnchor.constraint(equalToConstant: 96),
            thumbnailView.heightAnchor.constraint(equalToConstant: 64),

            durationBadge.trailingAnchor.constraint(equalTo: thumbnailView.trailingAnchor, constant: -4),
            durationBadge.bottomAnchor.constraint(equalTo: thumbnailView.bottomAnchor, constant: -4),
            durationBadge.heightAnchor.constraint(equalToConstant: 16),
            durationBadge.widthAnchor.constraint(greaterThanOrEqualToConstant: 34),

            titleLabel.leadingAnchor.constraint(equalTo: thumbnailView.trailingAnchor, constant: 12),
            titleLabel.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -8),
            titleLabel.bottomAnchor.constraint(equalTo: contentView.centerYAnchor, constant: -2),

            subtitleLabel.leadingAnchor.constraint(equalTo: titleLabel.leadingAnchor),
            subtitleLabel.trailingAnchor.constraint(equalTo: titleLabel.trailingAnchor),
            subtitleLabel.topAnchor.constraint(equalTo: contentView.centerYAnchor, constant: 2)
        ])
    }

    func configure(experiment: Experiment, dateText: String, timeText: String, durationText: String) {
        representedExperimentID = experiment.id
        titleLabel.text = dateText
        subtitleLabel.text = timeText
        durationBadge.text = " \(durationText) "
        setThumbnail(nil)
    }

    func setThumbnail(_ image: UIImage?) {
        if let image {
            thumbnailView.image = image
            thumbnailView.contentMode = .scaleAspectFill
        } else {
            thumbnailView.image = UIImage(systemName: "video.fill")
            thumbnailView.contentMode = .center
        }
    }
}
