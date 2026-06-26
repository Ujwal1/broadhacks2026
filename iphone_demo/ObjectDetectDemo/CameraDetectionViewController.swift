import AVFoundation
import CoreML
import Photos
import UIKit
import Vision

final class CameraDetectionViewController: UIViewController {
    private let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let movieOutput = AVCaptureMovieFileOutput()
    private let videoQueue = DispatchQueue(label: "camera.frames.queue", qos: .userInteractive)
    private let visionQueue = DispatchQueue(label: "vision.requests.queue", qos: .userInitiated)

    private var previewLayer: AVCaptureVideoPreviewLayer!
    private let overlayLayer = CALayer()
    private let statusLabel = UILabel()
    private let fpsLabel = UILabel()
    private let scanLabel = UILabel()
    private let protocolButton = UIButton(type: .system)
    private let detectionToggleButton = UIButton(type: .system)
    private let protocolPanel = UIView()
    private let protocolTitleLabel = UILabel()
    private let protocolTableView = UITableView(frame: .zero, style: .plain)
    private let doneButton = UIButton(type: .system)
    private let flagButton = UIButton(type: .system)
    private let uncertainButton = UIButton(type: .system)
    private let missedButton = UIButton(type: .system)
    private let memoButton = UIButton(type: .system)
    private let currentStepButton = UIButton(type: .system)
    private let resetProgressButton = UIButton(type: .system)
    private let recordButton = UIButton(type: .custom)
    private let recordButtonInnerView = UIView()
    private let pastExperimentsButton = UIButton(type: .system)
    private let recordingIndicatorLabel = UILabel()
    private var recordingTimer: Timer?
    private var recordingStartTime: Date?
    private var isRecording = false
    private var protocolPanelLeadingConstraint: NSLayoutConstraint?
    private let protocols = ProtocolLibrary.loadAll()
    private lazy var selectedProtocol: LabProtocol = protocols.first ?? LabProtocol.placeholder
    private lazy var protocolStates: [LabProtocol: ProtocolRunState] = Dictionary(
        uniqueKeysWithValues: protocols.map { ($0, ProtocolRunState(stepCount: $0.steps.count)) }
    )
    private var selectedStepIndex = 0
    private var isProtocolPanelOpen = false
    private var shouldShowDetectionBoxes = true
    private var lastDetectionTime: CFTimeInterval = 0
    private var lastCameraFPSUpdateTime: CFTimeInterval = 0
    private var cameraFrameIndex = 0
    private var inFlightFrameCount = 0
    private let maxInFlightFrames = 1
    private let minimumFrameInterval: CFTimeInterval = 1.0 / 10.0
    private let inferenceFrameStride = 6
    private let targetObjectLabels = TargetObjectLabels.all
    private let metalMatrixMultiplier = MetalMatrixMultiplier()
    private let metalQueue = DispatchQueue(label: "metal.matrix.queue", qos: .userInitiated)
    private var shouldRunMetalBenchmark = true

    #if DEBUG
    private var cameraFPSTracker = FPSTracker()
    #endif

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .black
        configureStatusLabel()
        configureFPSLabel()
        configureScanLabel()
        configureProtocolMenu()
        configureDetectionToggle()
        configureProtocolPanel()
        configureRecordingControls()
        configureProtocolGestures()
        startMetalMatrixBenchmark()
        checkCameraPermission()
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        previewLayer?.frame = view.bounds
        overlayLayer.frame = view.bounds
    }

    private func configureStatusLabel() {
        statusLabel.translatesAutoresizingMaskIntoConstraints = false
        statusLabel.text = ""
        statusLabel.textColor = .white
        statusLabel.font = .systemFont(ofSize: 14, weight: .semibold)
        statusLabel.backgroundColor = UIColor.black.withAlphaComponent(0.45)
        statusLabel.layer.cornerRadius = 8
        statusLabel.clipsToBounds = true
        statusLabel.textAlignment = .left
        statusLabel.lineBreakMode = .byTruncatingTail
        view.addSubview(statusLabel)

        NSLayoutConstraint.activate([
            statusLabel.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor, constant: 16),
            statusLabel.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -16),
            statusLabel.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -16),
            statusLabel.heightAnchor.constraint(equalToConstant: 36)
        ])
    }

    private func configureFPSLabel() {
        fpsLabel.translatesAutoresizingMaskIntoConstraints = false
        fpsLabel.textColor = .white
        fpsLabel.font = .monospacedDigitSystemFont(ofSize: 13, weight: .semibold)
        fpsLabel.backgroundColor = UIColor.black.withAlphaComponent(0.35)
        fpsLabel.layer.cornerRadius = 8
        fpsLabel.clipsToBounds = true
        fpsLabel.textAlignment = .center
        fpsLabel.text = "FPS --"
        view.addSubview(fpsLabel)

        NSLayoutConstraint.activate([
            fpsLabel.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 12),
            fpsLabel.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -12),
            fpsLabel.widthAnchor.constraint(equalToConstant: 76),
            fpsLabel.heightAnchor.constraint(equalToConstant: 32)
        ])

        #if !DEBUG
        fpsLabel.isHidden = true
        #endif
    }

    private func configureScanLabel() {
        scanLabel.translatesAutoresizingMaskIntoConstraints = false
        scanLabel.textColor = .white
        scanLabel.font = .systemFont(ofSize: 12, weight: .semibold)
        scanLabel.backgroundColor = UIColor.black.withAlphaComponent(0.35)
        scanLabel.layer.cornerRadius = 8
        scanLabel.clipsToBounds = true
        scanLabel.textAlignment = .center
        scanLabel.text = "Starting camera..."
        view.addSubview(scanLabel)

        NSLayoutConstraint.activate([
            scanLabel.topAnchor.constraint(equalTo: fpsLabel.bottomAnchor, constant: 8),
            scanLabel.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -12),
            scanLabel.widthAnchor.constraint(equalToConstant: 132),
            scanLabel.heightAnchor.constraint(equalToConstant: 32)
        ])
    }

    private func configureProtocolMenu() {
        protocolButton.translatesAutoresizingMaskIntoConstraints = false
        protocolButton.tintColor = .white
        protocolButton.backgroundColor = UIColor.black.withAlphaComponent(0.18)
        protocolButton.layer.cornerRadius = 8
        protocolButton.clipsToBounds = true
        protocolButton.contentHorizontalAlignment = .leading
        var configuration = UIButton.Configuration.plain()
        configuration.baseForegroundColor = .white
        configuration.contentInsets = NSDirectionalEdgeInsets(top: 8, leading: 10, bottom: 8, trailing: 10)
        protocolButton.configuration = configuration
        protocolButton.titleLabel?.font = .systemFont(ofSize: 12, weight: .semibold)
        protocolButton.titleLabel?.adjustsFontSizeToFitWidth = true
        protocolButton.titleLabel?.minimumScaleFactor = 0.78
        protocolButton.showsMenuAsPrimaryAction = true
        view.addSubview(protocolButton)

        NSLayoutConstraint.activate([
            protocolButton.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 12),
            protocolButton.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor, constant: 12),
            protocolButton.widthAnchor.constraint(equalToConstant: 220),
            protocolButton.heightAnchor.constraint(equalToConstant: 36)
        ])

        updateProtocolMenu()
    }

    private func configureDetectionToggle() {
        detectionToggleButton.translatesAutoresizingMaskIntoConstraints = false
        detectionToggleButton.backgroundColor = UIColor.black.withAlphaComponent(0.18)
        detectionToggleButton.layer.cornerRadius = 8
        detectionToggleButton.clipsToBounds = true
        detectionToggleButton.titleLabel?.font = .systemFont(ofSize: 12, weight: .semibold)
        detectionToggleButton.addTarget(self, action: #selector(toggleDetectionBoxes), for: .touchUpInside)
        view.addSubview(detectionToggleButton)

        NSLayoutConstraint.activate([
            detectionToggleButton.topAnchor.constraint(equalTo: scanLabel.bottomAnchor, constant: 8),
            detectionToggleButton.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -12),
            detectionToggleButton.widthAnchor.constraint(equalToConstant: 112),
            detectionToggleButton.heightAnchor.constraint(equalToConstant: 32)
        ])

        updateDetectionToggleTitle()
    }

    private func configureProtocolPanel() {
        protocolPanel.translatesAutoresizingMaskIntoConstraints = false
        protocolPanel.backgroundColor = UIColor.black.withAlphaComponent(0.45)
        protocolPanel.layer.cornerRadius = 8
        protocolPanel.clipsToBounds = true
        view.addSubview(protocolPanel)

        protocolTitleLabel.translatesAutoresizingMaskIntoConstraints = false
        protocolTitleLabel.textColor = .white
        protocolTitleLabel.font = .systemFont(ofSize: 16, weight: .bold)
        protocolTitleLabel.numberOfLines = 2
        protocolPanel.addSubview(protocolTitleLabel)

        protocolTableView.translatesAutoresizingMaskIntoConstraints = false
        protocolTableView.backgroundColor = .clear
        protocolTableView.separatorStyle = .none
        protocolTableView.rowHeight = UITableView.automaticDimension
        protocolTableView.estimatedRowHeight = 72
        protocolTableView.dataSource = self
        protocolTableView.delegate = self
        protocolTableView.register(ProtocolStepCell.self, forCellReuseIdentifier: ProtocolStepCell.reuseIdentifier)
        protocolPanel.addSubview(protocolTableView)

        let actionStack = UIStackView(arrangedSubviews: [doneButton, flagButton])
        actionStack.translatesAutoresizingMaskIntoConstraints = false
        actionStack.axis = .horizontal
        actionStack.spacing = 8
        actionStack.distribution = .fillEqually
        protocolPanel.addSubview(actionStack)

        let flagStack = UIStackView(arrangedSubviews: [uncertainButton, missedButton])
        flagStack.translatesAutoresizingMaskIntoConstraints = false
        flagStack.axis = .horizontal
        flagStack.spacing = 8
        flagStack.distribution = .fillEqually
        protocolPanel.addSubview(flagStack)

        let utilityStack = UIStackView(arrangedSubviews: [memoButton, currentStepButton])
        utilityStack.translatesAutoresizingMaskIntoConstraints = false
        utilityStack.axis = .horizontal
        utilityStack.spacing = 8
        utilityStack.distribution = .fillEqually
        protocolPanel.addSubview(utilityStack)

        let resetStack = UIStackView(arrangedSubviews: [resetProgressButton])
        resetStack.translatesAutoresizingMaskIntoConstraints = false
        resetStack.axis = .horizontal
        resetStack.spacing = 8
        resetStack.distribution = .fillEqually
        protocolPanel.addSubview(resetStack)

        configurePanelButton(doneButton, title: "Done", color: .systemGreen)
        configurePanelButton(flagButton, title: "Flag", color: .systemYellow, textColor: .black)
        configurePanelButton(uncertainButton, title: "Uncertain", color: .systemYellow, textColor: .black)
        configurePanelButton(missedButton, title: "Appears missed", color: .systemRed)
        configurePanelButton(memoButton, title: "Add memo", color: .systemBlue)
        configurePanelButton(currentStepButton, title: "Jump to current step", color: .darkGray)
        configurePanelButton(resetProgressButton, title: "Reset progress", color: .systemGray)

        doneButton.addTarget(self, action: #selector(markSelectedStepDone), for: .touchUpInside)
        flagButton.addTarget(self, action: #selector(showFlagOptions), for: .touchUpInside)
        uncertainButton.addTarget(self, action: #selector(markSelectedStepUncertain), for: .touchUpInside)
        missedButton.addTarget(self, action: #selector(markSelectedStepMissed), for: .touchUpInside)
        memoButton.addTarget(self, action: #selector(addMemoForSelectedStep), for: .touchUpInside)
        currentStepButton.addTarget(self, action: #selector(jumpToCurrentStep), for: .touchUpInside)
        resetProgressButton.addTarget(self, action: #selector(resetProtocolProgress), for: .touchUpInside)

        protocolPanelLeadingConstraint = protocolPanel.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: -332)
        protocolPanelLeadingConstraint?.isActive = true

        NSLayoutConstraint.activate([
            protocolPanel.topAnchor.constraint(equalTo: protocolButton.bottomAnchor, constant: 10),
            protocolPanel.widthAnchor.constraint(equalToConstant: 320),
            protocolPanel.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -64),

            protocolTitleLabel.topAnchor.constraint(equalTo: protocolPanel.topAnchor, constant: 14),
            protocolTitleLabel.leadingAnchor.constraint(equalTo: protocolPanel.leadingAnchor, constant: 14),
            protocolTitleLabel.trailingAnchor.constraint(equalTo: protocolPanel.trailingAnchor, constant: -14),

            protocolTableView.topAnchor.constraint(equalTo: protocolTitleLabel.bottomAnchor, constant: 6),
            protocolTableView.leadingAnchor.constraint(equalTo: protocolPanel.leadingAnchor, constant: 8),
            protocolTableView.trailingAnchor.constraint(equalTo: protocolPanel.trailingAnchor, constant: -8),
            protocolTableView.bottomAnchor.constraint(equalTo: utilityStack.topAnchor, constant: -8),

            utilityStack.leadingAnchor.constraint(equalTo: protocolPanel.leadingAnchor, constant: 10),
            utilityStack.trailingAnchor.constraint(equalTo: protocolPanel.trailingAnchor, constant: -10),
            utilityStack.bottomAnchor.constraint(equalTo: resetStack.topAnchor, constant: -8),
            utilityStack.heightAnchor.constraint(equalToConstant: 34),

            resetStack.leadingAnchor.constraint(equalTo: protocolPanel.leadingAnchor, constant: 10),
            resetStack.trailingAnchor.constraint(equalTo: protocolPanel.trailingAnchor, constant: -10),
            resetStack.bottomAnchor.constraint(equalTo: flagStack.topAnchor, constant: -8),
            resetStack.heightAnchor.constraint(equalToConstant: 34),

            flagStack.leadingAnchor.constraint(equalTo: protocolPanel.leadingAnchor, constant: 10),
            flagStack.trailingAnchor.constraint(equalTo: protocolPanel.trailingAnchor, constant: -10),
            flagStack.bottomAnchor.constraint(equalTo: actionStack.topAnchor, constant: -8),
            flagStack.heightAnchor.constraint(equalToConstant: 34),

            actionStack.leadingAnchor.constraint(equalTo: protocolPanel.leadingAnchor, constant: 10),
            actionStack.trailingAnchor.constraint(equalTo: protocolPanel.trailingAnchor, constant: -10),
            actionStack.bottomAnchor.constraint(equalTo: protocolPanel.bottomAnchor, constant: -10),
            actionStack.heightAnchor.constraint(equalToConstant: 38)
        ])

        renderSelectedProtocol()
    }

    private func configureProtocolGestures() {
        let openGesture = UIScreenEdgePanGestureRecognizer(target: self, action: #selector(handleOpenProtocolPanelGesture(_:)))
        openGesture.edges = .left
        view.addGestureRecognizer(openGesture)

        let closeGesture = UISwipeGestureRecognizer(target: self, action: #selector(handleCloseProtocolPanelGesture(_:)))
        closeGesture.direction = .left
        protocolPanel.addGestureRecognizer(closeGesture)
    }

    /// Panel/control buttons that get darkened and disabled while recording is in progress.
    private var dimmableControls: [UIView] {
        [
            protocolButton, detectionToggleButton, pastExperimentsButton,
            doneButton, flagButton, uncertainButton, missedButton,
            memoButton, currentStepButton, resetProgressButton
        ]
    }

    private func configureRecordingControls() {
        // Circular record button, bottom-center.
        recordButton.translatesAutoresizingMaskIntoConstraints = false
        recordButton.backgroundColor = .clear
        recordButton.layer.cornerRadius = 34
        recordButton.layer.borderWidth = 4
        recordButton.layer.borderColor = UIColor.white.cgColor
        recordButton.addTarget(self, action: #selector(toggleRecording), for: .touchUpInside)
        view.addSubview(recordButton)

        recordButtonInnerView.translatesAutoresizingMaskIntoConstraints = false
        recordButtonInnerView.backgroundColor = .systemRed
        recordButtonInnerView.layer.cornerRadius = 26
        recordButtonInnerView.isUserInteractionEnabled = false
        recordButton.addSubview(recordButtonInnerView)

        // "Past experiments" button, bottom-trailing.
        pastExperimentsButton.translatesAutoresizingMaskIntoConstraints = false
        pastExperimentsButton.backgroundColor = UIColor.black.withAlphaComponent(0.35)
        pastExperimentsButton.layer.cornerRadius = 8
        pastExperimentsButton.clipsToBounds = true
        pastExperimentsButton.tintColor = .white
        var pastConfig = UIButton.Configuration.plain()
        pastConfig.baseForegroundColor = .white
        pastConfig.image = UIImage(systemName: "tray.full")
        pastConfig.imagePadding = 5
        pastConfig.title = "Past"
        pastConfig.contentInsets = NSDirectionalEdgeInsets(top: 6, leading: 8, bottom: 6, trailing: 8)
        pastConfig.titleTextAttributesTransformer = UIConfigurationTextAttributesTransformer { attributes in
            var updated = attributes
            updated.font = .systemFont(ofSize: 12, weight: .semibold)
            return updated
        }
        pastExperimentsButton.configuration = pastConfig
        pastExperimentsButton.addTarget(self, action: #selector(openPastExperiments), for: .touchUpInside)
        view.addSubview(pastExperimentsButton)

        // Recording indicator (red dot + elapsed time), top-center.
        recordingIndicatorLabel.translatesAutoresizingMaskIntoConstraints = false
        recordingIndicatorLabel.textColor = .white
        recordingIndicatorLabel.font = .monospacedDigitSystemFont(ofSize: 14, weight: .bold)
        recordingIndicatorLabel.backgroundColor = UIColor.systemRed.withAlphaComponent(0.85)
        recordingIndicatorLabel.textAlignment = .center
        recordingIndicatorLabel.layer.cornerRadius = 8
        recordingIndicatorLabel.clipsToBounds = true
        recordingIndicatorLabel.isHidden = true
        view.addSubview(recordingIndicatorLabel)

        NSLayoutConstraint.activate([
            recordButton.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            recordButton.bottomAnchor.constraint(equalTo: statusLabel.topAnchor, constant: -16),
            recordButton.widthAnchor.constraint(equalToConstant: 68),
            recordButton.heightAnchor.constraint(equalToConstant: 68),

            recordButtonInnerView.centerXAnchor.constraint(equalTo: recordButton.centerXAnchor),
            recordButtonInnerView.centerYAnchor.constraint(equalTo: recordButton.centerYAnchor),
            recordButtonInnerView.widthAnchor.constraint(equalToConstant: 52),
            recordButtonInnerView.heightAnchor.constraint(equalToConstant: 52),

            pastExperimentsButton.centerYAnchor.constraint(equalTo: recordButton.centerYAnchor),
            pastExperimentsButton.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -16),
            pastExperimentsButton.heightAnchor.constraint(equalToConstant: 36),

            recordingIndicatorLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            recordingIndicatorLabel.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 12),
            recordingIndicatorLabel.heightAnchor.constraint(equalToConstant: 30),
            recordingIndicatorLabel.widthAnchor.constraint(greaterThanOrEqualToConstant: 96)
        ])
    }

    @objc private func openPastExperiments() {
        guard !isRecording else { return }
        let pastExperiments = PastExperimentsViewController()
        let navigation = UINavigationController(rootViewController: pastExperiments)
        navigation.modalPresentationStyle = .fullScreen
        present(navigation, animated: true)
    }

    @objc private func toggleRecording() {
        isRecording ? stopRecording() : startRecording()
    }

    private func startRecording() {
        guard session.isRunning, !movieOutput.isRecording else { return }
        guard movieOutput.connection(with: .video) != nil else { return }

        let outputURL = ExperimentStore.shared.newVideoURL()
        // newVideoURL() returns a fresh path, but AVFoundation refuses to overwrite.
        try? FileManager.default.removeItem(at: outputURL)

        recordingStartTime = Date()
        isRecording = true
        setControlsRecording(true)
        startRecordingTimer()
        movieOutput.startRecording(to: outputURL, recordingDelegate: self)
    }

    private func stopRecording() {
        guard movieOutput.isRecording else { return }
        movieOutput.stopRecording()
    }

    /// Dims and disables the panel controls while recording, and animates the record button
    /// between its idle (circle) and recording (rounded square) states.
    private func setControlsRecording(_ recording: Bool) {
        if recording {
            setProtocolPanel(open: false, animated: true)
        }

        UIView.animate(withDuration: 0.2) {
            for control in self.dimmableControls {
                control.alpha = recording ? 0.25 : 1.0
                control.isUserInteractionEnabled = !recording
            }
            self.recordButtonInnerView.layer.cornerRadius = recording ? 8 : 26
            self.recordButtonInnerView.transform = recording
                ? CGAffineTransform(scaleX: 0.62, y: 0.62)
                : .identity
        }

        recordingIndicatorLabel.isHidden = !recording
    }

    private func startRecordingTimer() {
        recordingIndicatorLabel.text = "● 0:00"
        recordingTimer?.invalidate()
        recordingTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
            guard let self, let start = self.recordingStartTime else { return }
            let elapsed = Int(Date().timeIntervalSince(start).rounded())
            self.recordingIndicatorLabel.text = String(format: "● %d:%02d", elapsed / 60, elapsed % 60)
        }
    }

    private func stopRecordingTimer() {
        recordingTimer?.invalidate()
        recordingTimer = nil
    }

    /// Persists a finished recording to the experiment store and offers to copy it into the
    /// system photo library. (Server upload is intentionally not wired up yet.)
    private func saveRecording(at outputURL: URL) {
        let duration = recordingStartTime.map { Date().timeIntervalSince($0) } ?? 0
        ExperimentStore.shared.addExperiment(
            videoURL: outputURL,
            protocolTitle: selectedProtocol.title,
            duration: duration,
            createdAt: recordingStartTime ?? Date()
        )
        recordingStartTime = nil
        saveToPhotoLibraryIfAuthorized(outputURL)
        presentRecordingSavedConfirmation()
    }

    private func saveToPhotoLibraryIfAuthorized(_ url: URL) {
        PHPhotoLibrary.requestAuthorization(for: .addOnly) { status in
            guard status == .authorized || status == .limited else { return }
            PHPhotoLibrary.shared().performChanges {
                PHAssetChangeRequest.creationRequestForAssetFromVideo(atFileURL: url)
            }
        }
    }

    private func presentRecordingSavedConfirmation() {
        let alert = UIAlertController(
            title: "Recording saved",
            message: "Saved to Past Experiments.",
            preferredStyle: .alert
        )
        alert.addAction(UIAlertAction(title: "OK", style: .default))
        alert.addAction(UIAlertAction(title: "View", style: .default) { [weak self] _ in
            self?.openPastExperiments()
        })
        present(alert, animated: true)
    }

    private func updateProtocolMenu() {
        protocolButton.configuration?.title = selectedProtocol.title
        protocolButton.menu = UIMenu(children: protocols.map { labProtocol in
            UIAction(title: labProtocol.title, state: labProtocol == selectedProtocol ? .on : .off) { [weak self] _ in
                self?.select(labProtocol)
            }
        })
    }

    private func select(_ labProtocol: LabProtocol) {
        selectedProtocol = labProtocol
        selectedStepIndex = state.currentStepIndex
        updateProtocolMenu()
        renderSelectedProtocol()
        updateCurrentStepSummary()
        setProtocolPanel(open: true, animated: true)
    }

    private func renderSelectedProtocol() {
        protocolTitleLabel.text = selectedProtocol.heading
        uncertainButton.isHidden = true
        missedButton.isHidden = true
        protocolTableView.reloadData()
        scrollToSelectedStep(animated: false)
        updateCurrentStepSummary()
    }

    private func configurePanelButton(_ button: UIButton, title: String, color: UIColor, textColor: UIColor = .white) {
        button.setTitle(title, for: .normal)
        button.setTitleColor(textColor, for: .normal)
        button.titleLabel?.font = .systemFont(ofSize: 13, weight: .bold)
        button.backgroundColor = color.withAlphaComponent(0.92)
        button.layer.cornerRadius = 8
        button.clipsToBounds = true
    }

    private var state: ProtocolRunState {
        get {
            protocolStates[selectedProtocol] ?? ProtocolRunState(stepCount: selectedProtocol.steps.count)
        }
        set {
            protocolStates[selectedProtocol] = newValue
        }
    }

    @objc private func toggleDetectionBoxes() {
        shouldShowDetectionBoxes.toggle()
        overlayLayer.isHidden = !shouldShowDetectionBoxes
        if !shouldShowDetectionBoxes {
            overlayLayer.sublayers?.forEach { $0.removeFromSuperlayer() }
        }
        updateDetectionToggleTitle()
    }

    private func updateDetectionToggleTitle() {
        let title = shouldShowDetectionBoxes ? "Boxes: On" : "Boxes: Off"
        detectionToggleButton.setTitle(title, for: .normal)
        detectionToggleButton.setTitleColor(.white, for: .normal)
    }

    @objc private func markSelectedStepDone() {
        setSelectedStepStatus(.done)
    }

    @objc private func showFlagOptions() {
        uncertainButton.isHidden.toggle()
        missedButton.isHidden = uncertainButton.isHidden
    }

    @objc private func markSelectedStepUncertain() {
        setSelectedStepStatus(.uncertain)
    }

    @objc private func markSelectedStepMissed() {
        setSelectedStepStatus(.missed)
    }

    private func setSelectedStepStatus(_ status: ProtocolStepStatus) {
        var nextState = state
        nextState.stepStatuses[selectedStepIndex] = status

        if selectedStepIndex == nextState.currentStepIndex {
            nextState.currentStepIndex = min(selectedStepIndex + 1, selectedProtocol.steps.count - 1)
        }

        state = nextState
        selectedStepIndex = nextState.currentStepIndex
        uncertainButton.isHidden = true
        missedButton.isHidden = true
        protocolTableView.reloadData()
        scrollToSelectedStep(animated: true)
        updateCurrentStepSummary()
    }

    @objc private func addMemoForSelectedStep() {
        let step = selectedProtocol.steps[selectedStepIndex]
        let memo = state.stepMemos[selectedStepIndex] ?? ""
        let alert = UIAlertController(title: "Step \(step.number) memo", message: step.text, preferredStyle: .alert)
        alert.addTextField { textField in
            textField.placeholder = "Note"
            textField.text = memo
            textField.clearButtonMode = .whileEditing
        }
        alert.addAction(UIAlertAction(title: "Cancel", style: .cancel))
        alert.addAction(UIAlertAction(title: "Save", style: .default) { [weak self, weak alert] _ in
            guard let self else { return }
            let note = alert?.textFields?.first?.text?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            var nextState = self.state
            nextState.stepMemos[self.selectedStepIndex] = note.isEmpty ? nil : note
            self.state = nextState
            self.protocolTableView.reloadRows(at: [IndexPath(row: self.selectedStepIndex, section: 0)], with: .automatic)
            self.updateCurrentStepSummary()
        })
        present(alert, animated: true)
    }

    @objc private func jumpToCurrentStep() {
        selectedStepIndex = state.currentStepIndex
        protocolTableView.reloadData()
        scrollToSelectedStep(animated: true)
        updateCurrentStepSummary()
    }

    @objc private func resetProtocolProgress() {
        state = ProtocolRunState(stepCount: selectedProtocol.steps.count)
        selectedStepIndex = 0
        uncertainButton.isHidden = true
        missedButton.isHidden = true
        protocolTableView.reloadData()
        scrollToSelectedStep(animated: true)
        updateCurrentStepSummary()
    }

    private func updateCurrentStepSummary() {
        let currentIndex = state.currentStepIndex
        guard selectedProtocol.steps.indices.contains(currentIndex) else {
            statusLabel.text = ""
            return
        }

        let step = selectedProtocol.steps[currentIndex]
        statusLabel.text = "Step \(step.number): \(step.text)"
    }

    private func scrollToSelectedStep(animated: Bool) {
        guard selectedStepIndex < selectedProtocol.steps.count else { return }
        let indexPath = IndexPath(row: selectedStepIndex, section: 0)
        protocolTableView.scrollToRow(at: indexPath, at: .middle, animated: animated)
    }

    @objc private func handleOpenProtocolPanelGesture(_ recognizer: UIScreenEdgePanGestureRecognizer) {
        guard recognizer.state == .ended else { return }
        setProtocolPanel(open: true, animated: true)
    }

    @objc private func handleCloseProtocolPanelGesture(_ recognizer: UISwipeGestureRecognizer) {
        setProtocolPanel(open: false, animated: true)
    }

    private func setProtocolPanel(open: Bool, animated: Bool) {
        isProtocolPanelOpen = open
        protocolPanelLeadingConstraint?.constant = open ? 12 : -332

        let animations = {
            self.view.layoutIfNeeded()
        }

        if animated {
            UIView.animate(withDuration: 0.22, delay: 0, options: [.curveEaseOut], animations: animations)
        } else {
            animations()
        }
    }

    private func startMetalMatrixBenchmark() {
        guard metalMatrixMultiplier != nil else { return }

        metalQueue.async { [weak self] in
            self?.runMetalMatrixBenchmarkLoop()
        }
    }

    private func runMetalMatrixBenchmarkLoop() {
        guard shouldRunMetalBenchmark, let metalMatrixMultiplier else { return }

        metalMatrixMultiplier.runOnce { [weak self] in
            guard let self else { return }
            self.metalQueue.asyncAfter(deadline: .now() + 0.05) { [weak self] in
                self?.runMetalMatrixBenchmarkLoop()
            }
        }
    }

    private func checkCameraPermission() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            configureCamera()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                DispatchQueue.main.async {
                    granted ? self?.configureCamera() : self?.showCameraDenied()
                }
            }
        default:
            showCameraDenied()
        }
    }

    private func showCameraDenied() {
        scanLabel.text = "Camera access required"
    }

    private func configureCamera() {
        session.beginConfiguration()
        session.sessionPreset = session.canSetSessionPreset(.hd1280x720) ? .hd1280x720 : .high

        guard
            let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
            let input = try? AVCaptureDeviceInput(device: camera),
            session.canAddInput(input)
        else {
            scanLabel.text = "Back camera unavailable"
            session.commitConfiguration()
            return
        }

        configureCameraFor60FPS(camera)
        session.addInput(input)

        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
        ]
        videoOutput.setSampleBufferDelegate(self, queue: videoQueue)

        guard session.canAddOutput(videoOutput) else {
            scanLabel.text = "Video output unavailable"
            session.commitConfiguration()
            return
        }

        session.addOutput(videoOutput)
        if let connection = videoOutput.connection(with: .video) {
            if connection.isVideoRotationAngleSupported(90) {
                connection.videoRotationAngle = 90
            }
        }

        addAudioInput()

        if session.canAddOutput(movieOutput) {
            session.addOutput(movieOutput)
            if let movieConnection = movieOutput.connection(with: .video),
               movieConnection.isVideoRotationAngleSupported(90) {
                movieConnection.videoRotationAngle = 90
            }
        }

        session.commitConfiguration()

        previewLayer = AVCaptureVideoPreviewLayer(session: session)
        previewLayer.videoGravity = .resizeAspectFill
        previewLayer.frame = view.bounds
        view.layer.insertSublayer(previewLayer, at: 0)

        overlayLayer.frame = view.bounds
        view.layer.insertSublayer(overlayLayer, above: previewLayer)

        scanLabel.text = "Scanning..."
        updateCurrentStepSummary()

        DispatchQueue.global(qos: .userInitiated).async { [session] in
            session.startRunning()
        }
    }

    private func addAudioInput() {
        guard
            let microphone = AVCaptureDevice.default(for: .audio),
            let audioInput = try? AVCaptureDeviceInput(device: microphone),
            session.canAddInput(audioInput)
        else { return }
        session.addInput(audioInput)
    }

    private func configureCameraFor60FPS(_ camera: AVCaptureDevice) {
        let targetFPS: Double = 60
        let targetDimensions = CMVideoDimensions(width: 1280, height: 720)

        let candidateFormats = camera.formats.compactMap { format -> (format: AVCaptureDevice.Format, dimensions: CMVideoDimensions, maxFPS: Double)? in
            let dimensions = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
            let maxFPS = format.videoSupportedFrameRateRanges.map(\.maxFrameRate).max() ?? 0
            guard maxFPS >= targetFPS else { return nil }
            return (format, dimensions, maxFPS)
        }

        let selectedFormat = candidateFormats
            .sorted {
                let lhsPenalty = abs(Int($0.dimensions.width - targetDimensions.width)) + abs(Int($0.dimensions.height - targetDimensions.height))
                let rhsPenalty = abs(Int($1.dimensions.width - targetDimensions.width)) + abs(Int($1.dimensions.height - targetDimensions.height))
                return lhsPenalty == rhsPenalty ? $0.maxFPS < $1.maxFPS : lhsPenalty < rhsPenalty
            }
            .first?.format

        do {
            try camera.lockForConfiguration()
            if let selectedFormat {
                camera.activeFormat = selectedFormat
            }
            let frameDuration = CMTime(value: 1, timescale: CMTimeScale(targetFPS))
            camera.activeVideoMinFrameDuration = frameDuration
            camera.activeVideoMaxFrameDuration = frameDuration
            if camera.isFocusModeSupported(.continuousAutoFocus) {
                camera.focusMode = .continuousAutoFocus
            }
            if camera.isExposureModeSupported(.continuousAutoExposure) {
                camera.exposureMode = .continuousAutoExposure
            }
            camera.unlockForConfiguration()
        } catch {
            scanLabel.text = "60 FPS unavailable"
        }
    }

    private func runDetection(on pixelBuffer: CVPixelBuffer) {
        var detections: [Detection] = []
        var firstError: Error?

        func capture(_ error: Error) {
            if firstError == nil {
                firstError = error
            }
        }

        do {
            detections.append(contentsOf: try detectPeopleAnimalsAndTargetObjects(in: pixelBuffer))
        } catch {
            capture(error)
        }

        do {
            detections.append(contentsOf: try detectHands(in: pixelBuffer))
        } catch {
            capture(error)
        }

        videoQueue.async { [weak self] in
            guard let self else { return }
            self.inFlightFrameCount -= 1

            let filteredDetections = self.mergeOverlappingDetections(detections.filter { $0.confidence >= 0.25 })
            DispatchQueue.main.async {
                self.draw(filteredDetections)

                if firstError != nil {
                    self.scanLabel.text = "Vision error"
                } else {
                    self.scanLabel.text = filteredDetections.isEmpty
                        ? "Scanning..."
                        : "\(filteredDetections.count) detected"
                }
            }
        }
    }

    private func detectPeopleAnimalsAndTargetObjects(in pixelBuffer: CVPixelBuffer) throws -> [Detection] {
        let humanRequest = VNDetectHumanRectanglesRequest()
        let animalRequest = VNRecognizeAnimalsRequest()
        let classificationRequest = VNClassifyImageRequest()
        let saliencyRequest = VNGenerateObjectnessBasedSaliencyImageRequest()
        configureVisionRequests([humanRequest, animalRequest, classificationRequest, saliencyRequest])
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .right, options: [:])

        try handler.perform([humanRequest, animalRequest, classificationRequest, saliencyRequest])

        var detections: [Detection] = []

        detections.append(contentsOf: (humanRequest.results ?? []).map {
            Detection(label: "person", confidence: $0.confidence, boundingBox: $0.boundingBox, source: .boxed)
        })

        detections.append(contentsOf: (animalRequest.results ?? []).compactMap { observation in
            guard let label = observation.labels.first else { return nil }
            return Detection(
                label: label.identifier,
                confidence: label.confidence,
                boundingBox: observation.boundingBox,
                source: .boxed
            )
        })

        let matchedLabels = (classificationRequest.results ?? [])
            .compactMap { observation -> (label: String, confidence: VNConfidence)? in
                guard let label = targetObjectLabels.match(for: observation.identifier) else { return nil }
                return (label, observation.confidence)
            }
            .filter { $0.confidence >= 0.25 }
            .prefix(3)

        if !matchedLabels.isEmpty {
            let saliencyBoxes = saliencyRequest.results?.first?.salientObjects?.map(\.boundingBox) ?? []
            let boxes = saliencyBoxes.isEmpty
                ? [CGRect(x: 0.08, y: 0.08, width: 0.84, height: 0.84)]
                : Array(saliencyBoxes.prefix(3))

            detections.append(contentsOf: matchedLabels.flatMap { label, confidence in
                boxes.map {
                    Detection(label: label, confidence: confidence, boundingBox: $0, source: .saliency)
                }
            })
        }

        return detections
    }

    private func detectPeopleAndAnimals(in pixelBuffer: CVPixelBuffer) throws -> [Detection] {
        let humanRequest = VNDetectHumanRectanglesRequest()
        let animalRequest = VNRecognizeAnimalsRequest()
        configureVisionRequests([humanRequest, animalRequest])
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .right, options: [:])

        try handler.perform([humanRequest, animalRequest])

        var detections: [Detection] = []

        let people = humanRequest.results ?? []
        detections.append(contentsOf: people.map {
            Detection(label: "person", confidence: $0.confidence, boundingBox: $0.boundingBox, source: .boxed)
        })

        let animals = animalRequest.results ?? []
        detections.append(contentsOf: animals.compactMap { observation in
            guard let label = observation.labels.first else { return nil }
            return Detection(
                label: label.identifier,
                confidence: label.confidence,
                boundingBox: observation.boundingBox,
                source: .boxed
            )
        })

        return detections
    }

    private func detectHands(in pixelBuffer: CVPixelBuffer) throws -> [Detection] {
        let request = VNDetectHumanHandPoseRequest()
        request.maximumHandCount = 4
        configureVisionRequests([request])

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .right, options: [:])
        try handler.perform([request])

        return try (request.results ?? []).compactMap { observation in
            let points = try observation.recognizedPoints(.all).values.filter { $0.confidence >= 0.2 }
            guard points.count >= 3 else { return nil }

            let minX = points.map(\.location.x).min() ?? 0
            let maxX = points.map(\.location.x).max() ?? 0
            let minY = points.map(\.location.y).min() ?? 0
            let maxY = points.map(\.location.y).max() ?? 0
            let confidence = points.map(\.confidence).reduce(0, +) / VNConfidence(points.count)
            let padding: CGFloat = 0.04

            let box = CGRect(
                x: max(0, minX - padding),
                y: max(0, minY - padding),
                width: min(1, maxX - minX + padding * 2),
                height: min(1, maxY - minY + padding * 2)
            )

            return Detection(label: "hand", confidence: confidence, boundingBox: box, source: .boxed)
        }
    }

    private func detectTargetObjectsWithSaliency(in pixelBuffer: CVPixelBuffer) throws -> [Detection] {
        let classificationRequest = VNClassifyImageRequest()
        let saliencyRequest = VNGenerateObjectnessBasedSaliencyImageRequest()
        configureVisionRequests([classificationRequest, saliencyRequest])
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .right, options: [:])

        try handler.perform([classificationRequest, saliencyRequest])

        let matchedLabels = (classificationRequest.results ?? [])
            .compactMap { observation -> (label: String, confidence: VNConfidence)? in
                guard let label = targetObjectLabels.match(for: observation.identifier) else { return nil }
                return (label, observation.confidence)
            }
            .filter { $0.confidence >= 0.25 }
            .prefix(3)

        guard !matchedLabels.isEmpty else { return [] }

        let saliencyBoxes = saliencyRequest.results?.first?.salientObjects?.map(\.boundingBox) ?? []
        let boxes = saliencyBoxes.isEmpty
            ? [CGRect(x: 0.08, y: 0.08, width: 0.84, height: 0.84)]
            : Array(saliencyBoxes.prefix(3))

        return matchedLabels.flatMap { label, confidence in
            boxes.map {
                Detection(label: label, confidence: confidence, boundingBox: $0, source: .saliency)
            }
        }
    }

    private func configureVisionRequests(_ requests: [VNRequest]) {
        for request in requests {
            request.preferBackgroundProcessing = false
            if let supportedDevices = try? request.supportedComputeStageDevices {
                for (stage, devices) in supportedDevices {
                    let preferredDevice = devices.first { device in
                        if case .neuralEngine = device {
                            return true
                        }
                        return false
                    } ?? devices.first { device in
                        if case .gpu = device {
                            return true
                        }
                        return false
                    }
                    request.setComputeDevice(preferredDevice, for: stage)
                }
            }
        }
    }

    private func mergeOverlappingDetections(_ detections: [Detection]) -> [Detection] {
        var selected: [Detection] = []

        for detection in detections.sorted(by: { $0.confidence > $1.confidence }) {
            let overlapsExisting = selected.contains {
                $0.label == detection.label && $0.boundingBox.intersectionOverUnion(with: detection.boundingBox) > 0.65
            }
            if !overlapsExisting {
                selected.append(detection)
            }
        }

        return Array(selected.prefix(12))
    }

    private func draw(_ detections: [Detection]) {
        overlayLayer.sublayers?.forEach { $0.removeFromSuperlayer() }
        guard shouldShowDetectionBoxes else { return }

        for detection in detections {
            let convertedRect = previewRect(for: detection.boundingBox)
            let color = color(for: detection.confidence)

            let boxLayer = CAShapeLayer()
            boxLayer.frame = convertedRect
            boxLayer.borderColor = color.cgColor
            boxLayer.borderWidth = 3
            boxLayer.cornerRadius = 4
            overlayLayer.addSublayer(boxLayer)

            let label = CATextLayer()
            label.string = "\(detection.label) \(Int(detection.confidence * 100))%"
            label.foregroundColor = UIColor.white.cgColor
            label.backgroundColor = color.withAlphaComponent(0.88).cgColor
            label.fontSize = 14
            label.contentsScale = UIScreen.main.scale
            label.alignmentMode = .center
            label.frame = CGRect(
                x: convertedRect.minX,
                y: max(0, convertedRect.minY - 24),
                width: min(140, max(84, convertedRect.width)),
                height: 24
            )
            overlayLayer.addSublayer(label)
        }
    }

    private func color(for confidence: VNConfidence) -> UIColor {
        let normalized = CGFloat(min(1, max(0, (confidence - 0.25) / 0.65)))
        return UIColor(
            red: 1 - normalized,
            green: normalized,
            blue: 0.08,
            alpha: 1
        )
    }

    private func previewRect(for visionRect: CGRect) -> CGRect {
        let metadataRect = CGRect(
            x: visionRect.minX,
            y: 1 - visionRect.maxY,
            width: visionRect.width,
            height: visionRect.height
        )
        return previewLayer.layerRectConverted(fromMetadataOutputRect: metadataRect)
    }
}

extension CameraDetectionViewController: UITableViewDataSource, UITableViewDelegate {
    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        selectedProtocol.steps.count
    }

    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let cell = tableView.dequeueReusableCell(
            withIdentifier: ProtocolStepCell.reuseIdentifier,
            for: indexPath
        ) as? ProtocolStepCell ?? ProtocolStepCell(style: .default, reuseIdentifier: ProtocolStepCell.reuseIdentifier)

        let state = state
        let status = state.stepStatuses[indexPath.row]
        let hasMemo = state.stepMemos[indexPath.row]?.isEmpty == false
        cell.configure(
            step: selectedProtocol.steps[indexPath.row],
            status: status,
            isCurrent: indexPath.row == state.currentStepIndex,
            isSelected: indexPath.row == selectedStepIndex,
            hasMemo: hasMemo
        )
        return cell
    }

    func tableView(_ tableView: UITableView, didSelectRowAt indexPath: IndexPath) {
        selectedStepIndex = indexPath.row
        uncertainButton.isHidden = true
        missedButton.isHidden = true
        protocolTableView.reloadData()

        if let memo = state.stepMemos[indexPath.row], !memo.isEmpty {
            let alert = UIAlertController(
                title: "Step \(selectedProtocol.steps[indexPath.row].number) memo",
                message: memo,
                preferredStyle: .alert
            )
            alert.addAction(UIAlertAction(title: "OK", style: .default))
            alert.addAction(UIAlertAction(title: "Edit", style: .default) { [weak self] _ in
                self?.addMemoForSelectedStep()
            })
            present(alert, animated: true)
        }
    }
}

extension CameraDetectionViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        let now = CACurrentMediaTime()

        #if DEBUG
        let cameraFPS = cameraFPSTracker.recordFrame(now: now)
        if now - lastCameraFPSUpdateTime >= 0.25 {
            lastCameraFPSUpdateTime = now
            DispatchQueue.main.async { [weak self] in
                self?.fpsLabel.text = String(format: "FPS %.1f", cameraFPS)
            }
        }
        #endif

        cameraFrameIndex = (cameraFrameIndex + 1) % inferenceFrameStride
        guard cameraFrameIndex == 0 else { return }
        guard now - lastDetectionTime >= minimumFrameInterval else { return }
        guard inFlightFrameCount < maxInFlightFrames else { return }
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        lastDetectionTime = now
        inFlightFrameCount += 1

        visionQueue.async { [weak self] in
            self?.runDetection(on: pixelBuffer)
        }
    }
}

extension CameraDetectionViewController: AVCaptureFileOutputRecordingDelegate {
    func fileOutput(
        _ output: AVCaptureFileOutput,
        didFinishRecordingTo outputFileURL: URL,
        from connections: [AVCaptureConnection],
        error: Error?
    ) {
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.isRecording = false
            self.stopRecordingTimer()
            self.setControlsRecording(false)

            // A nonzero error code can still accompany a usable file (e.g. it reached the
            // size limit); AVErrorRecordingSuccessfullyFinishedKey tells us if it's playable.
            let finishedCleanly = (error as NSError?)?
                .userInfo[AVErrorRecordingSuccessfullyFinishedKey] as? Bool ?? (error == nil)

            guard finishedCleanly else {
                try? FileManager.default.removeItem(at: outputFileURL)
                self.recordingStartTime = nil
                self.presentRecordingError()
                return
            }

            self.saveRecording(at: outputFileURL)
        }
    }

    private func presentRecordingError() {
        let alert = UIAlertController(
            title: "Recording failed",
            message: "The video could not be saved. Please try again.",
            preferredStyle: .alert
        )
        alert.addAction(UIAlertAction(title: "OK", style: .default))
        present(alert, animated: true)
    }
}

private struct Detection {
    let label: String
    let confidence: VNConfidence
    let boundingBox: CGRect
    let source: DetectionSource
}

private enum DetectionSource {
    case boxed
    case saliency
}

private final class ProtocolStepCell: UITableViewCell {
    static let reuseIdentifier = "ProtocolStepCell"

    private let markerView = UIView()
    private let numberLabel = UILabel()
    private let stepLabel = UILabel()
    private let memoLabel = UILabel()

    override init(style: UITableViewCell.CellStyle, reuseIdentifier: String?) {
        super.init(style: style, reuseIdentifier: reuseIdentifier)
        configure()
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        configure()
    }

    private func configure() {
        backgroundColor = .clear
        selectionStyle = .none
        contentView.backgroundColor = .clear

        markerView.translatesAutoresizingMaskIntoConstraints = false
        markerView.layer.cornerRadius = 5
        markerView.backgroundColor = .clear
        contentView.addSubview(markerView)

        numberLabel.translatesAutoresizingMaskIntoConstraints = false
        numberLabel.font = .monospacedDigitSystemFont(ofSize: 13, weight: .bold)
        numberLabel.textColor = .white
        numberLabel.textAlignment = .right
        contentView.addSubview(numberLabel)

        stepLabel.translatesAutoresizingMaskIntoConstraints = false
        stepLabel.font = .systemFont(ofSize: 13, weight: .regular)
        stepLabel.textColor = .white
        stepLabel.numberOfLines = 0
        contentView.addSubview(stepLabel)

        memoLabel.translatesAutoresizingMaskIntoConstraints = false
        memoLabel.font = .systemFont(ofSize: 11, weight: .semibold)
        memoLabel.textColor = .systemCyan
        memoLabel.text = "memo"
        contentView.addSubview(memoLabel)

        NSLayoutConstraint.activate([
            markerView.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 2),
            markerView.topAnchor.constraint(equalTo: contentView.topAnchor, constant: 12),
            markerView.widthAnchor.constraint(equalToConstant: 10),
            markerView.heightAnchor.constraint(equalToConstant: 10),

            numberLabel.leadingAnchor.constraint(equalTo: markerView.trailingAnchor, constant: 6),
            numberLabel.topAnchor.constraint(equalTo: contentView.topAnchor, constant: 8),
            numberLabel.widthAnchor.constraint(equalToConstant: 28),

            stepLabel.leadingAnchor.constraint(equalTo: numberLabel.trailingAnchor, constant: 8),
            stepLabel.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -8),
            stepLabel.topAnchor.constraint(equalTo: contentView.topAnchor, constant: 8),

            memoLabel.leadingAnchor.constraint(equalTo: stepLabel.leadingAnchor),
            memoLabel.topAnchor.constraint(equalTo: stepLabel.bottomAnchor, constant: 4),
            memoLabel.bottomAnchor.constraint(equalTo: contentView.bottomAnchor, constant: -8)
        ])
    }

    func configure(
        step: ProtocolStep,
        status: ProtocolStepStatus?,
        isCurrent: Bool,
        isSelected: Bool,
        hasMemo: Bool
    ) {
        numberLabel.text = "\(step.number)."
        stepLabel.text = step.text
        markerView.backgroundColor = status?.color ?? .clear
        memoLabel.isHidden = !hasMemo

        if status != nil {
            contentView.backgroundColor = UIColor.black.withAlphaComponent(0.42)
            stepLabel.textColor = UIColor.white.withAlphaComponent(0.58)
            numberLabel.textColor = UIColor.white.withAlphaComponent(0.58)
        } else {
            contentView.backgroundColor = isCurrent
                ? UIColor.systemBlue.withAlphaComponent(0.34)
                : UIColor.white.withAlphaComponent(0.06)
            stepLabel.textColor = .white
            numberLabel.textColor = .white
        }

        if isSelected {
            contentView.layer.borderColor = UIColor.white.withAlphaComponent(0.8).cgColor
            contentView.layer.borderWidth = 1
        } else {
            contentView.layer.borderWidth = 0
        }

        contentView.layer.cornerRadius = 8
        contentView.clipsToBounds = true
    }
}

private struct ProtocolRunState {
    var currentStepIndex = 0
    var stepStatuses: [Int: ProtocolStepStatus] = [:]
    var stepMemos: [Int: String] = [:]

    init(stepCount: Int) {}
}

private enum ProtocolStepStatus {
    case done
    case uncertain
    case missed

    var color: UIColor {
        switch self {
        case .done:
            return .systemGreen
        case .uncertain:
            return .systemYellow
        case .missed:
            return .systemRed
        }
    }
}

private struct TargetObjectLabels {
    static let all = TargetObjectLabels(labels: [
        "beaker": ["beaker", "cup", "glass", "flask"],
        "bottle": ["bottle", "vial", "test tube", "tube"],
        "pipette": ["pipette", "dropper", "syringe"],
        "microscope": ["microscope"],
        "plate": ["petri dish", "dish", "plate", "tray"],
        "rack": ["rack", "test tube rack"],
        "glove": ["glove"],
        "mask": ["mask"],
        "laptop": ["laptop", "computer", "notebook computer", "screen", "keyboard"],
        "phone": ["phone", "mobile phone", "cellular telephone"],
        "person": ["person", "human"],
        "hand": ["hand"]
    ])

    private let labels: [String: [String]]

    func match(for identifier: String) -> String? {
        let normalizedIdentifier = identifier.lowercased()

        for (label, aliases) in labels {
            if aliases.contains(where: { normalizedIdentifier.contains($0) }) {
                return label
            }
        }

        return nil
    }
}

private extension CGRect {
    func intersectionOverUnion(with other: CGRect) -> CGFloat {
        let intersectionArea = intersection(other).area
        guard intersectionArea > 0 else { return 0 }

        let unionArea = area + other.area - intersectionArea
        guard unionArea > 0 else { return 0 }

        return intersectionArea / unionArea
    }

    var area: CGFloat {
        width * height
    }
}

#if DEBUG
private struct FPSTracker {
    private var timestamps: [CFTimeInterval] = []

    mutating func recordFrame(now: CFTimeInterval = CACurrentMediaTime()) -> Double {
        timestamps.append(now)
        timestamps.removeAll { now - $0 > 1.0 }
        return Double(timestamps.count)
    }
}
#endif
