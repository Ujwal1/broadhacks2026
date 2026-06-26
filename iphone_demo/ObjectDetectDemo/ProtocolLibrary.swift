import Foundation

struct ProtocolStep {
    let number: Int
    let text: String
}

/// A lab protocol scraped from a `.txt` file in the bundled `Protocols` folder.
struct LabProtocol: Hashable, Identifiable {
    /// Stable identifier derived from the source file name.
    let id: String
    /// Short name shown in the protocol drop-down.
    let title: String
    /// Longer display heading shown at the top of the protocol panel.
    let heading: String
    let steps: [ProtocolStep]

    static func == (lhs: LabProtocol, rhs: LabProtocol) -> Bool { lhs.id == rhs.id }
    func hash(into hasher: inout Hasher) { hasher.combine(id) }

    /// Fallback used when no protocol files are found in the bundle.
    static let placeholder = LabProtocol(
        id: "none",
        title: "No protocols",
        heading: "No protocols found",
        steps: []
    )
}

/// Loads lab protocols from `.txt` files bundled in the `Protocols` folder so new protocols
/// can be added by dropping a file into that folder — no code changes required.
///
/// File format (leading metadata lines are optional):
/// ```
/// Title: Automated protein synthesis   ← drop-down name; defaults to the heading line
/// Order: 1                             ← sort key in the drop-down; defaults to end
/// PCR Mutagenesis Setup                ← heading (first non-metadata line)
///
/// 1. First step...
/// 2. Second step...
/// ```
/// Numbered lines become steps and are renumbered sequentially, so duplicate or skipped
/// numbers in the source file are tolerated.
enum ProtocolLibrary {
    private static let folderName = "Protocols"
    private static let defaultOrder = Int.max

    static func loadAll() -> [LabProtocol] {
        let urls = protocolFileURLs()
        let parsed = urls.compactMap { url -> (order: Int, protocol: LabProtocol)? in
            guard let contents = try? String(contentsOf: url, encoding: .utf8) else { return nil }
            return parse(contents: contents, fileName: url.lastPathComponent)
        }
        return parsed
            .sorted {
                $0.order == $1.order
                    ? $0.protocol.title.localizedCaseInsensitiveCompare($1.protocol.title) == .orderedAscending
                    : $0.order < $1.order
            }
            .map(\.protocol)
    }

    private static func protocolFileURLs() -> [URL] {
        // Works whether the folder is bundled as a folder reference (subdirectory) or its
        // files are added directly to the bundle root.
        if let directory = Bundle.main.url(forResource: folderName, withExtension: nil),
           let contents = try? FileManager.default.contentsOfDirectory(
               at: directory,
               includingPropertiesForKeys: nil
           ) {
            let txtFiles = contents.filter { $0.pathExtension.lowercased() == "txt" }
            if !txtFiles.isEmpty { return txtFiles }
        }

        return Bundle.main.urls(forResourcesWithExtension: "txt", subdirectory: folderName)
            ?? Bundle.main.urls(forResourcesWithExtension: "txt", subdirectory: nil)
            ?? []
    }

    private static func parse(contents: String, fileName: String) -> (order: Int, protocol: LabProtocol)? {
        var title: String?
        var order = defaultOrder
        var heading: String?
        var stepTexts: [String] = []

        for rawLine in contents.components(separatedBy: .newlines) {
            let line = rawLine.trimmingCharacters(in: .whitespaces)
            guard !line.isEmpty else { continue }

            if heading == nil, let value = metadataValue(in: line, key: "Title") {
                title = value
                continue
            }
            if heading == nil, let value = metadataValue(in: line, key: "Order") {
                order = Int(value) ?? defaultOrder
                continue
            }

            if let stepText = stepText(in: line) {
                stepTexts.append(stepText)
            } else if heading == nil {
                heading = line
            }
        }

        guard let resolvedHeading = heading, !stepTexts.isEmpty else { return nil }

        let steps = stepTexts.enumerated().map { ProtocolStep(number: $0.offset + 1, text: $0.element) }
        let id = (fileName as NSString).deletingPathExtension
        let labProtocol = LabProtocol(
            id: id,
            title: title ?? resolvedHeading,
            heading: resolvedHeading,
            steps: steps
        )
        return (order, labProtocol)
    }

    private static func metadataValue(in line: String, key: String) -> String? {
        let prefix = "\(key):"
        guard line.lowercased().hasPrefix(prefix.lowercased()) else { return nil }
        return String(line.dropFirst(prefix.count)).trimmingCharacters(in: .whitespaces)
    }

    /// Returns the text of a numbered step line (e.g. "3. Add buffer" → "Add buffer"), or nil.
    private static func stepText(in line: String) -> String? {
        var index = line.startIndex
        var sawDigit = false
        while index < line.endIndex, line[index].isNumber {
            sawDigit = true
            index = line.index(after: index)
        }
        guard sawDigit, index < line.endIndex, line[index] == "." else { return nil }
        let afterDot = line.index(after: index)
        let text = line[afterDot...].trimmingCharacters(in: .whitespaces)
        return text.isEmpty ? nil : text
    }
}
