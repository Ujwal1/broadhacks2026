import SwiftUI

struct CameraDetectionView: UIViewControllerRepresentable {
    func makeUIViewController(context: Context) -> CameraDetectionViewController {
        CameraDetectionViewController()
    }

    func updateUIViewController(_ uiViewController: CameraDetectionViewController, context: Context) {}
}
