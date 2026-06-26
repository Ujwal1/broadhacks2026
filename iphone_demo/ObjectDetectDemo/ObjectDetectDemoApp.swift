import SwiftUI

@main
struct ObjectDetectDemoApp: App {
    var body: some Scene {
        WindowGroup {
            CameraDetectionView()
                .ignoresSafeArea()
        }
    }
}
