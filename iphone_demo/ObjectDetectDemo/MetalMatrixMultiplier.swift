import Foundation
import Metal
import MetalPerformanceShaders

final class MetalMatrixMultiplier {
    private let matrixSize = 64
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let matrixMultiplication: MPSMatrixMultiplication
    private let matrixA: MTLBuffer
    private let matrixB: MTLBuffer
    private let matrixC: MTLBuffer
    private let matrixADescriptor: MPSMatrixDescriptor
    private let matrixBDescriptor: MPSMatrixDescriptor
    private let matrixCDescriptor: MPSMatrixDescriptor

    init?() {
        guard
            let device = MTLCreateSystemDefaultDevice(),
            MPSSupportsMTLDevice(device),
            let commandQueue = device.makeCommandQueue()
        else {
            return nil
        }

        let rowBytes = matrixSize * MemoryLayout<Float>.stride
        let byteCount = matrixSize * rowBytes

        guard
            let matrixA = device.makeBuffer(length: byteCount, options: .storageModeShared),
            let matrixB = device.makeBuffer(length: byteCount, options: .storageModeShared),
            let matrixC = device.makeBuffer(length: byteCount, options: .storageModeShared)
        else {
            return nil
        }

        self.device = device
        self.commandQueue = commandQueue
        self.matrixA = matrixA
        self.matrixB = matrixB
        self.matrixC = matrixC
        self.matrixADescriptor = MPSMatrixDescriptor(
            rows: matrixSize,
            columns: matrixSize,
            rowBytes: rowBytes,
            dataType: .float32
        )
        self.matrixBDescriptor = MPSMatrixDescriptor(
            rows: matrixSize,
            columns: matrixSize,
            rowBytes: rowBytes,
            dataType: .float32
        )
        self.matrixCDescriptor = MPSMatrixDescriptor(
            rows: matrixSize,
            columns: matrixSize,
            rowBytes: rowBytes,
            dataType: .float32
        )
        self.matrixMultiplication = MPSMatrixMultiplication(
            device: device,
            transposeLeft: false,
            transposeRight: false,
            resultRows: matrixSize,
            resultColumns: matrixSize,
            interiorColumns: matrixSize,
            alpha: 1,
            beta: 0
        )

        fill(matrixA, seed: 0.25)
        fill(matrixB, seed: 0.75)
    }

    func runOnce(completion: (() -> Void)? = nil) {
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            completion?()
            return
        }

        let leftMatrix = MPSMatrix(buffer: matrixA, descriptor: matrixADescriptor)
        let rightMatrix = MPSMatrix(buffer: matrixB, descriptor: matrixBDescriptor)
        let resultMatrix = MPSMatrix(buffer: matrixC, descriptor: matrixCDescriptor)

        matrixMultiplication.encode(
            commandBuffer: commandBuffer,
            leftMatrix: leftMatrix,
            rightMatrix: rightMatrix,
            resultMatrix: resultMatrix
        )

        commandBuffer.addCompletedHandler { _ in
            completion?()
        }
        commandBuffer.commit()
    }

    private func fill(_ buffer: MTLBuffer, seed: Float) {
        let values = buffer.contents().bindMemory(to: Float.self, capacity: matrixSize * matrixSize)

        for index in 0..<(matrixSize * matrixSize) {
            values[index] = seed + Float(index % matrixSize) / Float(matrixSize)
        }
    }
}
