// swift-tools-version:4.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
	name: "TensorFlow",
	products: [
		.executable(name: "OpProducer", targets: ["OpProducer"]),
		.library(name: "Proto", type: .static, targets: ["Proto"]),
		.library(name: "CAPI", type: .static, targets: ["CAPI"]),
		.library(name: "TensorFlowKit", type: .static, targets: ["TensorFlowKit"])
	],
	dependencies: [
        .package(url: "https://github.com/Octadero/CTensorFlow.git", from: "0.1.6"),
        .package(url: "https://github.com/Octadero/MNISTKit.git", from: "0.0.7"),
        .package(url: "https://github.com/Octadero/MemoryLayoutKit.git", from: "0.0.1"),
        .package(url: "https://github.com/apple/swift-protobuf.git", from: "1.0.0")
	],
	targets: [
		.target(
			name: "Proto",
			dependencies: ["SwiftProtobuf"]),
		.target(
			name: "OpProducer",
			dependencies: ["Proto", "CAPI"]),
		.target(
			name: "CAPI",
			dependencies: ["Proto"]),
		.target(
			name: "TensorFlowKit",
			dependencies: ["Proto", "CAPI", "MemoryLayoutKit"]),
		.testTarget(
			name: "CAPITests",
			dependencies: ["Proto", "CAPI"]),
        .testTarget(
            name: "MNISTTests",
            dependencies: ["Proto", "CAPI", "TensorFlowKit", "MNISTKit"]),
        .testTarget(
            name: "OptimizerTests",
            dependencies: ["Proto", "CAPI", "TensorFlowKit", "MNISTKit"]),
        .testTarget(
            name: "CAPIOperationsTests",
            dependencies: ["Proto", "CAPI"]),
        .testTarget(
			name: "TensorFlowKitTests",
			dependencies: ["Proto", "TensorFlowKit", "MNISTKit", "CAPI"]),
		]
)
