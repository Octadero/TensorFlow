// swift-tools-version:4.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
	name: "TensorFlow",
	products: [
		.executable(name: "OpProducer", targets: ["OpProducer"]),
		.library(name: "Proto", type: .static, targets: ["Proto"]),
		.library(name: "CAPI", type: .static, targets: ["CAPI"]),
		.library(name: "CCAPI", type: .static,  targets: ["CCAPI"]),
		.library(name: "TensorFlowKit", type: .static, targets: ["TensorFlowKit"])
	],
	dependencies: [
		.package(url: "git@github.com:Octadero/CCTensorFlow.git", from: "0.0.9"),
        .package(url: "git@github.com:Octadero/CTensorFlow.git", from: "0.1.6"),
		.package(url: "git@github.com:Octadero/CProtobuf.git", from: "3.4.1"),
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
			name: "CCAPI",
			dependencies: ["Proto", "CAPI"]),
		.target(
			name: "TensorFlowKit",
			dependencies: ["Proto", "CAPI", "CCAPI"]),
		.testTarget(
			name: "CAPITests",
			dependencies: ["Proto", "CAPI"]),
		.testTarget(
			name: "CCAPITests",
			dependencies: ["Proto", "CAPI", "CCAPI"]),
		.testTarget(
			name: "TensorFlowKitTests",
			dependencies: ["Proto", "CCAPI", "TensorFlowKit"]),
		]
)
