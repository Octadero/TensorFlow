// DO NOT EDIT.
//
// Generated by the Swift generator plugin for the protocol buffer compiler.
// Source: tensorflow/python/training/checkpoint_state.proto
//
// For information on using the generated types, please see the documenation:
//   https://github.com/apple/swift-protobuf/

import Foundation
import SwiftProtobuf

// If the compiler emits an error on this type, it is because this file
// was generated by a version of the `protoc` Swift plug-in that is
// incompatible with the version of SwiftProtobuf to which you are linking.
// Please ensure that your are building against the same version of the API
// that was used to generate this file.
fileprivate struct _GeneratedWithProtocGenSwiftVersion: SwiftProtobuf.ProtobufAPIVersionCheck {
  struct _2: SwiftProtobuf.ProtobufAPIVersion_2 {}
  typealias Version = _2
}

/// Protocol buffer representing the checkpoint state.
///
/// TODO(touts): Add other attributes as needed.
public struct Tensorflow_CheckpointState {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  /// Path to the most-recent model checkpoint.
  public var modelCheckpointPath: String = String()

  /// Paths to all not-yet-deleted model checkpoints, sorted from oldest to
  /// newest.
  /// Note that the value of model_checkpoint_path should be the last item in
  /// this list.
  public var allModelCheckpointPaths: [String] = []

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public init() {}
}

// MARK: - Code below here is support for the SwiftProtobuf runtime.

fileprivate let _protobuf_package = "tensorflow"

extension Tensorflow_CheckpointState: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = _protobuf_package + ".CheckpointState"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .standard(proto: "model_checkpoint_path"),
    2: .standard(proto: "all_model_checkpoint_paths"),
  ]

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      switch fieldNumber {
      case 1: try decoder.decodeSingularStringField(value: &self.modelCheckpointPath)
      case 2: try decoder.decodeRepeatedStringField(value: &self.allModelCheckpointPaths)
      default: break
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if !self.modelCheckpointPath.isEmpty {
      try visitor.visitSingularStringField(value: self.modelCheckpointPath, fieldNumber: 1)
    }
    if !self.allModelCheckpointPaths.isEmpty {
      try visitor.visitRepeatedStringField(value: self.allModelCheckpointPaths, fieldNumber: 2)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  public func _protobuf_generated_isEqualTo(other: Tensorflow_CheckpointState) -> Bool {
    if self.modelCheckpointPath != other.modelCheckpointPath {return false}
    if self.allModelCheckpointPaths != other.allModelCheckpointPaths {return false}
    if unknownFields != other.unknownFields {return false}
    return true
  }
}
