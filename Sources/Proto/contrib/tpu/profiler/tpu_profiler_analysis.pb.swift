// DO NOT EDIT.
//
// Generated by the Swift generator plugin for the protocol buffer compiler.
// Source: tensorflow/contrib/tpu/profiler/tpu_profiler_analysis.proto
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

public struct Tensorflow_NewProfileSessionRequest {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  public var request: Tensorflow_ProfileRequest {
    get {return _storage._request ?? Tensorflow_ProfileRequest()}
    set {_uniqueStorage()._request = newValue}
  }
  /// Returns true if `request` has been explicitly set.
  public var hasRequest: Bool {return _storage._request != nil}
  /// Clears the value of `request`. Subsequent reads from it will return its default value.
  public mutating func clearRequest() {_storage._request = nil}

  public var repositoryRoot: String {
    get {return _storage._repositoryRoot}
    set {_uniqueStorage()._repositoryRoot = newValue}
  }

  public var hosts: [String] {
    get {return _storage._hosts}
    set {_uniqueStorage()._hosts = newValue}
  }

  public var sessionID: String {
    get {return _storage._sessionID}
    set {_uniqueStorage()._sessionID = newValue}
  }

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public init() {}

  fileprivate var _storage = _StorageClass.defaultInstance
}

public struct Tensorflow_NewProfileSessionResponse {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  /// Auxiliary error_message.
  public var errorMessage: String = String()

  /// Whether all hosts had returned a empty trace.
  public var emptyTrace: Bool = false

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public init() {}
}

public struct Tensorflow_EnumProfileSessionsAndToolsRequest {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  public var repositoryRoot: String = String()

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public init() {}
}

public struct Tensorflow_ProfileSessionInfo {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  public var sessionID: String = String()

  /// Which tool data is available for consumption.
  public var availableTools: [String] = []

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public init() {}
}

public struct Tensorflow_EnumProfileSessionsAndToolsResponse {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  /// Auxiliary error_message.
  public var errorMessage: String = String()

  /// If success, the returned sessions information are stored here.
  public var sessions: [Tensorflow_ProfileSessionInfo] = []

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public init() {}
}

public struct Tensorflow_ProfileSessionDataRequest {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  public var repositoryRoot: String = String()

  public var sessionID: String = String()

  /// Which host the data is associated. if empty, data from all hosts are
  /// aggregated.
  public var hostName: String = String()

  /// Which tool
  public var toolName: String = String()

  /// Tool's specific parameters. e.g. TraceViewer's viewport etc
  public var parameters: Dictionary<String,String> = [:]

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public init() {}
}

public struct Tensorflow_ProfileSessionDataResponse {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  /// Auxiliary error_message.
  public var errorMessage: String = String()

  /// Output format. e.g. "json" or "proto" or "blob"
  public var outputFormat: String = String()

  /// TODO(jiesun): figure out whether to put bytes or oneof tool specific proto.
  public var output: Data = SwiftProtobuf.Internal.emptyData

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public init() {}
}

// MARK: - Code below here is support for the SwiftProtobuf runtime.

fileprivate let _protobuf_package = "tensorflow"

extension Tensorflow_NewProfileSessionRequest: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = _protobuf_package + ".NewProfileSessionRequest"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "request"),
    2: .standard(proto: "repository_root"),
    3: .same(proto: "hosts"),
    4: .standard(proto: "session_id"),
  ]

  fileprivate class _StorageClass {
    var _request: Tensorflow_ProfileRequest? = nil
    var _repositoryRoot: String = String()
    var _hosts: [String] = []
    var _sessionID: String = String()

    static let defaultInstance = _StorageClass()

    private init() {}

    init(copying source: _StorageClass) {
      _request = source._request
      _repositoryRoot = source._repositoryRoot
      _hosts = source._hosts
      _sessionID = source._sessionID
    }
  }

  fileprivate mutating func _uniqueStorage() -> _StorageClass {
    if !isKnownUniquelyReferenced(&_storage) {
      _storage = _StorageClass(copying: _storage)
    }
    return _storage
  }

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    _ = _uniqueStorage()
    try withExtendedLifetime(_storage) { (_storage: _StorageClass) in
      while let fieldNumber = try decoder.nextFieldNumber() {
        switch fieldNumber {
        case 1: try decoder.decodeSingularMessageField(value: &_storage._request)
        case 2: try decoder.decodeSingularStringField(value: &_storage._repositoryRoot)
        case 3: try decoder.decodeRepeatedStringField(value: &_storage._hosts)
        case 4: try decoder.decodeSingularStringField(value: &_storage._sessionID)
        default: break
        }
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    try withExtendedLifetime(_storage) { (_storage: _StorageClass) in
      if let v = _storage._request {
        try visitor.visitSingularMessageField(value: v, fieldNumber: 1)
      }
      if !_storage._repositoryRoot.isEmpty {
        try visitor.visitSingularStringField(value: _storage._repositoryRoot, fieldNumber: 2)
      }
      if !_storage._hosts.isEmpty {
        try visitor.visitRepeatedStringField(value: _storage._hosts, fieldNumber: 3)
      }
      if !_storage._sessionID.isEmpty {
        try visitor.visitSingularStringField(value: _storage._sessionID, fieldNumber: 4)
      }
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  public func _protobuf_generated_isEqualTo(other: Tensorflow_NewProfileSessionRequest) -> Bool {
    if _storage !== other._storage {
      let storagesAreEqual: Bool = withExtendedLifetime((_storage, other._storage)) { (_args: (_StorageClass, _StorageClass)) in
        let _storage = _args.0
        let other_storage = _args.1
        if _storage._request != other_storage._request {return false}
        if _storage._repositoryRoot != other_storage._repositoryRoot {return false}
        if _storage._hosts != other_storage._hosts {return false}
        if _storage._sessionID != other_storage._sessionID {return false}
        return true
      }
      if !storagesAreEqual {return false}
    }
    if unknownFields != other.unknownFields {return false}
    return true
  }
}

extension Tensorflow_NewProfileSessionResponse: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = _protobuf_package + ".NewProfileSessionResponse"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .standard(proto: "error_message"),
    2: .standard(proto: "empty_trace"),
  ]

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      switch fieldNumber {
      case 1: try decoder.decodeSingularStringField(value: &self.errorMessage)
      case 2: try decoder.decodeSingularBoolField(value: &self.emptyTrace)
      default: break
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if !self.errorMessage.isEmpty {
      try visitor.visitSingularStringField(value: self.errorMessage, fieldNumber: 1)
    }
    if self.emptyTrace != false {
      try visitor.visitSingularBoolField(value: self.emptyTrace, fieldNumber: 2)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  public func _protobuf_generated_isEqualTo(other: Tensorflow_NewProfileSessionResponse) -> Bool {
    if self.errorMessage != other.errorMessage {return false}
    if self.emptyTrace != other.emptyTrace {return false}
    if unknownFields != other.unknownFields {return false}
    return true
  }
}

extension Tensorflow_EnumProfileSessionsAndToolsRequest: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = _protobuf_package + ".EnumProfileSessionsAndToolsRequest"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .standard(proto: "repository_root"),
  ]

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      switch fieldNumber {
      case 1: try decoder.decodeSingularStringField(value: &self.repositoryRoot)
      default: break
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if !self.repositoryRoot.isEmpty {
      try visitor.visitSingularStringField(value: self.repositoryRoot, fieldNumber: 1)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  public func _protobuf_generated_isEqualTo(other: Tensorflow_EnumProfileSessionsAndToolsRequest) -> Bool {
    if self.repositoryRoot != other.repositoryRoot {return false}
    if unknownFields != other.unknownFields {return false}
    return true
  }
}

extension Tensorflow_ProfileSessionInfo: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = _protobuf_package + ".ProfileSessionInfo"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .standard(proto: "session_id"),
    2: .standard(proto: "available_tools"),
  ]

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      switch fieldNumber {
      case 1: try decoder.decodeSingularStringField(value: &self.sessionID)
      case 2: try decoder.decodeRepeatedStringField(value: &self.availableTools)
      default: break
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if !self.sessionID.isEmpty {
      try visitor.visitSingularStringField(value: self.sessionID, fieldNumber: 1)
    }
    if !self.availableTools.isEmpty {
      try visitor.visitRepeatedStringField(value: self.availableTools, fieldNumber: 2)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  public func _protobuf_generated_isEqualTo(other: Tensorflow_ProfileSessionInfo) -> Bool {
    if self.sessionID != other.sessionID {return false}
    if self.availableTools != other.availableTools {return false}
    if unknownFields != other.unknownFields {return false}
    return true
  }
}

extension Tensorflow_EnumProfileSessionsAndToolsResponse: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = _protobuf_package + ".EnumProfileSessionsAndToolsResponse"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .standard(proto: "error_message"),
    2: .same(proto: "sessions"),
  ]

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      switch fieldNumber {
      case 1: try decoder.decodeSingularStringField(value: &self.errorMessage)
      case 2: try decoder.decodeRepeatedMessageField(value: &self.sessions)
      default: break
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if !self.errorMessage.isEmpty {
      try visitor.visitSingularStringField(value: self.errorMessage, fieldNumber: 1)
    }
    if !self.sessions.isEmpty {
      try visitor.visitRepeatedMessageField(value: self.sessions, fieldNumber: 2)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  public func _protobuf_generated_isEqualTo(other: Tensorflow_EnumProfileSessionsAndToolsResponse) -> Bool {
    if self.errorMessage != other.errorMessage {return false}
    if self.sessions != other.sessions {return false}
    if unknownFields != other.unknownFields {return false}
    return true
  }
}

extension Tensorflow_ProfileSessionDataRequest: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = _protobuf_package + ".ProfileSessionDataRequest"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .standard(proto: "repository_root"),
    2: .standard(proto: "session_id"),
    5: .standard(proto: "host_name"),
    3: .standard(proto: "tool_name"),
    4: .same(proto: "parameters"),
  ]

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      switch fieldNumber {
      case 1: try decoder.decodeSingularStringField(value: &self.repositoryRoot)
      case 2: try decoder.decodeSingularStringField(value: &self.sessionID)
      case 3: try decoder.decodeSingularStringField(value: &self.toolName)
      case 4: try decoder.decodeMapField(fieldType: SwiftProtobuf._ProtobufMap<SwiftProtobuf.ProtobufString,SwiftProtobuf.ProtobufString>.self, value: &self.parameters)
      case 5: try decoder.decodeSingularStringField(value: &self.hostName)
      default: break
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if !self.repositoryRoot.isEmpty {
      try visitor.visitSingularStringField(value: self.repositoryRoot, fieldNumber: 1)
    }
    if !self.sessionID.isEmpty {
      try visitor.visitSingularStringField(value: self.sessionID, fieldNumber: 2)
    }
    if !self.toolName.isEmpty {
      try visitor.visitSingularStringField(value: self.toolName, fieldNumber: 3)
    }
    if !self.parameters.isEmpty {
      try visitor.visitMapField(fieldType: SwiftProtobuf._ProtobufMap<SwiftProtobuf.ProtobufString,SwiftProtobuf.ProtobufString>.self, value: self.parameters, fieldNumber: 4)
    }
    if !self.hostName.isEmpty {
      try visitor.visitSingularStringField(value: self.hostName, fieldNumber: 5)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  public func _protobuf_generated_isEqualTo(other: Tensorflow_ProfileSessionDataRequest) -> Bool {
    if self.repositoryRoot != other.repositoryRoot {return false}
    if self.sessionID != other.sessionID {return false}
    if self.hostName != other.hostName {return false}
    if self.toolName != other.toolName {return false}
    if self.parameters != other.parameters {return false}
    if unknownFields != other.unknownFields {return false}
    return true
  }
}

extension Tensorflow_ProfileSessionDataResponse: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = _protobuf_package + ".ProfileSessionDataResponse"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .standard(proto: "error_message"),
    2: .standard(proto: "output_format"),
    3: .same(proto: "output"),
  ]

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      switch fieldNumber {
      case 1: try decoder.decodeSingularStringField(value: &self.errorMessage)
      case 2: try decoder.decodeSingularStringField(value: &self.outputFormat)
      case 3: try decoder.decodeSingularBytesField(value: &self.output)
      default: break
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if !self.errorMessage.isEmpty {
      try visitor.visitSingularStringField(value: self.errorMessage, fieldNumber: 1)
    }
    if !self.outputFormat.isEmpty {
      try visitor.visitSingularStringField(value: self.outputFormat, fieldNumber: 2)
    }
    if !self.output.isEmpty {
      try visitor.visitSingularBytesField(value: self.output, fieldNumber: 3)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  public func _protobuf_generated_isEqualTo(other: Tensorflow_ProfileSessionDataResponse) -> Bool {
    if self.errorMessage != other.errorMessage {return false}
    if self.outputFormat != other.outputFormat {return false}
    if self.output != other.output {return false}
    if unknownFields != other.unknownFields {return false}
    return true
  }
}
