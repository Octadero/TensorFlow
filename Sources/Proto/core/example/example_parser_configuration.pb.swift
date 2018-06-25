// DO NOT EDIT.
//
// Generated by the Swift generator plugin for the protocol buffer compiler.
// Source: tensorflow/core/example/example_parser_configuration.proto
//
// For information on using the generated types, please see the documenation:
//   https://github.com/apple/swift-protobuf/

// Protocol messages for describing the configuration of the ExampleParserOp.

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

public struct Tensorflow_VarLenFeatureProto {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  public var dtype: Tensorflow_DataType = .dtInvalid

  public var valuesOutputTensorName: String = String()

  public var indicesOutputTensorName: String = String()

  public var shapesOutputTensorName: String = String()

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public init() {}
}

public struct Tensorflow_FixedLenFeatureProto {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  public var dtype: Tensorflow_DataType {
    get {return _storage._dtype}
    set {_uniqueStorage()._dtype = newValue}
  }

  public var shape: Tensorflow_TensorShapeProto {
    get {return _storage._shape ?? Tensorflow_TensorShapeProto()}
    set {_uniqueStorage()._shape = newValue}
  }
  /// Returns true if `shape` has been explicitly set.
  public var hasShape: Bool {return _storage._shape != nil}
  /// Clears the value of `shape`. Subsequent reads from it will return its default value.
  public mutating func clearShape() {_storage._shape = nil}

  public var defaultValue: Tensorflow_TensorProto {
    get {return _storage._defaultValue ?? Tensorflow_TensorProto()}
    set {_uniqueStorage()._defaultValue = newValue}
  }
  /// Returns true if `defaultValue` has been explicitly set.
  public var hasDefaultValue: Bool {return _storage._defaultValue != nil}
  /// Clears the value of `defaultValue`. Subsequent reads from it will return its default value.
  public mutating func clearDefaultValue() {_storage._defaultValue = nil}

  public var valuesOutputTensorName: String {
    get {return _storage._valuesOutputTensorName}
    set {_uniqueStorage()._valuesOutputTensorName = newValue}
  }

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public init() {}

  fileprivate var _storage = _StorageClass.defaultInstance
}

public struct Tensorflow_FeatureConfiguration {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  public var config: OneOf_Config? {
    get {return _storage._config}
    set {_uniqueStorage()._config = newValue}
  }

  public var fixedLenFeature: Tensorflow_FixedLenFeatureProto {
    get {
      if case .fixedLenFeature(let v)? = _storage._config {return v}
      return Tensorflow_FixedLenFeatureProto()
    }
    set {_uniqueStorage()._config = .fixedLenFeature(newValue)}
  }

  public var varLenFeature: Tensorflow_VarLenFeatureProto {
    get {
      if case .varLenFeature(let v)? = _storage._config {return v}
      return Tensorflow_VarLenFeatureProto()
    }
    set {_uniqueStorage()._config = .varLenFeature(newValue)}
  }

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public enum OneOf_Config: Equatable {
    case fixedLenFeature(Tensorflow_FixedLenFeatureProto)
    case varLenFeature(Tensorflow_VarLenFeatureProto)

    public static func ==(lhs: Tensorflow_FeatureConfiguration.OneOf_Config, rhs: Tensorflow_FeatureConfiguration.OneOf_Config) -> Bool {
      switch (lhs, rhs) {
      case (.fixedLenFeature(let l), .fixedLenFeature(let r)): return l == r
      case (.varLenFeature(let l), .varLenFeature(let r)): return l == r
      default: return false
      }
    }
  }

  public init() {}

  fileprivate var _storage = _StorageClass.defaultInstance
}

public struct Tensorflow_ExampleParserConfiguration {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  public var featureMap: Dictionary<String,Tensorflow_FeatureConfiguration> = [:]

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public init() {}
}

// MARK: - Code below here is support for the SwiftProtobuf runtime.

fileprivate let _protobuf_package = "tensorflow"

extension Tensorflow_VarLenFeatureProto: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = _protobuf_package + ".VarLenFeatureProto"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "dtype"),
    2: .standard(proto: "values_output_tensor_name"),
    3: .standard(proto: "indices_output_tensor_name"),
    4: .standard(proto: "shapes_output_tensor_name"),
  ]

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      switch fieldNumber {
      case 1: try decoder.decodeSingularEnumField(value: &self.dtype)
      case 2: try decoder.decodeSingularStringField(value: &self.valuesOutputTensorName)
      case 3: try decoder.decodeSingularStringField(value: &self.indicesOutputTensorName)
      case 4: try decoder.decodeSingularStringField(value: &self.shapesOutputTensorName)
      default: break
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if self.dtype != .dtInvalid {
      try visitor.visitSingularEnumField(value: self.dtype, fieldNumber: 1)
    }
    if !self.valuesOutputTensorName.isEmpty {
      try visitor.visitSingularStringField(value: self.valuesOutputTensorName, fieldNumber: 2)
    }
    if !self.indicesOutputTensorName.isEmpty {
      try visitor.visitSingularStringField(value: self.indicesOutputTensorName, fieldNumber: 3)
    }
    if !self.shapesOutputTensorName.isEmpty {
      try visitor.visitSingularStringField(value: self.shapesOutputTensorName, fieldNumber: 4)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  public func _protobuf_generated_isEqualTo(other: Tensorflow_VarLenFeatureProto) -> Bool {
    if self.dtype != other.dtype {return false}
    if self.valuesOutputTensorName != other.valuesOutputTensorName {return false}
    if self.indicesOutputTensorName != other.indicesOutputTensorName {return false}
    if self.shapesOutputTensorName != other.shapesOutputTensorName {return false}
    if unknownFields != other.unknownFields {return false}
    return true
  }
}

extension Tensorflow_FixedLenFeatureProto: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = _protobuf_package + ".FixedLenFeatureProto"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "dtype"),
    2: .same(proto: "shape"),
    3: .standard(proto: "default_value"),
    4: .standard(proto: "values_output_tensor_name"),
  ]

  fileprivate class _StorageClass {
    var _dtype: Tensorflow_DataType = .dtInvalid
    var _shape: Tensorflow_TensorShapeProto? = nil
    var _defaultValue: Tensorflow_TensorProto? = nil
    var _valuesOutputTensorName: String = String()

    static let defaultInstance = _StorageClass()

    private init() {}

    init(copying source: _StorageClass) {
      _dtype = source._dtype
      _shape = source._shape
      _defaultValue = source._defaultValue
      _valuesOutputTensorName = source._valuesOutputTensorName
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
        case 1: try decoder.decodeSingularEnumField(value: &_storage._dtype)
        case 2: try decoder.decodeSingularMessageField(value: &_storage._shape)
        case 3: try decoder.decodeSingularMessageField(value: &_storage._defaultValue)
        case 4: try decoder.decodeSingularStringField(value: &_storage._valuesOutputTensorName)
        default: break
        }
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    try withExtendedLifetime(_storage) { (_storage: _StorageClass) in
      if _storage._dtype != .dtInvalid {
        try visitor.visitSingularEnumField(value: _storage._dtype, fieldNumber: 1)
      }
      if let v = _storage._shape {
        try visitor.visitSingularMessageField(value: v, fieldNumber: 2)
      }
      if let v = _storage._defaultValue {
        try visitor.visitSingularMessageField(value: v, fieldNumber: 3)
      }
      if !_storage._valuesOutputTensorName.isEmpty {
        try visitor.visitSingularStringField(value: _storage._valuesOutputTensorName, fieldNumber: 4)
      }
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  public func _protobuf_generated_isEqualTo(other: Tensorflow_FixedLenFeatureProto) -> Bool {
    if _storage !== other._storage {
      let storagesAreEqual: Bool = withExtendedLifetime((_storage, other._storage)) { (_args: (_StorageClass, _StorageClass)) in
        let _storage = _args.0
        let other_storage = _args.1
        if _storage._dtype != other_storage._dtype {return false}
        if _storage._shape != other_storage._shape {return false}
        if _storage._defaultValue != other_storage._defaultValue {return false}
        if _storage._valuesOutputTensorName != other_storage._valuesOutputTensorName {return false}
        return true
      }
      if !storagesAreEqual {return false}
    }
    if unknownFields != other.unknownFields {return false}
    return true
  }
}

extension Tensorflow_FeatureConfiguration: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = _protobuf_package + ".FeatureConfiguration"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .standard(proto: "fixed_len_feature"),
    2: .standard(proto: "var_len_feature"),
  ]

  fileprivate class _StorageClass {
    var _config: Tensorflow_FeatureConfiguration.OneOf_Config?

    static let defaultInstance = _StorageClass()

    private init() {}

    init(copying source: _StorageClass) {
      _config = source._config
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
        case 1:
          var v: Tensorflow_FixedLenFeatureProto?
          if let current = _storage._config {
            try decoder.handleConflictingOneOf()
            if case .fixedLenFeature(let m) = current {v = m}
          }
          try decoder.decodeSingularMessageField(value: &v)
          if let v = v {_storage._config = .fixedLenFeature(v)}
        case 2:
          var v: Tensorflow_VarLenFeatureProto?
          if let current = _storage._config {
            try decoder.handleConflictingOneOf()
            if case .varLenFeature(let m) = current {v = m}
          }
          try decoder.decodeSingularMessageField(value: &v)
          if let v = v {_storage._config = .varLenFeature(v)}
        default: break
        }
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    try withExtendedLifetime(_storage) { (_storage: _StorageClass) in
      switch _storage._config {
      case .fixedLenFeature(let v)?:
        try visitor.visitSingularMessageField(value: v, fieldNumber: 1)
      case .varLenFeature(let v)?:
        try visitor.visitSingularMessageField(value: v, fieldNumber: 2)
      case nil: break
      }
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  public func _protobuf_generated_isEqualTo(other: Tensorflow_FeatureConfiguration) -> Bool {
    if _storage !== other._storage {
      let storagesAreEqual: Bool = withExtendedLifetime((_storage, other._storage)) { (_args: (_StorageClass, _StorageClass)) in
        let _storage = _args.0
        let other_storage = _args.1
        if _storage._config != other_storage._config {return false}
        return true
      }
      if !storagesAreEqual {return false}
    }
    if unknownFields != other.unknownFields {return false}
    return true
  }
}

extension Tensorflow_ExampleParserConfiguration: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = _protobuf_package + ".ExampleParserConfiguration"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .standard(proto: "feature_map"),
  ]

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      switch fieldNumber {
      case 1: try decoder.decodeMapField(fieldType: SwiftProtobuf._ProtobufMessageMap<SwiftProtobuf.ProtobufString,Tensorflow_FeatureConfiguration>.self, value: &self.featureMap)
      default: break
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if !self.featureMap.isEmpty {
      try visitor.visitMapField(fieldType: SwiftProtobuf._ProtobufMessageMap<SwiftProtobuf.ProtobufString,Tensorflow_FeatureConfiguration>.self, value: self.featureMap, fieldNumber: 1)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  public func _protobuf_generated_isEqualTo(other: Tensorflow_ExampleParserConfiguration) -> Bool {
    if self.featureMap != other.featureMap {return false}
    if unknownFields != other.unknownFields {return false}
    return true
  }
}
