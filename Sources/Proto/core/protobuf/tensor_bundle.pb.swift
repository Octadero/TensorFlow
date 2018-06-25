// DO NOT EDIT.
//
// Generated by the Swift generator plugin for the protocol buffer compiler.
// Source: tensorflow/core/protobuf/tensor_bundle.proto
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

/// Special header that is associated with a bundle.
///
/// TODO(zongheng,zhifengc): maybe in the future, we can add information about
/// which binary produced this checkpoint, timestamp, etc. Sometime, these can be
/// valuable debugging information. And if needed, these can be used as defensive
/// information ensuring reader (binary version) of the checkpoint and the writer
/// (binary version) must match within certain range, etc.
public struct Tensorflow_BundleHeaderProto {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  /// Number of data files in the bundle.
  public var numShards: Int32 {
    get {return _storage._numShards}
    set {_uniqueStorage()._numShards = newValue}
  }

  public var endianness: Tensorflow_BundleHeaderProto.Endianness {
    get {return _storage._endianness}
    set {_uniqueStorage()._endianness = newValue}
  }

  /// Versioning of the tensor bundle format.
  public var version: Tensorflow_VersionDef {
    get {return _storage._version ?? Tensorflow_VersionDef()}
    set {_uniqueStorage()._version = newValue}
  }
  /// Returns true if `version` has been explicitly set.
  public var hasVersion: Bool {return _storage._version != nil}
  /// Clears the value of `version`. Subsequent reads from it will return its default value.
  public mutating func clearVersion() {_storage._version = nil}

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  /// An enum indicating the endianness of the platform that produced this
  /// bundle.  A bundle can only be read by a platform with matching endianness.
  /// Defaults to LITTLE, as most modern platforms are little-endian.
  ///
  /// Affects the binary tensor data bytes only, not the metadata in protobufs.
  public enum Endianness: SwiftProtobuf.Enum {
    public typealias RawValue = Int
    case little // = 0
    case big // = 1
    case UNRECOGNIZED(Int)

    public init() {
      self = .little
    }

    public init?(rawValue: Int) {
      switch rawValue {
      case 0: self = .little
      case 1: self = .big
      default: self = .UNRECOGNIZED(rawValue)
      }
    }

    public var rawValue: Int {
      switch self {
      case .little: return 0
      case .big: return 1
      case .UNRECOGNIZED(let i): return i
      }
    }

  }

  public init() {}

  fileprivate var _storage = _StorageClass.defaultInstance
}

/// Describes the metadata related to a checkpointed tensor.
public struct Tensorflow_BundleEntryProto {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  /// The tensor dtype and shape.
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

  /// The binary content of the tensor lies in:
  ///   File "shard_id": bytes [offset, offset + size).
  public var shardID: Int32 {
    get {return _storage._shardID}
    set {_uniqueStorage()._shardID = newValue}
  }

  public var offset: Int64 {
    get {return _storage._offset}
    set {_uniqueStorage()._offset = newValue}
  }

  public var size: Int64 {
    get {return _storage._size}
    set {_uniqueStorage()._size = newValue}
  }

  /// The CRC32C checksum of the tensor bytes.
  public var crc32C: UInt32 {
    get {return _storage._crc32C}
    set {_uniqueStorage()._crc32C = newValue}
  }

  /// Iff present, this entry represents a partitioned tensor.  The previous
  /// fields are interpreted as follows:
  ///
  ///   "dtype", "shape": describe the full tensor.
  ///   "shard_id", "offset", "size", "crc32c": all IGNORED.
  ///      These information for each slice can be looked up in their own
  ///      BundleEntryProto, keyed by each "slice_name".
  public var slices: [Tensorflow_TensorSliceProto] {
    get {return _storage._slices}
    set {_uniqueStorage()._slices = newValue}
  }

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public init() {}

  fileprivate var _storage = _StorageClass.defaultInstance
}

// MARK: - Code below here is support for the SwiftProtobuf runtime.

fileprivate let _protobuf_package = "tensorflow"

extension Tensorflow_BundleHeaderProto: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = _protobuf_package + ".BundleHeaderProto"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .standard(proto: "num_shards"),
    2: .same(proto: "endianness"),
    3: .same(proto: "version"),
  ]

  fileprivate class _StorageClass {
    var _numShards: Int32 = 0
    var _endianness: Tensorflow_BundleHeaderProto.Endianness = .little
    var _version: Tensorflow_VersionDef? = nil

    static let defaultInstance = _StorageClass()

    private init() {}

    init(copying source: _StorageClass) {
      _numShards = source._numShards
      _endianness = source._endianness
      _version = source._version
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
        case 1: try decoder.decodeSingularInt32Field(value: &_storage._numShards)
        case 2: try decoder.decodeSingularEnumField(value: &_storage._endianness)
        case 3: try decoder.decodeSingularMessageField(value: &_storage._version)
        default: break
        }
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    try withExtendedLifetime(_storage) { (_storage: _StorageClass) in
      if _storage._numShards != 0 {
        try visitor.visitSingularInt32Field(value: _storage._numShards, fieldNumber: 1)
      }
      if _storage._endianness != .little {
        try visitor.visitSingularEnumField(value: _storage._endianness, fieldNumber: 2)
      }
      if let v = _storage._version {
        try visitor.visitSingularMessageField(value: v, fieldNumber: 3)
      }
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  public func _protobuf_generated_isEqualTo(other: Tensorflow_BundleHeaderProto) -> Bool {
    if _storage !== other._storage {
      let storagesAreEqual: Bool = withExtendedLifetime((_storage, other._storage)) { (_args: (_StorageClass, _StorageClass)) in
        let _storage = _args.0
        let other_storage = _args.1
        if _storage._numShards != other_storage._numShards {return false}
        if _storage._endianness != other_storage._endianness {return false}
        if _storage._version != other_storage._version {return false}
        return true
      }
      if !storagesAreEqual {return false}
    }
    if unknownFields != other.unknownFields {return false}
    return true
  }
}

extension Tensorflow_BundleHeaderProto.Endianness: SwiftProtobuf._ProtoNameProviding {
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    0: .same(proto: "LITTLE"),
    1: .same(proto: "BIG"),
  ]
}

extension Tensorflow_BundleEntryProto: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = _protobuf_package + ".BundleEntryProto"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "dtype"),
    2: .same(proto: "shape"),
    3: .standard(proto: "shard_id"),
    4: .same(proto: "offset"),
    5: .same(proto: "size"),
    6: .same(proto: "crc32c"),
    7: .same(proto: "slices"),
  ]

  fileprivate class _StorageClass {
    var _dtype: Tensorflow_DataType = .dtInvalid
    var _shape: Tensorflow_TensorShapeProto? = nil
    var _shardID: Int32 = 0
    var _offset: Int64 = 0
    var _size: Int64 = 0
    var _crc32C: UInt32 = 0
    var _slices: [Tensorflow_TensorSliceProto] = []

    static let defaultInstance = _StorageClass()

    private init() {}

    init(copying source: _StorageClass) {
      _dtype = source._dtype
      _shape = source._shape
      _shardID = source._shardID
      _offset = source._offset
      _size = source._size
      _crc32C = source._crc32C
      _slices = source._slices
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
        case 3: try decoder.decodeSingularInt32Field(value: &_storage._shardID)
        case 4: try decoder.decodeSingularInt64Field(value: &_storage._offset)
        case 5: try decoder.decodeSingularInt64Field(value: &_storage._size)
        case 6: try decoder.decodeSingularFixed32Field(value: &_storage._crc32C)
        case 7: try decoder.decodeRepeatedMessageField(value: &_storage._slices)
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
      if _storage._shardID != 0 {
        try visitor.visitSingularInt32Field(value: _storage._shardID, fieldNumber: 3)
      }
      if _storage._offset != 0 {
        try visitor.visitSingularInt64Field(value: _storage._offset, fieldNumber: 4)
      }
      if _storage._size != 0 {
        try visitor.visitSingularInt64Field(value: _storage._size, fieldNumber: 5)
      }
      if _storage._crc32C != 0 {
        try visitor.visitSingularFixed32Field(value: _storage._crc32C, fieldNumber: 6)
      }
      if !_storage._slices.isEmpty {
        try visitor.visitRepeatedMessageField(value: _storage._slices, fieldNumber: 7)
      }
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  public func _protobuf_generated_isEqualTo(other: Tensorflow_BundleEntryProto) -> Bool {
    if _storage !== other._storage {
      let storagesAreEqual: Bool = withExtendedLifetime((_storage, other._storage)) { (_args: (_StorageClass, _StorageClass)) in
        let _storage = _args.0
        let other_storage = _args.1
        if _storage._dtype != other_storage._dtype {return false}
        if _storage._shape != other_storage._shape {return false}
        if _storage._shardID != other_storage._shardID {return false}
        if _storage._offset != other_storage._offset {return false}
        if _storage._size != other_storage._size {return false}
        if _storage._crc32C != other_storage._crc32C {return false}
        if _storage._slices != other_storage._slices {return false}
        return true
      }
      if !storagesAreEqual {return false}
    }
    if unknownFields != other.unknownFields {return false}
    return true
  }
}
