// DO NOT EDIT.
//
// Generated by the Swift generator plugin for the protocol buffer compiler.
// Source: tensorflow/core/profiler/tfprof_options.proto
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

/// Refers to tfprof_options.h/cc for documentation.
/// Only used to pass tfprof options from Python to C++.
public struct Tensorflow_Tfprof_OptionsProto {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  public var maxDepth: Int64 {
    get {return _storage._maxDepth}
    set {_uniqueStorage()._maxDepth = newValue}
  }

  public var minBytes: Int64 {
    get {return _storage._minBytes}
    set {_uniqueStorage()._minBytes = newValue}
  }

  public var minPeakBytes: Int64 {
    get {return _storage._minPeakBytes}
    set {_uniqueStorage()._minPeakBytes = newValue}
  }

  public var minResidualBytes: Int64 {
    get {return _storage._minResidualBytes}
    set {_uniqueStorage()._minResidualBytes = newValue}
  }

  public var minOutputBytes: Int64 {
    get {return _storage._minOutputBytes}
    set {_uniqueStorage()._minOutputBytes = newValue}
  }

  public var minMicros: Int64 {
    get {return _storage._minMicros}
    set {_uniqueStorage()._minMicros = newValue}
  }

  public var minAcceleratorMicros: Int64 {
    get {return _storage._minAcceleratorMicros}
    set {_uniqueStorage()._minAcceleratorMicros = newValue}
  }

  public var minCpuMicros: Int64 {
    get {return _storage._minCpuMicros}
    set {_uniqueStorage()._minCpuMicros = newValue}
  }

  public var minParams: Int64 {
    get {return _storage._minParams}
    set {_uniqueStorage()._minParams = newValue}
  }

  public var minFloatOps: Int64 {
    get {return _storage._minFloatOps}
    set {_uniqueStorage()._minFloatOps = newValue}
  }

  public var minOccurrence: Int64 {
    get {return _storage._minOccurrence}
    set {_uniqueStorage()._minOccurrence = newValue}
  }

  public var step: Int64 {
    get {return _storage._step}
    set {_uniqueStorage()._step = newValue}
  }

  public var orderBy: String {
    get {return _storage._orderBy}
    set {_uniqueStorage()._orderBy = newValue}
  }

  public var accountTypeRegexes: [String] {
    get {return _storage._accountTypeRegexes}
    set {_uniqueStorage()._accountTypeRegexes = newValue}
  }

  public var startNameRegexes: [String] {
    get {return _storage._startNameRegexes}
    set {_uniqueStorage()._startNameRegexes = newValue}
  }

  public var trimNameRegexes: [String] {
    get {return _storage._trimNameRegexes}
    set {_uniqueStorage()._trimNameRegexes = newValue}
  }

  public var showNameRegexes: [String] {
    get {return _storage._showNameRegexes}
    set {_uniqueStorage()._showNameRegexes = newValue}
  }

  public var hideNameRegexes: [String] {
    get {return _storage._hideNameRegexes}
    set {_uniqueStorage()._hideNameRegexes = newValue}
  }

  public var accountDisplayedOpOnly: Bool {
    get {return _storage._accountDisplayedOpOnly}
    set {_uniqueStorage()._accountDisplayedOpOnly = newValue}
  }

  public var select: [String] {
    get {return _storage._select}
    set {_uniqueStorage()._select = newValue}
  }

  public var output: String {
    get {return _storage._output}
    set {_uniqueStorage()._output = newValue}
  }

  public var dumpToFile: String {
    get {return _storage._dumpToFile}
    set {_uniqueStorage()._dumpToFile = newValue}
  }

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public init() {}

  fileprivate var _storage = _StorageClass.defaultInstance
}

public struct Tensorflow_Tfprof_AdvisorOptionsProto {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  /// checker name -> a dict of key-value options.
  public var checkers: Dictionary<String,Tensorflow_Tfprof_AdvisorOptionsProto.CheckerOption> = [:]

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public struct CheckerOption {
    // SwiftProtobuf.Message conformance is added in an extension below. See the
    // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
    // methods supported on all messages.

    public var options: Dictionary<String,String> = [:]

    public var unknownFields = SwiftProtobuf.UnknownStorage()

    public init() {}
  }

  public init() {}
}

// MARK: - Code below here is support for the SwiftProtobuf runtime.

fileprivate let _protobuf_package = "tensorflow.tfprof"

extension Tensorflow_Tfprof_OptionsProto: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = _protobuf_package + ".OptionsProto"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .standard(proto: "max_depth"),
    2: .standard(proto: "min_bytes"),
    19: .standard(proto: "min_peak_bytes"),
    20: .standard(proto: "min_residual_bytes"),
    21: .standard(proto: "min_output_bytes"),
    3: .standard(proto: "min_micros"),
    22: .standard(proto: "min_accelerator_micros"),
    23: .standard(proto: "min_cpu_micros"),
    4: .standard(proto: "min_params"),
    5: .standard(proto: "min_float_ops"),
    17: .standard(proto: "min_occurrence"),
    18: .same(proto: "step"),
    7: .standard(proto: "order_by"),
    8: .standard(proto: "account_type_regexes"),
    9: .standard(proto: "start_name_regexes"),
    10: .standard(proto: "trim_name_regexes"),
    11: .standard(proto: "show_name_regexes"),
    12: .standard(proto: "hide_name_regexes"),
    13: .standard(proto: "account_displayed_op_only"),
    14: .same(proto: "select"),
    15: .same(proto: "output"),
    16: .standard(proto: "dump_to_file"),
  ]

  fileprivate class _StorageClass {
    var _maxDepth: Int64 = 0
    var _minBytes: Int64 = 0
    var _minPeakBytes: Int64 = 0
    var _minResidualBytes: Int64 = 0
    var _minOutputBytes: Int64 = 0
    var _minMicros: Int64 = 0
    var _minAcceleratorMicros: Int64 = 0
    var _minCpuMicros: Int64 = 0
    var _minParams: Int64 = 0
    var _minFloatOps: Int64 = 0
    var _minOccurrence: Int64 = 0
    var _step: Int64 = 0
    var _orderBy: String = String()
    var _accountTypeRegexes: [String] = []
    var _startNameRegexes: [String] = []
    var _trimNameRegexes: [String] = []
    var _showNameRegexes: [String] = []
    var _hideNameRegexes: [String] = []
    var _accountDisplayedOpOnly: Bool = false
    var _select: [String] = []
    var _output: String = String()
    var _dumpToFile: String = String()

    static let defaultInstance = _StorageClass()

    private init() {}

    init(copying source: _StorageClass) {
      _maxDepth = source._maxDepth
      _minBytes = source._minBytes
      _minPeakBytes = source._minPeakBytes
      _minResidualBytes = source._minResidualBytes
      _minOutputBytes = source._minOutputBytes
      _minMicros = source._minMicros
      _minAcceleratorMicros = source._minAcceleratorMicros
      _minCpuMicros = source._minCpuMicros
      _minParams = source._minParams
      _minFloatOps = source._minFloatOps
      _minOccurrence = source._minOccurrence
      _step = source._step
      _orderBy = source._orderBy
      _accountTypeRegexes = source._accountTypeRegexes
      _startNameRegexes = source._startNameRegexes
      _trimNameRegexes = source._trimNameRegexes
      _showNameRegexes = source._showNameRegexes
      _hideNameRegexes = source._hideNameRegexes
      _accountDisplayedOpOnly = source._accountDisplayedOpOnly
      _select = source._select
      _output = source._output
      _dumpToFile = source._dumpToFile
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
        case 1: try decoder.decodeSingularInt64Field(value: &_storage._maxDepth)
        case 2: try decoder.decodeSingularInt64Field(value: &_storage._minBytes)
        case 3: try decoder.decodeSingularInt64Field(value: &_storage._minMicros)
        case 4: try decoder.decodeSingularInt64Field(value: &_storage._minParams)
        case 5: try decoder.decodeSingularInt64Field(value: &_storage._minFloatOps)
        case 7: try decoder.decodeSingularStringField(value: &_storage._orderBy)
        case 8: try decoder.decodeRepeatedStringField(value: &_storage._accountTypeRegexes)
        case 9: try decoder.decodeRepeatedStringField(value: &_storage._startNameRegexes)
        case 10: try decoder.decodeRepeatedStringField(value: &_storage._trimNameRegexes)
        case 11: try decoder.decodeRepeatedStringField(value: &_storage._showNameRegexes)
        case 12: try decoder.decodeRepeatedStringField(value: &_storage._hideNameRegexes)
        case 13: try decoder.decodeSingularBoolField(value: &_storage._accountDisplayedOpOnly)
        case 14: try decoder.decodeRepeatedStringField(value: &_storage._select)
        case 15: try decoder.decodeSingularStringField(value: &_storage._output)
        case 16: try decoder.decodeSingularStringField(value: &_storage._dumpToFile)
        case 17: try decoder.decodeSingularInt64Field(value: &_storage._minOccurrence)
        case 18: try decoder.decodeSingularInt64Field(value: &_storage._step)
        case 19: try decoder.decodeSingularInt64Field(value: &_storage._minPeakBytes)
        case 20: try decoder.decodeSingularInt64Field(value: &_storage._minResidualBytes)
        case 21: try decoder.decodeSingularInt64Field(value: &_storage._minOutputBytes)
        case 22: try decoder.decodeSingularInt64Field(value: &_storage._minAcceleratorMicros)
        case 23: try decoder.decodeSingularInt64Field(value: &_storage._minCpuMicros)
        default: break
        }
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    try withExtendedLifetime(_storage) { (_storage: _StorageClass) in
      if _storage._maxDepth != 0 {
        try visitor.visitSingularInt64Field(value: _storage._maxDepth, fieldNumber: 1)
      }
      if _storage._minBytes != 0 {
        try visitor.visitSingularInt64Field(value: _storage._minBytes, fieldNumber: 2)
      }
      if _storage._minMicros != 0 {
        try visitor.visitSingularInt64Field(value: _storage._minMicros, fieldNumber: 3)
      }
      if _storage._minParams != 0 {
        try visitor.visitSingularInt64Field(value: _storage._minParams, fieldNumber: 4)
      }
      if _storage._minFloatOps != 0 {
        try visitor.visitSingularInt64Field(value: _storage._minFloatOps, fieldNumber: 5)
      }
      if !_storage._orderBy.isEmpty {
        try visitor.visitSingularStringField(value: _storage._orderBy, fieldNumber: 7)
      }
      if !_storage._accountTypeRegexes.isEmpty {
        try visitor.visitRepeatedStringField(value: _storage._accountTypeRegexes, fieldNumber: 8)
      }
      if !_storage._startNameRegexes.isEmpty {
        try visitor.visitRepeatedStringField(value: _storage._startNameRegexes, fieldNumber: 9)
      }
      if !_storage._trimNameRegexes.isEmpty {
        try visitor.visitRepeatedStringField(value: _storage._trimNameRegexes, fieldNumber: 10)
      }
      if !_storage._showNameRegexes.isEmpty {
        try visitor.visitRepeatedStringField(value: _storage._showNameRegexes, fieldNumber: 11)
      }
      if !_storage._hideNameRegexes.isEmpty {
        try visitor.visitRepeatedStringField(value: _storage._hideNameRegexes, fieldNumber: 12)
      }
      if _storage._accountDisplayedOpOnly != false {
        try visitor.visitSingularBoolField(value: _storage._accountDisplayedOpOnly, fieldNumber: 13)
      }
      if !_storage._select.isEmpty {
        try visitor.visitRepeatedStringField(value: _storage._select, fieldNumber: 14)
      }
      if !_storage._output.isEmpty {
        try visitor.visitSingularStringField(value: _storage._output, fieldNumber: 15)
      }
      if !_storage._dumpToFile.isEmpty {
        try visitor.visitSingularStringField(value: _storage._dumpToFile, fieldNumber: 16)
      }
      if _storage._minOccurrence != 0 {
        try visitor.visitSingularInt64Field(value: _storage._minOccurrence, fieldNumber: 17)
      }
      if _storage._step != 0 {
        try visitor.visitSingularInt64Field(value: _storage._step, fieldNumber: 18)
      }
      if _storage._minPeakBytes != 0 {
        try visitor.visitSingularInt64Field(value: _storage._minPeakBytes, fieldNumber: 19)
      }
      if _storage._minResidualBytes != 0 {
        try visitor.visitSingularInt64Field(value: _storage._minResidualBytes, fieldNumber: 20)
      }
      if _storage._minOutputBytes != 0 {
        try visitor.visitSingularInt64Field(value: _storage._minOutputBytes, fieldNumber: 21)
      }
      if _storage._minAcceleratorMicros != 0 {
        try visitor.visitSingularInt64Field(value: _storage._minAcceleratorMicros, fieldNumber: 22)
      }
      if _storage._minCpuMicros != 0 {
        try visitor.visitSingularInt64Field(value: _storage._minCpuMicros, fieldNumber: 23)
      }
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  public func _protobuf_generated_isEqualTo(other: Tensorflow_Tfprof_OptionsProto) -> Bool {
    if _storage !== other._storage {
      let storagesAreEqual: Bool = withExtendedLifetime((_storage, other._storage)) { (_args: (_StorageClass, _StorageClass)) in
        let _storage = _args.0
        let other_storage = _args.1
        if _storage._maxDepth != other_storage._maxDepth {return false}
        if _storage._minBytes != other_storage._minBytes {return false}
        if _storage._minPeakBytes != other_storage._minPeakBytes {return false}
        if _storage._minResidualBytes != other_storage._minResidualBytes {return false}
        if _storage._minOutputBytes != other_storage._minOutputBytes {return false}
        if _storage._minMicros != other_storage._minMicros {return false}
        if _storage._minAcceleratorMicros != other_storage._minAcceleratorMicros {return false}
        if _storage._minCpuMicros != other_storage._minCpuMicros {return false}
        if _storage._minParams != other_storage._minParams {return false}
        if _storage._minFloatOps != other_storage._minFloatOps {return false}
        if _storage._minOccurrence != other_storage._minOccurrence {return false}
        if _storage._step != other_storage._step {return false}
        if _storage._orderBy != other_storage._orderBy {return false}
        if _storage._accountTypeRegexes != other_storage._accountTypeRegexes {return false}
        if _storage._startNameRegexes != other_storage._startNameRegexes {return false}
        if _storage._trimNameRegexes != other_storage._trimNameRegexes {return false}
        if _storage._showNameRegexes != other_storage._showNameRegexes {return false}
        if _storage._hideNameRegexes != other_storage._hideNameRegexes {return false}
        if _storage._accountDisplayedOpOnly != other_storage._accountDisplayedOpOnly {return false}
        if _storage._select != other_storage._select {return false}
        if _storage._output != other_storage._output {return false}
        if _storage._dumpToFile != other_storage._dumpToFile {return false}
        return true
      }
      if !storagesAreEqual {return false}
    }
    if unknownFields != other.unknownFields {return false}
    return true
  }
}

extension Tensorflow_Tfprof_AdvisorOptionsProto: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = _protobuf_package + ".AdvisorOptionsProto"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "checkers"),
  ]

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      switch fieldNumber {
      case 1: try decoder.decodeMapField(fieldType: SwiftProtobuf._ProtobufMessageMap<SwiftProtobuf.ProtobufString,Tensorflow_Tfprof_AdvisorOptionsProto.CheckerOption>.self, value: &self.checkers)
      default: break
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if !self.checkers.isEmpty {
      try visitor.visitMapField(fieldType: SwiftProtobuf._ProtobufMessageMap<SwiftProtobuf.ProtobufString,Tensorflow_Tfprof_AdvisorOptionsProto.CheckerOption>.self, value: self.checkers, fieldNumber: 1)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  public func _protobuf_generated_isEqualTo(other: Tensorflow_Tfprof_AdvisorOptionsProto) -> Bool {
    if self.checkers != other.checkers {return false}
    if unknownFields != other.unknownFields {return false}
    return true
  }
}

extension Tensorflow_Tfprof_AdvisorOptionsProto.CheckerOption: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = Tensorflow_Tfprof_AdvisorOptionsProto.protoMessageName + ".CheckerOption"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "options"),
  ]

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      switch fieldNumber {
      case 1: try decoder.decodeMapField(fieldType: SwiftProtobuf._ProtobufMap<SwiftProtobuf.ProtobufString,SwiftProtobuf.ProtobufString>.self, value: &self.options)
      default: break
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if !self.options.isEmpty {
      try visitor.visitMapField(fieldType: SwiftProtobuf._ProtobufMap<SwiftProtobuf.ProtobufString,SwiftProtobuf.ProtobufString>.self, value: self.options, fieldNumber: 1)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  public func _protobuf_generated_isEqualTo(other: Tensorflow_Tfprof_AdvisorOptionsProto.CheckerOption) -> Bool {
    if self.options != other.options {return false}
    if unknownFields != other.unknownFields {return false}
    return true
  }
}
