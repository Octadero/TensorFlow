// DO NOT EDIT.
//
// Generated by the Swift generator plugin for the protocol buffer compiler.
// Source: tensorflow/compiler/xla/service/hlo_profile_printer_data.proto
//
// For information on using the generated types, please see the documenation:
//   https://github.com/apple/swift-protobuf/

// Copyright 2018 The TensorFlow Authors. All Rights Reserved.
//
//Licensed under the Apache License, Version 2.0 (the "License");
//you may not use this file except in compliance with the License.
//You may obtain a copy of the License at
//
//http://www.apache.org/licenses/LICENSE-2.0
//
//Unless required by applicable law or agreed to in writing, software
//distributed under the License is distributed on an "AS IS" BASIS,
//WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//See the License for the specific language governing permissions and
//limitations under the License.
//==============================================================================

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

/// Describes how to pretty-print a profile counter array gathered for a specific
/// HloModule.
public struct Xla_HloProfilePrinterData {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  /// HloComputationInfos for every HloComputation in the HloModule.
  public var computationInfos: [Xla_HloProfilePrinterData.HloComputationInfo] = []

  /// The size of the profile counters array we will pretty-print.
  public var profileCountersSize: Int64 = 0

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  /// Pretty-printer information about an HloInstruction.
  public struct HloInstructionInfo {
    // SwiftProtobuf.Message conformance is added in an extension below. See the
    // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
    // methods supported on all messages.

    public var longName: String = String()

    public var shortName: String = String()

    public var category: String = String()

    /// Metrics computed by HloCostAnalysis.
    public var flopCount: Float = 0

    public var transcendentalCount: Float = 0

    public var bytesAccessed: Float = 0

    public var optimalSeconds: Float = 0

    /// The index into the profile counters array for the HloInstruction
    /// corresponding to this HloInstructionInfo.
    public var profileIndex: Int64 = 0

    public var unknownFields = SwiftProtobuf.UnknownStorage()

    public init() {}
  }

  /// Pretty-printer information about an HloComputation.
  public struct HloComputationInfo {
    // SwiftProtobuf.Message conformance is added in an extension below. See the
    // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
    // methods supported on all messages.

    public var name: String = String()

    /// The index into the profile counters array for the HloComputation
    /// corresponding to this HloComputationInfo.
    public var profileIndex: Int64 = 0

    /// HloInstructionInfos for every HloInstruction in the HloComputation for
    /// corresponding to this HloComputattionInfo.
    public var instructionInfos: [Xla_HloProfilePrinterData.HloInstructionInfo] = []

    public var unknownFields = SwiftProtobuf.UnknownStorage()

    public init() {}
  }

  public init() {}
}

// MARK: - Code below here is support for the SwiftProtobuf runtime.

fileprivate let _protobuf_package = "xla"

extension Xla_HloProfilePrinterData: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = _protobuf_package + ".HloProfilePrinterData"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .standard(proto: "computation_infos"),
    2: .standard(proto: "profile_counters_size"),
  ]

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      switch fieldNumber {
      case 1: try decoder.decodeRepeatedMessageField(value: &self.computationInfos)
      case 2: try decoder.decodeSingularInt64Field(value: &self.profileCountersSize)
      default: break
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if !self.computationInfos.isEmpty {
      try visitor.visitRepeatedMessageField(value: self.computationInfos, fieldNumber: 1)
    }
    if self.profileCountersSize != 0 {
      try visitor.visitSingularInt64Field(value: self.profileCountersSize, fieldNumber: 2)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  public func _protobuf_generated_isEqualTo(other: Xla_HloProfilePrinterData) -> Bool {
    if self.computationInfos != other.computationInfos {return false}
    if self.profileCountersSize != other.profileCountersSize {return false}
    if unknownFields != other.unknownFields {return false}
    return true
  }
}

extension Xla_HloProfilePrinterData.HloInstructionInfo: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = Xla_HloProfilePrinterData.protoMessageName + ".HloInstructionInfo"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .standard(proto: "long_name"),
    2: .standard(proto: "short_name"),
    3: .same(proto: "category"),
    4: .standard(proto: "flop_count"),
    5: .standard(proto: "transcendental_count"),
    6: .standard(proto: "bytes_accessed"),
    7: .standard(proto: "optimal_seconds"),
    8: .standard(proto: "profile_index"),
  ]

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      switch fieldNumber {
      case 1: try decoder.decodeSingularStringField(value: &self.longName)
      case 2: try decoder.decodeSingularStringField(value: &self.shortName)
      case 3: try decoder.decodeSingularStringField(value: &self.category)
      case 4: try decoder.decodeSingularFloatField(value: &self.flopCount)
      case 5: try decoder.decodeSingularFloatField(value: &self.transcendentalCount)
      case 6: try decoder.decodeSingularFloatField(value: &self.bytesAccessed)
      case 7: try decoder.decodeSingularFloatField(value: &self.optimalSeconds)
      case 8: try decoder.decodeSingularInt64Field(value: &self.profileIndex)
      default: break
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if !self.longName.isEmpty {
      try visitor.visitSingularStringField(value: self.longName, fieldNumber: 1)
    }
    if !self.shortName.isEmpty {
      try visitor.visitSingularStringField(value: self.shortName, fieldNumber: 2)
    }
    if !self.category.isEmpty {
      try visitor.visitSingularStringField(value: self.category, fieldNumber: 3)
    }
    if self.flopCount != 0 {
      try visitor.visitSingularFloatField(value: self.flopCount, fieldNumber: 4)
    }
    if self.transcendentalCount != 0 {
      try visitor.visitSingularFloatField(value: self.transcendentalCount, fieldNumber: 5)
    }
    if self.bytesAccessed != 0 {
      try visitor.visitSingularFloatField(value: self.bytesAccessed, fieldNumber: 6)
    }
    if self.optimalSeconds != 0 {
      try visitor.visitSingularFloatField(value: self.optimalSeconds, fieldNumber: 7)
    }
    if self.profileIndex != 0 {
      try visitor.visitSingularInt64Field(value: self.profileIndex, fieldNumber: 8)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  public func _protobuf_generated_isEqualTo(other: Xla_HloProfilePrinterData.HloInstructionInfo) -> Bool {
    if self.longName != other.longName {return false}
    if self.shortName != other.shortName {return false}
    if self.category != other.category {return false}
    if self.flopCount != other.flopCount {return false}
    if self.transcendentalCount != other.transcendentalCount {return false}
    if self.bytesAccessed != other.bytesAccessed {return false}
    if self.optimalSeconds != other.optimalSeconds {return false}
    if self.profileIndex != other.profileIndex {return false}
    if unknownFields != other.unknownFields {return false}
    return true
  }
}

extension Xla_HloProfilePrinterData.HloComputationInfo: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = Xla_HloProfilePrinterData.protoMessageName + ".HloComputationInfo"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "name"),
    2: .standard(proto: "profile_index"),
    3: .standard(proto: "instruction_infos"),
  ]

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      switch fieldNumber {
      case 1: try decoder.decodeSingularStringField(value: &self.name)
      case 2: try decoder.decodeSingularInt64Field(value: &self.profileIndex)
      case 3: try decoder.decodeRepeatedMessageField(value: &self.instructionInfos)
      default: break
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if !self.name.isEmpty {
      try visitor.visitSingularStringField(value: self.name, fieldNumber: 1)
    }
    if self.profileIndex != 0 {
      try visitor.visitSingularInt64Field(value: self.profileIndex, fieldNumber: 2)
    }
    if !self.instructionInfos.isEmpty {
      try visitor.visitRepeatedMessageField(value: self.instructionInfos, fieldNumber: 3)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  public func _protobuf_generated_isEqualTo(other: Xla_HloProfilePrinterData.HloComputationInfo) -> Bool {
    if self.name != other.name {return false}
    if self.profileIndex != other.profileIndex {return false}
    if self.instructionInfos != other.instructionInfos {return false}
    if unknownFields != other.unknownFields {return false}
    return true
  }
}
