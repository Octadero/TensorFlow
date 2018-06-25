// DO NOT EDIT.
//
// Generated by the Swift generator plugin for the protocol buffer compiler.
// Source: tensorflow/core/kernels/boosted_trees/boosted_trees.proto
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

/// Node describes a node in a tree.
public struct Tensorflow_BoostedTrees_Node {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  public var node: OneOf_Node? {
    get {return _storage._node}
    set {_uniqueStorage()._node = newValue}
  }

  public var leaf: Tensorflow_BoostedTrees_Leaf {
    get {
      if case .leaf(let v)? = _storage._node {return v}
      return Tensorflow_BoostedTrees_Leaf()
    }
    set {_uniqueStorage()._node = .leaf(newValue)}
  }

  public var bucketizedSplit: Tensorflow_BoostedTrees_BucketizedSplit {
    get {
      if case .bucketizedSplit(let v)? = _storage._node {return v}
      return Tensorflow_BoostedTrees_BucketizedSplit()
    }
    set {_uniqueStorage()._node = .bucketizedSplit(newValue)}
  }

  public var metadata: Tensorflow_BoostedTrees_NodeMetadata {
    get {return _storage._metadata ?? Tensorflow_BoostedTrees_NodeMetadata()}
    set {_uniqueStorage()._metadata = newValue}
  }
  /// Returns true if `metadata` has been explicitly set.
  public var hasMetadata: Bool {return _storage._metadata != nil}
  /// Clears the value of `metadata`. Subsequent reads from it will return its default value.
  public mutating func clearMetadata() {_storage._metadata = nil}

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public enum OneOf_Node: Equatable {
    case leaf(Tensorflow_BoostedTrees_Leaf)
    case bucketizedSplit(Tensorflow_BoostedTrees_BucketizedSplit)

    public static func ==(lhs: Tensorflow_BoostedTrees_Node.OneOf_Node, rhs: Tensorflow_BoostedTrees_Node.OneOf_Node) -> Bool {
      switch (lhs, rhs) {
      case (.leaf(let l), .leaf(let r)): return l == r
      case (.bucketizedSplit(let l), .bucketizedSplit(let r)): return l == r
      default: return false
      }
    }
  }

  public init() {}

  fileprivate var _storage = _StorageClass.defaultInstance
}

/// NodeMetadata encodes metadata associated with each node in a tree.
public struct Tensorflow_BoostedTrees_NodeMetadata {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  /// The gain associated with this node.
  public var gain: Float {
    get {return _storage._gain}
    set {_uniqueStorage()._gain = newValue}
  }

  /// The original leaf node before this node was split.
  public var originalLeaf: Tensorflow_BoostedTrees_Leaf {
    get {return _storage._originalLeaf ?? Tensorflow_BoostedTrees_Leaf()}
    set {_uniqueStorage()._originalLeaf = newValue}
  }
  /// Returns true if `originalLeaf` has been explicitly set.
  public var hasOriginalLeaf: Bool {return _storage._originalLeaf != nil}
  /// Clears the value of `originalLeaf`. Subsequent reads from it will return its default value.
  public mutating func clearOriginalLeaf() {_storage._originalLeaf = nil}

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public init() {}

  fileprivate var _storage = _StorageClass.defaultInstance
}

/// Leaves can either hold dense or sparse information.
public struct Tensorflow_BoostedTrees_Leaf {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  public var leaf: OneOf_Leaf? {
    get {return _storage._leaf}
    set {_uniqueStorage()._leaf = newValue}
  }

  /// See third_party/tensorflow/contrib/decision_trees/
  /// proto/generic_tree_model.proto
  /// for a description of how vector and sparse_vector might be used.
  public var vector: Tensorflow_BoostedTrees_Vector {
    get {
      if case .vector(let v)? = _storage._leaf {return v}
      return Tensorflow_BoostedTrees_Vector()
    }
    set {_uniqueStorage()._leaf = .vector(newValue)}
  }

  public var sparseVector: Tensorflow_BoostedTrees_SparseVector {
    get {
      if case .sparseVector(let v)? = _storage._leaf {return v}
      return Tensorflow_BoostedTrees_SparseVector()
    }
    set {_uniqueStorage()._leaf = .sparseVector(newValue)}
  }

  public var scalar: Float {
    get {return _storage._scalar}
    set {_uniqueStorage()._scalar = newValue}
  }

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public enum OneOf_Leaf: Equatable {
    /// See third_party/tensorflow/contrib/decision_trees/
    /// proto/generic_tree_model.proto
    /// for a description of how vector and sparse_vector might be used.
    case vector(Tensorflow_BoostedTrees_Vector)
    case sparseVector(Tensorflow_BoostedTrees_SparseVector)

    public static func ==(lhs: Tensorflow_BoostedTrees_Leaf.OneOf_Leaf, rhs: Tensorflow_BoostedTrees_Leaf.OneOf_Leaf) -> Bool {
      switch (lhs, rhs) {
      case (.vector(let l), .vector(let r)): return l == r
      case (.sparseVector(let l), .sparseVector(let r)): return l == r
      default: return false
      }
    }
  }

  public init() {}

  fileprivate var _storage = _StorageClass.defaultInstance
}

public struct Tensorflow_BoostedTrees_Vector {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  public var value: [Float] = []

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public init() {}
}

public struct Tensorflow_BoostedTrees_SparseVector {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  public var index: [Int32] = []

  public var value: [Float] = []

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public init() {}
}

public struct Tensorflow_BoostedTrees_BucketizedSplit {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  /// Float feature column and split threshold describing
  /// the rule feature <= threshold.
  public var featureID: Int32 = 0

  public var threshold: Int32 = 0

  /// Node children indexing into a contiguous
  /// vector of nodes starting from the root.
  public var leftID: Int32 = 0

  public var rightID: Int32 = 0

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public init() {}
}

/// Tree describes a list of connected nodes.
/// Node 0 must be the root and can carry any payload including a leaf
/// in the case of representing the bias.
/// Note that each node id is implicitly its index in the list of nodes.
public struct Tensorflow_BoostedTrees_Tree {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  public var nodes: [Tensorflow_BoostedTrees_Node] = []

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public init() {}
}

public struct Tensorflow_BoostedTrees_TreeMetadata {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  /// Number of layers grown for this tree.
  public var numLayersGrown: Int32 = 0

  /// Whether the tree is finalized in that no more layers can be grown.
  public var isFinalized: Bool = false

  /// If tree was finalized and post pruning happened, it is possible that cache
  /// still refers to some nodes that were deleted or that the node ids changed
  /// (e.g. node id 5 became node id 2 due to pruning of the other branch).
  /// The mapping below allows us to understand where the old ids now map to and
  /// how the values should be adjusted due to post-pruning.
  /// The size of the list should be equal to the number of nodes in the tree
  /// before post-pruning happened.
  /// If the node was pruned, it will have new_node_id equal to the id of a node
  /// that this node was collapsed into. For a node that didn't get pruned, it is
  /// possible that its id still changed, so new_node_id will have the
  /// corresponding id in the pruned tree.
  /// If post-pruning didn't happen, or it did and it had no effect (e.g. no
  /// nodes got pruned), this list will be empty.
  public var postPrunedNodesMeta: [Tensorflow_BoostedTrees_TreeMetadata.PostPruneNodeUpdate] = []

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public struct PostPruneNodeUpdate {
    // SwiftProtobuf.Message conformance is added in an extension below. See the
    // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
    // methods supported on all messages.

    public var newNodeID: Int32 = 0

    public var logitChange: Float = 0

    public var unknownFields = SwiftProtobuf.UnknownStorage()

    public init() {}
  }

  public init() {}
}

public struct Tensorflow_BoostedTrees_GrowingMetadata {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  /// Number of trees that we have attempted to build. After pruning, these
  /// trees might have been removed.
  public var numTreesAttempted: Int64 = 0

  /// Number of layers that we have attempted to build. After pruning, these
  /// layers might have been removed.
  public var numLayersAttempted: Int64 = 0

  /// The start (inclusive) and end (exclusive) ids of the nodes in the latest
  /// layer of the latest tree.
  public var lastLayerNodeStart: Int32 = 0

  public var lastLayerNodeEnd: Int32 = 0

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public init() {}
}

/// TreeEnsemble describes an ensemble of decision trees.
public struct Tensorflow_BoostedTrees_TreeEnsemble {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  public var trees: [Tensorflow_BoostedTrees_Tree] {
    get {return _storage._trees}
    set {_uniqueStorage()._trees = newValue}
  }

  public var treeWeights: [Float] {
    get {return _storage._treeWeights}
    set {_uniqueStorage()._treeWeights = newValue}
  }

  public var treeMetadata: [Tensorflow_BoostedTrees_TreeMetadata] {
    get {return _storage._treeMetadata}
    set {_uniqueStorage()._treeMetadata = newValue}
  }

  /// Metadata that is used during the training.
  public var growingMetadata: Tensorflow_BoostedTrees_GrowingMetadata {
    get {return _storage._growingMetadata ?? Tensorflow_BoostedTrees_GrowingMetadata()}
    set {_uniqueStorage()._growingMetadata = newValue}
  }
  /// Returns true if `growingMetadata` has been explicitly set.
  public var hasGrowingMetadata: Bool {return _storage._growingMetadata != nil}
  /// Clears the value of `growingMetadata`. Subsequent reads from it will return its default value.
  public mutating func clearGrowingMetadata() {_storage._growingMetadata = nil}

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public init() {}

  fileprivate var _storage = _StorageClass.defaultInstance
}

// MARK: - Code below here is support for the SwiftProtobuf runtime.

fileprivate let _protobuf_package = "tensorflow.boosted_trees"

extension Tensorflow_BoostedTrees_Node: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = _protobuf_package + ".Node"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "leaf"),
    2: .standard(proto: "bucketized_split"),
    777: .same(proto: "metadata"),
  ]

  fileprivate class _StorageClass {
    var _node: Tensorflow_BoostedTrees_Node.OneOf_Node?
    var _metadata: Tensorflow_BoostedTrees_NodeMetadata? = nil

    static let defaultInstance = _StorageClass()

    private init() {}

    init(copying source: _StorageClass) {
      _node = source._node
      _metadata = source._metadata
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
          var v: Tensorflow_BoostedTrees_Leaf?
          if let current = _storage._node {
            try decoder.handleConflictingOneOf()
            if case .leaf(let m) = current {v = m}
          }
          try decoder.decodeSingularMessageField(value: &v)
          if let v = v {_storage._node = .leaf(v)}
        case 2:
          var v: Tensorflow_BoostedTrees_BucketizedSplit?
          if let current = _storage._node {
            try decoder.handleConflictingOneOf()
            if case .bucketizedSplit(let m) = current {v = m}
          }
          try decoder.decodeSingularMessageField(value: &v)
          if let v = v {_storage._node = .bucketizedSplit(v)}
        case 777: try decoder.decodeSingularMessageField(value: &_storage._metadata)
        default: break
        }
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    try withExtendedLifetime(_storage) { (_storage: _StorageClass) in
      switch _storage._node {
      case .leaf(let v)?:
        try visitor.visitSingularMessageField(value: v, fieldNumber: 1)
      case .bucketizedSplit(let v)?:
        try visitor.visitSingularMessageField(value: v, fieldNumber: 2)
      case nil: break
      }
      if let v = _storage._metadata {
        try visitor.visitSingularMessageField(value: v, fieldNumber: 777)
      }
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  public func _protobuf_generated_isEqualTo(other: Tensorflow_BoostedTrees_Node) -> Bool {
    if _storage !== other._storage {
      let storagesAreEqual: Bool = withExtendedLifetime((_storage, other._storage)) { (_args: (_StorageClass, _StorageClass)) in
        let _storage = _args.0
        let other_storage = _args.1
        if _storage._node != other_storage._node {return false}
        if _storage._metadata != other_storage._metadata {return false}
        return true
      }
      if !storagesAreEqual {return false}
    }
    if unknownFields != other.unknownFields {return false}
    return true
  }
}

extension Tensorflow_BoostedTrees_NodeMetadata: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = _protobuf_package + ".NodeMetadata"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "gain"),
    2: .standard(proto: "original_leaf"),
  ]

  fileprivate class _StorageClass {
    var _gain: Float = 0
    var _originalLeaf: Tensorflow_BoostedTrees_Leaf? = nil

    static let defaultInstance = _StorageClass()

    private init() {}

    init(copying source: _StorageClass) {
      _gain = source._gain
      _originalLeaf = source._originalLeaf
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
        case 1: try decoder.decodeSingularFloatField(value: &_storage._gain)
        case 2: try decoder.decodeSingularMessageField(value: &_storage._originalLeaf)
        default: break
        }
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    try withExtendedLifetime(_storage) { (_storage: _StorageClass) in
      if _storage._gain != 0 {
        try visitor.visitSingularFloatField(value: _storage._gain, fieldNumber: 1)
      }
      if let v = _storage._originalLeaf {
        try visitor.visitSingularMessageField(value: v, fieldNumber: 2)
      }
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  public func _protobuf_generated_isEqualTo(other: Tensorflow_BoostedTrees_NodeMetadata) -> Bool {
    if _storage !== other._storage {
      let storagesAreEqual: Bool = withExtendedLifetime((_storage, other._storage)) { (_args: (_StorageClass, _StorageClass)) in
        let _storage = _args.0
        let other_storage = _args.1
        if _storage._gain != other_storage._gain {return false}
        if _storage._originalLeaf != other_storage._originalLeaf {return false}
        return true
      }
      if !storagesAreEqual {return false}
    }
    if unknownFields != other.unknownFields {return false}
    return true
  }
}

extension Tensorflow_BoostedTrees_Leaf: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = _protobuf_package + ".Leaf"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "vector"),
    2: .standard(proto: "sparse_vector"),
    3: .same(proto: "scalar"),
  ]

  fileprivate class _StorageClass {
    var _leaf: Tensorflow_BoostedTrees_Leaf.OneOf_Leaf?
    var _scalar: Float = 0

    static let defaultInstance = _StorageClass()

    private init() {}

    init(copying source: _StorageClass) {
      _leaf = source._leaf
      _scalar = source._scalar
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
          var v: Tensorflow_BoostedTrees_Vector?
          if let current = _storage._leaf {
            try decoder.handleConflictingOneOf()
            if case .vector(let m) = current {v = m}
          }
          try decoder.decodeSingularMessageField(value: &v)
          if let v = v {_storage._leaf = .vector(v)}
        case 2:
          var v: Tensorflow_BoostedTrees_SparseVector?
          if let current = _storage._leaf {
            try decoder.handleConflictingOneOf()
            if case .sparseVector(let m) = current {v = m}
          }
          try decoder.decodeSingularMessageField(value: &v)
          if let v = v {_storage._leaf = .sparseVector(v)}
        case 3: try decoder.decodeSingularFloatField(value: &_storage._scalar)
        default: break
        }
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    try withExtendedLifetime(_storage) { (_storage: _StorageClass) in
      switch _storage._leaf {
      case .vector(let v)?:
        try visitor.visitSingularMessageField(value: v, fieldNumber: 1)
      case .sparseVector(let v)?:
        try visitor.visitSingularMessageField(value: v, fieldNumber: 2)
      case nil: break
      }
      if _storage._scalar != 0 {
        try visitor.visitSingularFloatField(value: _storage._scalar, fieldNumber: 3)
      }
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  public func _protobuf_generated_isEqualTo(other: Tensorflow_BoostedTrees_Leaf) -> Bool {
    if _storage !== other._storage {
      let storagesAreEqual: Bool = withExtendedLifetime((_storage, other._storage)) { (_args: (_StorageClass, _StorageClass)) in
        let _storage = _args.0
        let other_storage = _args.1
        if _storage._leaf != other_storage._leaf {return false}
        if _storage._scalar != other_storage._scalar {return false}
        return true
      }
      if !storagesAreEqual {return false}
    }
    if unknownFields != other.unknownFields {return false}
    return true
  }
}

extension Tensorflow_BoostedTrees_Vector: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = _protobuf_package + ".Vector"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "value"),
  ]

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      switch fieldNumber {
      case 1: try decoder.decodeRepeatedFloatField(value: &self.value)
      default: break
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if !self.value.isEmpty {
      try visitor.visitPackedFloatField(value: self.value, fieldNumber: 1)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  public func _protobuf_generated_isEqualTo(other: Tensorflow_BoostedTrees_Vector) -> Bool {
    if self.value != other.value {return false}
    if unknownFields != other.unknownFields {return false}
    return true
  }
}

extension Tensorflow_BoostedTrees_SparseVector: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = _protobuf_package + ".SparseVector"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "index"),
    2: .same(proto: "value"),
  ]

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      switch fieldNumber {
      case 1: try decoder.decodeRepeatedInt32Field(value: &self.index)
      case 2: try decoder.decodeRepeatedFloatField(value: &self.value)
      default: break
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if !self.index.isEmpty {
      try visitor.visitPackedInt32Field(value: self.index, fieldNumber: 1)
    }
    if !self.value.isEmpty {
      try visitor.visitPackedFloatField(value: self.value, fieldNumber: 2)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  public func _protobuf_generated_isEqualTo(other: Tensorflow_BoostedTrees_SparseVector) -> Bool {
    if self.index != other.index {return false}
    if self.value != other.value {return false}
    if unknownFields != other.unknownFields {return false}
    return true
  }
}

extension Tensorflow_BoostedTrees_BucketizedSplit: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = _protobuf_package + ".BucketizedSplit"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .standard(proto: "feature_id"),
    2: .same(proto: "threshold"),
    3: .standard(proto: "left_id"),
    4: .standard(proto: "right_id"),
  ]

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      switch fieldNumber {
      case 1: try decoder.decodeSingularInt32Field(value: &self.featureID)
      case 2: try decoder.decodeSingularInt32Field(value: &self.threshold)
      case 3: try decoder.decodeSingularInt32Field(value: &self.leftID)
      case 4: try decoder.decodeSingularInt32Field(value: &self.rightID)
      default: break
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if self.featureID != 0 {
      try visitor.visitSingularInt32Field(value: self.featureID, fieldNumber: 1)
    }
    if self.threshold != 0 {
      try visitor.visitSingularInt32Field(value: self.threshold, fieldNumber: 2)
    }
    if self.leftID != 0 {
      try visitor.visitSingularInt32Field(value: self.leftID, fieldNumber: 3)
    }
    if self.rightID != 0 {
      try visitor.visitSingularInt32Field(value: self.rightID, fieldNumber: 4)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  public func _protobuf_generated_isEqualTo(other: Tensorflow_BoostedTrees_BucketizedSplit) -> Bool {
    if self.featureID != other.featureID {return false}
    if self.threshold != other.threshold {return false}
    if self.leftID != other.leftID {return false}
    if self.rightID != other.rightID {return false}
    if unknownFields != other.unknownFields {return false}
    return true
  }
}

extension Tensorflow_BoostedTrees_Tree: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = _protobuf_package + ".Tree"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "nodes"),
  ]

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      switch fieldNumber {
      case 1: try decoder.decodeRepeatedMessageField(value: &self.nodes)
      default: break
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if !self.nodes.isEmpty {
      try visitor.visitRepeatedMessageField(value: self.nodes, fieldNumber: 1)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  public func _protobuf_generated_isEqualTo(other: Tensorflow_BoostedTrees_Tree) -> Bool {
    if self.nodes != other.nodes {return false}
    if unknownFields != other.unknownFields {return false}
    return true
  }
}

extension Tensorflow_BoostedTrees_TreeMetadata: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = _protobuf_package + ".TreeMetadata"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    2: .standard(proto: "num_layers_grown"),
    3: .standard(proto: "is_finalized"),
    4: .standard(proto: "post_pruned_nodes_meta"),
  ]

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      switch fieldNumber {
      case 2: try decoder.decodeSingularInt32Field(value: &self.numLayersGrown)
      case 3: try decoder.decodeSingularBoolField(value: &self.isFinalized)
      case 4: try decoder.decodeRepeatedMessageField(value: &self.postPrunedNodesMeta)
      default: break
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if self.numLayersGrown != 0 {
      try visitor.visitSingularInt32Field(value: self.numLayersGrown, fieldNumber: 2)
    }
    if self.isFinalized != false {
      try visitor.visitSingularBoolField(value: self.isFinalized, fieldNumber: 3)
    }
    if !self.postPrunedNodesMeta.isEmpty {
      try visitor.visitRepeatedMessageField(value: self.postPrunedNodesMeta, fieldNumber: 4)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  public func _protobuf_generated_isEqualTo(other: Tensorflow_BoostedTrees_TreeMetadata) -> Bool {
    if self.numLayersGrown != other.numLayersGrown {return false}
    if self.isFinalized != other.isFinalized {return false}
    if self.postPrunedNodesMeta != other.postPrunedNodesMeta {return false}
    if unknownFields != other.unknownFields {return false}
    return true
  }
}

extension Tensorflow_BoostedTrees_TreeMetadata.PostPruneNodeUpdate: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = Tensorflow_BoostedTrees_TreeMetadata.protoMessageName + ".PostPruneNodeUpdate"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .standard(proto: "new_node_id"),
    2: .standard(proto: "logit_change"),
  ]

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      switch fieldNumber {
      case 1: try decoder.decodeSingularInt32Field(value: &self.newNodeID)
      case 2: try decoder.decodeSingularFloatField(value: &self.logitChange)
      default: break
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if self.newNodeID != 0 {
      try visitor.visitSingularInt32Field(value: self.newNodeID, fieldNumber: 1)
    }
    if self.logitChange != 0 {
      try visitor.visitSingularFloatField(value: self.logitChange, fieldNumber: 2)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  public func _protobuf_generated_isEqualTo(other: Tensorflow_BoostedTrees_TreeMetadata.PostPruneNodeUpdate) -> Bool {
    if self.newNodeID != other.newNodeID {return false}
    if self.logitChange != other.logitChange {return false}
    if unknownFields != other.unknownFields {return false}
    return true
  }
}

extension Tensorflow_BoostedTrees_GrowingMetadata: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = _protobuf_package + ".GrowingMetadata"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .standard(proto: "num_trees_attempted"),
    2: .standard(proto: "num_layers_attempted"),
    3: .standard(proto: "last_layer_node_start"),
    4: .standard(proto: "last_layer_node_end"),
  ]

  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      switch fieldNumber {
      case 1: try decoder.decodeSingularInt64Field(value: &self.numTreesAttempted)
      case 2: try decoder.decodeSingularInt64Field(value: &self.numLayersAttempted)
      case 3: try decoder.decodeSingularInt32Field(value: &self.lastLayerNodeStart)
      case 4: try decoder.decodeSingularInt32Field(value: &self.lastLayerNodeEnd)
      default: break
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if self.numTreesAttempted != 0 {
      try visitor.visitSingularInt64Field(value: self.numTreesAttempted, fieldNumber: 1)
    }
    if self.numLayersAttempted != 0 {
      try visitor.visitSingularInt64Field(value: self.numLayersAttempted, fieldNumber: 2)
    }
    if self.lastLayerNodeStart != 0 {
      try visitor.visitSingularInt32Field(value: self.lastLayerNodeStart, fieldNumber: 3)
    }
    if self.lastLayerNodeEnd != 0 {
      try visitor.visitSingularInt32Field(value: self.lastLayerNodeEnd, fieldNumber: 4)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  public func _protobuf_generated_isEqualTo(other: Tensorflow_BoostedTrees_GrowingMetadata) -> Bool {
    if self.numTreesAttempted != other.numTreesAttempted {return false}
    if self.numLayersAttempted != other.numLayersAttempted {return false}
    if self.lastLayerNodeStart != other.lastLayerNodeStart {return false}
    if self.lastLayerNodeEnd != other.lastLayerNodeEnd {return false}
    if unknownFields != other.unknownFields {return false}
    return true
  }
}

extension Tensorflow_BoostedTrees_TreeEnsemble: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let protoMessageName: String = _protobuf_package + ".TreeEnsemble"
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "trees"),
    2: .standard(proto: "tree_weights"),
    3: .standard(proto: "tree_metadata"),
    4: .standard(proto: "growing_metadata"),
  ]

  fileprivate class _StorageClass {
    var _trees: [Tensorflow_BoostedTrees_Tree] = []
    var _treeWeights: [Float] = []
    var _treeMetadata: [Tensorflow_BoostedTrees_TreeMetadata] = []
    var _growingMetadata: Tensorflow_BoostedTrees_GrowingMetadata? = nil

    static let defaultInstance = _StorageClass()

    private init() {}

    init(copying source: _StorageClass) {
      _trees = source._trees
      _treeWeights = source._treeWeights
      _treeMetadata = source._treeMetadata
      _growingMetadata = source._growingMetadata
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
        case 1: try decoder.decodeRepeatedMessageField(value: &_storage._trees)
        case 2: try decoder.decodeRepeatedFloatField(value: &_storage._treeWeights)
        case 3: try decoder.decodeRepeatedMessageField(value: &_storage._treeMetadata)
        case 4: try decoder.decodeSingularMessageField(value: &_storage._growingMetadata)
        default: break
        }
      }
    }
  }

  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    try withExtendedLifetime(_storage) { (_storage: _StorageClass) in
      if !_storage._trees.isEmpty {
        try visitor.visitRepeatedMessageField(value: _storage._trees, fieldNumber: 1)
      }
      if !_storage._treeWeights.isEmpty {
        try visitor.visitPackedFloatField(value: _storage._treeWeights, fieldNumber: 2)
      }
      if !_storage._treeMetadata.isEmpty {
        try visitor.visitRepeatedMessageField(value: _storage._treeMetadata, fieldNumber: 3)
      }
      if let v = _storage._growingMetadata {
        try visitor.visitSingularMessageField(value: v, fieldNumber: 4)
      }
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  public func _protobuf_generated_isEqualTo(other: Tensorflow_BoostedTrees_TreeEnsemble) -> Bool {
    if _storage !== other._storage {
      let storagesAreEqual: Bool = withExtendedLifetime((_storage, other._storage)) { (_args: (_StorageClass, _StorageClass)) in
        let _storage = _args.0
        let other_storage = _args.1
        if _storage._trees != other_storage._trees {return false}
        if _storage._treeWeights != other_storage._treeWeights {return false}
        if _storage._treeMetadata != other_storage._treeMetadata {return false}
        if _storage._growingMetadata != other_storage._growingMetadata {return false}
        return true
      }
      if !storagesAreEqual {return false}
    }
    if unknownFields != other.unknownFields {return false}
    return true
  }
}
