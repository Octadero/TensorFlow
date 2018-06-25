/* Copyright 2017 The Octadero Authors. All Rights Reserved.
Created by Volodymyr Pavliukevych on 2017.

Licensed under the GPL License, Version 3.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.gnu.org/licenses/gpl-3.0.txt

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

import CTensorFlow
import Foundation
import Proto

/// Represents a computation graph.  Graphs may be shared between sessions.
/// Graphs are thread-safe when used as directed below.

/// Return a new TF graph object.
public func newGraph() -> TF_Graph {
	return TF_NewGraph()
}

/// Destroy an options object.  Graph will be deleted once no more
/// TFSession's are referencing it.
public func delete(graph: TF_Graph!) {
	return TF_DeleteGraph(graph)
}

/// Operation being built. The underlying graph must outlive this.

/// Operation that has been added to the graph. Valid until the graph is
/// deleted -- in particular adding a new operation to the graph does not
/// invalidate old TF_Operation* pointers.

/// Represents a specific input of an operation.
/*public struct TF_Input {

public var oper: OpaquePointer!

public var index: Int32 /// The index of the input within oper.

public init(){
self = TF_Input()
}

public init(oper: OpaquePointer!, index: Int32){
self = TF_Input(oper,index)
}
}*/

/// Represents a specific output of an operation.
/*public struct TF_Output {

public var oper: OpaquePointer!

public var index: Int32 /// The index of the output within oper.

public init(){
self = TF_Output
}

public init(oper: OpaquePointer!, index: Int32){
self = TF_Output(oper,index)
}
}*/

/// Sets the shape of the Tensor referenced by `output` in `graph` to
/// the shape described by `dims` and `num_dims`.
///
/// If the number of dimensions is unknown, `num_dims` must be
/// set to -1 and dims can be null. If a dimension is unknown,
/// the corresponding entry in the `dims` array must be -1.
///
/// This does not overwrite the existing shape associated with `output`,
/// but merges the input shape with the existing shape.  For example,
/// setting a shape of [-1, 2] with an existing shape [2, -1] would set
/// a final shape of [2, 2] based on shape merging semantics.
///
/// Returns an error into `status` if:
///   * `output` is not in `graph`.
///   * An invalid shape is being set (e.g., the shape being set
///     is incompatible with the existing shape).
public func setTensorShape(in graph: TF_Graph!, output: TF_Output, dimensions: UnsafePointer<Int64>!, dimensionsNumber: Int32, status: TF_Status!) {
	TF_GraphSetTensorShape(graph, output, dimensions, dimensionsNumber, status)
}

/// Returns the number of dimensions of the Tensor referenced by `output`
/// in `graph`.
///
/// If the number of dimensions in the shape is unknown, returns -1.
///
/// Returns an error into `status` if:
///   * `output` is not in `graph`.
public func getTensorDimensionsNumber(in graph: TF_Graph!, output: TF_Output, status: TF_Status!) -> Int32 {
	return TF_GraphGetTensorNumDims(graph, output, status)
}

/// Returns the shape of the Tensor referenced by `output` in `graph`
/// into `dims`. `dims` must be an array large enough to hold `num_dims`
/// entries (e.g., the return value of TF_GraphGetTensorNumDims).
///
/// If the number of dimensions in the shape is unknown or the shape is
/// a scalar, `dims` will remain untouched. Otherwise, each element of
/// `dims` will be set corresponding to the size of the dimension. An
/// unknown dimension is represented by `-1`.
///
/// Returns an error into `status` if:
///   * `output` is not in `graph`.
///   * `num_dims` does not match the actual number of dimensions.
public func getTensorShape(in graph: TF_Graph!, output: TF_Output, dimensions: UnsafeMutablePointer<Int64>!, dimensionsNumber: Int32, status: TF_Status!) {
	return TF_GraphGetTensorShape(graph, output, dimensions, dimensionsNumber, status)
}

/// Operation will only be added to *graph when TF_FinishOperation() is
/// called (assuming TF_FinishOperation() does not return an error).
/// *graph must not be deleted until after TF_FinishOperation() is
/// called.
public func newOperation(in graph: TF_Graph!, operationType: String, operationName: String) -> TF_OperationDescription! {
	return TF_NewOperation(graph, operationType.cString(using: .utf8), operationName.cString(using: .utf8))
}

/// Specify the device for `desc`.  Defaults to empty, meaning unconstrained.
public func set(operationDescription: TF_OperationDescription!, for device: String) {
	TF_SetDevice(operationDescription, device.cString(using: .utf8))
}

/// The calls to TF_AddInput and TF_AddInputList must match (in number,
/// order, and type) the op declaration.  For example, the "Concat" op
/// has registration:
///   REGISTER_OP("Concat")
///       .Input("concat_dim: int32")
///       .Input("values: N * T")
///       .Output("output: T")
///       .Attr("N: int >= 2")
///       .Attr("T: type");
/// that defines two inputs, "concat_dim" and "values" (in that order).
/// You must use TF_AddInput() for the first input (since it takes a
/// single tensor), and TF_AddInputList() for the second input (since
/// it takes a list, even if you were to pass a list with a single
/// tensor), as in:
///   TF_OperationDescription* desc = TF_NewOperation(graph, "Concat", "c");
///   TF_Output concat_dim_input = {...};
///   TF_AddInput(desc, concat_dim_input);
///   TF_Output values_inputs[5] = {{...}, ..., {...}};
///   TF_AddInputList(desc, values_inputs, 5);

/// For inputs that take a single tensor.
public func add(input: TF_Output, for operationDescription: TF_OperationDescription!) {
	TF_AddInput(operationDescription, input)
}

/// For inputs that take a list of tensors.
/// inputs must point to TF_Output[num_inputs].
public func add(inputs: [TF_Output], for operationDescription: TF_OperationDescription!) throws {
    try inputs.withUnsafeBufferPointer { (bufferPointer: UnsafeBufferPointer<TF_Output>) in
        guard let pointer = bufferPointer.baseAddress else {
            throw CAPIError.canNotComputPointer(functionName: #function)
        }
        TF_AddInputList(operationDescription, pointer, Int32(inputs.count))
    }
}

/// Call once per control input to `desc`.
public func add(controlInput: TF_Operation, for operationDescription: TF_OperationDescription!) {
	TF_AddControlInput(operationDescription, controlInput)
}

/// Request that `desc` be co-located on the device where `op`
/// is placed.
///
/// Use of this is discouraged since the implementation of device placement is
/// subject to change. Primarily intended for internal libraries
public func colocate(description: TF_OperationDescription!, operation: TF_Operation!) {
	TF_ColocateWith(description, operation)
}

/// Call some TF_SetAttr*() function for every attr that is not
/// inferred from an input and doesn't have a default value you wish to
/// keep.

/// `value` must point to a string of length `length` bytes.
public func setAttribute(value: String, by name: String, for operationDescription: TF_OperationDescription!) {
	let stringLength = value.count
	value.withCString { (pointer) in
		TF_SetAttrString(operationDescription, name.cString(using: .utf8), pointer, stringLength)
	}
}

/// `values` and `lengths` each must have lengths `num_values`.
/// `values[i]` must point to a string of length `lengths[i]` bytes.
public func setAttribute(values: [String], by name: String, for operationDescription: TF_OperationDescription!){
    let lengths = values.map { Int($0.count) }
    var localValues = values
    let lengthsUnsafePointer = lengths.withUnsafeBufferPointer { $0.baseAddress! }
    
    let unsafeRawPointer = withUnsafePointer(to: &localValues) { (pointer) -> UnsafePointer<UnsafeRawPointer?> in
        return pointer.withMemoryRebound(to: UnsafeRawPointer?.self, capacity: values.count, { (unsafeRawPointer) -> UnsafePointer<UnsafeRawPointer?> in
            return unsafeRawPointer
        })
    }
    
	TF_SetAttrStringList(operationDescription,
                         name.cString(using: .utf8),
                         unsafeRawPointer,
                         lengthsUnsafePointer,
                         Int32(values.count))
}

public func setAttribute(value: Int64, by name: String, for operationDescription: TF_OperationDescription!) {
	TF_SetAttrInt(operationDescription, name.cString(using: .utf8), value)
}

public func setAttribute(value: Int32, by name: String, for operationDescription: TF_OperationDescription!) {
    TF_SetAttrInt(operationDescription, name.cString(using: .utf8), Int64(value))
}

public func setAttribute(value: Int16, by name: String, for operationDescription: TF_OperationDescription!) {
    TF_SetAttrInt(operationDescription, name.cString(using: .utf8), Int64(value))
}

public func setAttribute(value: Int8, by name: String, for operationDescription: TF_OperationDescription!) {
    TF_SetAttrInt(operationDescription, name.cString(using: .utf8), Int64(value))
}

public func setAttribute(value: UInt8, by name: String, for operationDescription: TF_OperationDescription!) {
    TF_SetAttrInt(operationDescription, name.cString(using: .utf8), Int64(value))
}

public func setAttribute(values: [Int64], by name: String, for operationDescription: TF_OperationDescription!) {
    values.withUnsafeBufferPointer { bufferPointer in
        TF_SetAttrIntList(operationDescription, name.cString(using: .utf8), bufferPointer.baseAddress, Int32(values.count))
    }
}

public func setAttribute(value: Float, by name: String, for operationDescription: TF_OperationDescription!) {
	TF_SetAttrFloat(operationDescription, name.cString(using: .utf8), value)
}

public func setAttribute(values: [Float], by name: String, for operationDescription: TF_OperationDescription!) {
    values.withUnsafeBufferPointer { bufferPointer in
        TF_SetAttrFloatList(operationDescription, name.cString(using: .utf8), bufferPointer.baseAddress, Int32(values.count))
    }
}

/// Set Bool attribute for Operation Description.
public func setAttribute(value: Bool, by name: String, for operationDescription: TF_OperationDescription!) {
	TF_SetAttrBool(operationDescription, name.cString(using: .utf8), value ? 1 : 0)
}

/// Set list of Boolean attributes for Operation Description.
public func setAttribute(values: [Bool], by name: String, for operationDescription: TF_OperationDescription!) {
    values.withUnsafeBufferPointer { bufferPointer in
        bufferPointer.baseAddress?.withMemoryRebound(to: UInt8.self, capacity: values.count, { pointer in
            TF_SetAttrBoolList(operationDescription, name.cString(using: .utf8), pointer, Int32(values.count))
        })
    }
}

public func setAttribute(type: TF_DataType, by name: String, for operationDescription: TF_OperationDescription!) {
	TF_SetAttrType(operationDescription, name.cString(using: .utf8), type)
}

public func setAttribute(types: [TF_DataType], by name: String, for operationDescription: TF_OperationDescription!) {
    types.withUnsafeBufferPointer { bufferPointer in
        TF_SetAttrTypeList(operationDescription, name.cString(using: .utf8), bufferPointer.baseAddress, Int32(types.count))
    }
}

/// Set `dimensions` to nil to represent "unknown rank".
/// Read Tensorflow_TensorShapeProto and tensor_shaper.pb for more details.
/// From C API:
/// 	Set `num_dims` to -1 to represent "unknown rank".  Otherwise,
/// 	`dims` points to an array of length `num_dims`.  `dims[i]` must be
/// 	>= -1, with -1 meaning "unknown dimension".
public func setAttributeShape(dimensions: [Int64]?, by name: String, for operationDescription: TF_OperationDescription!) {
	if let dimensions = dimensions {
		dimensions.withUnsafeBufferPointer { bufferPointer in
			TF_SetAttrShape(operationDescription, name.cString(using: .utf8), bufferPointer.baseAddress, Int32(dimensions.count))
		}
	} else {
		TF_SetAttrShape(operationDescription, name.cString(using: .utf8), nil, Int32(-1))
	}
}

/// `dims` and `num_dims` must point to arrays of length `num_shapes`.
/// Set `num_dims[i]` to -1 to represent "unknown rank".  Otherwise,
/// `dims[i]` points to an array of length `num_dims[i]`.  `dims[i][j]`
/// must be >= -1, with -1 meaning "unknown dimension".
public func setAttributesShape(dimensions: UnsafePointer<UnsafePointer<Int64>?>!, by name: String, dimensionsNumber: UnsafePointer<Int32>!, shapesNumber: Int32, for operationDescription: TF_OperationDescription!) {
	TF_SetAttrShapeList(operationDescription, name.cString(using: .utf8), dimensions, dimensionsNumber, shapesNumber)
}

/// `proto` must point to an array of `proto_len` bytes representing a
/// binary-serialized TensorShapeProto.
public func setAttributeTensorShape(proto: UnsafeRawPointer!, by name: String, protoLength: Int, for operationDescription: TF_OperationDescription!, status: TF_Status!) {
	TF_SetAttrTensorShapeProto(operationDescription, name.cString(using: .utf8), proto, protoLength, status)
}

/// `protos` and `proto_lens` must point to arrays of length `num_shapes`.
/// `protos[i]` must point to an array of `proto_lens[i]` bytes
/// representing a binary-serialized TensorShapeProto.
public func setAttributesTensorShape(protos: UnsafePointer<UnsafeRawPointer?>!, by name: String, protoLength: UnsafePointer<Int>!, shapesNumber: Int32, for operationDescription: TF_OperationDescription!, status: TF_Status!) {
	TF_SetAttrTensorShapeProtoList(operationDescription, name.cString(using: .utf8), protos, protoLength, shapesNumber, status)
}

public func setAttribute(tensor: TF_Tensor!, by name: String, for operationDescription: TF_OperationDescription!) throws {
	let status = newStatus()
    defer {
        delete(status: status)
    }
	TF_SetAttrTensor(operationDescription, name.cString(using: .utf8), tensor, status)
	if let status = status, let error = StatusError(tfStatus: status) {
		throw error
	}
}
public func setAttributes(tensors: UnsafePointer<TF_Tensor?>!, by name: String, tensorsNumber: Int32, for operationDescription: TF_OperationDescription!, status: TF_Status!) {
	fatalError("\(#function): Not implemented.")
}

/// `proto` should point to a sequence of bytes of length `proto_len`
/// representing a binary serialization of an AttrValue protocol
/// buffer.
public func setAttribute(proto: Tensorflow_AttrValue, by name: String, for operationDescription: TF_OperationDescription!) throws {
    var data = try proto.serializedData()
    
    let attrName = name.cString(using: .utf8)
    let status = newStatus()
    defer {
        delete(status: status)
    }
    data.withUnsafeBytes { (pointer: UnsafePointer<UInt8>) in
        TF_SetAttrValueProto(operationDescription,
                             attrName,
                             UnsafeRawPointer(pointer),
                             data.count, status)
    }
    if let status = status, let error = StatusError(tfStatus: status) {
        throw error
    }
}

/// If this function succeeds:
///   * *status is set to an OK value,
///   * a TF_Operation is added to the graph,
///   * a non-null value pointing to the added operation is returned --
///     this value is valid until the underlying graph is deleted.
/// Otherwise:
///   * *status is set to a non-OK value,
///   * the graph is not modified,
///   * a null value is returned.
/// In either case, it deletes `desc`.
public func finish(operationDescription: TF_OperationDescription!) throws -> TF_Operation {
	let status = newStatus()
    defer {
        delete(status: status)
    }
	let operation = TF_FinishOperation(operationDescription, status)
	if let status = status, let error = StatusError(tfStatus: status) {
		throw error
	}
	guard let op = operation else {
		throw StatusError.unknown(message: "TF_FinishOperation returned nil insted pointer.")
	}
	return op
}

/// TF_Operation functions.  Operations are immutable once created, so
/// these are all query functions.
public func name(of operation: TF_Operation!) -> String {
	let str: UnsafePointer<Int8> = TF_OperationName(operation)
	return  String(cString:str)
}

public func type(of operation: TF_Operation!) -> String {
	let str: UnsafePointer<Int8> = TF_OperationOpType(operation)
	return  String(cString:str)
	
}

public func device(of operation: TF_Operation!) -> String {
	let str: UnsafePointer<Int8> = TF_OperationDevice(operation)
	return  String(cString:str)
}

public func numberOfOutputs(at operation: TF_Operation!) -> Int32 {
	return  TF_OperationNumOutputs(operation)
}

public func type(of operationOutput: TF_Output) -> TF_DataType {
	return  TF_OperationOutputType(operationOutput)
}

public func outputListLength(at operation: TF_Operation!, argumentName: UnsafePointer<Int8>!, status: TF_Status!) -> Int32 {
	return  TF_OperationOutputListLength(operation, argumentName, status)
}

public func numberOfInputs(at operation: TF_Operation!) -> Int32 {
	return  TF_OperationNumInputs(operation)
}

public func type(of operationInput: TF_Input) -> TF_DataType {
	return  TF_OperationInputType(operationInput)
}

public func inputListLength(at operation: TF_Operation!, argumentName: UnsafePointer<Int8>!, status: TF_Status!) -> Int32 {
	return  TF_OperationInputListLength(operation, argumentName, status)
}

/// In this code:
///   TF_Output producer = TF_OperationInput(consumer);
/// There is an edge from producer.oper's output (given by
/// producer.index) to consumer.oper's input (given by consumer.index).
public func output(for operationInput: TF_Input) -> TF_Output {
	return TF_OperationInput(operationInput)
}

/// Get the number of current consumers of a specific output of an
/// operation.  Note that this number can change when new operations
/// are added to the graph.
public func numberOfOperationOutputConsumers(of operationOutput: TF_Output) -> Int32 {
	return TF_OperationOutputNumConsumers(operationOutput)
}

/// Get list of all current consumers of a specific output of an
/// operation.  `consumers` must point to an array of length at least
/// `max_consumers` (ideally set to
/// TF_OperationOutputNumConsumers(oper_out)).  Beware that a concurrent
/// modification of the graph can increase the number of consumers of
/// an operation.  Returns the number of output consumers (should match
/// TF_OperationOutputNumConsumers(oper_out)).
public func operationOutputConsumers(of operationOutput: TF_Output, consumers: UnsafeMutablePointer<TF_Input>!, maxConsumers: Int32) -> Int32 {
	return TF_OperationOutputConsumers(operationOutput, consumers, maxConsumers)
}

/// Get the number of control inputs to an operation.
public func numberOfOperationControlInputs(of operation: TF_Operation!) -> Int32 {
	return TF_OperationNumControlInputs(operation)
}

/// Get list of all control inputs to an operation.  `control_inputs` must
/// point to an array of length `max_control_inputs` (ideally set to
/// TF_OperationNumControlInputs(oper)).  Returns the number of control
/// inputs (should match TF_OperationNumControlInputs(oper)).
public func getControlInputs(for operation: TF_Operation!, controlInputs: UnsafeMutablePointer<TF_Operation?>!, maxControlInputs: Int32) -> Int32 {
	return TF_OperationGetControlInputs(operation, controlInputs, maxControlInputs)
}

/// Get the number of operations that have `*oper` as a control input.
/// Note that this number can change when new operations are added to
/// the graph.
public func numberOfControlOutputs(of operation: TF_Operation!) -> Int32 {
	return TF_OperationNumControlOutputs(operation)
}

/// Get the list of operations that have `*oper` as a control input.
/// `control_outputs` must point to an array of length at least
/// `max_control_outputs` (ideally set to
/// TF_OperationNumControlOutputs(oper)). Beware that a concurrent
/// modification of the graph can increase the number of control
/// outputs.  Returns the number of control outputs (should match
/// TF_OperationNumControlOutputs(oper)).
public func getControlOutputs(of operation: TF_Operation!, controlOutputs: UnsafeMutablePointer<TF_Operation?>!, maxControlOutputs: Int32) -> Int32 {
	return TF_OperationGetControlOutputs(operation, controlOutputs, maxControlOutputs)
}

/// TF_AttrType describes the type of the value of an attribute on an operation.
/*public struct TF_AttrType : RawRepresentable, Equatable {

public init(_ rawValue: UInt32){
self = TF_AttrType(rawValue)
}

public init(rawValue: UInt32){
self =  TF_AttrType(rawValue)
}

public var rawValue: UInt32
}*/

/// TF_AttrMetadata describes the value of an attribute on an operation.
/*public struct TF_AttrMetadata {

/// A boolean: 1 if the attribute value is a list, 0 otherwise.
public var is_list: UInt8


/// Length of the list if is_list is true. Undefined otherwise.
public var list_size: Int64


/// Type of elements of the list if is_list != 0.
/// Type of the single value stored in the attribute if is_list == 0.
public var type: TF_AttrType


/// Total size the attribute value.
/// The units of total_size depend on is_list and type.
/// (1) If type == TF_ATTR_STRING and is_list == 0
///     then total_size is the byte size of the string
///     valued attribute.
/// (2) If type == TF_ATTR_STRING and is_list == 1
///     then total_size is the cumulative byte size
///     of all the strings in the list.
/// (3) If type == TF_ATTR_SHAPE and is_list == 0
///     then total_size is the number of dimensions
///     of the shape valued attribute, or -1
///     if its rank is unknown.
/// (4) If type == TF_ATTR_SHAPE and is_list == 1
///     then total_size is the cumulative number
///     of dimensions of all shapes in the list.
/// (5) Otherwise, total_size is undefined.
public var total_size: Int64

public init(){
self = TF_AttrMetadata()
}

public init(is_list: UInt8, list_size: Int64, type: TF_AttrType, total_size: Int64){
self = TF_AttrMetadata.init(is_list,list_size,type,total_size)
}
}*/

/// Returns metadata about the value of the attribute `attr_name` of `oper`.
public func getAttributeMetadata(of operation : TF_Operation!, by name: String, status: TF_Status!) -> TF_AttrMetadata {
	return TF_OperationGetAttrMetadata(operation, name.cString(using: .utf8), status)
}

/// Fills in `value` with the value of the attribute `attr_name`.  `value` must
/// point to an array of length at least `max_length` (ideally set to
/// TF_AttrMetadata.total_size from TF_OperationGetAttrMetadata(oper,
/// attr_name)).
public func getAttributeString(of operation: TF_Operation!, by name: String, value: UnsafeMutableRawPointer!, maxLength: Int, status: TF_Status!) {
	TF_OperationGetAttrString(operation, name.cString(using: .utf8), value, maxLength, status)
}

/// Get the list of strings in the value of the attribute `attr_name`.  Fills in
/// `values` and `lengths`, each of which must point to an array of length at
/// least `max_values`.
///
/// The elements of values will point to addresses in `storage` which must be at
/// least `storage_size` bytes in length.  Ideally, max_values would be set to
/// TF_AttrMetadata.list_size and `storage` would be at least
/// TF_AttrMetadata.total_size, obtained from TF_OperationGetAttrMetadata(oper,
/// attr_name).
///
/// Fails if storage_size is too small to hold the requested number of strings.
public func getAttributeString(of operation: TF_Operation!,
                                by name: String,
                                values: UnsafeMutablePointer<UnsafeMutableRawPointer?>!,
                                lengths: UnsafeMutablePointer<Int>!,
                                maxValues: Int32,
                                storage: UnsafeMutableRawPointer!,
                                storageSize: Int,
                                status: TF_Status!) {
	
	TF_OperationGetAttrStringList(operation, name.cString(using: .utf8), values, lengths, maxValues, storage, storageSize, status)
}

public func getAttributeInt(of operation: TF_Operation!,
                            by name: String,
                            value: UnsafeMutablePointer<Int64>!,
                            status: TF_Status!) {
	TF_OperationGetAttrInt(operation, name.cString(using: .utf8), value, status)
}

/// Fills in `values` with the value of the attribute `attr_name` of `oper`.
/// `values` must point to an array of length at least `max_values` (ideally set
/// TF_AttrMetadata.list_size from TF_OperationGetAttrMetadata(oper,
/// attr_name)).
public func getAttributeInt(of operation: TF_Operation!, by name: String, values: UnsafeMutablePointer<Int64>!, maxValues: Int32, status: TF_Status!) {
	fatalError("\(#function): Not implemented.")
}

public func getAttributeFloat(of operation: TF_Operation!, by name: String, value: UnsafeMutablePointer<Float>!, status: TF_Status!) {
	fatalError("\(#function): Not implemented.")
}

/// Fills in `values` with the value of the attribute `attr_name` of `oper`.
/// `values` must point to an array of length at least `max_values` (ideally set
/// to TF_AttrMetadata.list_size from TF_OperationGetAttrMetadata(oper,
/// attr_name)).
public func getAttributeFloat(of operation: TF_Operation!, by name: String, values: UnsafeMutablePointer<Float>!, maxValues: Int32, status: TF_Status!) {
	fatalError("\(#function): Not implemented.")
}

public func getAttributeBool(of operation: TF_Operation!, by name: String, value: UnsafeMutablePointer<UInt8>!, status: TF_Status!) {
	fatalError("\(#function): Not implemented.")
}

/// Fills in `values` with the value of the attribute `attr_name` of `oper`.
/// `values` must point to an array of length at least `max_values` (ideally set
/// to TF_AttrMetadata.list_size from TF_OperationGetAttrMetadata(oper,
/// attr_name)).
public func getAttributeBool(of operation: TF_Operation!, by name: String, values: UnsafeMutablePointer<UInt8>!, maxValues: Int32, status: TF_Status!) {
	fatalError("\(#function): Not implemented.")
}

public func getAttributeType(of operation: TF_Operation!, by name: String) throws -> TF_DataType {
	var tfType = TF_DataType(0)
    
    let status = newStatus()
    defer {
        delete(status: status)
    }
    withUnsafeMutablePointer(to: &tfType) { (pointer: UnsafeMutablePointer<TF_DataType>) in
        TF_OperationGetAttrType(operation, name.cString(using: .utf8), pointer, status)
    }
    if let status = status, let error = StatusError(tfStatus: status) {
        throw error
    }
    return tfType
}

/// Fills in `values` with the value of the attribute `attr_name` of `oper`.
/// `values` must point to an array of length at least `max_values` (ideally set
/// to TF_AttrMetadata.list_size from TF_OperationGetAttrMetadata(oper,
/// attr_name)).
public func getAttributeType(of operation: TF_Operation!, by name: String, values: UnsafeMutablePointer<TF_DataType>!, maxValues: Int32, status: TF_Status!) {
	fatalError("\(#function): Not implemented.")
}

/// Fills in `value` with the value of the attribute `attr_name` of `oper`.
/// `values` must point to an array of length at least `num_dims` (ideally set to
/// TF_Attr_Meta.size from TF_OperationGetAttrMetadata(oper, attr_name)).
public func getAttributeShape(of operation: TF_Operation!, by name: String, value: UnsafeMutablePointer<Int64>!, dimensionsNumber: Int32, status: TF_Status!) {
	fatalError("\(#function): Not implemented.")
}

/// Fills in `dims` with the list of shapes in the attribute `attr_name` of
/// `oper` and `num_dims` with the corresponding number of dimensions. On return,
/// for every i where `num_dims[i]` > 0, `dims[i]` will be an array of
/// `num_dims[i]` elements. A value of -1 for `num_dims[i]` indicates that the
/// i-th shape in the list is unknown.
///
/// The elements of `dims` will point to addresses in `storage` which must be
/// large enough to hold at least `storage_size` int64_ts.  Ideally, `num_shapes`
/// would be set to TF_AttrMetadata.list_size and `storage_size` would be set to
/// TF_AttrMetadata.total_size from TF_OperationGetAttrMetadata(oper,
/// attr_name).
///
/// Fails if storage_size is insufficient to hold the requested shapes.
public func getAttributeShape(of operation: TF_Operation!,
                              by name: String,
                              dimensions: UnsafeMutablePointer<UnsafeMutablePointer<Int64>?>!,
                              dimensionsNumber: UnsafeMutablePointer<Int32>!,
                              shapesNumber: Int32,
                              storage: UnsafeMutablePointer<Int64>!,
                              storageSize: Int32,
                              status: TF_Status!) {
	fatalError("\(#function): Not implemented.")
}

/// Sets `value` to the binary-serialized TensorShapeProto of the value of
/// `attr_name` attribute of `oper`'.
public func getAttributeTensorShapeProto(of operation: TF_Operation!, by name: String, value: UnsafeMutablePointer<TF_Buffer>!, status: TF_Status!) {
	fatalError("\(#function): Not implemented.")
}

/// Fills in `values` with binary-serialized TensorShapeProto values of the
/// attribute `attr_name` of `oper`. `values` must point to an array of length at
/// least `num_values` (ideally set to TF_AttrMetadata.list_size from
/// TF_OperationGetAttrMetadata(oper, attr_name)).
public func getAttributeTensorShapeProto(of operation: TF_Operation!,
                                         by name: String,
                                         values: UnsafeMutablePointer<UnsafeMutablePointer<TF_Buffer>?>!,
                                         maxValues: Int32,
                                         status: TF_Status!) {
	fatalError("\(#function): Not implemented.")
}

/// Gets the TF_Tensor valued attribute of `attr_name` of `oper`.
///
/// Allocates a new TF_Tensor which the caller is expected to take
/// ownership of (and can deallocate using TF_DeleteTensor).
public func getAttributeTensor(of operation: TF_Operation!, by name: String) throws -> TF_Tensor? {
	
    let pointer = UnsafeMutablePointer<TF_Tensor?>.allocate(capacity: 1)
    
    let status = newStatus()
    defer {
        delete(status: status)
    }
    TF_OperationGetAttrTensor(operation, name.cString(using: .utf8), pointer, status)

    if let status = status, let error = StatusError(tfStatus: status) {
        throw error
    }
    return pointer.pointee
}

/// Fills in `values` with the TF_Tensor values of the attribute `attr_name` of
/// `oper`. `values` must point to an array of TF_Tensor* of length at least
/// `max_values` (ideally set to TF_AttrMetadata.list_size from
/// TF_OperationGetAttrMetadata(oper, attr_name)).
///
/// The caller takes ownership of all the non-null TF_Tensor* entries in `values`
/// (which can be deleted using TF_DeleteTensor(values[i])).
public func getAttributeTensor(of operation: TF_Operation!, by name: String, values: UnsafeMutablePointer<OpaquePointer?>!, maxValues: Int32, status: TF_Status!) {
    fatalError("\(#function): Not implemented.")
}

/// Sets `output_attr_value` to the binary-serialized AttrValue proto
/// representation of the value of the `attr_name` attr of `oper`.
public func getAttributeValueProto(of operation: TF_Operation!, by name: String, value: UnsafeMutablePointer<TF_Buffer>!, status: TF_Status!) {
	fatalError("\(#function): Not implemented.")
}

/// Returns the operation in the graph with `oper_name`. Returns nullptr if
/// no operation found.
public func operation(in graph: TF_Graph!, by name: String) -> OpaquePointer! {
    return name.withCString { pointer -> OpaquePointer? in
        return TF_GraphOperationByName(graph, pointer)
    }
}

/// Returns list of all operations in `TF_Graph`, using TF_GraphNextOperation api inside.
/// Iterate through the operations of a graph.  To use:
/// size_t pos = 0;
/// TF_Operation* oper;
/// while ((oper = TF_GraphNextOperation(graph, &pos)) != nullptr) {
///   DoSomethingWithOperation(oper);
/// }
public func operations(of graph: TF_Graph) -> [TF_Operation] {
    var operations = [TF_Operation]()
    let position = UnsafeMutablePointer<Int>.allocate(capacity: 1)
    position.initialize(to: 0)
    while let pointer = TF_GraphNextOperation(graph, position) {
        operations.append(pointer)
    }
    position.deallocate()
	return operations
}

/// Write out a serialized representation of `graph` (as a GraphDef protocol
/// message) to `output_graph_def` (allocated by TF_NewBuffer()).
/// `output_graph_def`'s underlying buffer will be freed when TF_DeleteBuffer()
/// is called.
///
/// May fail on very large graphs in the future.
public func graphDef(of graph: TF_Graph!, graphDef: UnsafeMutablePointer<TF_Buffer>!) throws {
	let status = newStatus()
    defer {
        delete(status: status)
    }
	TF_GraphToGraphDef(graph, graphDef, status)
	if let status = status, let error = StatusError(tfStatus: status) {
		throw error
	}
}

/// TF_ImportGraphDefOptions holds options that can be passed to
/// TF_GraphImportGraphDef.
public func newImportGraphDefOptions() -> OpaquePointer! {
	return TF_NewImportGraphDefOptions()
}
public func delete(importGraphDefOptions: TF_ImportGraphDefOptions!) {
	TF_DeleteImportGraphDefOptions(importGraphDefOptions)
}

/// Set the prefix to be prepended to the names of nodes in `graph_def` that will
/// be imported into `graph`.
public func setPrefix(into importGraphDefOptions: TF_ImportGraphDefOptions!, prefix: String) {
	let cPrefix = UnsafePointer<Int8>(prefix)
	TF_ImportGraphDefOptionsSetPrefix(importGraphDefOptions, cPrefix)
}

/// Set any imported nodes with input `src_name:src_index` to have that input
/// replaced with `dst`. `src_name` refers to a node in the graph to be imported,
/// `dst` references a node already existing in the graph being imported into.
public func addInputMapping(into importGraphDefOptions: TF_ImportGraphDefOptions!, sourceName: String, sourceIndex: Int32, output: TF_Output) {
	let cSrcName = UnsafePointer<Int8>(sourceName)
	TF_ImportGraphDefOptionsAddInputMapping(importGraphDefOptions, cSrcName, sourceIndex, output)
}

/// Set any imported nodes with control input `src_name` to have that input
/// replaced with `dst`. `src_name` refers to a node in the graph to be imported,
/// `dst` references an operation already existing in the graph being imported
/// into.
public func remapControlDependency(in importGraphDefOptions: TF_ImportGraphDefOptions!, sourceName: String, destination: TF_Operation!) {
	let cSrcName = UnsafePointer<Int8>(sourceName)
	TF_ImportGraphDefOptionsRemapControlDependency(importGraphDefOptions, cSrcName, destination)
}

/// Cause the imported graph to have a control dependency on `oper`. `oper`
/// should exist in the graph being imported into.
public func addControlDependency(in importGraphDefOptions: TF_ImportGraphDefOptions!, operation: TF_Operation!) {
	TF_ImportGraphDefOptionsAddControlDependency(importGraphDefOptions, operation)
}

/// Add an output in `graph_def` to be returned via the `return_outputs` output
/// parameter of TF_GraphImportGraphDef(). If the output is remapped via an input
/// mapping, the corresponding existing tensor in `graph` will be returned.
public func addReturnOutput(in importGraphDefOptions: TF_ImportGraphDefOptions!, operationName: UnsafePointer<Int8>!, index: Int32) {
	TF_ImportGraphDefOptionsAddReturnOutput(importGraphDefOptions, operationName, index)
}

/// Returns the number of return outputs added via
/// TF_ImportGraphDefOptionsAddReturnOutput().
public func numberOfReturnOutputs(importGraphDefOptions: TF_ImportGraphDefOptions!) -> Int32 {
	return TF_ImportGraphDefOptionsNumReturnOutputs(importGraphDefOptions)
}

/// Import the graph serialized in `graph_def` into `graph`.
///
/// `num_return_outputs` must be the number of return outputs added (i.e. the
/// result of TF_ImportGraphDefOptionsNumReturnOutputs()).  If
/// `num_return_outputs` is non-zero, `return_outputs` must be of length
/// `num_return_outputs`. Otherwise it can be null.
public func `import`(graph: TF_Graph!,
                     in graphDef: UnsafePointer<TF_Buffer>!,
                     sessionOptions: TF_SessionOptions!,
                     returnOutputs: UnsafeMutablePointer<TF_Output>!,
                     numberOfReturnOutputs: Int32,
                     status: TF_Status!) {
	fatalError("\(#function): Not implemented.")
	/*TF_GraphImportGraphDefWithReturnOutputs(<#T##graph: OpaquePointer!##OpaquePointer!#>, <#T##graph_def: UnsafePointer<TF_Buffer>!##UnsafePointer<TF_Buffer>!#>, <#T##options: OpaquePointer!##OpaquePointer!#>, <#T##return_outputs: UnsafeMutablePointer<TF_Output>!##UnsafeMutablePointer<TF_Output>!#>, <#T##num_return_outputs: Int32##Int32#>, <#T##status: OpaquePointer!##OpaquePointer!#>)*/
}

/// Import the graph serialized in `graph_def` into `graph`.
/// Convenience function for when no return outputs have been added.
public func `import`(graph: TF_Graph!, in graphDef: UnsafePointer<TF_Buffer>!, sessionOptions: TF_SessionOptions!) throws {
    let status = newStatus()
    defer {
        delete(status: status)
    }
    TF_GraphImportGraphDef(graph, graphDef, sessionOptions, status)
    if let status = status, let error = StatusError(tfStatus: status) {
        throw error
    }
}

/// Note: The following function may fail on very large protos in the future.
public func operationToNodeDef(operation: TF_Operation!, outputNodeDef: UnsafeMutablePointer<TF_Buffer>!, status: TF_Status!) {
	TF_OperationToNodeDef(operation, outputNodeDef, status)
}
