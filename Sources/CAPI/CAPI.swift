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
import Proto

// C API for TensorFlow.
//
// The API leans towards simplicity and uniformity instead of convenience
// since most usage will be by language specific wrappers.
//
// Conventions:
// * We use the prefix TF_ for everything in the API.
// * Objects are always passed around as pointers to opaque structs
//   and these structs are allocated/deallocated via the API.
// * TF_Status holds error information.  It is an object type
//   and therefore is passed around as a pointer to an opaque
//   struct as mentioned above.
// * Every call that has a TF_Status* argument clears it on success
//   and fills it with error info on failure.
// * unsigned char is used for booleans (instead of the 'bool' type).
//   In C++ bool is a keyword while in C99 bool is a macro defined
//   in stdbool.h. It is possible for the two to be inconsistent.
//   For example, neither the C99 nor the C++11 standard force a byte
//   size on the bool type, so the macro defined in stdbool.h could
//   be inconsistent with the bool keyword in C++. Thus, the use
//   of stdbool.h is avoided and unsigned char is used instead.
// * size_t is used to represent byte sizes of objects that are
//   materialized in the address space of the calling process.
// * int is used as an index into arrays.
//
// Questions left to address:
// * Might at some point need a way for callers to provide their own Env.
// * Maybe add TF_TensorShape that encapsulates dimension info.
//
// Design decisions made:
// * Backing store for tensor memory has an associated deallocation
//   function.  This deallocation function will point to client code
//   for tensors populated by the client.  So the client can do things
//   like shadowing a numpy array.
// * We do not provide TF_OK since it is not strictly necessary and we
//   are not optimizing for convenience.
// * We make assumption that one session has one graph.  This should be
//   fine since we have the ability to run sub-graphs.
// * We could allow NULL for some arguments (e.g., NULL options arg).
//   However since convenience is not a primary goal, we don't do this.
// * Devices are not in this API.  Instead, they are created/used internally
//   and the API just provides high level controls over the number of
//   devices of each type.


/// <#Title#>
///
/// <#Description#>
/// <#Description new line#>
///
///		<#Code Exable line#>
///		<#Code Exable line#>
///		<#Code Exable line#>
///		<#Code Exable line#>
///		<#Code Exable line#>
///
/// - Parameter <#Argument#>: <#Argument description#>.
///
/// - Parameter <#Argument#>: <#Argument description#>.
///
/// - Returns: <#Return description#>.
///
/// - Complexity: <#Difficulty#>

// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/c_api.h

/// C pointer for session representation
public typealias TF_Session = OpaquePointer
/// C pointer for tensor representation
public typealias TF_Tensor = OpaquePointer
/// C pointer for status representation
public typealias TF_Status = OpaquePointer
/// C pointer for session options representation
public typealias TF_SessionOptions = OpaquePointer
/// C pointer for grapht representation
public typealias TF_Graph = OpaquePointer
/// C pointer for library representation
public typealias TF_Library = OpaquePointer
/// C pointer for deprecated session representation
public typealias TF_DeprecatedSession = OpaquePointer
/// C pointer for roperation description epresentation
public typealias TF_OperationDescription = OpaquePointer
/// C pointer for operation representation
public typealias TF_Operation = OpaquePointer
/// C pointer for import operation representation
public typealias TF_ImportGraphDefOptions = OpaquePointer

public typealias TensorflowNameAttrList = Tensorflow_NameAttrList
public typealias Byte = UInt8

/// Swift Error type for CAPI needs.
public enum CAPIError: Error {
	case cancelled(message: String?)
    case canNotComputPointer(functionName: String)
    case canNotComputDataFromString
    
	public var localizedDescription: String {
		switch self {
		case .cancelled(let message):
			return "Error: \(message ?? "cancelled")"
        case .canNotComputPointer(let functionName):
            return "Error: Can not comput pointer at \(functionName) function"
        case .canNotComputDataFromString:
            return "Error: Can not comput Data from string"

        }
	}
}


/// TF_Version returns a string describing version information of the
/// TensorFlow library. TensorFlow using semantic versioning.
public func version() -> String {
	let str: UnsafePointer<Int8> = TF_Version()
	return  String(cString:str)
}

/// TF_DataTypeSize returns the sizeof() for the underlying type corresponding
/// to the given TF_DataType enum value. Returns 0 for variable length types
/// (eg. TF_STRING) or on failure.
func dataTypeSize(_ dt: TF_DataType) -> Int {
	let typeSize =  TF_DataTypeSize(dt) as Int
	return typeSize
}
