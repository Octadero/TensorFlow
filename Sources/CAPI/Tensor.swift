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

public typealias DeallocatorCallback = (@convention(c) (UnsafeMutableRawPointer?, Int, UnsafeMutableRawPointer?) -> Swift.Void)!

/// TF_Tensor holds a multi-dimensional array of elements of a single data type.
/// For all types other than TF_STRING, the data buffer stores elements
/// in row major order.  E.g. if data is treated as a vector of TF_DataType:
//
///   element 0:   index (0, ..., 0)
///   element 1:   index (0, ..., 1)
///   ...
//
/// The format for TF_STRING tensors is:
///   start_offset: array[uint64]
///   data:         byte[...]
//
///   The string length (as a varint), followed by the contents of the string
///   is encoded at data[start_offset[i]]]. TF_StringEncode and TF_StringDecode
///   facilitate this encoding.

/// Return a new tensor that holds the bytes data[0,len-1].
//
/// The data will be deallocated by a subsequent call to TF_DeleteTensor via:
///      (*deallocator)(data, len, deallocator_arg)
/// Clients must provide a custom deallocator function so they can pass in
/// memory managed by something like numpy.
public func newTensor(dataType: TF_DataType,
                      dimensions: UnsafePointer<Int64>!,
                      dimensionsNumber: Int32,
                      data: UnsafeMutableRawPointer!,
                      length: Int,
                      deallocator: DeallocatorCallback,
                      deallocatorArg: UnsafeMutableRawPointer!) -> TF_Tensor! {
	return TF_NewTensor(dataType, dimensions, dimensionsNumber, data, length, deallocator, deallocatorArg)
}

public func newTensor(dataType: Tensorflow_DataType,
                      dimensions: UnsafePointer<Int64>!,
                      dimensionsNumber: Int32,
                      data: UnsafeMutableRawPointer!,
                      length: Int,
                      deallocator: DeallocatorCallback,
                      deallocatorArg: UnsafeMutableRawPointer!) -> TF_Tensor! {
	let tfDataType = unsafeBitCast(dataType.rawValue, to: TF_DataType.self)
	
	return TF_NewTensor(tfDataType, dimensions, dimensionsNumber, data, length, deallocator, deallocatorArg)
}

/// Allocate and return a new Tensor.
//
/// This function is an alternative to TF_NewTensor and should be used when
/// memory is allocated to pass the Tensor to the C API. The allocated memory
/// satisfies TensorFlow's memory alignment preferences and should be preferred
/// over calling malloc and free.
//
/// The caller must set the Tensor values by writing them to the pointer returned
/// by TF_TensorData with length TF_TensorByteSize.
public func allocateTensor(dataType: TF_DataType, dimensions: [Int64], length: Int) -> TF_Tensor? {
    let dimensionsPointer = UnsafeRawPointer(dimensions).assumingMemoryBound(to: Int64.self)
    return TF_AllocateTensor(dataType, dimensionsPointer, Int32(dimensions.count), length)
}

/// Destroy a tensor.
public func delete(tensor :TF_Tensor!){
	return TF_DeleteTensor(tensor)
}

/// Return the type of a tensor element.
public func dataType(of tensor:TF_Tensor!) -> TF_DataType {
	return TF_TensorType(tensor)
}

/// Return the number of dimensions that the tensor has.
public func numberOfDimensions(in tensor: TF_Tensor!) -> Int32 {
	return TF_NumDims(tensor)
}

/// Return the length of the tensor in the "dim_index" dimension.
/// REQUIRES: 0 <= dim_index < TF_NumDims(tensor)
public func dimension(of tensor: TF_Tensor!, at dimensionIndex: Int32) -> Int64 {
	return TF_Dim(tensor, dimensionIndex)
}

/// Return the size of the underlying data in bytes.
public func byteSize(of tensor: TF_Tensor!) -> Int {
	return  TF_TensorByteSize(tensor)
}

/// Return a pointer to the underlying data buffer.
public func data(in tensor: TF_Tensor!) -> UnsafeMutableRawPointer! {
	return TF_TensorData(tensor)
}

/// Helper to encourage use of the proto Tensorflow_DataType type instead of c primitive
public func dataType(of tensor: TF_Tensor!) -> Tensorflow_DataType {
	let primitive = TF_TensorType(tensor)
    
    if let dataType = Tensorflow_DataType(rawValue: Int(primitive.rawValue)) {
		return dataType
	}
	return .dtInvalid
}

