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
import Foundation
import CAPI

typealias Byte = UInt8

/// Tensor holds a multi-dimensional array of elements of a single data type.
public enum TensorError: Error {
    case canNotAllocateTensor
	case canNotComputeDataPointer
	case incorrectDataSize
}

public protocol Value: Comparable, CustomStringConvertible, Hashable {}

extension Double: Value {}
extension Float: Value {}
extension Int: Value {}

public class Tensor: CustomStringConvertible {
    var tfTensor: TF_Tensor
    let dimensions: [Int64]
    public let dtType: TF_DataType
	let size: Int
	init(tfTensor: TF_Tensor) throws {
		self.tfTensor = tfTensor
		
		let numberOfDimensions = CAPI.numberOfDimensions(in: tfTensor)
		var dims = Array<Int64>(repeating: 0, count: Int(numberOfDimensions))
		
		for index in 0..<numberOfDimensions {
			let dimension = CAPI.dimension(of: tfTensor, at: index)
			dims[Int(index)] = dimension
		}
		self.dimensions = dims
		self.dtType = CAPI.dataType(of: tfTensor)
		size = CAPI.byteSize(of: tfTensor)
	}

    public convenience init<T: Value>(scalar: T) throws {
		try self.init(dimensions: Array<Int64>(), values: [scalar])
	}
	
    public convenience init<T: Value>(dimensions: [Int], values: [T]) throws {
        try self.init(dimensions: dimensions.map {Int64($0)}, values: values)
    }
    
	public init<T>(dimensions: [Int64], values: [T]) throws {
        self.dimensions = dimensions
		
        dtType = try TF_DataType(for: T.self)
        
        size = MemoryLayout<T>.size * values.count
        let tensorPointer = allocateTensor(dataType: dtType, dimensions: dimensions, length: size)
        
        guard let tfTensor = tensorPointer else {
            throw TensorError.canNotAllocateTensor
        }
        self.tfTensor = tfTensor
        memcpy(TF_TensorData(tfTensor), values, size)
    }
	
	public var description: String {
        return "Tensor \(dimensions) type: \(self.dtType)"
	}
    
    //MARK: - Working with data
	public func pullData() throws -> Data {
		let size = CAPI.byteSize(of: tfTensor)
		let count = size / MemoryLayout<Byte>.size
		guard let pointer = CAPI.data(in: tfTensor) else {
			throw TensorError.canNotComputeDataPointer
		}
		
		let bytePointer =  pointer.bindMemory(to: Byte.self, capacity: count)
		let array = Array(UnsafeBufferPointer(start: bytePointer, count: count))
		return Data(array)
	}

	public func push(data: Data) throws {
		guard data.count == size else {
			throw TensorError.incorrectDataSize
		}
		_ = data.withUnsafeBytes { pointer in
			memcpy(TF_TensorData(tfTensor), UnsafeRawPointer(pointer), size)
		}
	}
    
    public func pullCollection<T: Value> () throws -> [T] {
        let data = try pullData()
        let collection = data.withUnsafeBytes({ (pointer: UnsafePointer<T>) -> Array<T> in
            return Array(UnsafeBufferPointer<T>(start: pointer, count: data.count / MemoryLayout<T>.size))
        })
        return collection
    }

    deinit {
        delete(tensor: tfTensor)
    }
}
