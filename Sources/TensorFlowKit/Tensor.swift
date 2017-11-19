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
    case canNotComputeTensorPointer
	case incorrectDataSize
    case incorrectShape
}

public protocol Value: Comparable, CustomStringConvertible, Hashable {}

extension Double: Value {}
extension Float: Value {}
extension Int: Value {}
extension Int32: Value {}
extension Int64: Value {}
extension String: Value {}
extension UInt8: Value {}
extension Int8: Value {}


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

    public convenience init<T: Value>(shape: Shape, values: [T]) throws {
        switch shape {
        case .dimensions(let dimensions):
            try self.init(dimensions: dimensions, values: values)
            break
        case .unknown:
            // TODO: In that case dimensions coud be [-1]
            // need to check.
            try self.init(dimensions: [Int64](), values: values)
            break
        }
    }
    
    /// Calculate number of elements in `Tensor`
    public func numElements() -> Int64 {
       return dimensions.reduce(1, *)
    }
    
    public convenience init<T: Value>(dimensions: [Int], values: [T]) throws {
        try self.init(dimensions: dimensions.map {Int64($0)}, values: values)
    }
    
    public init<T: Value>(dimensions: [Int64], values: [T]) throws {
        self.dimensions = dimensions
		
        guard dimensions.reduce(1, *) == values.count else {
            throw TensorError.incorrectShape
        }
        
        dtType = try TF_DataType(for: T.self)
        if T.self == String.self {
            /// Calculate size of encoded strings.
            var encodedSize = MemoryLayout<UInt64>.size * values.count
            try values.forEach({ value in
                guard  let string = value as? String else { return }
                encodedSize += try CAPI.encodedSize(of: string)
            })
            size = encodedSize
        } else {
            size = MemoryLayout<T>.size * values.count
        }
        
        let tensorPointer = allocateTensor(dataType: dtType, dimensions: dimensions, length: size)
        
        guard let tfTensor = tensorPointer else {
            throw TensorError.canNotAllocateTensor
        }
        self.tfTensor = tfTensor
        guard let tensorStoragePointer = CAPI.data(in: tfTensor) else { throw TensorError.canNotComputeDataPointer }
        
        if T.self == String.self {
            /// offset for writing data
            var offset = MemoryLayout<UInt64>.size * values.count
            /// size for header table
            var encodedSize = 0
            /// header table
            var headers = Array<UInt64>()
            
            try values.forEach({ value in
                guard  let string = value as? String else { return }
                /// Add element in header storage table
                headers.append(UInt64(encodedSize))
                let lenhth = try CAPI.encode(string: string, writeAt: tensorStoragePointer + offset)
                offset += lenhth
                encodedSize += lenhth
            })
            
            /// Write header table for storage at index 0
            memcpy(tensorStoragePointer, headers, MemoryLayout<UInt64>.size * values.count)

        } else {
            memcpy(tensorStoragePointer, values, size)
        }
    }
	
	public var description: String {
        return "Tensor \(dimensions) type: \(self.dtType)"
	}
    
    //MARK: - Working with data
	public func pullData() throws -> Data {
		let size = CAPI.byteSize(of: tfTensor)
		let count = size / MemoryLayout<Byte>.size
        guard let pointer: UnsafeMutableRawPointer = CAPI.data(in: tfTensor) else {
			throw TensorError.canNotComputeDataPointer
		}

        let result = Data.init(bytes: pointer, count: count)
        if self.dtType == TF_STRING {
            let offset = MemoryLayout<UInt64>.size * Int(self.numElements())
            let decoded = CAPI.decode(data: result[offset..<count])
            return decoded
        }
        
		return result
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
