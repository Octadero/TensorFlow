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

/// Encode the string `src` (`src_len` bytes long) into `dst` in the format
/// required by TF_STRING tensors. Does not write to memory more than `dst_len`
/// bytes beyond `*dst`. `dst_len` should be at least
/// TF_StringEncodedSize(src_len).
///
/// On success returns the size in bytes of the encoded string.
/// Returns an error into `status` otherwise.
public func encode(string: String, encoding: String.Encoding = .ascii, writeAt destinationRawPointer: UnsafeMutableRawPointer) throws -> Int {
    guard let stringData = string.data(using: encoding) else {
        throw CAPIError.canNotComputDataFromString
    }
    
    return try CAPI.encode(stringData: stringData, writeAt: destinationRawPointer)
}

public func encode(stringData: Data, writeAt destinationRawPointer: UnsafeMutableRawPointer) throws -> Int {
    let status = newStatus()
    defer {
        delete(status: status)
    }
    let sourceLength = stringData.count
    let destinationLength = try encodedSize(of: stringData)
    let destinationPointer = destinationRawPointer.assumingMemoryBound(to: Int8.self)
    
    let result = stringData.withUnsafeBytes { (sourcePointer: UnsafePointer<Int8>) -> Int in
        TF_StringEncode(sourcePointer, sourceLength, destinationPointer, destinationLength, status)
    }
    
    if let status = status, let error = StatusError(tfStatus: status) {
        throw error
    }
    return result
}

/// Decode a string encoded using TF_StringEncode.
///
/// On success, sets `*dst` to the start of the decoded string and `*dst_len` to
/// its length. Returns the number of bytes starting at `src` consumed while
/// decoding. `*dst` points to memory within the encoded buffer.  On failure,
/// `*dst` and `*dst_len` are undefined and an error is set in `status`.
///
/// Does not read memory more than `src_len` bytes beyond `src`.
public func decode(data: Data) throws -> Data {
    var data = data
    let status = newStatus()
    defer {
        delete(status: status)
    }
    let destinationPointer = UnsafeMutablePointer<UnsafePointer<Int8>?>.allocate(capacity: data.count)
    let destinationLength = UnsafeMutablePointer<Int>.allocate(capacity: 1)
    
    let result = data.withUnsafeBytes { (pointer: UnsafePointer<Int8>) -> Data in
        let _ = TF_StringDecode(pointer, data.count, destinationPointer, destinationLength, status)
        let count: Int = destinationLength.pointee
        let buffer = UnsafeBufferPointer(start: destinationPointer.pointee, count: count)
        return Data(buffer: buffer)
    }
    
    
    destinationLength.deallocate(capacity: 1)
    destinationPointer.deallocate(capacity: data.count)
    
    if let status = status, let error = StatusError(tfStatus: status) {
        throw error
    }
    return result
}

/// Return the size in bytes required to encode a string `len` bytes long into a
/// TF_STRING tensor.
public func stringEncodedSize(length: Int) -> Int {
	return TF_StringEncodedSize(length)
}


///Return the size in bytes required to encode a string `string` bytes long into a
/// TF_STRING tensor.
public func encodedSize(`of` string: String, encoding: String.Encoding = .ascii) throws -> Int {
    guard let stringData = string.data(using: encoding) else {
        throw CAPIError.canNotComputDataFromString
    }
    let length = stringData.count
    return TF_StringEncodedSize(length)
}

public func encodedSize(`of` stringData: Data) throws -> Int {
    let length = stringData.count
    return TF_StringEncodedSize(length)
}
