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

/// Makes a copy of the input and sets an appropriate deallocator.  Useful for
/// passing in read-only, input protobufs.
public func newBuffer(from cString: UnsafeRawPointer!, length: Int) -> UnsafeMutablePointer<TF_Buffer>! {
	return TF_NewBufferFromString(cString, length)
}

/// Useful for passing *out* a protobuf.
public func newBuffer() -> UnsafeMutablePointer<TF_Buffer>! {
	return TF_NewBuffer()
}

/// Deleting TF_Buffer from memory by pointer.
public func deleteBuffer(_ unsafePointer: UnsafeMutablePointer<TF_Buffer>!) {
	TF_DeleteBuffer(unsafePointer)
}

/// Extract TF_Buffer from memory by pointer.
public func buffer(_ buffer: UnsafeMutablePointer<TF_Buffer>!) -> TF_Buffer {
	return TF_GetBuffer(buffer)
}

/// Hellper. Provides Swift `Data` from allocated TF_Buffer.
public func allocAndProcessBuffer(process:(_ bufferPointer: UnsafeMutablePointer<TF_Buffer>?) throws -> Void) rethrows -> Data {
	let bufferPointer = TF_NewBuffer()
	try process(bufferPointer)
	let tfBuffer = TF_GetBuffer(bufferPointer)
	let data = Data(bytes: tfBuffer.data, count: tfBuffer.length)
	TF_DeleteBuffer(bufferPointer)
	return data
}
