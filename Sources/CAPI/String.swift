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
public func encode(source: UnsafePointer<Int8>!,
                   sourceLength: Int,
                   destination: UnsafeMutablePointer<Int8>!,
                   destinationLength: Int,
                   status: TF_Status!) -> Int {
	return TF_StringEncode(source, sourceLength, destination, destinationLength, status)
}

/// Decode a string encoded using TF_StringEncode.
///
/// On success, sets `*dst` to the start of the decoded string and `*dst_len` to
/// its length. Returns the number of bytes starting at `src` consumed while
/// decoding. `*dst` points to memory within the encoded buffer.  On failure,
/// `*dst` and `*dst_len` are undefined and an error is set in `status`.
///
/// Does not read memory more than `src_len` bytes beyond `src`.
public func decode(source: UnsafePointer<Int8>!,
                   sourceLength: Int,
                   destination: UnsafeMutablePointer<UnsafePointer<Int8>?>!,
                   destinationLength: UnsafeMutablePointer<Int>!,
                   status: TF_Status!) -> Int {
	return TF_StringDecode(source, sourceLength, destination, destinationLength, status)
}

/// Return the size in bytes required to encode a string `len` bytes long into a
/// TF_STRING tensor.
public func stringEncodedSize(length: Int) -> Int {
	return TF_StringEncodedSize(length)
}

