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

import Foundation

/// Shape represents the (possibly partially known) shape of a tensor that will
/// be produced by an operation.
///
/// The zero-value of a Shape represents a shape with an unknown number of
/// dimensions.
public enum Shape: CustomDebugStringConvertible {
	case unknown
	case dimensions(value: [Int64])
	
    public var elements: Int64?     {
        switch self {
        case .dimensions(let value):
            return value.reduce(1, *)
        default:
            return nil
        }
    }
    
	public var debugDescription: String {
		switch self {
		case .unknown:
			return "Shape with unknown size."
		case .dimensions(let value):
			return "Shape: \(value)"
		}
	}
}

