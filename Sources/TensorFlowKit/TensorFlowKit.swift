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

import CAPI
import CTensorFlow
import Foundation

public enum TensorFlowKitError: Hashable, CustomStringConvertible, Error {
	case some
	case newSome
	case library(code: UInt32?, message: String?)
	case noValueNoErrorCase
	
	public var hashValue: Int {
		
		switch self {
		case .some: return 1 << 1
		case .newSome: return 1 << 2
		case .noValueNoErrorCase: return 1 << 3
		
		case .library(let code, _):
			let basis = 1 << 99
			if let code = code {
				return basis + Int(code)
			}
			return basis
		}
	}
	
	public var description: String {
		switch self {
		case .some:
			return "some"
						
		case .library(let code, let message):
			return "Error at library: \(code ?? 0) message: \(message ?? "")"
			
		case .noValueNoErrorCase:
			return "Function can't return no error nither value."
			
		default:
			return self.localizedDescription
			
		}
	}
	
	public static func ==(lel : TensorFlowKitError, rel : TensorFlowKitError) -> Bool {
		return lel.hashValue == rel.hashValue
	}
}


///MARK: - DataType
public enum TypeRepresentableError: Error, CustomStringConvertible, CustomDebugStringConvertible {
    case notSuitableType(type: Any.Type)
    case notSuitableTFType(type: TF_DataType)
    
    /// Getting description.
    public var description: String {
        switch self {
        case .notSuitableType(let type):
            return "Can't convert \(type) to TF_DataType."
        case .notSuitableTFType(let type):
            return "Can't convert TF_DataType: \(type) to swift type."
        }
    }
    
    /// Getting description.
    public var localizedDescription: String {
        return self.description
    }
    
    /// Getting description.
    public var debugDescription: String {
        return self.description
    }
}

public protocol SwiftTypeRepresentable {
    func swiftType() throws -> Any.Type
}

extension TF_DataType {
    public init(for swiftType: Any.Type) throws {
        if swiftType == Float.self {
            rawValue = TF_FLOAT.rawValue
			return
        } else if swiftType == Double.self {
            rawValue = TF_DOUBLE.rawValue
			return
        } else if swiftType == Int32.self {
            rawValue = TF_INT32.rawValue
			return
        } else if swiftType == UInt8.self {
            rawValue = TF_UINT8.rawValue
			return
        } else if swiftType == Int16.self {
            rawValue = TF_INT16.rawValue
			return
        } else if swiftType == Int8.self {
            rawValue = TF_INT8.rawValue
			return
        } else if swiftType == Int64.self {
            rawValue = TF_INT64.rawValue
			return
        } else if swiftType == Bool.self {
            rawValue = TF_BOOL.rawValue
			return
        } else if swiftType == UInt16.self {
            rawValue = TF_UINT16.rawValue
			return
        } else if swiftType == Int.self {
            rawValue = TF_INT32.rawValue
            return
        } else if swiftType == String.self {
            rawValue = TF_STRING.rawValue
            return
        }
        throw TypeRepresentableError.notSuitableType(type: swiftType)
    }
}

extension TF_DataType: SwiftTypeRepresentable {
    public func swiftType() throws -> Any.Type {
        if self == TF_FLOAT {
            return Float.self
        } else if self == TF_DOUBLE {
            return Double.self
        } else if self == TF_INT32 {
            return Int32.self
        } else if self == TF_UINT8 {
            return UInt8.self
        } else if self == TF_INT16 {
            return Int16.self
        } else if self == TF_INT8 {
            return Int8.self
        } else if self == TF_STRING {
            return String.self
        } else if self == TF_COMPLEX64 {
            throw TypeRepresentableError.notSuitableTFType(type: self)
        } else if self == TF_COMPLEX {
            throw TypeRepresentableError.notSuitableTFType(type: self)
        }else if self == TF_INT64 {
            return Int64.self
        }else if self == TF_BOOL {
            return Bool.self
        }else if self == TF_QINT8 {
            throw TypeRepresentableError.notSuitableTFType(type: self)
        }else if self == TF_QUINT8 {
            throw TypeRepresentableError.notSuitableTFType(type: self)
        }else if self == TF_QINT32 {
            throw TypeRepresentableError.notSuitableTFType(type: self)
        }else if self == TF_BFLOAT16 {
            throw TypeRepresentableError.notSuitableTFType(type: self)
        }else if self == TF_QINT16 {
            throw TypeRepresentableError.notSuitableTFType(type: self)
        }else if self == TF_QUINT16 {
            throw TypeRepresentableError.notSuitableTFType(type: self)
        }else if self == TF_UINT16 {
            return UInt16.self
        }else if self == TF_COMPLEX128 {
            throw TypeRepresentableError.notSuitableTFType(type: self)
        }else if self == TF_HALF {
            throw TypeRepresentableError.notSuitableTFType(type: self)
        }else if self == TF_RESOURCE {
            throw TypeRepresentableError.notSuitableTFType(type: self)
        }
        throw TypeRepresentableError.notSuitableTFType(type: self)
    }
}
