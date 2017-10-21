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

/// Swift `Error` structure for TF_Status representation.
public enum StatusError: Error, CustomStringConvertible, CustomDebugStringConvertible {
    /// Getting description.
    public var description: String {
        switch self {
        case .cancelled(let message):
            return "Error: \(message ?? "cancelled")"
        case .unknown(let message):
            return "Error: \(message ?? "unknown")"
        case .invalid_argument(let message):
            return "Error: \(message ?? "invalid_argument")"
        case .deadline_exceeded(let message):
            return "Error: \(message ?? "deadline_exceeded")"
        case .not_found(let message):
            return "Error: \(message ?? "not_found")"
        case .already_exists(let message):
            return "Error: \(message ?? "already_exists")"
        case .permission_denied(let message):
            return "Error: \(message ?? "permission_denied")"
        case .resource_exhausted(let message):
            return "Error: \(message ?? "resource_exhausted")"
        case .failed_precondition(let message):
            return "Error: \(message ?? "failed_precondition")"
        case .aborted(let message):
            return "Error: \(message ?? "aborted")"
        case .out_of_range(let message):
            return "Error: \(message ?? "out_of_range")"
        case .unimplemented(let message):
            return "Error: \(message ?? "unimplemented")"
        case .`internal`(let message):
            return "Error: \(message ?? "internal")"
        case .unavailable(let message):
            return "Error: \(message ?? "unavailable")"
        case .data_loss(let message):
            return "Error: \(message ?? "data_loss")"
        case .unauthenticated(let message):
            return "Error: \(message ?? "unauthenticated")"
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

    
	/*case ok = 0*/
    case cancelled(message: String?)
	case unknown(message: String?)
	case invalid_argument(message: String?)
	case deadline_exceeded(message: String?)
	case not_found(message: String?)
	case already_exists(message: String?)
	case permission_denied(message: String?)
	case resource_exhausted(message: String?)
	case failed_precondition(message: String?)
	case aborted(message: String?)
	case out_of_range(message: String?)
	case unimplemented(message: String?)
	case `internal`(message: String?)
	case unavailable(message: String?)
	case data_loss(message: String?)
	case unauthenticated(message: String?)
    
    init?(tfStatus: TF_Status) {
        let message = String(cString: TF_Message(tfStatus))
        let code = Int(TF_GetCode(tfStatus).rawValue)
        
        if code == 1 { self = .cancelled(message: message) } else
        if code == 2 { self = .unknown(message: message) } else
        if code == 3 { self = .invalid_argument(message: message) } else
        if code == 4 { self = .deadline_exceeded(message: message) } else
        if code == 5 { self = .not_found(message: message) } else
        if code == 6 { self = .already_exists(message: message) } else
        if code == 7 { self = .permission_denied(message: message) } else
        if code == 8 { self = .resource_exhausted(message: message) } else
        if code == 9 { self = .failed_precondition(message: message) } else
        if code == 10 { self = .aborted(message: message) } else
        if code == 11 { self = .out_of_range(message: message) } else
        if code == 12 { self = .unimplemented(message: message) } else
        if code == 13 { self = .`internal`(message: message) } else
        if code == 14 { self = .unavailable(message: message) } else
        if code == 15 { self = .data_loss(message: message) } else  
        if code == 16 { self = .unauthenticated(message: message) } else
        {
            return nil
        }
        
        
	}
}

/// TF_Status holds error information.  It either has an OK code, or
/// else an error code with an associated error message.

/// Return a new status object.
public func newStatus() -> TF_Status! {
	return TF_NewStatus()
}

/// Delete a previously created status object.
public func delete(status pointer: TF_Status!) {
	TF_DeleteStatus(pointer)
}

/// Record <code, msg> in *s.  Any previous information is lost.
/// A common use is to clear a status: TF_SetStatus(s, TF_OK, "");
public func set(status: TF_Status!, code: TF_Code, message: UnsafePointer<Int8>!) {
	TF_SetStatus(status, code, message)
}

/// Return the code record in *s.
public func code(for status: TF_Status!) -> TF_Code {
	return TF_GetCode(status)
}

/// Return a pointer to the (null-terminated) error message in *s.  The
/// return value points to memory that is only usable until the next
/// mutation to *s.  Always returns an empty string if TF_GetCode(s) is
/// TF_OK.
public func message(for status: TF_Status!) -> String {
	return  String(cString:TF_Message(status))
}




