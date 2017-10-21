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
import Dispatch

/// status holds error information returned by TensorFlow. We convert all
/// TF statuses to Swift errors.
public class Status  {
    public private(set) var tfStatus:TF_Status!
	
	/// Create `Status` calling TF_NewStatus()
	public init() {
		tfStatus = CAPI.newStatus()
	}
	
	/// Returns `TF_Code` of status.
	public func code()-> TF_Code {
		return CAPI.code(for: tfStatus)
	}
	
	/// Returns message (description) of status.
	public func message() -> String {
		return CAPI.message(for: tfStatus)
	}
	/// Returns Swift `Error`
	public func error() -> Error? {
		let errorCode = code()
		guard errorCode != TF_OK else {
			return nil
		}
		return TensorFlowKitError.library(code: errorCode.rawValue, message: message())
	}
	
	/// Returns `Tensorflow_Error_Code` code.
	public func errorCode() -> Tensorflow_Error_Code {
		let code:TF_Code = CAPI.code(for: tfStatus)
		if let code = Tensorflow_Error_Code(rawValue: Int(code.rawValue)){
			return code
		}
		return Tensorflow_Error_Code(rawValue: 2)! //unknown
	}
	
	deinit {
		delete(status: tfStatus)
	}
}
