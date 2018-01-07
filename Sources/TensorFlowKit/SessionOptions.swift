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
import CTensorFlow
import Proto
import CAPI

public class SessionOptions  {
	/// Target indicates the TensorFlow runtime to connect to.
	///
	/// If 'target' is empty or unspecified, the local TensorFlow runtime
	/// implementation will be used.  Otherwise, the TensorFlow engine
	/// defined by 'target' will be used to perform all computations.
	///
	/// "target" can be either a single entry or a comma separated list
	/// of entries. Each entry is a resolvable address of one of the
	/// following formats:
	///   local
	///   ip:port
	///   host:port
	///   ... other system-specific formats to identify tasks and jobs ...
	///
	/// NOTE: at the moment 'local' maps to an in-process service-based
	/// runtime.
	///
	/// Upon creation, a single session affines itself to one of the
	/// remote processes, with possible load balancing choices when the
	/// "target" resolves to a list of possible processes.
	///
	/// If the session disconnects from the remote process during its
	/// lifetime, session calls may fail immediately.
	var target: String = ""
	
	/// Config is a binary-serialized representation of the
	/// tensorflow.ConfigProto protocol message
	/// (https://www.tensorflow.org/code/tensorflow/core/protobuf/config.proto).
    
    var config: Tensorflow_ConfigProto?
	
	///private var configProto: Tensorflow_ConfigProto = Tensorflow_ConfigProto()
	public private(set) var tfSessionOptions: TF_SessionOptions
	
    public init(target: String? = nil, configProto: Tensorflow_ConfigProto? = nil) throws {
        guard let tfSessionOptions = newSessionOptions() else {
            throw TensorFlowKitError.library(code: 0, message: "TF_SessionOptions can't be nil")
        }
        self.tfSessionOptions = tfSessionOptions

        if let target = target {
			self.target = target
		}
        
        if let configProto = configProto {
            self.configProto = configProto
        }

		set(target: self.target, for: tfSessionOptions)
	}
	
	public var configProto: Tensorflow_ConfigProto? {
		set {
			self.config = newValue
			do {
                if let value = newValue {
                    try setConfig(config:value)
                }
			} catch {
				fatalError("Can't set Tensorflow_ConfigProto to SessionOptions.")
			}
		}
		get {
			return self.config
		}
	}
	
	private func setConfig(config: Tensorflow_ConfigProto) throws {
		let data = try config.serializedData()
		try data.withUnsafeBytes {(pointer: UnsafePointer<UInt8>) in
			let rawPointer = UnsafeRawPointer(pointer)
			try CAPI.setConfig(options: self.tfSessionOptions, proto: rawPointer, protoLength: data.count)
		}
	}
	
	deinit {
		delete(sessionOptions: tfSessionOptions)
	}
}
