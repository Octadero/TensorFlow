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

public typealias SessionCompletionClosure = () -> (Void)

/// Session drives a TensorFlow graph computation.
///
/// When a Session is created with a given target, a new Session object is bound
/// to the universe of resources specified by that target. Those resources are
/// available to this session to perform computation described in the GraphDef.
/// After creating the session with a graph, the caller uses the Run() API to
/// perform the computation and potentially fetch outputs as Tensors.
/// A Session allows concurrent calls to Run().
public class Session {
    var tfSession: TF_Session
	public private(set) var sessionOptions: SessionOptions
	/// return all devices for this session in a dictionary.
	/// each key in the dictionary represents a device name,
	/// and the value is a tuple of device type and its memory size, in bytes
	public var devices: [TF_Device] {
		do {
			return try CAPI.devices(in: self.tfSession)
		} catch {
			debugPrint(error)
			return []
		}
	}
	
	/// Creates a `Session` with the specified `TF_Session` pointer.
    init(tfSession: TF_Session, sessionOptions: SessionOptions) {
        self.tfSession = tfSession
        self.sessionOptions = sessionOptions
	}
	
	/// NewSession creates a new execution session with the associated graph.
	/// options may be nil to use the default options.
	public init(graph: Graph, sessionOptions: SessionOptions? = nil) throws {
        if let sessionOptions = sessionOptions {
            self.sessionOptions = sessionOptions
        } else {
            self.sessionOptions = try SessionOptions()
        }
		
        self.tfSession = try newSession(graph: graph.tfGraph, sessionOptions: self.sessionOptions.tfSessionOptions)
	}
	/// Run a session with the specified inputs and outputs.
	/// - Parameter inputs: Array of `Outputs`.
	/// - Parameter values: Array of `Tensor` for feeding inputs at `Graph`.
	/// - Parameter targetOperations: Array of `Operation`.
	/// - Return: Array of computed `Tensor`.
    public func run(inputs: [Output], values: [Tensor], outputs: [Output], targetOperations: [Operation]) throws -> [Tensor] {

		let tfTensors = try CAPI.run(session: self.tfSession,
		                             runOptions: nil,
		                             inputs: inputs.map { $0.tfOutput() },
		                             inputsValues: values.map { $0.tfTensor },
		                             outputs: outputs.map { $0.tfOutput() },
		                             targetOperations: targetOperations.map { $0.tfOperation },
		                             metadata: nil)
	
		return try tfTensors.map { try Tensor(tfTensor: $0) }
    }
    
    public func run(runOptions: String, inputNames: [String], inputs: [Tensor?], outputNames: [String], targetOperationsNames:[String] ) throws -> (outputs: [Tensor], metaDataGraph: Tensorflow_MetaGraphDef?) {
        
        var metaGraphDef: Tensorflow_MetaGraphDef? = nil
        let bufferHandler = { (bufferPointer: UnsafeMutablePointer<TF_Buffer>? ) in
            
            let tfBuffer = TF_GetBuffer(bufferPointer)
            let data = Data(bytes: tfBuffer.data, count: tfBuffer.length)
            metaGraphDef = try Tensorflow_MetaGraphDef(serializedData: data)
        }
        
        let tfTensors = try CAPI.run(session: self.tfSession,
                                     runOptions: runOptions,
                                     inputNames: inputNames,
                                     inputs: inputs.map { $0?.tfTensor },
                                     outputNames: outputNames,
                                     targetOperationsNames: targetOperationsNames,
                                     metaDataGraphDefInjection: bufferHandler)
        
        
        let outputs = try tfTensors.map { try Tensor(tfTensor: $0) }
        return (outputs, metaGraphDef)
    }
    
	
	//TODO: - Load from model
	public init(modelPath: URL) throws {
		fatalError("Is in progress")
		//FT_LoadSessionFromSavedModel
	}
	
	/// Create a partial session with the specified inputs and outputs.
	/// - Parameter inputs: Array of `Outputs`.
	/// - Parameter values: Array of `Tensor` for feeding inputs at `Graph`.
	/// - Parameter targetOperations: Array of `Operation`.
	/// - Return: Array of computed `Tensor`.
	public func portial(inputs: [Output]? = nil, outputs: [Output]? = nil, targetOperations: [TensorFlowKit.Operation]? = nil) throws -> PartialSession {
		return try PartialSession(session: self, inputs: inputs, outputs: outputs, targetOperations: targetOperations)
	}
	
	/// Close openned session.
	public func close() throws {
		try CAPI.close(session: self.tfSession)
	}
	
    deinit {
		do {
			try close()
		} catch {
			debugPrint(error)
		}
    }
}
 
public class PartialSession {
	
	let handle: UnsafePointer<CChar>
	public let parent: Session
	
	/// Set up the graph with the intended feeds (inputs) and fetches (outputs) for a sequence of partial run calls.
	/// On success, returns a handle that is used for subsequent PRun calls. The handle should be deleted with TF_DeletePRunHandle when it is no longer needed.
	/// On failure, out_status contains a tensorflow::Status with an error message. NOTE: This is EXPERIMENTAL and subject to change.
	/// - parameters:
	///   - session: the parent session
	///   - inputs: an array of Output as inputs
	///   - outputs: an array of Output as outputs
	///   - targets: target operations in an array
	/// - throws: Panic
	public init(session: Session, inputs: [Output]? = nil, outputs: [Output]? = nil, targetOperations: [TensorFlowKit.Operation]? = nil) throws {
		self.parent = session
		
		var tfInputs = [TF_Output]()
		var tfOutputs = [TF_Output]()
		var tfTargetOperations = [TF_Operation]()
		
		if let inputs = inputs {
			tfInputs.append(contentsOf: inputs.map { $0.tfOutput() })
		}
		if let outputs = outputs {
			tfOutputs.append(contentsOf: outputs.map { $0.tfOutput() })
		}
		if let targetOperations = targetOperations {
			tfTargetOperations.append(contentsOf: targetOperations.map{ $0.tfOperation })
		}
		
		self.handle = try CAPI.sessionPartialRunSetup(session: self.parent.tfSession, inputs: tfInputs, outputs: tfOutputs, targetOperations: tfTargetOperations)
		
	}
	
	deinit {
		deletePartialRun(handle: handle)
	}
	
	/// Run a session with the specified inputs and outputs.
	/// - Parameter inputs: Array of `Outputs`.
	/// - Parameter values: Array of `Tensor` for feeding inputs at `Graph`.
	/// - Parameter targetOperations: Array of `Operation`.
	/// - Return: Array of computed `Tensor`.
	public func run(inputs: [Output], values: [Tensor], outputs: [Output], targetOperations: [Operation]) throws -> [Tensor] {
		let tfTensors = try CAPI.sessionPartialRun(session: self.parent.tfSession,
		                                           handle: self.handle,
		                                           inputs: inputs.map { $0.tfOutput() },
		                                           inputsValues: values.map { $0.tfTensor },
		                                           outputs: outputs.map { $0.tfOutput() },
		                                           targetOperations: targetOperations.map { $0.tfOperation })
		
		
		return try tfTensors.map { try Tensor(tfTensor: $0) }
	}
}

