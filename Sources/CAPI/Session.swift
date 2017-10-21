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

/// Representation of devices in session
public struct TF_Device {
	let name: String?
	let type: String?
	let memorySize: Int64
}

/// API for driving Graph execution.

/// Return a new execution session with the associated graph, or NULL on error.
//
/// *graph must be a valid graph (not deleted or nullptr).  This function will
/// prevent the graph from being deleted until TF_DeleteSession() is called.
/// Does not take ownership of opts.
public func newSession(graph: TF_Graph!, sessionOptions: TF_SessionOptions!) throws -> TF_Session! {
	let status = TF_NewStatus()
	let session: TF_Session = TF_NewSession(graph, sessionOptions, status)
	if let status = status, let error = StatusError(tfStatus: status) {
		throw error
	}
	return session
}

/// This function creates a new TF_Session (which is created on success) using
/// `session_options`, and then initializes state (restoring tensors and other
/// assets) using `run_options`.
//
/// Any NULL and non-NULL value combinations for (`run_options, `meta_graph_def`)
/// are valid.
//
/// - `export_dir` must be set to the path of the exported SavedModel.
/// - `tags` must include the set of tags used to identify one MetaGraphDef in
///    the SavedModel.
/// - `graph` must be a graph newly allocated with TF_NewGraph().
//
/// If successful, populates `graph` with the contents of the Graph and
/// `meta_graph_def` with the MetaGraphDef of the loaded model.
public func loadSessionFromSavedModel(sessionOptions: TF_SessionOptions!,
                                      runOptions: UnsafePointer<TF_Buffer>!,
                                      exportDir: UnsafePointer<Int8>!,
                                      tags: UnsafePointer<UnsafePointer<Int8>?>!,
                                      tagsLength: Int32,
                                      graph: TF_Graph!,
                                      metaGraphDef: UnsafeMutablePointer<TF_Buffer>!,
                                      status: TF_Status!) -> TF_Session! {
	
	return TF_LoadSessionFromSavedModel(sessionOptions,
	                                    runOptions,
	                                    exportDir,
	                                    tags,
	                                    tagsLength,
	                                    graph,
	                                    metaGraphDef,
	                                    status)
}

/// Close a session.
//
/// Contacts any other processes associated with the session, if applicable.
/// May not be called after TF_DeleteSession().
public func close(session: TF_Session!) throws {
	let status = TF_NewStatus()
	TF_CloseSession(session, status)
	if let status = status, let error = StatusError(tfStatus: status) {
		throw error
	}
}

/// Destroy a session object.
//
/// Even if error information is recorded in *status, this call discards all
/// local resources associated with the session.  The session may not be used
/// during or after this call (and the session drops its reference to the
/// corresponding graph).
public func delete(session: TF_Session!, status: TF_Status!) {
	TF_DeleteSession(session, status)
}

/// Run the graph associated with the session starting with the supplied inputs
/// (inputs[0,ninputs-1] with corresponding values in input_values[0,ninputs-1]).
//
/// Any NULL and non-NULL value combinations for (`run_options`,
/// `run_metadata`) are valid.
//
///    - `run_options` may be NULL, in which case it will be ignored; or
///      non-NULL, in which case it must point to a `TF_Buffer` containing the
///      serialized representation of a `RunOptions` protocol buffer.
///    - `run_metadata` may be NULL, in which case it will be ignored; or
///      non-NULL, in which case it must point to an empty, freshly allocated
///      `TF_Buffer` that may be updated to contain the serialized representation
///      of a `RunMetadata` protocol buffer.
//
/// The caller retains ownership of `input_values` (which can be deleted using
/// TF_DeleteTensor). The caller also retains ownership of `run_options` and/or
/// `run_metadata` (when not NULL) and should manually call TF_DeleteBuffer on
/// them.
//
/// On success, the tensors corresponding to outputs[0,noutputs-1] are placed in
/// output_values[]. Ownership of the elements of output_values[] is transferred
/// to the caller, which must eventually call TF_DeleteTensor on them.
//
/// On failure, output_values[] contains NULLs.
public func run(session: TF_Session,
                runOptions: UnsafePointer<TF_Buffer>?,
                inputs: [TF_Output],
                inputsValues: [TF_Tensor?],
                outputs: [TF_Output],
                targetOperations: [TF_Operation?],
                metadata: UnsafeMutablePointer<TF_Buffer>?) throws -> [TF_Tensor] {
	
    guard inputsValues.count == inputs.count else {
        throw CAPIError.cancelled(message: "Incorrect number of inputs and thirs values")
    }
	
	let numberOfInputs = Int32(inputs.count)
	let numberOfOutputs = Int32(outputs.count)
	let numberOfTargets = Int32(targetOperations.count)
	let status = TF_NewStatus()
	/// Inputs
	let inputsPointer = inputs.withUnsafeBufferPointer {$0.baseAddress}
	let inputsValuesPointer = inputsValues.withUnsafeBufferPointer {$0.baseAddress}
	/// Outputs
	let outputsPointer = outputs.withUnsafeBufferPointer {$0.baseAddress}
	/// Targets
	let targetOperationsPointer = targetOperations.withUnsafeBufferPointer { $0.baseAddress }
	
	var outputsValuesPointer: UnsafeMutablePointer<TF_Tensor?>?
	if numberOfOutputs > 0 {
		outputsValuesPointer = UnsafeMutablePointer<TF_Tensor?>.allocate(capacity: Int(numberOfOutputs))
	} else {
		outputsValuesPointer = UnsafeMutablePointer<TF_Tensor?>(bitPattern: 0)
	}
	TF_SessionRun(session,
	              runOptions,
	              inputsPointer,
	              inputsValuesPointer,
	              numberOfInputs,
	              outputsPointer,
	              outputsValuesPointer,
	              numberOfOutputs,
	              targetOperationsPointer,
	              numberOfTargets,
	              metadata,
	              status)
	
	if let status = status, let error = StatusError(tfStatus: status) {
		throw error
	}
	
	if numberOfOutputs > 0, let pointer = outputsValuesPointer {
		return UnsafeMutableBufferPointer<TF_Tensor?>(start: pointer, count: Int(numberOfOutputs)).flatMap{ $0 }
	} else {
		return [TF_Tensor]()
	}
}

/// RunOptions

/// Input tensors

/// Output tensors

/// Target operations

/// RunMetadata

/// Output status

/// Set up the graph with the intended feeds (inputs) and fetches (outputs) for a
/// sequence of partial run calls.
//
/// On success, returns a handle that is used for subsequent PRun calls. The
/// handle should be deleted with TF_DeletePRunHandle when it is no longer
/// needed.
//
/// On failure, out_status contains a tensorflow::Status with an error
/// message.
/// NOTE: This is EXPERIMENTAL and subject to change.
public func sessionPartialRunSetup(session: TF_Session,
                                   inputs: [TF_Output],
                                   outputs: [TF_Output],
                                   targetOperations: [TF_Operation?]) throws -> UnsafePointer<Int8> {
	
	
	let numberOfInputs = Int32(inputs.count)
	let numberOfOutputs = Int32(outputs.count)
	let numberOfTargets = Int32(targetOperations.count)
	
	let inputsPointer = inputs.withUnsafeBufferPointer {$0.baseAddress}
	let outputsPointer = outputs.withUnsafeBufferPointer {$0.baseAddress}
	let targetOperationsPointer = targetOperations.withUnsafeBufferPointer { $0.baseAddress }
	
	var handle = UnsafePointer<CChar>(bitPattern: 0)

	let status = TF_NewStatus()
	
	TF_SessionPRunSetup(session, inputsPointer, numberOfInputs, outputsPointer, numberOfOutputs, targetOperationsPointer, numberOfTargets, &handle, status)
	
	if let status = status, let error = StatusError(tfStatus: status) {
		throw error
	}
	
	guard let result = handle else {
		throw CAPIError.cancelled(message: "Can't produce handle pointer at TF_SessionPRunSetup call.")
	}
	
	return result
}

/// Input names

/// Output names

/// Target operations

/// Output handle

/// Output status

/// Continue to run the graph with additional feeds and fetches. The
/// execution state is uniquely identified by the handle.
/// NOTE: This is EXPERIMENTAL and subject to change.
public func sessionPartialRun(session: TF_Session,
                              handle: UnsafePointer<Int8>,
                              inputs: [TF_Output],
                              inputsValues: [TF_Tensor?],
                              outputs: [TF_Output],
                              targetOperations: [TF_Operation?]) throws -> [TF_Tensor] {
	
	guard inputsValues.count == inputs.count else {
		throw CAPIError.cancelled(message: "Incorrect number of inputs and thirs values")
	}
	
	let numberOfInputs = Int32(inputs.count)
	let numberOfOutputs = Int32(outputs.count)
	let numberOfTargets = Int32(targetOperations.count)
	let status = TF_NewStatus()
	/// Inputs
	let inputsPointer = inputs.withUnsafeBufferPointer {$0.baseAddress}
	let inputsValuesPointer = inputsValues.withUnsafeBufferPointer {$0.baseAddress}
	/// Outputs
	let outputsPointer = outputs.withUnsafeBufferPointer {$0.baseAddress}
	/// Targets
	let targetOperationsPointer = targetOperations.withUnsafeBufferPointer { $0.baseAddress }
	
	var outputsValuesPointer: UnsafeMutablePointer<TF_Tensor?>?
	if numberOfOutputs > 0 {
		outputsValuesPointer = UnsafeMutablePointer<TF_Tensor?>.allocate(capacity: Int(numberOfOutputs))
	} else {
		outputsValuesPointer = UnsafeMutablePointer<TF_Tensor?>(bitPattern: 0)
	}
	
	TF_SessionPRun(session,
	               handle,
	               inputsPointer,
	               inputsValuesPointer,
	               numberOfInputs,
	               outputsPointer,
	               outputsValuesPointer,
	               numberOfOutputs,
	               targetOperationsPointer,
	               numberOfTargets,
	               status)
	
    if let status = status, let error = StatusError(tfStatus: status) {
        throw error
    }
    
	if numberOfOutputs > 0, let pointer = outputsValuesPointer {
        return UnsafeMutableBufferPointer<TF_Tensor?>(start: pointer, count: Int(numberOfOutputs)).flatMap { $0 }
	} else {
		return [TF_Tensor]()
	}
}

/// Input tensors

/// Output tensors

/// Target operations

/// Output status

/// Deletes a handle allocated by TF_SessionPRunSetup.
/// Once called, no more calls to TF_SessionPRun should be made.
public func deletePartialRun(handle: UnsafePointer<Int8>!) {
	return TF_DeletePRunHandle(handle)
}

/// The deprecated session API. Please switch to the above instead of
/// TF_ExtendGraph(). This deprecated API can be removed at any time without
/// notice.
public func newDeprecatedSession(options: TF_SessionOptions!, status: TF_Status!) -> TF_DeprecatedSession! {
	return TF_NewDeprecatedSession(options, status)
}

public func closeDeprecated(session: TF_DeprecatedSession!, status: TF_Status!) {
	fatalError("\(#function): Not implemented.")
}

public func deleteDeprecated(session: TF_DeprecatedSession!, status: TF_Status!) {
	fatalError("\(#function): Not implemented.")
}

public func reset(options: TF_SessionOptions!, containers: UnsafeMutablePointer<UnsafePointer<Int8>?>!, containersNumber: Int32, status: TF_Status!) {
	fatalError("\(#function): Not implemented.")
}

/// Treat the bytes proto[0,proto_len-1] as a serialized GraphDef and
/// add the nodes in that GraphDef to the graph for the session.
//
/// Prefer use of TF_Session and TF_GraphImportGraphDef over this.
public func extendGraph(oPointer:OpaquePointer!, _ proto: UnsafeRawPointer!, _ proto_len: Int, status: TF_Status!) {
	fatalError("\(#function): Not implemented.")
	/* TF_Reset(const TF_SessionOptions* opt, const char** containers, int ncontainers, TF_Status* status); */
}

/// See TF_SessionRun() above.
public func run(session:TF_Session!,
                runOptions: UnsafePointer<TF_Buffer>!,
                inputNames: UnsafeMutablePointer<UnsafePointer<Int8>?>!,
                inputs: UnsafeMutablePointer<OpaquePointer?>!,
                inputsNumber: Int32,
                outputNames: UnsafeMutablePointer<UnsafePointer<Int8>?>!,
                outputs: UnsafeMutablePointer<OpaquePointer?>!,
                outputsNumber: Int32,
                targetOperationsNames: UnsafeMutablePointer<UnsafePointer<Int8>?>!,
                targetsNumbers: Int32,
                runMetadata: UnsafeMutablePointer<TF_Buffer>!,
                status: TF_Status!) throws {
	
	let status = TF_NewStatus()
	/// CALL
	if let status = status, let error = StatusError(tfStatus: status) {
		throw error
	}
}

/// See TF_SessionPRunSetup() above.
public func partialRunSetup(session:TF_Session,
                            inputNames: [String],
                            inputs: [TF_Output],
                            outputs: [TF_Output],
                            targetOperations: [TF_Operation]? = nil) throws -> UnsafePointer<Int8> {
	
	fatalError("Not ready")
	
}

/// See TF_SessionPRun above.
public func partialRun(sessiong: TF_DeprecatedSession!,
                       handle: UnsafePointer<Int8>!,
                       inputNames: UnsafeMutablePointer<UnsafePointer<Int8>?>!,
                       inputs: UnsafeMutablePointer<OpaquePointer?>!,
                       inputsNumber: Int32,
                       outputNames: UnsafeMutablePointer<UnsafePointer<Int8>?>!,
                       outputs: UnsafeMutablePointer<OpaquePointer?>!,
                       outputsNumber: Int32,
                       targetOperationsNames: UnsafeMutablePointer<UnsafePointer<Int8>?>!,
                       targetsNumber: Int32,
                       status: TF_Status!) {
	TF_PRun(sessiong, handle, inputNames, inputs, inputsNumber, outputNames, outputs, outputsNumber, targetOperationsNames, targetsNumber, status)
}

/// TF_SessionOptions holds options that can be passed during session creation.

/// Return a new options object.
public func newSessionOptions() -> TF_SessionOptions! {
	return TF_NewSessionOptions()
}

/// Set the target in TF_SessionOptions.options.
/// target can be empty, a single entry, or a comma separated list of entries.
/// Each entry is in one of the following formats :
/// "local"
/// ip:port
/// host:port
public func set(target: UnsafePointer<Int8>!, for options: TF_SessionOptions!) {
	TF_SetTarget(options, target)
}

/// Set the config in TF_SessionOptions.options.
/// config should be a serialized tensorflow.ConfigProto proto.
/// If config was not parsed successfully as a ConfigProto, record the
/// error information in *status.
public func setConfig(options: TF_SessionOptions!, proto: UnsafeRawPointer!, protoLength: Int) throws {
	let status = TF_NewStatus()
	TF_SetConfig(options, proto, protoLength, status)
	if let status = status, let error = StatusError(tfStatus: status) {
		throw error
	}
}

/// Destroy an options object.
public func delete(sessionOptions: TF_SessionOptions!) {
	TF_DeleteSessionOptions(sessionOptions)
}


/// Lists all devices in a TF_Session.
///
/// Caller takes ownership of the returned TF_DeviceList* which must eventually
/// be freed with a call to TF_DeleteDeviceList.
public func devices(`in` session: TF_Session) throws -> [TF_Device] {
	let status = TF_NewStatus()
	let list = TF_SessionListDevices(session, status)
	if let status = status, let error = StatusError(tfStatus: status) {
		throw error
	}
	var devices = [TF_Device]()
	let count = TF_DeviceListCount(list)
	for index in 0..<count {
		guard let deviceName = String(utf8String: TF_DeviceListName(list, index, status)) else {
			if let status = status, let error = StatusError(tfStatus: status) {
				debugPrint("Error at getting device list: \(error).")
			}
			continue
		}
		guard let deviceType = String(utf8String: TF_DeviceListType(list, index, status)) else {
			if let status = status, let error = StatusError(tfStatus: status) {
				debugPrint("Error at getting device list: \(error).")
			}
			continue
		}
		
		let memorySize = TF_DeviceListMemoryBytes(list, index, status)
		if let status = status, let error = StatusError(tfStatus: status) {
			debugPrint("Error at getting device list: \(error).")
			continue
		}
		
		devices.append(TF_Device(name: deviceName,
		                         type: deviceType,
		                         memorySize: memorySize))
	}
	
	TF_DeleteDeviceList(list)
	
	return devices
}
