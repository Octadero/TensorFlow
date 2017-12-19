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
import CAPI
import Foundation

/// Swift `Error` represents error occurred at `Operation` manipulation.
public enum OperationError: Error {
	case tfOperationPointerIsEmpty
}

/// Operation that has been added to the graph.
public struct Operation  {
    var tfOperation: TF_Operation
    public var graph: Graph
	
	/// Creates a Operation with the specified `TF_Operation` pointer and `Graph`
	public init(tfOperation: TF_Operation, graph: Graph)  {
        self.tfOperation = tfOperation
        self.graph = graph
    }
	
	/// Name returns the name of the operation.
    public var name: String  {
		return  CAPI.name(of: self.tfOperation)
	}
	
	/// Type returns the name of the operator used by this operation.
    public var type: String {
		return CAPI.type(of: self.tfOperation)
	}
	
	/// Returns the number of outputs of op.
	/// - Return: Number of outputs of Operation.
    public var numberOfOutputs: Int32 {
		return CAPI.numberOfOutputs(at: self.tfOperation)
	}
	
	/// OutputListSize returns the size of the list of Outputs that is produced by a
	/// named output of op.
	///
	/// An Operation has multiple named outputs, each of which produces either
	/// a single tensor or a list of tensors. This method returns the size of
	/// the list of tensors for a specific output of the operation, identified
	/// by its name.
	func  outputListSize(output: String) throws -> Int32 {
		let cname = output.cString(using: .utf8)
		defer{
			//FIXME: Do free cname
			debugPrint("String should be dealloced.")
		}
		
		let status = Status()
		let size =  outputListLength(at: self.tfOperation, argumentName: cname, status: status.tfStatus)
		if let error = status.error() {
			throw error
		}
		return size
	}
	
	/// Output returns the i-th output of op.
	public func output(at index: Int) -> Output {
		return Output(in: self, at: index)
	}
    
    /// Output at index 0
    public var defaultOutput: Output {
        return Output(in: self, at: 0)
    }
    
    public func attributeType(by name: String) throws -> TF_DataType {
        return try getAttributeType(of: self.tfOperation, by: name)
    }
    
    public func attributeTensor(by name: String) throws -> Tensor? {
        guard let tfTensor = try getAttributeTensor(of: self.tfOperation, by: name) else {
            return nil
        }
        return try Tensor(tfTensor: tfTensor)
    }
}
