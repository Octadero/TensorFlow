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

/// Output represents one of the outputs of an operation in the graph. Has a
/// DataType (and eventually a Shape).  May be passed as an input argument to a
/// function for adding operations to a graph, or to a Session's Run() method to
/// fetch that output as a tensor.
public struct Output  {
	
	/// Op is the Operation that produces this Output.
	public private(set) var operation: Operation
	
	/// Index specifies the index of the output within the Operation.
	public private(set) var index: Int
	
	/// Creates a `Output` with the specified `Operation` and index.
	public init(in operation: Operation, at index:Int) {
		self.operation = operation
		self.index = index
	}
	
	/// Type returns the type of elements in the tensor produced by p.
	public func tfType() -> TF_DataType {
		return TF_OperationOutputType(tfOutput())
	}
	
	/// Obtain TF_Output structure which represents outout of operation.
	public func tfOutput() -> TF_Output {
		return TF_Output(oper: self.operation.tfOperation, index: CInt(self.index))
	}
	
	/// API is nor ready.
	/// Leads to fatalError.
	public func canBeAnInput() -> Bool {
		fatalError("Not ready.")
	}
}
