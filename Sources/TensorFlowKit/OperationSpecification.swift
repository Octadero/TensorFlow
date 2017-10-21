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
import Foundation

/// OpSpec is the specification of an Operation to be added to a Graph
/// (using Graph.AddOperation).
public struct OpSpec  {
	/// Type of the operation (e.g., "Add", "MatMul").
	public var type: String
	
	/// Name by which the added operation will be referred to in the Graph.
	/// If omitted, defaults to Type.
	public var name: String = "Type"
	
	/// Inputs to this operation, which in turn must be outputs
	/// of other operations already added to the Graph.
	///
	/// An operation may have multiple inputs with individual inputs being
	/// either a single tensor produced by another operation or a list of
	/// tensors produced by multiple operations. For example, the "Concat"
	/// operation takes two inputs: (1) the dimension along which to
	/// concatenate and (2) a list of tensors to concatenate. Thus, for
	/// Concat, len(Input) must be 2, with the first element being an Output
	/// and the second being an OutputList.
	///private outputList =
	public var input: [Any]
	
	/// Map from attribute name to its value that will be attached to this
	/// operation.
	public var attrs = [String : Any]()
	
	/// Other possible fields: Device, ColocateWith, ControlInputs.
	public init(type: String, name: String, input: [Any], attrs: [String : Any]) {
		self.type = type
		self.name = name
		self.input = input
		self.attrs = attrs
	}
}
