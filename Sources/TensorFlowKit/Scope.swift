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
import CAPI
import Proto

/// Scope encapsulates common operation properties when building a Graph.
///
/// A Scope object (and its derivates, e.g., obtained from Scope.SubScope)
/// act as a builder for graphs. They allow common properties (such as
/// a name prefix) to be specified for multiple operations being added
/// to the graph.
///
/// A Scope object and all its derivates (e.g., obtained from Scope.SubScope)
/// are not safe for concurrent use by multiple goroutines.
public class Scope {
    public var graph: Graph
	var namemap = [String : Int]()
    var namespace: String?
	
	public init(graph: Graph? = nil, namespace: String? = nil) {
		if let graph = graph {
			self.graph = graph
		} else {
			self.graph = Graph()
		}
		self.namespace = namespace
	}
	
	/// Finalize returns the Graph on which this scope operates on and renders s
	/// unusable. If there was an error during graph construction, that error is
	/// returned instead.
	func finalize() throws -> Graph {
		return self.graph
	}
	
	/// AddOperation adds the operation to the Graph managed by s.
	///
	/// If there is a name prefix associated with s (such as if s was created
	/// by a call to SubScope), then this prefix will be applied to the name
	/// of the operation being added. See also Graph.AddOperation.
	public func addOperation(specification: OpSpec) throws -> Operation {
		var specification = specification
		
		if specification.name.isEmpty {
			specification.name = specification.type
		}
		
		if let namespace = self.namespace {
			specification.name = namespace + "/" + specification.name
		}
        
        let operation = try self.graph.addOperation(specification: specification)
        return operation
        
    }
	
	/// SubScope returns a new Scope which will cause all operations added to the
	/// graph to be namespaced with 'namespace'.  If namespace collides with an
	/// existing namespace within the scope, then a suffix will be added.
	func subScope(namespace: String) -> Scope {
		var namespace = self.uniqueName(namespace)
		if let selfNamespace = self.namespace {
			namespace = selfNamespace + "/" + namespace
		}
		return Scope(graph: graph, namespace: namespace)
	}

	func uniqueName(_ name:String) -> String {
		if let count = self.namemap[name], count > 0{
			return"\(name)_\(count)"
		}
		return name
	}
	
	func opName(type:String) -> String {
		guard let namespace = self.namespace else {
			return type
		}
		return namespace + "/" + type
	}
	
	public func graphDef() throws -> Tensorflow_GraphDef {
		return try self.graph.graphDef()
	}
	
	/// Save graph at folder.
	///		Using EventsWriter core feature to represent grapht Data.
	/// - Parameters:
	///		- folder: folder where file will be stored.
	///		- fileName: file name prefix for file.
	///		- wallTime: time will be showing at tensorboard as wall time. If you sen nil, save with `now` time.
	///		- step: step of process.
	public func save(at folder: URL, fileName: String, wallTime specialTime: Date? = nil, step: Int64) throws {
		try self.graph.save(at: folder, fileName: fileName, wallTime: specialTime, step: step)
	}

}
