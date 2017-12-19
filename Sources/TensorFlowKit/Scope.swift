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
import CTensorFlow
import Dispatch

public enum ScopeError: Error {
    case operationNotFoundByname
}

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
    var controlDependencies = [Operation]()
    
	public init(graph: Graph? = nil, namespace: String? = nil) {
		if let graph = graph {
			self.graph = graph
		} else {
			self.graph = Graph()
		}
		self.namespace = namespace
	}
    
    //FIXME: Add mutex https://www.cocoawithlove.com/blog/2016/06/02/threads-and-mutexes.html
    /// Add controlDependencies to scope.
    public func with<T>(controlDependencies: [Operation], scopeClosure: (_ scope: Scope) throws -> T) throws -> T {
        self.controlDependencies = controlDependencies
        let result = try scopeClosure(self)
        self.controlDependencies.removeAll()
        return result
    }
    
    /// Add controlDependencies to scope.
    public func with<T>(controlDependencyNames: [String], scopeClosure: (_ scope: Scope) throws -> T) throws -> T {
        let operations = try controlDependencyNames.map({ (name) -> Operation in
            guard let operation = try self.graph.operation(by: name) else {
                throw ScopeError.operationNotFoundByname
            }
            return operation
        })
        return try self.with(controlDependencies: operations, scopeClosure: scopeClosure)
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
	public func addOperation(specification: OpSpec, controlDependencies: [Operation]? = nil) throws -> Operation {
		var specification = specification
		
		if specification.name.isEmpty {
			specification.name = specification.type
		}
		
		if let namespace = self.namespace {
			specification.name = namespace + "/" + specification.name
		}
        
        let operation = try self.graph.addOperation(specification: specification, controlDependencies: controlDependencies)
        return operation
        
    }
	
	/// SubScope returns a new Scope which will cause all operations added to the
    /// graph to be namespaced with 'namespace'.  If namespace collides with an
    /// existing namespace within the scope, then a suffix will be added.
    public func subScope(namespace: String) -> Scope {
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
}
