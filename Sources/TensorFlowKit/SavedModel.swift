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


/// SavedModel represents the contents of loaded SavedModel.
/// TODO: Add and document metagraphdef when we pre-generate protobufs.
struct SavedModel  {
    var session: Session
    var graph: Graph
    
    public init(session: Session, graph:Graph) {
        self.session = session
        self.graph = graph
    }
	
	/// LoadSavedModel creates a new SavedModel from a model previously
	/// exported to a directory on disk.
	///
	/// Exported models contain a set of graphs and, optionally, variable values.
	/// Tags in the model identify a single graph. LoadSavedModel initializes a
	/// session with the identified graph and with variables initialized to from the
	/// checkpoints on disk.
	///
	/// The tensorflow package currently does not have the ability to export a model
	/// to a directory from Go. This function thus currently targets loading models
	/// exported in other languages, such as using CAPI.saved_model.builder in Python.
	/// See: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md#tags
	public func load(exportPath: String, tags: [String?], options: SessionOptions) throws -> SavedModel {
		let status = Status()
		let tfSessiongOptions = options.tfSessionOptions

		if let cExportDir:[CChar] = exportPath.cString(using: .utf8) {
			let graph = Graph()
			let cTags = tags.map { $0.flatMap { UnsafePointer<Int8>(strdup($0)) } }
			
			
			if let cSession = loadSessionFromSavedModel(sessionOptions: tfSessiongOptions,
			                                            runOptions: nil,
			                                            exportDir: cExportDir,
			                                            tags: cTags,
			                                            tagsLength: Int32(cTags.count),
			                                            graph: graph.tfGraph,
			                                            metaGraphDef: nil,
			                                            status: status.tfStatus) {
				let session = Session(tfSession: cSession)
				//FIXME: Do finalizer
				//runtime.SetFinalizer(s, func(s *Session) { s.Close() })
				return SavedModel(session: session, graph: graph)
			}
		}
		
		if let error = status.error() {
			throw error
		} else {
			throw TensorFlowKitError.noValueNoErrorCase
		}
	}
}


