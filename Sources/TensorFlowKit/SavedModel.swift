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

/// Simple Checkpoint files names maper.
public enum CheckpointFile: String {
    case meta
    case data
    case index
    
    var postfix: String {
        return self.rawValue
    }
}

/// List of available error cases.
public enum SavedModelError: Error {
    case exportFolderNotFound
    case saveOperationFound
    case saveOperationNotFound
    case noVariablesFound
    case canNotExtractMetaGraph
}


/// SavedModel represents the contents of loaded SavedModel.
/// TODO: Add and document metagraphdef when we pre-generate protobufs.
public class SavedModel  {
    public let session: Session
    public let graph: Graph
    public let scope: Scope
    
    public let saveOp: Operation
    public let exportPath: String
    
    public private(set) var metaGraphDef: Tensorflow_MetaGraphDef
    public private(set) var saveModel: Tensorflow_SavedModel
    
    var scopeName: String
    
    /// List of constants.
    public enum Constants: String {
        case filenamePb = "saved_model.pb"
        case filenamePbTxt = "saved_model.pbtxt"
        case assetsDirectory = "assets"
        case assetsExtraDirectory = "assets.extra"
        case assetsKey = "saved_model_assets"
        case mainOpKey = "saved_model_main_op"
//        case variablesDirectory = "variables"
        case variablesFilenameOrDirectory = "variables"
        
        case tagGpu = "gpu"
        case tagServe = "serve"
        case tagTrain = "train"
        
        case scopeName = "save"
        case saveOperationName = "SaveV2"
        case restoreOperationName = "restore_all"
        
        public var key: String {
            return self.rawValue
        }
    }
    /// Constructor. 
    public init(session: Session, graph: Graph, exportPath: String) throws {
        self.session = session
        self.graph = graph
        self.scope = Scope(graph: graph, namespace: nil)
        
        scopeName = SavedModel.lookingForSaveScope(at: self.graph).suggestion
        
        self.exportPath = exportPath
        self.saveOp = try SavedModel.createSaveRestoreOps(at: scopeName, at: graph, mainScope: scope, exportPath: exportPath)
        
        self.metaGraphDef = Tensorflow_MetaGraphDef()
        self.metaGraphDef.graphDef = try graph.graphDef()
        
        // Prepare meta info
        // Filter operations from graph list.
        var metaInfo = Tensorflow_MetaGraphDef.MetaInfoDef()
        var opList = try CAPI.opList()
        let operationsInGraph = graph.operations.map { $0.type }
        let existedOperations = opList.op.filter { operationsInGraph.contains($0.name) }
        opList.op = existedOperations
        metaInfo.strippedOpList = opList
        metaInfo.tags = [Constants.tagServe.key]
        metaInfo.tensorflowVersion = CAPI.version()
        metaInfo.tensorflowGitVersion = "Octadero swift version."
        self.metaGraphDef.metaInfoDef = metaInfo
        
        // saver_def
        var saverDef = Tensorflow_SaverDef()
        saverDef.filenameTensorName = scopeName + "/Const:0"
        saverDef.saveTensorName = scopeName + "/control_dependency:0"
        saverDef.restoreOpName = scopeName + "/" + Constants.restoreOperationName.key
        saverDef.maxToKeep = 5
        saverDef.keepCheckpointEveryNHours = 10000
        saverDef.version = .v2
        saverDef.sharded = true
        self.metaGraphDef.saverDef = saverDef
        
        //FIXME: Curruntly collectionDef and signatureDef not finished.
        //self.metaGraphDef.collectionDef
        //self.metaGraphDef.signatureDef
        
        self.saveModel = Tensorflow_SavedModel()
        self.saveModel.metaGraphs = [self.metaGraphDef]
        self.saveModel.savedModelSchemaVersion = 1
    }
	
    static func lookingForSaveScope(at graph: Graph) -> (found: String?, suggestion: String) {
        let existedOperationsName = graph.operations.map { $0.name }
        guard existedOperationsName.contains(Constants.scopeName.key + "/" + Constants.saveOperationName.key) else {
            return (nil, Constants.scopeName.key)
        }
        var foundScopeName = String()
        for number in UInt.min..<UInt.max {
            let name = Constants.scopeName.key + "_" + String(number)
            if existedOperationsName.contains(name + "/" + Constants.saveOperationName.key) {
                foundScopeName = name
                continue
            } else {
                let notFoundScopeName = name
                return (foundScopeName, notFoundScopeName)
            }
        }
        return (nil, Constants.scopeName.key + "_" + String(UInt64.max) + "/" + Constants.saveOperationName.key)
    }
    
    /// Adds save and restore operations in the `Graph`.
    static func createSaveRestoreOps(at namespace: String, at graph: Graph, mainScope: Scope, exportPath: String) throws  -> Operation {
        let scope = mainScope.subScope(namespace: namespace)
        
        let saveFilePathConst = try scope.addConst(values: [exportPath + "/" + Constants.variablesFilenameOrDirectory.key + "/" + Constants.variablesFilenameOrDirectory.key], dimensions: [], as: "Const")

        let saveOp = try createSaveOps(at: graph, saveScope: scope, saveFilePathConst: saveFilePathConst)
        try createRestoreOps(at: graph, saveScope: scope, saveFilePathConst: saveFilePathConst)
        return saveOp
    }
    
    /// Adds save operations in the `Graph`.
    static func createSaveOps(at graph: Graph, saveScope scope: Scope, saveFilePathConst: Operation) throws -> Operation {
        let variables = graph.variables
        guard !variables.isEmpty else {
            throw SavedModelError.noVariablesFound
        }
        
        let types = try variables.map { try $0.attributeType(by: "dtype") }
        
        let save = try scope.saveV2(operationName: Constants.saveOperationName.key,
                                    prefix: saveFilePathConst.defaultOutput,
                                    tensorNames: scope.addConst(strings: variables.map { $0.name }, as: "\(Constants.saveOperationName.key)/tensor_names").defaultOutput,
                                    shapeAndSlices: scope.addConst(strings: Array<String>(repeating: "", count: variables.count), as: "\(Constants.saveOperationName.key)/shape_and_slices").defaultOutput,
                                    tensors: variables.map {$0.defaultOutput},
                                    dtypes: try types.map { try $0.swiftType() })
        
       let _ = try scope.with(controlDependencies: [save], scopeClosure: { (scope) in
            try scope.identity(operationName: "control_dependency", input: saveFilePathConst.defaultOutput)
        })
        
        return save
    }
    
    /// Adds restore operations in the `Graph`.
    static func createRestoreOps(at graph: Graph, saveScope scope: Scope, saveFilePathConst: Operation) throws {
        let variables = graph.variables
        guard !variables.isEmpty else {
            throw SavedModelError.noVariablesFound
        }
        
        var assignOps = [Operation]()
        for (index, variable) in variables.enumerated() {
            let restoreOpName = "RestoreV2_\(index)"
            let type = try variable.attributeType(by: "dtype").swiftType()
            let restore = try scope.restoreV2(operationName: restoreOpName,
                                              prefix: saveFilePathConst.defaultOutput,
                                              tensorNames: scope.addConst(strings: [variable.name], as: "\(restoreOpName)/tensor_names").defaultOutput,
                                              shapeAndSlices: scope.addConst(strings: [""], as: "\(restoreOpName)/shape_and_slices").defaultOutput,
                                              dtypes: [type])
            
            let restoreAssign = try scope.assign(operationName: "Assign_\(index)", ref: variable.defaultOutput, value: restore, validateShape: true, useLocking: true)
            assignOps.append(restoreAssign.operation)
        }
        
        let _: TensorFlowKit.Operation = try scope.with(controlDependencies: assignOps, scopeClosure: { (scope) in
            return try scope.noOp(operationName: Constants.restoreOperationName.key)
        })
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
    public static func load(into scope: Scope, exportPath: String, tags: [String], options: SessionOptions) throws -> (session: Session, metaGraphDef: Tensorflow_MetaGraphDef?) {
        
        var metaGraphDef: Tensorflow_MetaGraphDef? = nil
        let bufferHandler = { (bufferPointer: UnsafeMutablePointer<TF_Buffer>? ) in
            
            let tfBuffer = TF_GetBuffer(bufferPointer)
            let data = Data(bytes: tfBuffer.data, count: tfBuffer.length)
            metaGraphDef = try Tensorflow_MetaGraphDef(serializedData: data)
        }
        
        let cSession = try loadSessionFromSavedModel(sessionOptions: options.tfSessionOptions,
                                                     runOptions: nil,
                                                     exportPath: exportPath,
                                                     tags: tags,
                                                     graph: scope.graph.tfGraph,
                                                     metaDataGraphDefInjection: bufferHandler)
        
        let session = Session(tfSession: cSession, sessionOptions: options)
        guard let meta = metaGraphDef else {
            throw SavedModelError.canNotExtractMetaGraph
        }
        return (session: session, metaGraphDef: meta)
	}
    
    /// Create folder for exporting checkpoints
    func createExportFolder() throws {
        let exportFolderURL = URL(fileURLWithPath: exportPath + "/" + Constants.variablesFilenameOrDirectory.key, isDirectory: true)
        try FileManager.default.createDirectory(at: exportFolderURL, withIntermediateDirectories: true, attributes: nil)
    }
    
    /// Save current state as checkpoint.
    public func save() throws {
        var isDirectory : ObjCBool = false
        if !FileManager.default.fileExists(atPath: exportPath + "/" + Constants.variablesFilenameOrDirectory.key, isDirectory: &isDirectory) {
            try createExportFolder()
        } else {
            #if os(Linux)
                if !isDirectory {
                    try createExportFolder()
                }
            #else
                if !isDirectory.boolValue {
                    try createExportFolder()
                }
            #endif
        }
        
        let _ = try session.run(inputs: [], values: [], outputs: [], targetOperations: [saveOp])
        
        let savedModelData = try self.saveModel.serializedData()
        try savedModelData.write(to: URL(fileURLWithPath: self.exportPath).appendingPathComponent(Constants.filenamePb.key))
    }
    
    /// Restore `Graph` and `Session` from checkpoint saved by tf.train.Saver or other SaverDef implemintation.
    ///     *From Python:*
    ///     saver = tf.train.Saver(tf.all_variables(), write_version=saver_pb2.SaverDef.V2)
    ///     saver.save(session, '/tmp/your_checkpoint_folder/', global_step=episode_number)
    ///     https://www.tensorflow.org/programmers_guide/saved_model
    ///     See details: *Structure of a SavedModel directory*
    ///
    public static func restore(into scope: Scope, exportPath: URL, checkpoint: String) throws -> (session: Session, metaGraphDef: Tensorflow_MetaGraphDef?) {
        let temporaryFolder = NSTemporaryDirectory() + UUID().uuidString + "/"
        let variablesFolder = temporaryFolder + Constants.variablesFilenameOrDirectory.key
        
        try FileManager.default.createDirectory(atPath: temporaryFolder, withIntermediateDirectories: true, attributes: nil)
        try FileManager.default.createDirectory(atPath: variablesFolder, withIntermediateDirectories: true, attributes: nil)

        let folder = FileManager.default.enumerator(at: exportPath, includingPropertiesForKeys: nil)
        
        guard let folderFiles = folder else {
            throw SavedModelError.exportFolderNotFound
        }
        
        for file in folderFiles {
            guard let fileURL = file as? URL else { continue }
            if fileURL.lastPathComponent.hasPrefix(checkpoint) {
                let dstURL = URL(fileURLWithPath: variablesFolder).appendingPathComponent(Constants.variablesFilenameOrDirectory.key).appendingPathExtension(fileURL.pathExtension)
                try FileManager.default.copyItem(at: fileURL, to: dstURL)
            }
        }
        
        let metaPath = exportPath.appendingPathComponent(checkpoint).appendingPathExtension(CheckpointFile.meta.postfix)
        let metaGraphDefUrl = URL(fileURLWithPath: metaPath.absoluteString)

        let metaGraphDefData = try Data(contentsOf: metaGraphDefUrl)
        var metaGraphDef = try Tensorflow_MetaGraphDef(serializedData: metaGraphDefData)

        metaGraphDef.metaInfoDef.tags = [Constants.tagServe.key]
        var savedModel = Tensorflow_SavedModel()
        savedModel.metaGraphs = [metaGraphDef]
        let savedModelData = try savedModel.serializedData()
        let savedModelFilePath = temporaryFolder + Constants.filenamePb.key
        try savedModelData.write(to: URL(fileURLWithPath: savedModelFilePath))
        
        let savedModelResult = try self.load(into: scope, exportPath: temporaryFolder, tags: [Constants.tagServe.key], options: SessionOptions())
        
        try FileManager.default.removeItem(atPath: temporaryFolder)
        
        return savedModelResult
    }
}


