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
import Proto
import Dispatch

public enum FileWriterError : Error {
    case canNotComputeFileURL
    case canNotCreateFile
}

public class FileWriter {
    /// List of stored records.
    var records = [Record]()
    var fileOffcet: UInt64 = 0
    /// Path to storage file.
    public private(set) var fileURL: URL?
    /// Path to storage folder.
    public private(set) var folderURL: URL?
    public private(set) var identifier: String?
    /// Writing queue.
    fileprivate let dataQueue = DispatchQueue(label: "com.octadero.TensorFlowKit", qos: .userInteractive, attributes: .concurrent)
    
    /// Constructor, should receive url path to storage folder
    /// Also you can set some identifier for your file.
    public init(folder url: URL, identifier : String? = "", graph: Graph? = nil) throws {
        try prepareFile(folder: url, identifier: identifier)
        if let graph = graph {
            try add(graph: graph)
        }
        try flush()
    }
    
    internal func prepareFile(folder url: URL, identifier : String? = "") throws {
        self.identifier = identifier
        let eventRecord = EventRecord.fileEvent()
        let fileEventRecord = try eventRecord.record()
        records.append(fileEventRecord)
        
        let session : String = identifier ?? "Event"
        guard let computedFileURL = URL(string: "events.out.tfevents.\(Date().timeIntervalSince1970).\(session)", relativeTo: url) else {
            throw FileWriterError.canNotComputeFileURL
        }
        
        if url.scheme == nil {
            folderURL = URL(string: "file://" + url.absoluteString)!
        } else {
            folderURL = url
        }
        
        fileURL = computedFileURL
    }
    
    /// Checking is folder available.
    internal func folderPreparation() throws {
        guard let folderURL = folderURL else { throw FileWriterError.canNotComputeFileURL }
        var isDirectory : ObjCBool = false
        if FileManager.default.fileExists(atPath: folderURL.absoluteString, isDirectory: &isDirectory) {
            #if os(Linux)
                if isDirectory {
                    return
                }
            #else
                if isDirectory.boolValue {
                    return
                }
            #endif
        }
        
        try FileManager.default.createDirectory(at: folderURL, withIntermediateDirectories: true, attributes: nil)
    }
    
    /// After records accumulated, you should save them on file system.
    public func flush() throws {
        guard let fileURL = fileURL else { throw FileWriterError.canNotCreateFile }
        try folderPreparation()
        
        if !FileManager.default.fileExists(atPath: fileURL.path) {
            /// Clear file always
            guard FileManager.default.createFile(atPath: fileURL.absoluteString, contents: nil, attributes: nil) else {
                throw FileWriterError.canNotCreateFile
            }
        }
        
        let fileHandle = try FileHandle(forUpdating: fileURL)
        
        dataQueue.sync(flags: .barrier) {
            records.forEach { (record : Record) in
                fileHandle.seek(toFileOffset: fileOffcet)
                var mutableRecord = record
                var data = mutableRecord.header.encode()
                data.append(mutableRecord.data)
                data.append(mutableRecord.footer.encode())
                fileHandle.write(data)
                fileOffcet += UInt64(data.count)
            }
            records.removeAll()
            fileHandle.closeFile()
        }
    }
    
    /// Add Graph to events list to store it on file system
    internal func add(graph: Graph) throws {
        var eventRecord = EventRecord(defaultKind: .value)
        eventRecord.event.summary = Tensorflow_Summary()
        eventRecord.event.graphDef = try graph.data()
        let record = try eventRecord.record()
        dataQueue.sync(flags: .barrier) {
            records.append(record)
        }
    }
    
    /// Add serialized `Summary` to events list to store it on file system
    internal func add(summary: Data) throws {
        let record = try SummaryRecord(proto: summary).record()
        dataQueue.sync(flags: .barrier) {
            records.append(record)
        }
    }
	
	/// One more feature to track some
	public func add(scalar: Float, tag: String, step: Int64, time: TimeInterval = Date().timeIntervalSince1970) throws {
		var summary = Tensorflow_Summary()
		
		var summaryValue = Tensorflow_Summary.Value()
		summaryValue.simpleValue = scalar
		summaryValue.tag = tag
		summary.value.append(summaryValue)
		
		var eventRecord = EventRecord(defaultKind: .value)
		eventRecord.event.wallTime = time
		eventRecord.event.summary = summary
		eventRecord.event.step = step
		let record = try eventRecord.record()
		
		dataQueue.sync(flags: .barrier) {
			records.append(record)
		}
		try flush()
	}
	
    /// Add summary as serialized proto buffer stored in `Tensor`.
    public func addSummary(tensor: Tensor, step: Int64, time: TimeInterval = Date().timeIntervalSince1970) throws {
        let summary = try Tensorflow_Summary(serializedData: tensor.pullData())
        
        var eventRecord = EventRecord(defaultKind: .value)
        eventRecord.event.wallTime = time
        eventRecord.event.summary = summary
        eventRecord.event.step = step
        let record = try eventRecord.record()
        
        dataQueue.sync(flags: .barrier) {
            records.append(record)
        }
        try flush()
    }
}


