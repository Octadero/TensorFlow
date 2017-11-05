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

public enum EventWriterError : Error {
    case canNotComputeFileURL
    case canNotCreateFile
}

public class EventWriter {
    /// List of stored records.
    var records = [Record]()
    /// Path to storage file.
    public private(set) var fileURL: URL?
    /// Path to storage folder.
    public private(set) var folderURL: URL?
    public private(set) var identifier: String?
    /// Writing queue.
    fileprivate let dataQueue = DispatchQueue(label: "com.octadero.TensorFlowKit", qos: .userInteractive, attributes: .concurrent)
    
    /// Constructor, should receive url path to storage folder
    /// Also you can set some identifier for your file.
    public init(folder url: URL, identifier : String? = "") throws {
        try prepareFile(folder: url, identifier: identifier)
    }
    
    internal func prepareFile(folder url: URL, identifier : String? = "") throws {
        self.identifier = identifier
        let eventRecord = EventRecord.fileEvent()
        let fileEventRecord = try eventRecord.record()
        records.append(fileEventRecord)
        
        let session : String = identifier ?? "Event"
        guard let computedFileURL = URL(string: "events.out.tfevents.\(Date().timeIntervalSince1970).\(session)", relativeTo: url) else {
            throw EventWriterError.canNotComputeFileURL
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
        guard let folderURL = folderURL else { throw EventWriterError.canNotComputeFileURL }
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
        guard let fileURL = fileURL else { throw EventWriterError.canNotCreateFile }
        guard let folderURL = folderURL else { throw EventWriterError.canNotComputeFileURL }

        try folderPreparation()
        
        /// Clear file always
        guard FileManager.default.createFile(atPath: fileURL.absoluteString, contents: nil, attributes: nil) else {
            throw EventWriterError.canNotCreateFile
        }
        
        let fileHandle = try FileHandle(forUpdating: fileURL)
        var offset : Int = 0
        
        dataQueue.sync(flags: .barrier) {
            records.forEach { (record : Record) in
                fileHandle.seek(toFileOffset: UInt64(offset))
                var mutableRecord = record
                var data = mutableRecord.header.encode()
                data.append(mutableRecord.data)
                data.append(mutableRecord.footer.encode())
                fileHandle.write(data)
                offset += data.count
            }
            records.removeAll()
        }
        fileHandle.closeFile()
        try prepareFile(folder: folderURL, identifier: self.identifier)
    }
    
    /// Temporary functions
    public func track(graph: Graph, time : TimeInterval, step : Int64) throws {        
        var eventRecord = EventRecord(defaultKind: .value)
        eventRecord.event.wallTime = time
        eventRecord.event.summary = Tensorflow_Summary()
        eventRecord.event.step = step
        eventRecord.event.graphDef = try graph.data()
        let record = try eventRecord.record()
        
        dataQueue.sync(flags: .barrier) {
            records.append(record)
        }
    }
    
    /// Temporary functions
    public func track(tag : String, value : Float, time : TimeInterval, step : Int64) throws {
        var summary = Tensorflow_Summary()
        
        var summaryValue = Tensorflow_Summary.Value()
        summaryValue.simpleValue = value
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
    }
    /// Temporary functions
    public func track(tag : String, values : [Double], time : TimeInterval, step : Int64) throws {
        var summary = Tensorflow_Summary()
        
        var summaryValue = Tensorflow_Summary.Value()
        summaryValue.histo.bucketLimit = values
        var bucket : [Double] = Array<Double>(repeating: 0.0, count: values.count)
        for value_index in 0..<values.count {
            for limit_index in 0..<summaryValue.histo.bucketLimit.count {
                if values[value_index] < summaryValue.histo.bucketLimit[limit_index] {
                    bucket[limit_index] += 1
                    break
                }
            }
        }
        
        summaryValue.histo.bucket = bucket//.filter{ return $0 != 0.0}
        
        if let max = values.max() {
            summaryValue.histo.max = max
        }
        
        if let min = values.min() {
            summaryValue.histo.min = min
        }
        
        summaryValue.histo.num = Double(values.count)
        summaryValue.histo.sum = values.reduce(0, +)
        summaryValue.histo.sumSquares = values.map{$0 * $0}.reduce(0, +)
        
        
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
    }
    /// Temporary functions
    public func track(tag : String, values : [Float], time : TimeInterval, step : Int64) throws {
        let array = Array<Double>(values.map{ Double($0)})
        try track(tag: tag, values: array, time: time, step: step)
    }
}


