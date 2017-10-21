import XCTest

import CAPI
import CCAPI
import Proto
import CTensorFlow
import CCTensorFlow

class CCAPITests: XCTestCase {
	let tfStatus = TF_NewStatus()
	let tfGraph = newGraph()
	var namespace: String?
	var operation: TF_Operation?

	func test1GraphWriter() {
		let bufferPointer = TF_NewBuffer()
		let path = "/tmp/some.file."
		
		try? CAPI.graphDef(of: tfGraph, graphDef: bufferPointer)
		
		let tfBuffer = TF_GetBuffer(bufferPointer)
		let data = Data(bytes: tfBuffer.data, count: tfBuffer.length)
		do {
			let _ = try Tensorflow_GraphDef(serializedData: data)
			
		} catch {
			XCTFail("Graph should not be nil. \(error)")
		}
		CCAPI.createEventWriter(tfBuffer.data, UInt(tfBuffer.length), UnsafeMutablePointer(mutating: path.cString(using: .utf8)), 100.00, 1)
		TF_DeleteBuffer(bufferPointer)
	}
	
    static var allTests = [
		("test1GraphWriter", test1GraphWriter)
    ]
}
