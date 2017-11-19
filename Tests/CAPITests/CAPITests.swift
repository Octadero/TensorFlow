import XCTest

import CAPI
import Proto
import CTensorFlow
import MemoryLayoutKit

class CAPITests: XCTestCase {
    func testStringEncodeDecode() {
        do {
            let string = "string"
            let destinationLength = try encodedSize(of: string, encoding: .utf8)
            let destinationPointer = UnsafeMutablePointer<Int8>.allocate(capacity: destinationLength)
            let size = try CAPI.encode(string: string, writeAt: destinationPointer)
            let result = String(cString: destinationPointer)
            print("encode result: '\(result)' \(size) bytes")
            
            guard let originData = string.data(using: .utf8) else {
                XCTFail()
                return
            }
            
            let encodedData = Data(buffer: UnsafeBufferPointer<Int8>(start: destinationPointer, count: destinationLength))
            let decodedData = CAPI.decode(data: encodedData)
            
            
            print("originData: \t\t", originData.hexEncodedString())
            print("encodedData: \t", encodedData.hexEncodedString())
            print("decodedData: \t\t", decodedData.hexEncodedString())

            if originData != decodedData {
                XCTFail("Incorrect decod or encode result.")
            }
            
        } catch {
            print(error)
        }
    }
    
    func testVersion() {
		let version = CAPI.version()
		print("TensorFlow library version: ",version)
		XCTAssert(!version.isEmpty, "Version string can't be empty.")
    }
    
    static var allTests = [
        ("testVersion", testVersion),
        ("testString", testStringEncodeDecode)

    ]
}
