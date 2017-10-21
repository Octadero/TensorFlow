import XCTest

import CAPI
import Proto
import CTensorFlow

class CAPITests: XCTestCase {
	
    func testVersion() {
		let version = CAPI.version()
		print("TensorFlow library version: ",version)
		XCTAssert(!version.isEmpty, "Version string can't be empty.")
    }

    static var allTests = [
        ("testVersion", testVersion)		
    ]
}
