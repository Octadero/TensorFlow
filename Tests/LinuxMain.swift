import XCTest
@testable import TensorFlowKitTests
@testable import CAPITests
@testable import CCAPITests


XCTMain([
	testCase(TensorFlowKitTests.allTests),
	testCase(CCAPITests.allTests),
	testCase(CAPITests.allTests),
	testCase(CAPIOperations.allTests)
])
