import XCTest
@testable import TensorFlowKitTests
@testable import CAPITests


XCTMain([
	testCase(TensorFlowKitTests.allTests),
	testCase(CAPITests.allTests),
	testCase(CAPIOperations.allTests)
])
