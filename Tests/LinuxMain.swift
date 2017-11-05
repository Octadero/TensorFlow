import XCTest
@testable import TensorFlowKitTests
@testable import CAPITests
@testable import MNISTTests
@testable import OptimizerTests

XCTMain([
	testCase(TensorFlowKitTests.allTests),
	testCase(CAPITests.allTests),
	testCase(CAPIOperations.allTests),
	testCase(MNISTTests.allTests),
	testCase(OptimizerTests.allTests)
])
