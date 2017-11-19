import XCTest
@testable import TensorFlowKitTests
@testable import CAPITests
@testable import MNISTTests
@testable import OptimizerTests
@testable import CAPIOperationsTests

XCTMain([
	testCase(TensorFlowKitTests.allTests),
	testCase(CAPITests.allTests),
	testCase(CAPIOperationsTests.allTests),
	testCase(MNISTTests.allTests),
	testCase(OptimizerTests.allTests)
])
