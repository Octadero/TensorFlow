import XCTest

import CAPI
import Proto
import CTensorFlow

extension Tensorflow_DataType {

	enum SwiftDataTypeError: Error {
		case unknownType
		case unsutableType
	}
	
	func swiftType() throws -> Any.Type {
		switch self {
		case .dtFloat:
			return Float.self
		default:
			throw SwiftDataTypeError.unknownType
		}
	}
	
	func isSutable(type: Any.Type) throws {
		let dataType = try self.swiftType()
		guard type == dataType else {
			throw SwiftDataTypeError.unsutableType
		}
	}
}

enum TestError: Error {
	case errorCase
	case incorrectDataType
}

public struct OpSpec  {
	var type: String
	var name: String = "Type"
	var input: [Any]
	var attrs = [String : Any]()
}

class CAPIOperationsTests: XCTestCase {
	let tfStatus = TF_NewStatus()
	let tfGraph = newGraph()
	var namespace: String?
	var operation: TF_Operation?
	
	func matMul(graph: TF_Graph, tfStatus: TF_Status, a: TF_Operation, b: TF_Operation, name: String, transpose_a: Bool? = false, transpose_b: Bool? = false) -> TF_Operation {
		let desc: TF_OperationDescription = TF_NewOperation(graph, "MatMul".cString(using: .utf8), name.cString(using: .utf8));
		
		if let value = transpose_a, value == true {
			TF_SetAttrBool(desc, "transpose_a".cString(using: .utf8), 1);
		}
		if let value = transpose_b, value == true {
			TF_SetAttrBool(desc, "transpose_b".cString(using: .utf8), 1);
		}
		
		CAPI.add(input: TF_Output(oper: a, index: 0), for: desc)
		CAPI.add(input: TF_Output(oper: b, index: 0), for: desc)
		let operation: TF_Operation = TF_FinishOperation(desc, tfStatus)
		return operation
	}
	
	func floatTensor2x2(values: [Float]) -> TF_Tensor {
		let dims: [Int64] = [2, 2]
		let pointer = UnsafeRawPointer(dims)
		let len = MemoryLayout<Float>.size * values.count
		let tfTensor = TF_AllocateTensor(TF_FLOAT, pointer.assumingMemoryBound(to: Int64.self), 2, len)
		memcpy(TF_TensorData(tfTensor), values, len);
		return tfTensor!;
	}
	
	func floatConst2x2(tfGraph: TF_Graph, tfStatus: TF_Status, values: [Float], name: String) throws -> TF_Operation {
		let tfTensor = floatTensor2x2(values: values)
		defer {
			TF_DeleteTensor(tfTensor)
		}
		
		let desc: TF_OperationDescription = newOperation(in: tfGraph, operationType: "Const", operationName: name)
		TF_SetAttrTensor(desc, "value".cString(using: .utf8), tfTensor, tfStatus)

		guard TF_GetCode(tfStatus) == TF_OK else {
			XCTFail(String(cString:TF_Message(tfStatus)))
			throw TestError.errorCase
		}
	
		TF_SetAttrType(desc, "dtype".cString(using: .utf8), TF_FLOAT)
		let operation: TF_Operation = TF_FinishOperation(desc, tfStatus)
		
		guard TF_GetCode(tfStatus) == TF_OK else {
			XCTFail(String(cString:TF_Message(tfStatus)))
			throw TestError.errorCase
		}
		
		return operation
	}
	
	func test0BuildSuccessGraph() {
		let const0_val: [Float] = [1.0, 2.0, 3.0, 4.0];
		let const1_val: [Float] = [1.0, 0.0, 0.0, 1.0];
		let tfStatus = TF_NewStatus()
		do {
			let const0Operation = try floatConst2x2(tfGraph: tfGraph, tfStatus: tfStatus!, values: const0_val, name: "Const_0")
			let const1Operation = try floatConst2x2(tfGraph: tfGraph, tfStatus: tfStatus!, values: const1_val, name: "Const_1")
			
			let matMult = matMul(graph: tfGraph,
			                     tfStatus: tfStatus!,
			                     a: const0Operation,
			                     b: const1Operation,
			                     name: "MatMulName",
			                     transpose_a: false,
			                     transpose_b: false)
			
			
			let name = String(cString: TF_OperationName(matMult)!)
			let opType = String(cString: TF_OperationOpType(matMult)!)
			
			XCTAssert(name == "MatMulName", "Incorrect operation built.")
			XCTAssert(opType == "MatMul", "Incorrect operation built.")
			
		} catch {
			XCTFail(error.localizedDescription)
		}
		
		guard TF_GetCode(tfStatus) == TF_OK else {
			XCTFail(String(cString:TF_Message(tfStatus)))
			return
		}
	}
	
    static var allTests = [
		("test0BuildSuccessGraph", test0BuildSuccessGraph),
    ]
}
