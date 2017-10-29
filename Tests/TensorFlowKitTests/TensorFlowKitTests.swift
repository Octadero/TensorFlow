import XCTest

import CAPI
import Proto
import CTensorFlow
import TensorFlowKit


class TensorFlowKitTests: XCTestCase {
	
	func testScope() {
		let scope = Scope()
		do {
			let _ = try scope.graphDef()
		}catch {
			XCTFail(error.localizedDescription)
		}
	}
	
	func testGraph() {
		let graph = Graph()
		do {
			let _ = try graph.graphDef()
		}catch {
			XCTFail(error.localizedDescription)
		}
	}
	
	func applyGradientDescentFunc(scope: Scope,`var`: Output, alpha: Output, delta: Output, useLocking: Bool, name: String) throws -> Output {
		var attrs = [String : Any]()
		attrs["use_locking"] = useLocking
		
		let opspec = OpSpec(
			type: "ApplyGradientDescent",
			name: name,
			input: [`var`, alpha, delta],
			attrs: attrs
		)
		let op = try scope.addOperation(specification: opspec)
		return op.output(at: 0)
	}
	
	func subFun(scope: Scope, x: Output, y: Output, name: String) throws -> Output {
		let attrs = [String : Any]()
		
		let opspec = OpSpec(
			type: "Sub",
			name: name,
			input: [x, y],
			attrs: attrs
		)
		let op = try scope.addOperation(specification: opspec)
		return op.output(at: 0)
	}
	
	func substr(scope: Scope, input: Output, pos: Output, len: Output ) throws -> Output {
		let attrs = [String : Any]()
		
		let opspec = OpSpec(
			type: "Substr",
			name: "Type",
			input: [ input, pos, len],
			attrs: attrs
		)
		let op = try scope.addOperation(specification: opspec)
		return op.output(at: 0)
	}
	
	func placeholderFunc(scope: Scope, shape: Shape, name: String, type: Any.Type) throws -> Output {
		var attrs = [String : Any]()
		attrs["shape"] = shape
		attrs["dtype"] = try TF_DataType(for: type)
		
		let opspec = OpSpec(
			type: "Placeholder",
			name: name,
			input: [ ],
			attrs: attrs
		)
		let op = try scope.addOperation(specification: opspec)
		return op.output(at: 0)
	}
	
	func placeholderV2Func(scope: Scope, shape: Shape) throws -> Output? {
		var attrs = [String : Any]()
		attrs["shape"] = shape
		let opspec = OpSpec(
			type: "PlaceholderV2",
			name: "Type",
			input: [ ],
			attrs: attrs
		)
		let op = try scope.addOperation(specification: opspec)
		return op.output(at: 0)
	}
	
	func assignFunc(scope: Scope, ref: Output, value: Output, validateShape: Bool, useLocking : Bool, name: String) throws -> Output {
		var attrs = [String : Any]()
		attrs["validate_shape"] = validateShape
		attrs["use_locking"] = useLocking
		let opspec = OpSpec(
			type: "Assign",
			name: name,
			input: [ref, value],
			attrs: attrs
		)
		let op = try scope.addOperation(specification: opspec)
		return op.output(at: 0)
	}
	
	func variableV2Func(scope: Scope, shape: Shape, container: String, sharedName: String, type: Any.Type, name: String) throws -> Output {
		var attrs = [String : Any]()
		attrs["shape"] = shape
		attrs["container"] = container
		attrs["shared_name"] = sharedName
		attrs["dtype"] = try TF_DataType(for: type)
		
		let opspec = OpSpec(
			type: "VariableV2",
			name: name,
			input: [],
			attrs: attrs
		)
		let op = try scope.addOperation(specification: opspec)
		return op.output(at: 0)
	}
	
	func matMulFunc(scope: Scope, a: Output, b: Output, transposeA: Bool, transposeB: Bool, name: String) throws -> Output {
		var attrs = [String : Any]()
		attrs["transpose_a"] = transposeA
		attrs["transpose_b"] = transposeB
		
		let opspec = OpSpec(
			type: "MatMul",
			name: name,
			input: [a, b],
			attrs: attrs
		)
		
		let operation = try scope.addOperation(specification: opspec)
		return operation.output(at: 0)
	}
		
	func const(scope: Scope, value: Any, name: String) throws -> TensorFlowKit.Operation {
		var attrs = [String : Any]()
		attrs["value"] = value
		attrs["dtype"] = try TF_DataType(for: type(of: value))
		let specification = OpSpec(type: "Const", name: name, input: [ ], attrs: attrs)
		return try scope.addOperation(specification: specification)
	}
    
	func testComputedGraph() {
		let scope = Scope()
		do {
            let wTensor0 = try Tensor(dimensions: [3, 1], values: [1.0, 1.0, 1.0])
			
			let w = try variableV2Func(scope: scope, shape: .dimensions(value: [3, 1]), container: "", sharedName: "", type: Double.self, name: "W")
			let wInit = try assignFunc(scope: scope,
			                           ref: w,
			                           value: scope.addConst(tensor: wTensor0, as: "Const/Const").defaultOutput,
			                           validateShape: true,
			                           useLocking: true,
			                           name: "initW")
			
			let x = try scope.placeholder(operationName: "x", dtype: Double.self, shape: .unknown)
			let y = try scope.placeholder(operationName: "y", dtype: Double.self, shape: .unknown)
            let output = try scope.matMul(operationName: "output", a: x, b: w, transposeA: false, transposeB: false)
            let z = try scope.sub(operationName: "z", x: y, y: output)

            let loss = try matMulFunc(scope: scope,
			                            a: z,
			                            b: z,
			                            transposeA: true,
			                            transposeB: false,
			                            name: "loss")
			
            let gradientsOperations = try scope.addGradients(yOutputs: [loss], xOutputs: [w])

            guard let gradientsOperation = gradientsOperations.first else {
                fatalError("gradOutputs is empty")
            }

			let lossTensor = try Tensor(scalar: Double(0.05))

			let apply_grad_W = try applyGradientDescentFunc(scope: scope,
			                                                `var`: w,
			                                                alpha: scope.addConst(tensor: lossTensor, as: "Const_1/Const").defaultOutput,
			                                                delta: gradientsOperation.output(at: 0),
			                                                useLocking: false,
			                                                name: "ApplyGD")

			guard let url = URL(string: "/tmp/") else {
				XCTFail("Can't compute folder path")
				return
			}
			
			try scope.save(at: url, fileName: "testComputedGraph", step: 0)
            
			let session = try Session(graph: scope.graph, sessionOptions: SessionOptions())
            
            if let _ = try scope.graph.operation(by: "initW") {
                let initResult: [Tensor] = try session.run(inputs: [], values: [], outputs: [], targetOperations: [wInit.operation])
                print(initResult)
            }
            
            let xs = [1.0, -1.0, 3.0,  1.0, 2.0, 1.0,  1.0, -2.0, -2.0, 1.0, 0.0, 2.0]
            let ys = [14.0, 15.0, -9.0, 13.0]

			for index in 0..<210 {
                let xTensor0 = try Tensor(dimensions: [4, 3], values: xs)
                let yTensor0 = try Tensor(dimensions: [ys.count, 1], values: ys)

                let resultOutput = try session.run(inputs: [x, y],
                                                   values: [xTensor0, yTensor0],
                                                   outputs:  [loss, apply_grad_W],
                                                   targetOperations: [])
                resultOutput.forEach({tensor in
                    do {
                        let collection: [Double] = try tensor.pullCollection()
                        if index > 200 {
                            print(collection)
                        }
                    } catch {
                        print(error)
                    }
                })
            }
            
			
		} catch {
			XCTFail(String(describing: error))
		}
	}
	
    func mulFunc(scope: Scope, x: Output, y: Output) throws -> Output {
        let attrs = [String : Any]()
        
        let opspec = OpSpec(
            type: "Mul",
            name: "Type",
            input: [x, y],
            attrs: attrs
        )
        let op = try scope.addOperation(specification: opspec)
        return op.output(at: 0)
    }
    
	func testSimpleSession() {
		do {
			let scope = Scope()
			let x = try placeholderFunc(scope: scope, shape: .unknown, name: "x", type: Double.self)
			let y = try placeholderFunc(scope: scope, shape: .unknown, name: "y", type: Double.self)
			
			let result = try mulFunc(scope: scope, x: x, y: y)
			
			
			let session = try Session(graph: scope.graph, sessionOptions: SessionOptions())
			let feed = [x, y]
			let fetches = [result]
			
			let xValueTensor = try Tensor(scalar: Double(3.33333))
			let yValueTensor = try Tensor(scalar: Double(2.32323))
			
			let portialSession = try session.portial(inputs: feed, outputs: fetches, targetOperations: nil)
			
			let runResult = try portialSession.run(inputs: [x, y], values: [xValueTensor, yValueTensor], outputs: [result], targetOperations: [])

			print(runResult.map { String(describing: $0.description) })
			
		} catch {
			print(error)
		}
	}
	
	func testCreateScopeGraphConstsFunction() {
		let scope = Scope()
		do {
            let tensor0: Tensor = try Tensor(dimensions: [2, 2], values: [1.0, 2.0, 3.0, 4.0])
			let tensor1: Tensor = try Tensor(dimensions: [2, 2], values: [2.0, 3.0, 4.0, 5.0])
			
			var attrs0 = [String : Any]()
			attrs0["value"] = tensor0
			attrs0["dtype"] = TF_FLOAT
			let specification0 = OpSpec(type: "Const", name: "Const_0", input: [ ], attrs: attrs0)
			
			var attrs1 = [String : Any]()
			attrs1["value"] = tensor1
			attrs1["dtype"] = TF_FLOAT
			let specification1 = OpSpec(type: "Const", name: "Const_1", input: [ ], attrs: attrs1)
			
			let operation0 = try scope.addOperation(specification: specification0)
			let operation1 = try scope.addOperation(specification: specification1)
			
			let _ = try matMulFunc(scope: scope,
			                       a: operation0.output(at: 0),
			                       b: operation1.output(at: 0),
			                       transposeA: false,
			                       transposeB: false,
			                       name: "matMulFunc")
			
			guard let url = URL(string: "/tmp/") else {
				XCTFail("Can't compute folder path")
				return
			}
			
			try scope.save(at: url, fileName: "graphTest", step: 0)
			
		} catch {
			XCTFail(error.localizedDescription)
		}
	}
	
	func testDeviceList() {
		let scope = Scope()
		do {
			let session = try Session(graph: scope.graph, sessionOptions: SessionOptions())
			XCTAssert(session.devices.count != 0, "Device list can't be empty.")
		} catch {
			XCTFail(error.localizedDescription)
		}
	}
	
    func testEvent() {

        let scope = Scope()
        do {
            let tensor0: Tensor = try Tensor(dimensions: [2, 2], values: [1.0, 2.0, 3.0, 4.0])
            let tensor1: Tensor = try Tensor(dimensions: [2, 2], values: [2.0, 3.0, 4.0, 5.0])
            
            var attrs0 = [String : Any]()
            attrs0["value"] = tensor0
            attrs0["dtype"] = TF_FLOAT
            let specification0 = OpSpec(type: "Const", name: "Const_0", input: [ ], attrs: attrs0)
            
            var attrs1 = [String : Any]()
            attrs1["value"] = tensor1
            attrs1["dtype"] = TF_FLOAT
            let specification1 = OpSpec(type: "Const", name: "Const_1", input: [ ], attrs: attrs1)
            
            let operation0 = try scope.addOperation(specification: specification0)
            let operation1 = try scope.addOperation(specification: specification1)
            
            let _ = try matMulFunc(scope: scope,
                                   a: operation0.output(at: 0),
                                   b: operation1.output(at: 0),
                                   transposeA: false,
                                   transposeB: false,
                                   name: "matMulFunc")
            
            var event = Tensorflow_Event()
            event.fileVersion = "0.3"
            event.wallTime = Date().timeIntervalSince1970
            event.step = 0
            
            let graphDefData = try allocAndProcessBuffer { (bufferPointer) in
                try CAPI.graphDef(of:  scope.graph.tfGraph, graphDef: bufferPointer)
            }
            event.graphDef = graphDefData
            
            let value = try Tensorflow_Summary.Value(serializedData: try event.serializedData())
            
            var summury = Tensorflow_Summary()
            
            summury.value = [value]
            let summuryData = try summury.serializedData()
            guard let url = URL(string: "file:///tmp/writed.summury.data") else { XCTFail(); return }
            try summuryData.write(to: url)
            
            
        } catch {
            XCTFail(error.localizedDescription)
        }
        

    }
	static var allTests = [
        ("testEvent", testEvent),
		("testScope", testScope),
		("testGraph", testGraph),
        ("testComputedGraph", testComputedGraph),
        ("testSimpleSession", testSimpleSession),
        ("testCreateScopeGraphConstsFunction", testCreateScopeGraphConstsFunction),
        ("testDeviceList", testDeviceList)
    ]
}
