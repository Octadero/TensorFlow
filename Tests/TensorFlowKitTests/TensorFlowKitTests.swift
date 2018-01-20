import XCTest

import CAPI
import Proto
import CTensorFlow
import TensorFlowKit
import Unarchiver

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
			
            let gradientsOutputs = try scope.addGradients(yOutputs: [loss], xOutputs: [w])

            guard let gradientsOutput = gradientsOutputs.first else {
                fatalError("gradOutputs is empty")
            }

			let lossTensor = try Tensor(scalar: Double(0.05))

			let apply_grad_W = try applyGradientDescentFunc(scope: scope,
			                                                `var`: w,
			                                                alpha: scope.addConst(tensor: lossTensor, as: "Const_1/Const").defaultOutput,
			                                                delta: gradientsOutput,
			                                                useLocking: false,
			                                                name: "ApplyGD")

			let url = URL(fileURLWithPath: "/tmp/graph.data")
			
			try scope.graph.save(at: url)
            
            guard let writerURL = URL(string: "/tmp/") else {
                XCTFail("Can't compute folder url.")
                return
            }
            
            let logger = try FileWriter(folder: writerURL, identifier: "iMac", graph: scope.graph)
            try logger.flush()
            
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
	
    func testComputedGraphByNames() {
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
            
            let gradientsOutputs = try scope.addGradients(yOutputs: [loss], xOutputs: [w])
            
            guard let gradientsOutput = gradientsOutputs.first else {
                fatalError("gradOutputs is empty")
            }
            
            let lossTensor = try Tensor(scalar: Double(0.05))
            
            let _ = try applyGradientDescentFunc(scope: scope,
                                                            `var`: w,
                                                            alpha: scope.addConst(tensor: lossTensor, as: "Const_1/Const").defaultOutput,
                                                            delta: gradientsOutput,
                                                            useLocking: false,
                                                            name: "ApplyGD")
            
            let url = URL(fileURLWithPath: "/tmp/graph.data")
            
            try scope.graph.save(at: url)
            
            guard let writerURL = URL(string: "/tmp/") else {
                XCTFail("Can't compute folder url.")
                return
            }
            
            let logger = try FileWriter(folder: writerURL, identifier: "iMac", graph: scope.graph)
            try logger.flush()
            
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
                let resultOutput = try session.run(runOptions: nil,
                                                   inputNames: ["x", "y"],
                                                   inputs: [xTensor0, yTensor0],
                                                   outputNames: ["loss", "ApplyGD"],
                                                   targetOperationsNames: [])
                
                resultOutput.outputs.forEach({tensor in
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
			let url = URL(fileURLWithPath: "/tmp/graph.data")
			
			try scope.graph.save(at: url)
			
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
	
    func testEventSummury() {
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
            event.fileVersion = "1.3.0"
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
            guard let url = URL(string: "file:///tmp/events.out.tfevents.1509444510.72209.writed.summury.data") else { XCTFail(); return }
            try summuryData.write(to: url)
            
            
            guard let writerURL = URL(string: "/tmp/log/") else {
                XCTFail("Can't compute folder url.")
                return
            }
            
            let logger = try FileWriter(folder: writerURL, identifier: "iMac", graph: scope.graph)
            try logger.flush()
            
        } catch {
            XCTFail(error.localizedDescription)
        }
    }
    
    func testEventWriter() {
        guard let url = URL(string: "/tmp/log/") else {
            XCTFail("Can't compute folder url.")
            return
        }
        
        do {
            let logger = try FileWriter(folder: url, identifier: "iMac")
            try logger.flush()
        } catch {
            XCTFail(error.localizedDescription)
        }
    }
    
    
    func testStringTensor() {
        do {
            guard let writerURL = URL(string: "/tmp/\(#function)/") else {
                XCTFail("Can't compute folder url.")
                return
            }
            
            let scope = Scope()
            
            let tensorInt0 = try Tensor(dimensions: [2, 2], values: [1,2,3,4])
            let constInt0 = try scope.addConst(tensor: tensorInt0, as: "TensorConstInt0").defaultOutput
            
            let tensorInt1 = try Tensor(dimensions: [2, 2], values: [5,6,7,8])
            let constInt1 = try scope.addConst(tensor: tensorInt1, as: "TensorConstInt1").defaultOutput

            let _ = try scope.matMul(operationName: "Mult", a: constInt0, b: constInt1, transposeA: false, transposeB: false)
            
            let tensor0 = try Tensor(dimensions: [2, 2], values: ["1-s", "2-s", "3-s", "4-s"])
            let const0 = try scope.addConst(tensor: tensor0, as: "TensorConst0").defaultOutput

            let tensor1 = try Tensor(dimensions: [2, 2], values: ["5 s", "6 st", "7 str", "8 stri"])
            let const1 = try scope.addConst(tensor: tensor1, as: "TensorConst1").defaultOutput

            
            let _ = try scope.stringJoin(operationName: "Join", inputs: [const0, const1], n: 2, separator: ",")
            
            let fileWriter = try FileWriter(folder: writerURL, identifier: "iMac", graph: scope.graph)
            try fileWriter.flush()
            
        } catch {
            XCTFail(error.localizedDescription)
        }
    }

    func testGraphEnumirator() {
        do {
            let scope = Scope()
            let tensorInt0 = try Tensor(dimensions: [2, 2], values: [1,2,3,4])
            let constInt0 = try scope.addConst(tensor: tensorInt0, as: "TensorConstInt0").defaultOutput
            
            let tensorInt1 = try Tensor(dimensions: [2, 2], values: [5,6,7,8])
            let constInt1 = try scope.addConst(tensor: tensorInt1, as: "TensorConstInt1").defaultOutput
            
            let _ = try scope.matMul(operationName: "Mult", a: constInt0, b: constInt1, transposeA: false, transposeB: false)
            
            let tensor0 = try Tensor(dimensions: [2, 2], values: ["1-s", "2-s", "3-s", "4-s"])
            let const0 = try scope.addConst(tensor: tensor0, as: "TensorConst0").defaultOutput
            
            let tensor1 = try Tensor(dimensions: [2, 2], values: ["5 s", "6 st", "7 str", "8 stri"])
            let const1 = try scope.addConst(tensor: tensor1, as: "TensorConst1").defaultOutput
            
            let _ = try scope.stringJoin(operationName: "Join", inputs: [const0, const1], n: 2, separator: ",")
            print("Operations: ", scope.graph.operations.map { $0.name })
            XCTAssert(scope.graph.operations.count > 0, "Incorrect list of operations.")
        } catch {
            XCTFail(error.localizedDescription)
        }
    }
    
    func testImageSummary() {
        do {
            let scope = Scope()

            guard let writerURL = URL(string: "/tmp/\(#function)/") else {
                XCTFail("Can't compute folder url.")
                return
            }
            
            let summary = Summary(scope: scope)
            
            var values = Array<Float>()
            let numberOfImages = 10
            for i in 0..<(25 * 25 * numberOfImages) {
                values.append(0.1 * Float(i))
            }
            
            try summary.images(name: "ImageTest",
                               batchSize: numberOfImages,
                               size: Summary.ImageSize(width: 25, height: 25),
                               values: values,
                               maxImages: 255,
                               badColor: Summary.BadColor(channel: .grayscale, colorComponents: [UInt8(100)]))
            
            let merge = try summary.merged()
            
            let fileWriter = try FileWriter(folder: writerURL, identifier: "Summary-test", graph: scope.graph)

            let session = try Session(graph: scope.graph, sessionOptions: SessionOptions())
            let resultOutput = try session.run(inputs: [],
                                               values: [],
                                               outputs: [merge],
                                               targetOperations: [])
            if let image = resultOutput.first {
                try fileWriter.addSummary(tensor: image, step: 0)
            }
            
        } catch {
            XCTFail(error.localizedDescription)
        }
    }

    
    func testTensorTransformation() {
        do {
            let scope = Scope()

            
            var values = Array<Float>()
            let numberOfElements = 100
            for i in 0..<numberOfElements {
                values.append(Float(i))
            }
            
            let input = try scope.addConst(values: values, dimensions: [Int64(values.count)], as: "Const")
     
            let output = try scope.stridedSlice(operationName: "StridedSlice",
                                                input: input.defaultOutput,
                                                begin: try scope.addConst(values: [Int(0)], dimensions: [1], as: "Begin").defaultOutput,
                                                end: try scope.addConst(values: [Int(numberOfElements)], dimensions: [1], as: "End").defaultOutput,
                                                strides: try scope.addConst(values: [Int(5)], dimensions: [1], as: "Strides").defaultOutput,
                                                index: Int.self,
                                                beginMask: 0,
                                                endMask: 0,
                                                ellipsisMask: 0,
                                                newAxisMask: 0,
                                                shrinkAxisMask: 0)
            
            
            let session = try Session(graph: scope.graph, sessionOptions: SessionOptions())
            let resultOutput = try session.run(inputs: [],
                                               values: [],
                                               outputs: [output],
                                               targetOperations: [])
            
            if let tensor = resultOutput.first {
                let transformedValue: [Float] = try tensor.pullCollection()
                for value in transformedValue {
                    guard Int(value) % 5 == 0 else {
                        XCTFail("Incorrect value.")
                        return
                    }
                }
            }
            
        } catch {
            XCTFail(error.localizedDescription)
        }
    }
    
    func testGraph0Save() {
        do {
            let scope = Scope()
            let tensorInt0 = try Tensor(dimensions: [2, 2], values: [1,2,3,4])
            let constInt0 = try scope.addConst(tensor: tensorInt0, as: "TensorConstInt0").defaultOutput
            
            let tensorInt1 = try Tensor(dimensions: [2, 2], values: [5,6,7,8])
            let constInt1 = try scope.addConst(tensor: tensorInt1, as: "TensorConstInt1").defaultOutput
            
            let _ = try scope.matMul(operationName: "Mult", a: constInt0, b: constInt1, transposeA: false, transposeB: false)
            
            let tensor0 = try Tensor(dimensions: [2, 2], values: ["1-s", "2-s", "3-s", "4-s"])
            let const0 = try scope.addConst(tensor: tensor0, as: "TensorConst0").defaultOutput
            
            let tensor1 = try Tensor(dimensions: [2, 2], values: ["5 s", "6 st", "7 str", "8 stri"])
            let const1 = try scope.addConst(tensor: tensor1, as: "TensorConst1").defaultOutput
            
            let _ = try scope.stringJoin(operationName: "Join", inputs: [const0, const1], n: 2, separator: ",")
            
            guard let pbURL = URL(string: "file:///tmp/graph.pb") else {
                XCTFail("Can't compute folder url.")
                return
            }
            
            guard let pbtxtURL = URL(string: "file:///tmp/graph.pbtxt") else {
                XCTFail("Can't compute folder url.")
                return
            }

            try scope.graph.save(at: pbURL)
            try scope.graph.save(at: pbtxtURL, asText: true)

        } catch {
            XCTFail(error.localizedDescription)
        }
    }
 
    func testDiffImage() {
        do {
            let scope = Scope()
            let size: Int64 = 9
            
            let input = try scope.placeholder(dtype: Float.self, shape: Shape.dimensions(value: [size]))
            let x = try scope.variableV2(shape: Shape.dimensions(value: [size]), dtype: Float.self, container: "", sharedName: "")
            
            let assign = try scope.assign(ref: x,
                                          value: try scope.addConst(values: Array<Float>(repeating: 15.0, count: Int(size)), dimensions: [size], as: "Zero").defaultOutput,
                                          validateShape: true,
                                          useLocking: true)
            
            let _ = try scope.assignSub(ref: x, value: input, useLocking: true)
            
            let session = try Session(graph: scope.graph)
            let _ = try session.run(inputs: [], values: [], outputs: [], targetOperations: [assign.operation])
            
            for i in 0..<5 {
                let tensor = try Tensor(dimensions: [size], values: Array<Float>.init(repeating: Float(i), count: Int(size)))
                let out = try session.run(inputNames: ["Placeholder"], inputs: [tensor], outputNames: ["AssignSub"], targetOperationsNames: [])
                let result: [Float] = try out.outputs[0].pullCollection()
                print(result.debugDescription)
            }
            guard let fileWriterURL = URL(string: "/tmp/") else {
                XCTFail("Can't compute folder url.")
                return
            }
            let _ = try FileWriter(folder: fileWriterURL, identifier: "iMac", graph: scope.graph)
        } catch {
            XCTFail(error.localizedDescription)
        }
    }
    
    func testControlDependency() {
        do {
            let scope = Scope()
            let size: Int64 = 9
            
            let input = try scope.placeholder(dtype: Float.self, shape: Shape.dimensions(value: [size]))
            let x = try scope.variableV2(shape: Shape.dimensions(value: [size]), dtype: Float.self, container: "", sharedName: "")
            
            let initAssign = try scope.assign(ref: x,
                                              value: try scope.addConst(values: Array<Float>(repeating: -2.0, count: Int(size)), dimensions: [size], as: "Zero").defaultOutput,
                                              validateShape: true,
                                              useLocking: true)
            
            let sub = try scope.sub(x: input, y: x)

            let _ = try scope.with(controlDependencies: [sub.operation], scopeClosure: { (scope) -> Output in
                return try scope.assign(operationName: "diffAssign", ref: x, value: input, validateShape: true, useLocking: true)
            })
            let session = try Session(graph: scope.graph)
            let _ = try session.run(inputs: [], values: [], outputs: [], targetOperations: [initAssign.operation])
            
            for i in 0..<5 {
                let tensor = try Tensor(dimensions: [size], values: Array<Float>(repeating: Float(i * 2), count: Int(size)))
                let out = try session.run(inputNames: ["Placeholder"], inputs: [tensor], outputNames: ["Sub"], targetOperationsNames: ["diffAssign"])
                let result: [Float] = try out.outputs[0].pullCollection()
                let a = result.reduce(0, +) / Float(size)
                XCTAssert(a == 2.0, "Incorrect result.")
            }
            
            guard let fileWriterURL = URL(string: "/tmp/\(#function)/") else {
                XCTFail("Can't compute folder url.")
                return
            }
            let _ = try FileWriter(folder: fileWriterURL, identifier: "iMac", graph: scope.graph)
        } catch {
            XCTFail(error.localizedDescription)
        }
    }

    func testGraph1RestorePb() {
        do {
            let scope = Scope()
            
            guard let pbURL = URL(string: "file:///tmp/graph.pb") else {
                XCTFail("Can't compute folder url.")
                return
            }

            try scope.graph.import(from: pbURL, prefix: "")
            
            XCTAssert(scope.graph.operations.count > 0, "Incorrect list of operations.")

            guard let fileWriterURL = URL(string: "/tmp/") else {
                XCTFail("Can't compute folder url.")
                return
            }
            
            let _ = try FileWriter(folder: fileWriterURL, identifier: "iMac", graph: scope.graph)
            
            XCTAssert(scope.graph.operations.count > 0, "Incorrect list of operations.")
        } catch {
            XCTFail(error.localizedDescription)
        }
    }
    
    func testGraph1RestorePbTxt() {
        do {
            let scope = Scope()
            
            guard let pbtxtURL = URL(string: "file:///tmp/graph.pbtxt") else {
                XCTFail("Can't compute folder url.")
                return
            }
            
            try scope.graph.import(from: pbtxtURL, prefix: "", asText: true)
            
            XCTAssert(scope.graph.operations.count > 0, "Incorrect list of operations.")
            
            guard let fileWriterURL = URL(string: "/tmp/") else {
                XCTFail("Can't compute folder url.")
                return
            }
            
            let _ = try FileWriter(folder: fileWriterURL, identifier: "iMac", graph: scope.graph)
            
            XCTAssert(scope.graph.operations.count > 0, "Incorrect list of operations.")
        } catch {
            XCTFail(error.localizedDescription)
        }
    }
    
    func testSaveModelRestore() {
        do {
            let dataPath = "https://storage.googleapis.com/api.octadero.com/tests/data/checkpoint.ckpt-10.data-00000-of-00001"
            let indexPath = "https://storage.googleapis.com/api.octadero.com/tests/data/checkpoint.ckpt-10.index"
            let metaPath = "https://storage.googleapis.com/api.octadero.com/tests/data/checkpoint.ckpt-10.meta"
            guard let dataURL = URL(string: dataPath) else { XCTFail("Can't compute url"); return }
            guard let indexURL = URL(string: indexPath) else { XCTFail("Can't compute url"); return }
            guard let metaURL = URL(string: metaPath) else { XCTFail("Can't compute url"); return }
            
            let tmp = "/tmp/" + UUID().uuidString + "/"
            try FileManager.default.createDirectory(atPath:tmp, withIntermediateDirectories: true, attributes: nil)

            
            let metaData = try Data(contentsOf: metaURL)
            let dataData = try Data(contentsOf: dataURL)
            let dataIndex = try Data(contentsOf: indexURL)
            
            try metaData.write(to: URL(fileURLWithPath: tmp + "checkpoint.ckpt-10.meta"))
            try dataData.write(to: URL(fileURLWithPath: tmp + "checkpoint.ckpt-10.data-00000-of-00001"))
            try dataIndex.write(to: URL(fileURLWithPath: tmp + "checkpoint.ckpt-10.index"))
            
            let scope = Scope()
            let savedModel = try SavedModel.restore(into: scope, exportPath: URL(string:tmp)!, checkpoint: "checkpoint.ckpt-10")
            guard !scope.graph.operations.isEmpty else {
                XCTFail("graph operations can't be empty.")
                return
            }
            let value = try savedModel.session.run(runOptions: nil,
                                                   inputNames: [],
                                                   inputs: [],
                                                   outputNames: ["layer_one/W1", "layer_two/W2"],
                                                   targetOperationsNames: [])
            
            let w1: [Float] = try value.outputs[0].pullCollection()
            let w2: [Float] = try value.outputs[1].pullCollection()
            
            guard !w1.isEmpty && !w2.isEmpty else {
                XCTFail("Restored variables can't be empty.")
                return
            }
        } catch {
            XCTFail(error.localizedDescription)
        }
    }
    
    func testStringTensorAPI() {
        do {
            let _ = Scope()
            let tensor = try Tensor(dimensions: [Int64(3), Int64(2)], values: ["layer_one/W1_1", "layer_two/W2_2222", "layer_tree/W3_333333333", "111", "222", "333"])
            let strings: [String] = try tensor.pullCollection()
            if strings.count != 6 {
                XCTFail("Incorrect size.")
            }
            
            var t = Tensorflow_TensorProto()
            t.stringVal = ["first".data(using: .ascii)!, "second".data(using: .ascii)!]
            print(t.textFormatString())
            print("\n")
            print(try t.jsonString())
            
            var t2 = Tensorflow_TensorProto()
            t2.floatVal = [1.0, 2.0, 3.0]
            print(t2.textFormatString())
            print("\n")
            print(try t2.jsonString())
            
        } catch {
            XCTFail(error.localizedDescription)
        }
    }
    
    
    func testZ0AddSaveRestoreOperation() {
        let scope = Scope()
        do {
            // Create Graph
            let w = try scope.variableV2(operationName: "W", shape: .dimensions(value: [3, 1]), dtype: Float.self, container: "", sharedName: "")
            let wInit = try scope.assign(operationName: "initW",
                                         ref: w,
                                         value: scope.addConst(values: Array<Float>([1.0, 1.0, 1.0]), dimensions: Array<Int64>([3, 1]), as: "Const/Const").defaultOutput,
                                         validateShape: true,
                                         useLocking: true)
            
            let x = try scope.placeholder(operationName: "x", dtype: Float.self, shape: .unknown)
            let y = try scope.placeholder(operationName: "y", dtype: Float.self, shape: .unknown)
            let output = try scope.matMul(operationName: "output", a: x, b: w, transposeA: false, transposeB: false)
            let z = try scope.sub(operationName: "z", x: y, y: output)
            
            let loss = try scope.matMul(operationName: "loss", a: z, b: z, transposeA: true, transposeB: false)
            
            let gradientsOutputs = try scope.addGradients(yOutputs: [loss], xOutputs: [w])
            
            guard let gradientsOutput = gradientsOutputs.first else {
                fatalError("gradOutputs is empty")
            }
            
            let lossTensor = try Tensor(scalar: Float(0.05))
            
            let _ = try scope.applyGradientDescent(operationName: "ApplyGD",
                                                   `var`: w,
                                                   alpha: scope.addConst(tensor: lossTensor, as: "Const_1/Const").defaultOutput,
                                                   delta: gradientsOutput,
                                                   useLocking: false)
            
            // Init
            let session = try Session(graph: scope.graph, sessionOptions: SessionOptions())
            
            if let _ = try scope.graph.operation(by: "initW") {
                let initResult: [Tensor] = try session.run(inputs: [], values: [], outputs: [], targetOperations: [wInit.operation])
                print(initResult)
            }

            
            let saver = try SavedModel(session: session, graph: scope.graph, exportPath: "/tmp/load_restore/exportPath/")
            
            // Save and visualize Graph
            try FileManager.default.createDirectory(at: URL(fileURLWithPath: "/tmp/save_restore_test/"), withIntermediateDirectories: true, attributes: nil)
            try scope.graph.save(at: URL(fileURLWithPath: "/tmp/save_restore_test/graph.pbtxt"), asText: true)
            
            guard let writerURL = URL(string: "/tmp/save_restore_test/") else {
                XCTFail("Can't compute folder url.")
                return
            }
            let _ = try FileWriter(folder: writerURL, identifier: "iMac", graph: scope.graph)
            
            
            // Training
            let xs = Array<Float>([1.0, -1.0, 3.0,  1.0, 2.0, 1.0,  1.0, -2.0, -2.0, 1.0, 0.0, 2.0])
            let ys = Array<Float>([14.0, 15.0, -9.0, 13.0])
            let xTensor0 = try Tensor(dimensions: [4, 3], values: xs)
            let yTensor0 = try Tensor(dimensions: [ys.count, 1], values: ys)
            
            for index in 0..<101 {
                
                let resultOutput = try session.run(runOptions: nil,
                                                   inputNames: ["x", "y"],
                                                   inputs: [xTensor0, yTensor0],
                                                   outputNames: ["loss", "ApplyGD"],
                                                   targetOperationsNames: [])
                if index % 10 == 0 {
                let lossError: [Float] = try resultOutput.outputs[0].pullCollection()
                    print("index:\(index) loss: \(String(describing: lossError.first))")
                }
            }
            try saver.save()
        } catch {
            XCTFail(String(describing: error))
        }
    }
    
    func testZ1LoadSavedModel() {
        do {
            let scope = Scope()
            let savedModel = try SavedModel.load(into: scope, exportPath: "/tmp/load_restore/exportPath/", tags: ["serve"], options: SessionOptions())
            
            // Continue Training
            let xs = Array<Float>([1.0, -1.0, 3.0,  1.0, 2.0, 1.0,  1.0, -2.0, -2.0, 1.0, 0.0, 2.0])
            let ys = Array<Float>([14.0, 15.0, -9.0, 13.0])
            let xTensor0 = try Tensor(dimensions: [4, 3], values: xs)
            let yTensor0 = try Tensor(dimensions: [ys.count, 1], values: ys)
            
            var loss : Float = 1.0
            
            for index in 0..<10 {
                let resultOutput = try savedModel.session.run(runOptions: nil,
                                                              inputNames: ["x", "y"],
                                                              inputs: [xTensor0, yTensor0],
                                                              outputNames: ["loss", "ApplyGD"],
                                                              targetOperationsNames: [])
                
                if index > 1 {
                    let lossError: [Float] = try resultOutput.outputs[0].pullCollection()
                    loss = lossError.first!
                }
            }
            if loss >= 1.0 {
                XCTFail("Restored model is not pre trained. los: \(loss)")
            }
            
        } catch {
            XCTFail(String(describing: error))
        }
    }
    
    func testLoadPreTrainadModel() {
        do {
            let path = "https://github.com/tensorflow/tensorflow/raw/master/tensorflow/cc/saved_model/testdata/half_plus_two/00000123/"
            
            guard let savedModelURL = URL(string: path + "saved_model.pb") else { XCTFail("Incorrect url"); return }
            guard let dataURL = URL(string: path + "variables/variables.data-00000-of-00001") else { XCTFail("Incorrect url"); return }
            guard let indexURL = URL(string: path + "variables/variables.index") else { XCTFail("Incorrect url"); return }
            guard let assetsURL = URL(string: path + "assets/foo.txt") else { XCTFail("Incorrect url"); return }

            let dataSavedModel = try Data(contentsOf: savedModelURL)
            let dataData = try Data(contentsOf: dataURL)
            let dataIndex = try Data(contentsOf: indexURL)
            let dataAssets = try Data(contentsOf: assetsURL)

            try FileManager.default.createDirectory(atPath: "/tmp/variables/", withIntermediateDirectories: true, attributes: nil)
            try FileManager.default.createDirectory(atPath: "/tmp/assets/", withIntermediateDirectories: true, attributes: nil)
            
            try dataSavedModel.write(to: URL(fileURLWithPath: "/tmp/saved_model.pb"))
            try dataData.write(to: URL(fileURLWithPath: "/tmp/variables/variables.data-00000-of-00001"))
            try dataIndex.write(to: URL(fileURLWithPath: "/tmp/variables/variables.index"))
            try dataAssets.write(to: URL(fileURLWithPath: "/tmp/assets/foo.txt"))

            let scope = Scope()
            let savedModel = try SavedModel.load(into: scope, exportPath: "/tmp/", tags: ["serve"], options: SessionOptions())
            guard !scope.graph.operations.isEmpty else {
                XCTFail("graph operations can't be empty.")
                return
            }
            
            let value = try savedModel.session.run(runOptions: nil,
                                                   inputNames: [],
                                                   inputs: [],
                                                   outputNames: ["a", "b"],
                                                   targetOperationsNames: [])
            
            let a: [Float] = try value.outputs[0].pullCollection()
            let b: [Float] = try value.outputs[1].pullCollection()
            
            guard let writerURL = URL(string: "/tmp/pattern/") else {
                XCTFail("Can't compute folder url.")
                return
            }
            
            let _ = try FileWriter(folder: writerURL, identifier: "iMac", graph: scope.graph)
            
            guard !a.isEmpty && !b.isEmpty else {
                XCTFail("Restored variables can't be empty.")
                return
            }
            
        } catch {
            XCTFail(error.localizedDescription)
        }
    }
    
    static var allTests = [
        ("testControlDependency", testControlDependency),
        ("testDiffImage", testDiffImage),
        ("testScope", testScope),
        ("testGraph", testGraph),
        ("testComputedGraph", testComputedGraph),
        ("testComputedGraphByNames", testComputedGraphByNames),
        ("testSimpleSession", testSimpleSession),
        ("testCreateScopeGraphConstsFunction", testCreateScopeGraphConstsFunction),
        ("testDeviceList", testDeviceList),
        ("testEventSummury", testEventSummury),
        ("testEventWriter", testEventWriter),
        ("testStringTensor", testStringTensor),
        ("testGraphEnumirator", testGraphEnumirator),
        ("testImageSummary", testImageSummary),
        ("testTensorTransformation", testTensorTransformation),
        ("testGraph0Save", testGraph0Save),
        ("testGraph1RestorePb", testGraph1RestorePb),
        ("testGraph1RestorePbTxt", testGraph1RestorePbTxt),
        ("testSaveModelRestore", testSaveModelRestore),
        ("testStringTensorAPI", testStringTensorAPI),
        ("testZ0AddSaveRestoreOperation", testZ0AddSaveRestoreOperation),
        ("testZ1LoadSavedModel", testZ1LoadSavedModel),
        ("testLoadPreTrainadModel", testLoadPreTrainadModel)
    ]
}
