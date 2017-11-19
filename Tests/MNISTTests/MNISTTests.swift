import XCTest

import CAPI
import Proto
import CTensorFlow
import TensorFlowKit
import MNISTKit

typealias ActivationFunction = (_ scope: Scope, _ name: String, _ output: Output) throws -> Output

class MNISTTests: XCTestCase {
    var dataset: MNISTDataset?
    
    enum MNISTTestsError: Error {
        case operationNotFound(name: String)
        case datasetNotReady
    }
    
    func loadDataset(callback: @escaping (_ error: Error?) -> Void) {
        if dataset == nil {
            dataset = MNISTDataset(callback: callback)
        }
    }
    
    //MARK: - weight and biases
    /// Create a bias variable with appropriate initialization.
    func biasVariable(at scope: Scope, name: String, shape: Shape) throws -> (output: Output, variable: Output) {
        let scope = scope.subScope(namespace: name)
        
        let biasConst = try scope.addConst(tensor: Tensor(shape: shape, values: Array<Float>(repeating: 0.0001, count: Int(shape.elements ?? 0))), as: "zero").defaultOutput
        let bias = try scope.variableV2(operationName: "Variable", shape: shape, dtype: Float.self, container: "", sharedName: "")
        let _ = try scope.assign(operationName: "Assign", ref: bias, value: biasConst, validateShape: true, useLocking: true)
        let read = try scope.identity(operationName: "read", input: bias)
        return (read, bias)
    }
    
    /// We can't initialize these variables to 0 - the network will get stuck.
    /// Create a weight variable with appropriate initialization.
    func weightVariable(at scope: Scope, name: String, shape: Shape) throws -> (output: Output, variable: Output) {
        let scope = scope.subScope(namespace: name)
        
        let zeros = try scope.addConst(tensor: Tensor(shape: shape, values: Array<Float>(repeating: 0.0001, count: Int(shape.elements ?? 0))), as: "zero").defaultOutput
        let weight = try scope.variableV2(operationName: "Variable", shape: shape, dtype: Float.self, container: "", sharedName: "")
        let _ = try scope.assign(operationName: "Assign", ref: weight, value: zeros, validateShape: true, useLocking: true)
        let read = try scope.identity(operationName: "read", input: weight)
        return (read, weight)
    }
    
    //MARK: - Neuron (W * x) + b
    func neuron(at scope: Scope, name: String, x: Output, w: Output, bias: Output) throws -> Output {
        let scope = scope.subScope(namespace: name)
        let matMult = try scope.matMul(operationName: "MatMul", a: x, b: w, transposeA: false, transposeB: false)
        let preactivate = try scope.add(operationName: "add", x: matMult, y: bias)
        return preactivate
    }
    
    /// Main build graph function.
    func buildGraph() throws -> Scope {
        let scope = Scope()

        //MARK: Input sub scope
        let inputScope = scope.subScope(namespace: "input")
        let x = try inputScope.placeholder(operationName: "x-input", dtype: Float.self, shape: Shape.dimensions(value: [-1, 784]))
        let yLabels = try inputScope.placeholder(operationName: "y-input", dtype: Float.self, shape: Shape.dimensions(value: [-1, 10]))
        
        let weights = try weightVariable(at: scope, name: "weights", shape: Shape.dimensions(value: [784, 10]))
        let bias = try biasVariable(at: scope, name: "biases", shape: Shape.dimensions(value: [10]))
        
        let neuron = try self.neuron(at: scope, name: "layer", x: x, w: weights.output, bias: bias.output)
        let softmax = try scope.softmax(operationName: "Softmax", logits: neuron)
        
        let log = try scope.log(operationName: "Log", x: softmax)
        let mul = try scope.mul(operationName: "Mul", x: yLabels, y: log)
        let reductionIndices = try scope.addConst(tensor: Tensor(dimensions: [1], values: [Int(1)]), as: "reduction_indices").defaultOutput
        let sum = try scope.sum(operationName: "Sum", input: mul, reductionIndices: reductionIndices, keepDims: false, tidx: Int32.self)
        let neg = try scope.neg(operationName: "Neg", x: sum)

        let meanReductionIndices = try scope.addConst(tensor: Tensor(dimensions: [1], values: [Int(0)]), as: "mean_reduction_indices").defaultOutput
        let cross_entropy = try scope.mean(operationName: "Mean", input: neg, reductionIndices: meanReductionIndices, keepDims: false, tidx: Int32.self)
        
        let gradientsOutputs = try scope.addGradients(yOutputs: [cross_entropy], xOutputs: [weights.variable, bias.variable])
        
        let learningRate = try scope.addConst(tensor: try Tensor(scalar: Float(0.05)), as: "learningRate").defaultOutput
        
        let _ = try scope.applyGradientDescent(operationName: "applyGradientDescent_W",
                                               `var`: weights.variable,
                                               alpha: learningRate,
                                               delta: gradientsOutputs[0],
                                               useLocking: false)
        
        let _ = try scope.applyGradientDescent(operationName: "applyGradientDescent_B",
                                               `var`: bias.variable,
                                               alpha: learningRate,
                                               delta: gradientsOutputs[1],
                                               useLocking: false)
        
        return scope
    }
    
    func learn(scope: Scope) throws {
        let session = try Session(graph: scope.graph, sessionOptions: SessionOptions())
        
        guard let wAssign = try scope.graph.operation(by: "weights/Assign") else { throw MNISTTestsError.operationNotFound(name: "weights/Assign") }
        guard let bAssign = try scope.graph.operation(by: "biases/Assign") else { throw MNISTTestsError.operationNotFound(name: "biases/Assign") }
        let _ = try session.run(inputs: [], values: [], outputs: [], targetOperations: [wAssign, bAssign])
		
        guard let x = try scope.graph.operation(by: "input/x-input")?.defaultOutput else { throw MNISTTestsError.operationNotFound(name: "input/x-input") }
        guard let y = try scope.graph.operation(by: "input/y-input")?.defaultOutput else { throw MNISTTestsError.operationNotFound(name: "input/y-input") }

        guard let loss = try scope.graph.operation(by: "Mean")?.defaultOutput else { throw MNISTTestsError.operationNotFound(name: "Mean") }
        guard let applyGradW = try scope.graph.operation(by: "applyGradientDescent_W")?.defaultOutput else { throw MNISTTestsError.operationNotFound(name:  "applyGradientDescent_W") }

        guard let applyGradB = try scope.graph.operation(by: "applyGradientDescent_B")?.defaultOutput else { throw MNISTTestsError.operationNotFound(name:  "applyGradientDescent_B") }
        
        guard let dataset = dataset else { throw MNISTTestsError.datasetNotReady }
        guard let images = dataset.files(for: .image(stride: .train)).first as? MNISTImagesFile else { throw MNISTTestsError.datasetNotReady }
        guard let labels = dataset.files(for: .label(stride: .train)).first as? MNISTLabelsFile else { throw MNISTTestsError.datasetNotReady }
        
        print("Load dataset ...")
        let bach = 2000
        let steps = 5000
        let xs = images.images[0..<bach].flatMap { $0 }
        
        var ys = [Float]()
        labels.labels[0..<bach].forEach { index in
            var label = Array<Float>(repeating: 0, count: 10)
            label[Int(index)] = 1.0
            ys.append(contentsOf: label)
        }
        
        let xTensorInput = try Tensor(dimensions: [bach, 784], values: xs)
        let yTensorInput = try Tensor(dimensions: [bach, 10], values: ys)
        var lossValueResult: Float = Float(Int.max)
        for index in 0..<steps {

            let resultOutput = try session.run(inputs: [x, y],
                                               values: [xTensorInput, yTensorInput],
                                               outputs: [loss, applyGradW, applyGradB],
                                               targetOperations: [])
            
            if index % 100 == 0 {
                let lossTensor = resultOutput[0]
                let lossValues: [Float] = try lossTensor.pullCollection()
                guard let lossValue = lossValues.first else { continue }
                print("[\(index)] loss: ", lossValue)
                lossValueResult = lossValue
            }
        }
        
        XCTAssert(lossValueResult < 0.2 , "Accuracy value not reached.")
    }
    
    func testMNISTModel() {
        let anExpectation = expectation(description: "TestInvalidatingWithExecution \(#function)")
        
        loadDataset { (error) in
            if let error = error {
                XCTFail(error.localizedDescription)
            }
            anExpectation.fulfill()
        }
        waitForExpectations(timeout: 300) { error in
            XCTAssertNil(error, "Download timeout.")
        }
        print("Dataset is ready, go!")
        
        //MARK: Create a multilayer model.
        do {
            let scope = try buildGraph()
            try learn(scope: scope)
        } catch {
            XCTFail(error.localizedDescription)
        }
        
    }
    
    static var allTests = [
        ("testMNISTModel", testMNISTModel)
    ]
}
