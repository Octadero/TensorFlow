import XCTest

import CAPI
import Proto
import CTensorFlow
import TensorFlowKit
import MNISTKit

typealias ActivationFunction = (_ scope: Scope, _ name: String, _ output: Output) throws -> Output

/// Port from https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py
class MNISTTests: XCTestCase {
    var dataset: MNISTDataset?
    
    func dumpGraph(graph: Graph) throws {
        guard let writerURL = URL(string: "/tmp/tensorflow/mnist/logs/com.octadero.tensorflowkit/") else {
            XCTFail("Can't compute folder url.")
            return
        }
        
        let logger = try EventWriter(folder: writerURL, identifier: "iMac")
        try logger.track(graph: graph, time: Date().timeIntervalSince1970, step: 1)
        try logger.flush()
    }
    
    func loadDataset(callback: @escaping (_ error: Error?) -> Void) {
        if dataset == nil {
            dataset = MNISTDataset(callback: callback)
        }
    }
    
    //MARK: - weight and biases
    /// Create a bias variable with appropriate initialization.
    func biasVariable(at scope: Scope, name: String, shape: Shape) throws -> Output {
        let scope = scope.subScope(namespace: name)
        let variableScope = scope.subScope(namespace: "Variable")
        // IT SHOULD BE "float_val":0.10000000149011612
        let biasConst = try scope.addConst(tensor: Tensor(shape: shape, values: [Float(0.1)]), as: "Const").defaultOutput
        let bias = try variableScope.variableV2(operationName: "(Variable)", shape: shape, dtype: Float.self, container: "", sharedName: "")
        let _ = try variableScope.assign(operationName: "Assign", ref: bias, value: biasConst, validateShape: true, useLocking: true)
        let read = try variableScope.identity(operationName: "read", input: bias)
        return read
    }
    
    /// We can't initialize these variables to 0 - the network will get stuck.
    /// Create a weight variable with appropriate initialization.
    func weightVariable(at scope: Scope, name: String, shape: Shape) throws -> Output {
        let scope = scope.subScope(namespace: name)
        
        let truncatedNormal = try scope.truncatedNormal(name: "truncated_normal", shape: shape, stddev: 0.1)
        let variableScope = scope.subScope(namespace: "Variable")
        let weight = try variableScope.variableV2(operationName: "(Variable)", shape: shape, dtype: Float.self, container: "", sharedName: "")
        let _ = try variableScope.assign(operationName: "Assign", ref: weight, value: truncatedNormal, validateShape: true, useLocking: true)
        let read = try variableScope.identity(operationName: "read", input: weight)
        return read
    }
    
    //MARK: - Activation functions
    func relu(at scope: Scope, name: String, output: Output) throws -> Output {
        return try scope.relu(operationName: name, features: output)
    }

    func identity(at scope: Scope, name: String, output: Output) throws -> Output {
        return try scope.identity(operationName: name, input: output)
    }
    
    //MARK: - Neuron (W * x) + b
    func neuron(at scope: Scope, name: String, x: Output, w: Output, bias: Output) throws -> Output {
        let scope = scope.subScope(namespace: name)
        let matMult = try scope.matMul(operationName: "MatMul", a: x, b: w, transposeA: false, transposeB: false)
        let preactivate = try scope.add(operationName: "add", x: matMult, y: bias)
        return preactivate
    }
    
    func randomUniform(at scope: Scope, name: String, inputShape: Output) throws -> Output {
        let scope = scope.subScope(namespace: name)
        let randomUniform = try scope.randomUniform(operationName: "RandomUniform", shape: inputShape, seed: 0, seed2: 0, dtype: Float.self)
        
        let maxTensor = try Tensor(scalar: Float(1))
        let minTensor = try Tensor(scalar: Float(0))
        
        let max = try scope.addConst(tensor: maxTensor, as: "max").defaultOutput
        let min = try scope.addConst(tensor: minTensor, as: "min").defaultOutput
        let sub = try scope.sub(operationName: "sub", x: max, y: min)
        let mul = try scope.mul(operationName: "mul", x: randomUniform, y: sub)
        
        let add = try scope.add(operationName: "(random_uniform)", x: mul, y: min)
        return add
    }
    
    /// Dopout feature
    func dropout(at scope: Scope, name: String, input: Output) throws -> Output {
        let scope = scope.subScope(namespace: name)
        
        let placeholder = try scope.placeholder(operationName: "Placeholder", dtype: Float.self, shape: Shape.unknown)
        let shapeOpration = try scope.shape(operationName: "Shape", input: input, outType: Int32.self)
        let randomUniform = try self.randomUniform(at: scope, name: "random_uniform", inputShape: shapeOpration)
        let add = try scope.add(operationName: "add", x: randomUniform, y: placeholder)
        let floor = try scope.floor(operationName: "Floor", x: add)
        let realDiv = try scope.realDiv(operationName: "div", x: placeholder, y: input)
        let mul = try scope.mul(operationName: "mul", x: realDiv, y: floor)
        return mul
    }
    
    func reduceMean(at scope: Scope, name: String, input: Output) throws {
        let scope = scope.subScope(namespace: name)
        let reductionIndices = try scope.addConst(tensor: Tensor(dimensions: [1], values: [Int(0)]), as: "Const").defaultOutput
        let _ = try scope.mean(operationName: "Mean", input: input, reductionIndices: reductionIndices, keepDims: false, tidx: Int32.self)
    }
    
    func slice(at scope: Scope, name: String, input: Output, size: Int, inversion: Bool) throws -> Output {
        let scope = scope.subScope(namespace: name)

        let y = try scope.addConst(tensor: Tensor(scalar: Int(1)), as: "y").defaultOutput
        let rank = try scope.addConst(tensor: Tensor(scalar: Int(2)), as: "Rank").defaultOutput
        let sub = try scope.sub(operationName: "Sub", x: rank, y: y)
        let shapeOperation = try scope.shape(operationName: "Shape", input: input, outType: Int32.self)
        let brgin = try scope.pack(operationName: "begin", values: [sub], n: 1, axis: 0)
        
        let size = try scope.addConst(tensor: Tensor(dimensions: [1], values: [size]), as: "size").defaultOutput
        
        if inversion == true {
            return try scope.slice(operationName: "(Slice)", input: shapeOperation, begin: size, size: brgin, index: Int32.self)
        }
        let slice = try scope.slice(operationName: "(Slice)", input: shapeOperation, begin: brgin, size: size, index: Int32.self)
        return slice
    }
    
    func reshape(at scope: Scope, name: String, slice: Output, input: Output) throws -> Output {
        let scope = scope.subScope(namespace: name)
        
        let axis = try scope.addConst(tensor: Tensor(scalar: Int(0)), as: "axis").defaultOutput
        let values0 = try scope.addConst(tensor: Tensor(dimensions: [1], values: [Int(-1)]), as: "values_0").defaultOutput
        let concat = try scope.concatV2(operationName: "concat", values: [values0, slice], axis: axis, n: 2, tidx: Int32.self)
        
        let reshape = try scope.reshape(operationName: name, tensor: input, shape: concat, tshape: Int32.self)
        return reshape
    }
    
    /// Cross entropy
    /// The raw formulation of cross-entropy,
    ///
    /// tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
    ///                               reduction_indices=[1]))
    ///
    /// can be numerically unstable.
    ///
    /// So here we use tf.nn.softmax_cross_entropy_with_logits on the
    /// raw outputs of the nn_layer above, and then average across
    /// the batch.
    func crossEntropy(at scope: Scope, name: String, features: Output, label: Output) throws {
        let scope = scope.subScope(namespace: name)
        
        let slice1 = try slice(at: scope, name: "Slice1", input: features, size: 1, inversion: false)
        let reshape1 = try reshape(at: scope, name: "Reshape1", slice: slice1, input: features)
        
        let slice2 = try slice(at: scope, name: "Slice2", input: label, size: 1, inversion: false)
        let reshape2 = try reshape(at: scope, name: "Reshape2", slice: slice2, input: label)

        let diff = try scope.softmaxCrossEntropyWithLogits(operationName: "SoftmaxCrossEntropyWithLogits", features: reshape1, labels: reshape2)

        let slice3 = try slice(at: scope, name: "Slice3", input: features, size: 0, inversion: true)
        let reshape3 = try scope.reshape(operationName: "Reshape3", tensor: diff.loss, shape: slice3, tshape: Int32.self)

        try self.reduceMean(at: scope, name: "total", input: reshape3)        
    }
    
    /// Reusable code for making a simple neural net layer.
    ///     It does a matrix multiply, bias add, and then uses ReLU to nonlinearize.
    ///     It also sets up name scoping so that the resultant graph is easy to read,
    ///     and adds a number of summary ops.
    func nnLayer(name: String, at scope: Scope, inputTensor: Output, inputDim: Int64, outputDim: Int64, activation: ActivationFunction) throws -> Output {
        //MARK: Adding a name scope ensures logical grouping of the layers in the graph.
        let layerScope = scope.subScope(namespace: name)
        
        // This Variable will hold the state of the weights for the layer
        let weights = try weightVariable(at: layerScope, name: "weights", shape: Shape.dimensions(value: [inputDim, outputDim]))
        let bias = try biasVariable(at: layerScope, name: "biases", shape: Shape.dimensions(value: [outputDim]))
        let preactivate = try neuron(at: layerScope, name: "Wx_plus_b", x: inputTensor, w: weights, bias: bias)
        let activation = try activation(layerScope, "activation", preactivate)
        return activation
    }
    
    func train(name: String, at scope: Scope) throws {
        let scope = scope.subScope(namespace: name)

    }
    
    /// Main build graph function.
    func buildGraph() throws -> Scope {
        let scope = Scope()
        
        //MARK: Input sub scope
        let inputScope = scope.subScope(namespace: "input")
        let x = try inputScope.placeholder(operationName: "x-input", dtype: Float.self, shape: Shape.dimensions(value: [-1, 784]))
        let yLabels = try inputScope.placeholder(operationName: "y-input", dtype: Float.self, shape: Shape.dimensions(value: [-1, 10]))
        
        let relu = {(scope: Scope, name: String, output: Output) throws -> Output in
            return try self.relu(at: scope, name: name, output: output)
        }
        
        let hidden1 = try nnLayer(name: "layer1", at: scope, inputTensor: x, inputDim: 784, outputDim: 500, activation: relu)
        let dropout = try self.dropout(at: scope, name: "dropout", input: hidden1)
        
        ///Do not apply softmax activation yet, see below.
        let identity = {(scope: Scope, name: String, output: Output) throws -> Output in
            return try self.identity(at: scope, name: name, output: output)
        }
        
        let yFeatures = try nnLayer(name: "layer2", at: scope, inputTensor: dropout, inputDim: 500, outputDim: 10, activation: identity)
        try crossEntropy(at: scope, name: "cross_entropy", features: yFeatures, label: yLabels)
        
        let train = try self.train(name: "train", at: scope)
        
        return scope
    }
    
    func testMNISTModel() {
        let anExpectation = expectation(description: "TestInvalidatingWithExecution \(#function)")
        
        loadDataset { (error) in
            if let error = error {
                XCTFail(error.localizedDescription)
            }
            anExpectation.fulfill()
        }
        waitForExpectations(timeout: 30) { error in
            XCTAssertNil(error, "Download timeout.")
        }
        print("Dataset is ready, go!")

        //MARK: Create a multilayer model.
        do {
            let scope = try buildGraph()
            try dumpGraph(graph: scope.graph)
        } catch {
            XCTFail(error.localizedDescription)
        }
        
    }
    
	static var allTests = [
        ("testMNISTModel", testMNISTModel)
    ]
}
