import XCTest

import CAPI
import CCAPI
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
	
	func sub(scope: Scope, x: Output, y: Output, name: String) throws -> Output {
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
	
	func substr( scope:Scope, input: Output, pos: Output, len: Output ) throws -> Output {
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
	
	func tensorASOperation(scope: Scope, tensor: Tensor, name: String) throws -> TensorFlowKit.Operation {
		var attrs = [String : Any]()
		attrs["value"] = tensor
		attrs["dtype"] = tensor.dtType
		let specification = OpSpec(type: "Const", name: name, input: [ ], attrs: attrs)
		return try scope.addOperation(specification: specification)
	}
	
	func testComputedGraph() {
		let scope = Scope()
		do {
			let wValue0: [Double] = [1.0, 1.0, 1.0]
			let wTensor0 = try Tensor(dimensions: [3, 1], values: wValue0)
			let tensorOutput0 = try tensorASOperation(scope: scope, tensor: wTensor0, name: "Const/Const").output(at: 0)
			
			let w = try variableV2Func(scope: scope, shape: .dimensions(value: [3, 1]), container: "", sharedName: "", type: Double.self, name: "W")
			let wInit = try assignFunc(scope: scope,
			                           ref: w,
			                           value: tensorOutput0,
			                           validateShape: true,
			                           useLocking: true,
			                           name: "initW")
			
			
			
			let x = try placeholderFunc(scope: scope, shape: .unknown, name: "x", type: Double.self)
			let y = try placeholderFunc(scope: scope, shape: .unknown, name: "y", type: Double.self)
			
			let output = try matMulFunc(scope: scope,
			                            a: x,
			                            b: w,
			                            transposeA: false,
			                            transposeB: false,
			                            name: "output")
			let z = try sub(scope: scope, x: y, y: output, name: "z")
			
			let loss = try matMulFunc(scope: scope,
			                            a: z,
			                            b: z,
			                            transposeA: true,
			                            transposeB: false,
			                            name: "loss")
			
			
			let lossOutput = loss.tfOutput()
			let wOutput = w.tfOutput()
			let status = Status()
			let fake = TF_Output(oper: nil, index: 0)
			var gradOutputs = Array<TF_Output>(repeating: fake, count: 1)
			
			let l = [lossOutput]
			let ww = [wOutput]
			
			gradOutputs.withUnsafeMutableBufferPointer({ bufferPointer in
				TF_AddGradients(scope.graph.tfGraph,
				                UnsafeMutablePointer(mutating:l),
				                Int32(1),
				                UnsafeMutablePointer(mutating:ww),
				                Int32(1),
				                nil,
				                status.tfStatus,
				                bufferPointer.baseAddress)
			})
            
            
			let lossTensor = try Tensor(dimensions: [], values: [Double(0.5)])
			let lossTensorOutput = try tensorASOperation(scope: scope, tensor: lossTensor, name: "Const_1/Const").output(at: 0)

			
			guard let grad_output: TF_Output = gradOutputs.first else {
				fatalError("gradOutputs is empty")
			}
			
			let operation = try TensorFlowKit.Operation(tfOperation: grad_output.oper, graph: scope.graph)
			
			let apply_grad_W = try applyGradientDescentFunc(scope: scope,
			                                                `var`: w,
			                                                alpha: lossTensorOutput,
			                                                delta: Output(in: operation, at: Int(grad_output.index)),
			                                                useLocking: false,
			                                                name: "ApplyGD")

			
			guard let url = URL(string: "/tmp/current/") else {
				XCTFail("Can't compute folder path")
				return
			}
			
			try scope.save(at: url, fileName: "testComputedGraph", step: 0)
            
			let session = try Session(graph: scope.graph, sessionOptions: SessionOptions())
            
            if let initOperation = try scope.graph.operation(by: "initW") {
                let initResult = try session.run(inputs: [], values: [], outputs: [], targetOperations: [initOperation])
                print(initResult)
            }
            
            let xs = [[1.0, -1.0, 3.0], [1.0, 2.0, 1.0], [1.0, -2.0, -2.0], [1.0, 0.0, 2.0]]
            let ys = [14.0, 15.0, -9.0, 13.0]

            
			/*
            https://stackoverflow.com/questions/44305647/segmentation-fault-when-using-tf-sessionrun-to-run-tensorflow-graph-in-c-not-c
            */

			for _ in 0..<10 {
                let xTensor0 = try Tensor(dimensions: [1, 3], values: xs[0])
                let yTensor0 = try Tensor(scalar: ys[0])
                
                let xTensor1 = try Tensor(dimensions: [1, 3], values: xs[1])
                let yTensor1 = try Tensor(scalar: ys[1])

                let xTensor2 = try Tensor(dimensions: [1, 3], values: xs[2])
                let yTensor2 = try Tensor(scalar: ys[2])

                let xTensor3 = try Tensor(dimensions: [1, 3], values: xs[3])
                let yTensor3 = try Tensor(scalar: ys[3])

                let resultOutput = try session.run(inputs: [x, y, x, y, x, y, x, y],
                                                   values: [xTensor0, yTensor0, xTensor1, yTensor1, xTensor2, yTensor2, xTensor3, yTensor3],
                                                   outputs:  [loss, apply_grad_W],
                                                   targetOperations: [])
                resultOutput.forEach({tensor in
                    do {
                        let collection: [Double] = try tensor.pullCollection()
                        print(collection)
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
		let values0: [Float] = [1.0, 2.0, 3.0, 4.0]
		let values1: [Float] = [2.0, 3.0, 4.0, 5.0]
		
		do {
			let tensor0 = try Tensor(dimensions: [2, 2], values: values0)
			let tensor1 = try Tensor(dimensions: [2, 2], values: values1)
			
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
	
	static var allTests = [
		("testScope", testScope),
		("testGraph", testGraph),
        ("testComputedGraph", testComputedGraph),
        ("testSimpleSession", testSimpleSession),
        ("testCreateScopeGraphConstsFunction", testCreateScopeGraphConstsFunction),
        ("testDeviceList", testDeviceList)
    ]
}
/*
#include "tensorflow/c/c_api.h"

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <string.h>
#include <assert.h>
#include <vector>
#include <algorithm>
#include <iterator>
#include <iostream>


TF_Buffer* read_file(const char* file);

void free_buffer(void* data, size_t length) {
    free(data);
}

static void Deallocator(void* data, size_t length, void* arg) {
    free(data);
    // *reinterpret_cast<bool*>(arg) = true;
}

int main() {
    // Use read_file to get graph_def as TF_Buffer*
    TF_Buffer* graph_def = read_file("tensorflow_model/constant_graph_weights.pb");
    TF_Graph* graph = TF_NewGraph();
    
    // Import graph_def into graph
    TF_Status* status = TF_NewStatus();
    TF_ImportGraphDefOptions* graph_opts = TF_NewImportGraphDefOptions();
    TF_GraphImportGraphDef(graph, graph_def, graph_opts, status);
    if (TF_GetCode(status) != TF_OK) {
        fprintf(stderr, "ERROR: Unable to import graph %s", TF_Message(status));
        return 1;
    }
    else {
        fprintf(stdout, "Successfully imported graph\n");
    }
    
    // Create variables to store the size of the input and output variables
    const int num_bytes_in = 3 * sizeof(float);
    const int num_bytes_out = 9 * sizeof(float);
    
    // Set input dimensions - this should match the dimensionality of the input in
    // the loaded graph, in this case it's three dimensional.
    int64_t in_dims[] = {1, 1, 3};
    int64_t out_dims[] = {1, 9};
    
    // ######################
    // Set up graph inputs
    // ######################
    
    // Create a variable containing your values, in this case the input is a
    // 3-dimensional float
    float values[3] = {-1.04585315e+03,   1.25702492e+02,   1.11165466e+02};
    
    // Create vectors to store graph input operations and input tensors
    std::vector<TF_Output> inputs;
    std::vector<TF_Tensor*> input_values;
    
    // Pass the graph and a string name of your input operation
    // (make sure the operation name is correct)
    TF_Operation* input_op = TF_GraphOperationByName(graph, "lstm_1_input");
    TF_Output input_opout = {input_op, 0};
    inputs.push_back(input_opout);
    
    // Create the input tensor using the dimension (in_dims) and size (num_bytes_in)
    // variables created earlier
    TF_Tensor* input = TF_NewTensor(TF_FLOAT, in_dims, 3, values, num_bytes_in, &Deallocator, 0);
    input_values.push_back(input);
    
    // Optionally, you can check that your input_op and input tensors are correct
    // by using some of the functions provided by the C API.
    std::cout << "Input op info: " << TF_OperationNumOutputs(input_op) << "\n";
    std::cout << "Input data info: " << TF_Dim(input, 0) << "\n";
    
    // ######################
    // Set up graph outputs (similar to setting up graph inputs)
    // ######################
    
    // Create vector to store graph output operations
    std::vector<TF_Output> outputs;
    TF_Operation* output_op = TF_GraphOperationByName(graph, "output_node0");
    TF_Output output_opout = {output_op, 0};
    outputs.push_back(output_opout);
    
    // Create TF_Tensor* vector
    std::vector<TF_Tensor*> output_values(outputs.size(), nullptr);
    
    // Similar to creating the input tensor, however here we don't yet have the
    // output values, so we use TF_AllocateTensor()
    TF_Tensor* output_value = TF_AllocateTensor(TF_FLOAT, out_dims, 2, num_bytes_out);
    output_values.push_back(output_value);
    
    // As with inputs, check the values for the output operation and output tensor
    std::cout << "Output: " << TF_OperationName(output_op) << "\n";
    std::cout << "Output info: " << TF_Dim(output_value, 0) << "\n";
    
    // ######################
    // Run graph
    // ######################
    fprintf(stdout, "Running session...\n");
    TF_SessionOptions* sess_opts = TF_NewSessionOptions();
    TF_Session* session = TF_NewSession(graph, sess_opts, status);
    assert(TF_GetCode(status) == TF_OK);
    
    // Call TF_SessionRun
    TF_SessionRun(session, nullptr,
    &inputs[0], &input_values[0], inputs.size(),
    &outputs[0], &output_values[0], outputs.size(),
    nullptr, 0, nullptr, status);
    
    // Assign the values from the output tensor to a variable and iterate over them
    float* out_vals = static_cast<float*>(TF_TensorData(output_values[0]));
    for (int i = 0; i < 9; ++i)
    {
        std::cout << "Output values info: " << *out_vals++ << "\n";
    }
    
    fprintf(stdout, "Successfully run session\n");
    
    // Delete variables
    TF_CloseSession(session, status);
    TF_DeleteSession(session, status);
    TF_DeleteSessionOptions(sess_opts);
    TF_DeleteImportGraphDefOptions(graph_opts);
    TF_DeleteGraph(graph);
    TF_DeleteStatus(status);
    return 0;
}

TF_Buffer* read_file(const char* file) {
    FILE *f = fopen(file, "rb");
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);  //same as rewind(f);
    
    void* data = malloc(fsize);
    fread(data, fsize, 1, f);
    fclose(f);
    
    TF_Buffer* buf = TF_NewBuffer();
    buf->data = data;
    buf->length = fsize;
    buf->data_deallocator = free_buffer;
    return buf;
}
*/

