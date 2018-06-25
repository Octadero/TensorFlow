/* Copyright 2017 The Octadero Authors. All Rights Reserved.
 Created by Volodymyr Pavliukevych on 2017.
 
 Licensed under the GPL License, Version 3.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 
 http://www.gnu.org/licenses/gpl-3.0.txt
 
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */

import CAPI
import Proto
import CTensorFlow

extension Scope {
    ///Outputs random values from a truncated normal distribution.
    /// The generated values follow a normal distribution with specified mean and standard deviation,
    /// except that values whose magnitude is more than 2 standard deviations
    /// from the mean are dropped and re-picked.
    /// See the guide: [Constants, Sequences, and Random Values > Random Tensors](https://www.tensorflow.org/api_guides/python/constant_op#Random_Tensors)
    public func truncatedNormal(name: String, shape: Shape, stddev stddevValue: Float, mean meanValue: Float = 0) throws -> Output {
        let scope = self.subScope(namespace: name)
        
        var shapeValue = [Int32]()
        
        switch shape {
        case .dimensions(let value):
            shapeValue = value.map { Int32($0) }
            break
        default:
            break
        }
        
        let shapeTensor = try Tensor(dimensions: [2], values: Array<Int32>(shapeValue))
        let stddev = try scope.addConst(tensor: try Tensor(scalar: stddevValue), as: "stddev").defaultOutput
        let mean = try scope.addConst(tensor: try Tensor(scalar: meanValue), as: "mean").defaultOutput
        
        let truncatedNormal = try scope.truncatedNormal(operationName: "TruncatedNormal",
                                                        shape: scope.addConst(tensor: shapeTensor, as: "shape").defaultOutput,
                                                        seed: 0,
                                                        seed2: 0,
                                                        dtype: Float.self)
        
        let mul = try scope.mul(operationName: "mul", x: truncatedNormal, y: stddev)
        return try scope.add(operationName: "(truncated_normal)", x: mul, y: mean)
    }
	/// Adds operations to compute the partial derivatives of sum of `y`s w.r.t `x`s,
	/// i.e., d(y_1 + y_2 + ...)/dx_1, d(y_1 + y_2 + ...)/dx_2...
	/// `dx` are used as initial gradients (which represent the symbolic partial
	/// derivatives of some loss function `L` w.r.t. `y`).
	/// `dx` must be nullptr or have size `ny`.
	/// If `dx` is nullptr, the implementation will use dx of `OnesLike` for all
	/// shapes in `y`.
	/// The partial derivatives are returned in `dy`. `dy` should be allocated to
	/// size `nx`.
	///
	/// WARNING: This function does not yet support all the gradients that python
	/// supports. See
	/// https://www.tensorflow.org/code/tensorflow/cc/gradients/README.md
	/// for instructions on how to add C++ more gradients.
	public func addGradients(yOutputs: [Output], xOutputs: [Output]) throws -> [Output] {
		
		let tfOutputs = try CAPI.addGradients(graph: self.graph.tfGraph,
		                                      yOutputs: yOutputs.map { $0.tfOutput()},
		                                      xOutputs: xOutputs.map { $0.tfOutput() })
		
		return tfOutputs.map({ (tfOutput) -> Output in
			let operation = TensorFlowKit.Operation(tfOperation: tfOutput.oper, graph: self.graph)
			return Output(in: operation, at: Int(tfOutput.index))
		})
	}
	
	/// Add const `Tensor` to scope
	public func addConst(tensor: Tensor, `as` name: String) throws -> TensorFlowKit.Operation {
		var attrs = [String : Any]()
		attrs["value"] = tensor
		attrs["dtype"] = tensor.dtType
		let specification = OpSpec(type: "Const",
		                           name: name,
		                           input: [ ],
		                           attrs: attrs)
		
		return try addOperation(specification: specification)
	}
	
    /// Add const array of `String` to scope
    public func addConst(strings: [String], `as` name: String) throws -> TensorFlowKit.Operation {
        var tensorProto = Tensorflow_TensorProto()
        tensorProto.stringVal = strings.compactMap { $0.data(using: .utf8) }
        tensorProto.dtype = Tensorflow_DataType.dtString
        
        var shape = Tensorflow_TensorShapeProto()
        var dim = Tensorflow_TensorShapeProto.Dim()

        dim.size = Int64(strings.count)
        shape.dim = [dim]
        tensorProto.tensorShape = shape
        var attrValue = Tensorflow_AttrValue()
        attrValue.tensor = tensorProto
        
        var attrs = [String : Any]()
        attrs["value"] = attrValue
        attrs["dtype"] = TF_STRING
        let specification = OpSpec(type: "Const",
                                   name: name,
                                   input: [ ],
                                   attrs: attrs)
        
        return try addOperation(specification: specification)
    }
    
	/// Add const from value.
	public func addConst<T: Value>(values: [T], dimensions: [Int64], `as` name: String) throws -> TensorFlowKit.Operation {
		let tensor = try Tensor(dimensions: dimensions, values: values)
		return try addConst(tensor: tensor, as: name)
	}
}
