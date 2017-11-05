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
}
