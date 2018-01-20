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

import CTensorFlow
import Foundation
import Proto

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
public func addGradients(graph: TF_Graph, yOutputs: [TF_Output], xOutputs: [TF_Output]) throws -> [TF_Output] {
    let status = newStatus()
    defer {
        delete(status: status)
    }
    let dyOutputPointer = UnsafeMutablePointer<TF_Output>.allocate(capacity: xOutputs.count)
    
    TF_AddGradients(graph,
                    UnsafeMutablePointer(mutating:yOutputs),
                    Int32(yOutputs.count),
                    UnsafeMutablePointer(mutating:xOutputs),
                    Int32(xOutputs.count),
                    nil,
                    status,
                    dyOutputPointer)
    
    if let status = status, let error = StatusError(tfStatus: status) {
        throw error
    }

    return Array(UnsafeBufferPointer(start: dyOutputPointer, count: xOutputs.count))
}

