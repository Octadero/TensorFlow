/* HEADER */
import Foundation
import Proto
extension Scope {

///Does nothing. Only useful as a placeholder for control edges.
public func noOp(operationName: String? = nil) throws -> Operation { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "NoOp",
		name: (operationName ?? "Type"),
		input: [],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Computes the gradient function for function f via backpropagation.
/// - Parameter input: a list of input tensors of size N + M;
/// - Parameter tin: the type list for the input list.
/// - Parameter tout: the type list for the input list.
/// - Parameter f: The function we want to compute the gradient for.
/// 
/// The function 'f' must be a numerical function which takes N inputs and
/// produces M outputs. Its gradient function 'g', which is computed by
/// this SymbolicGradient op is a function taking N + M inputs and
/// produces N outputs.
/// 
/// I.e. if we have
///    (y1, y2, ..., y_M) = f(x1, x2, ..., x_N),
/// then, g is
///    (dL/dx1, dL/dx2, ..., dL/dx_N) = g(x1, x2, ..., x_N,
///                                      dL/dy1, dL/dy2, ..., dL/dy_M),
/// 
/// where L is a scalar-value function of (x1, x2, ..., xN) (e.g., the
/// loss function). dL/dx_i is the partial derivative of L with respect
/// to x_i.
/// 
/// (Needs some math expert to say the comment above better.)
/// - Returns: 
///	output: a list of output tensors of size N;
public func symbolicGradient(operationName: String? = nil, input: Output, tin: [Any.Type], tout: [Any.Type], f: Tensorflow_NameAttrList) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Tin"] = tin
	attrs["Tout"] = tout
	attrs["f"] = f
	let opspec = OpSpec(
		type: "SymbolicGradient",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Converts an array of tensors to a list of tensors.
/// - Parameter input: 
/// - Parameter n: 
/// - Parameter outTypes: 
/// - Returns: 
///	output: 
public func arrayToList(operationName: String? = nil, input: [Output], n: UInt8, outTypes: [Any.Type]) throws -> Output { 
	var attrs = [String : Any]()
	attrs["N"] = n
	attrs["out_types"] = outTypes
	let opspec = OpSpec(
		type: "_ArrayToList",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Converts a list of tensors to an array of tensors.
/// - Parameter input: 
/// - Parameter tin: 
/// - Parameter n: 
/// - Returns: 
///	output: 
public func listToArray(operationName: String? = nil, input: Output, tin: [Any.Type], n: UInt8) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Tin"] = tin
	attrs["N"] = n
	let opspec = OpSpec(
		type: "_ListToArray",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///A graph node which represents a return value of a function.
/// - Parameter input: The return value.
/// - Parameter index: This return value is the index-th return value of the function.
public func retval(operationName: String? = nil, input: Output, index: UInt8) throws -> Operation { 
	var attrs = [String : Any]()
	attrs["index"] = index
	let opspec = OpSpec(
		type: "_Retval",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///A graph node which represents an argument to a function.
/// - Parameter index: This argument is the index-th argument of the function.
/// - Returns: 
///	output: The argument.
public func arg(operationName: String? = nil, index: UInt8) throws -> Output { 
	var attrs = [String : Any]()
	attrs["index"] = index
	let opspec = OpSpec(
		type: "_Arg",
		name: (operationName ?? "Type"),
		input: [],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Quantized Batch normalization.
///This op is deprecated and will be removed in the future. Prefer
/// `tf.nn.batch_normalization`.
/// - Parameter t: A 4D input Tensor.
/// - Parameter tMin: The value represented by the lowest quantized input.
/// - Parameter tMax: The value represented by the highest quantized input.
/// - Parameter m: A 1D mean Tensor with size matching the last dimension of t.
/// This is the first output from tf.nn.moments,
/// or a saved moving average thereof.
/// - Parameter mMin: The value represented by the lowest quantized mean.
/// - Parameter mMax: The value represented by the highest quantized mean.
/// - Parameter v: A 1D variance Tensor with size matching the last dimension of t.
/// This is the second output from tf.nn.moments,
/// or a saved moving average thereof.
/// - Parameter vMin: The value represented by the lowest quantized variance.
/// - Parameter vMax: The value represented by the highest quantized variance.
/// - Parameter beta: A 1D beta Tensor with size matching the last dimension of t.
/// An offset to be added to the normalized tensor.
/// - Parameter betaMin: The value represented by the lowest quantized offset.
/// - Parameter betaMax: The value represented by the highest quantized offset.
/// - Parameter gamma: A 1D gamma Tensor with size matching the last dimension of t.
/// If "scale_after_normalization" is true, this tensor will be multiplied
/// with the normalized tensor.
/// - Parameter gammaMin: The value represented by the lowest quantized gamma.
/// - Parameter gammaMax: The value represented by the highest quantized gamma.
/// - Parameter tinput: 
/// - Parameter outType: 
/// - Parameter varianceEpsilon: A small float number to avoid dividing by 0.
/// - Parameter scaleAfterNormalization: A bool indicating whether the resulted tensor
/// needs to be multiplied with gamma.
/// - Returns: 
///	result: 
///	result_min: 
///	result_max: 
public func quantizedBatchNormWithGlobalNormalization(operationName: String? = nil, t: Output, tMin: Output, tMax: Output, m: Output, mMin: Output, mMax: Output, v: Output, vMin: Output, vMax: Output, beta: Output, betaMin: Output, betaMax: Output, gamma: Output, gammaMin: Output, gammaMax: Output, tinput: Any.Type, outType: Any.Type, varianceEpsilon: Float, scaleAfterNormalization: Bool) throws -> (result: Output, resultMin: Output, resultMax: Output) { 
	var attrs = [String : Any]()
	attrs["Tinput"] = tinput
	attrs["out_type"] = outType
	attrs["variance_epsilon"] = varianceEpsilon
	attrs["scale_after_normalization"] = scaleAfterNormalization
	let opspec = OpSpec(
		type: "QuantizedBatchNormWithGlobalNormalization",
		name: (operationName ?? "Type"),
		input: [t, tMin, tMax, m, mMin, mMax, v, vMin, vMax, beta, betaMin, betaMax, gamma, gammaMin, gammaMax],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (result: op.output(at: 0), resultMin: op.output(at: 1), resultMax: op.output(at: 2))
} 

///Computes Quantized Rectified Linear 6: `min(max(features, 0), 6)`
/// - Parameter features: 
/// - Parameter minFeatures: The float value that the lowest quantized value represents.
/// - Parameter maxFeatures: The float value that the highest quantized value represents.
/// - Parameter tinput: 
/// - Parameter outType: 
/// - Returns: 
///	activations: Has the same output shape as "features".
///	min_activations: The float value that the lowest quantized value represents.
///	max_activations: The float value that the highest quantized value represents.
public func quantizedRelu6(operationName: String? = nil, features: Output, minFeatures: Output, maxFeatures: Output, tinput: Any.Type, outType: Any.Type) throws -> (activations: Output, minActivations: Output, maxActivations: Output) { 
	var attrs = [String : Any]()
	attrs["Tinput"] = tinput
	attrs["out_type"] = outType
	let opspec = OpSpec(
		type: "QuantizedRelu6",
		name: (operationName ?? "Type"),
		input: [features, minFeatures, maxFeatures],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (activations: op.output(at: 0), minActivations: op.output(at: 1), maxActivations: op.output(at: 2))
} 

///Computes gradient of the FractionalMaxPool function.
/// - Parameter origInput: Original input for `fractional_max_pool`
/// - Parameter origOutput: Original output for `fractional_max_pool`
/// - Parameter outBackprop: 4-D with shape `[batch, height, width, channels]`.  Gradients
/// w.r.t. the output of `fractional_max_pool`.
/// - Parameter rowPoolingSequence: row pooling sequence, form pooling region with
/// col_pooling_sequence.
/// - Parameter colPoolingSequence: column pooling sequence, form pooling region with
/// row_pooling sequence.
/// - Parameter overlapping: When set to True, it means when pooling, the values at the boundary
/// of adjacent pooling cells are used by both cells. For example:
/// 
/// `index  0  1  2  3  4`
/// 
/// `value  20 5  16 3  7`
/// 
/// If the pooling sequence is [0, 2, 4], then 16, at index 2 will be used twice.
/// The result would be [20, 16] for fractional max pooling.
/// - Returns: 
///	output: 4-D.  Gradients w.r.t. the input of `fractional_max_pool`.
public func fractionalMaxPoolGrad(operationName: String? = nil, origInput: Output, origOutput: Output, outBackprop: Output, rowPoolingSequence: Output, colPoolingSequence: Output, overlapping: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["overlapping"] = overlapping
	let opspec = OpSpec(
		type: "FractionalMaxPoolGrad",
		name: (operationName ?? "Type"),
		input: [origInput, origOutput, outBackprop, rowPoolingSequence, colPoolingSequence],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Says whether the targets are in the top `K` predictions.
///This outputs a `batch_size` bool array, an entry `out[i]` is `true` if the
/// prediction for the target class is among the top `k` predictions among
/// all predictions for example `i`. Note that the behavior of `InTopK` differs
/// from the `TopK` op in its handling of ties; if multiple classes have the
/// same prediction value and straddle the top-`k` boundary, all of those
/// classes are considered to be in the top `k`.
/// 
/// More formally, let
/// 
///   \\(predictions_i\\) be the predictions for all classes for example `i`,
///   \\(targets_i\\) be the target class for example `i`,
///   \\(out_i\\) be the output for example `i`,
/// 
/// $$out_i = predictions_{i, targets_i} \in TopKIncludingTies(predictions_i)$$
/// - Parameter predictions: A `batch_size` x `classes` tensor.
/// - Parameter targets: A `batch_size` vector of class ids.
/// - Parameter k: Number of top elements to look at for computing precision.
/// - Returns: 
///	precision: Computed Precision at `k` as a `bool Tensor`.
public func inTopK(operationName: String? = nil, predictions: Output, targets: Output, k: UInt8) throws -> Output { 
	var attrs = [String : Any]()
	attrs["k"] = k
	let opspec = OpSpec(
		type: "InTopK",
		name: (operationName ?? "Type"),
		input: [predictions, targets],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes softmax cross entropy cost and gradients to backpropagate.
///Inputs are the logits, not probabilities.
/// - Parameter features: batch_size x num_classes matrix
/// - Parameter labels: batch_size x num_classes matrix
/// The caller must ensure that each batch of labels represents a valid
/// probability distribution.
/// - Returns: 
///	loss: Per example loss (batch_size vector).
///	backprop: backpropagated gradients (batch_size x num_classes matrix).
public func softmaxCrossEntropyWithLogits(operationName: String? = nil, features: Output, labels: Output) throws -> (loss: Output, backprop: Output) { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "SoftmaxCrossEntropyWithLogits",
		name: (operationName ?? "Type"),
		input: [features, labels],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (loss: op.output(at: 0), backprop: op.output(at: 1))
} 

///Computes log softmax activations.
///For each batch `i` and class `j` we have
/// 
///     logsoftmax[i, j] = logits[i, j] - log(sum(exp(logits[i])))
/// - Parameter logits: 2-D with shape `[batch_size, num_classes]`.
/// - Returns: 
///	logsoftmax: Same shape as `logits`.
public func logSoftmax(operationName: String? = nil, logits: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "LogSoftmax",
		name: (operationName ?? "Type"),
		input: [logits],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes softsign gradients for a softsign operation.
/// - Parameter gradients: The backpropagated gradients to the corresponding softsign operation.
/// - Parameter features: The features passed as input to the corresponding softsign operation.
/// - Returns: 
///	backprops: The gradients: `gradients / (1 + abs(features))  *  *  2`.
public func softsignGrad(operationName: String? = nil, gradients: Output, features: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "SoftsignGrad",
		name: (operationName ?? "Type"),
		input: [gradients, features],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes softplus: `log(exp(features) + 1)`.
/// - Parameter features: 
/// - Returns: 
///	activations: 
public func softplus(operationName: String? = nil, features: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Softplus",
		name: (operationName ?? "Type"),
		input: [features],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes gradients for the exponential linear (Elu) operation.
/// - Parameter gradients: The backpropagated gradients to the corresponding Elu operation.
/// - Parameter outputs: The outputs of the corresponding Elu operation.
/// - Returns: 
///	backprops: The gradients: `gradients  *  (outputs + 1)` if outputs < 0,
/// `gradients` otherwise.
public func eluGrad(operationName: String? = nil, gradients: Output, outputs: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "EluGrad",
		name: (operationName ?? "Type"),
		input: [gradients, outputs],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes exponential linear: `exp(features) - 1` if < 0, `features` otherwise.
///See [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)
/// ](http://arxiv.org/abs/1511.07289)
/// - Parameter features: 
/// - Returns: 
///	activations: 
public func elu(operationName: String? = nil, features: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Elu",
		name: (operationName ?? "Type"),
		input: [features],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes rectified linear 6: `min(max(features, 0), 6)`.
/// - Parameter features: 
/// - Returns: 
///	activations: 
public func relu6(operationName: String? = nil, features: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Relu6",
		name: (operationName ?? "Type"),
		input: [features],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes rectified linear gradients for a Relu operation.
/// - Parameter gradients: The backpropagated gradients to the corresponding Relu operation.
/// - Parameter features: The features passed as input to the corresponding Relu operation, OR
/// the outputs of that operation (both work equivalently).
/// - Returns: 
///	backprops: `gradients  *  (features > 0)`.
public func reluGrad(operationName: String? = nil, gradients: Output, features: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "ReluGrad",
		name: (operationName ?? "Type"),
		input: [gradients, features],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes the gradient of morphological 2-D dilation with respect to the input.
/// - Parameter input: 4-D with shape `[batch, in_height, in_width, depth]`.
/// - Parameter filter: 3-D with shape `[filter_height, filter_width, depth]`.
/// - Parameter outBackprop: 4-D with shape `[batch, out_height, out_width, depth]`.
/// - Parameter strides: 1-D of length 4. The stride of the sliding window for each dimension of
/// the input tensor. Must be: `[1, stride_height, stride_width, 1]`.
/// - Parameter rates: 1-D of length 4. The input stride for atrous morphological dilation.
/// Must be: `[1, rate_height, rate_width, 1]`.
/// - Parameter padding: The type of padding algorithm to use.
/// - Returns: 
///	in_backprop: 4-D with shape `[batch, in_height, in_width, depth]`.
public func dilation2DBackpropInput(operationName: String? = nil, input: Output, filter: Output, outBackprop: Output, strides: [Int64], rates: [Int64], padding: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["strides"] = strides
	attrs["rates"] = rates
	attrs["padding"] = padding
	let opspec = OpSpec(
		type: "Dilation2DBackpropInput",
		name: (operationName ?? "Type"),
		input: [input, filter, outBackprop],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Sends the named tensor from send_device to recv_device.
/// - Parameter tensor: The tensor to send.
/// - Parameter tensorName: The name of the tensor to send.
/// - Parameter sendDevice: The name of the device sending the tensor.
/// - Parameter sendDeviceIncarnation: The current incarnation of send_device.
/// - Parameter recvDevice: The name of the device receiving the tensor.
/// - Parameter clientTerminated: If set to true, this indicates that the node was added
/// to the graph as a result of a client-side feed or fetch of Tensor data,
/// in which case the corresponding send or recv is expected to be managed
/// locally by the caller.
public func send(operationName: String? = nil, tensor: Output, tensorName: String, sendDevice: String, sendDeviceIncarnation: UInt8, recvDevice: String, clientTerminated: Bool) throws -> Operation { 
	var attrs = [String : Any]()
	attrs["tensor_name"] = tensorName
	attrs["send_device"] = sendDevice
	attrs["send_device_incarnation"] = sendDeviceIncarnation
	attrs["recv_device"] = recvDevice
	attrs["client_terminated"] = clientTerminated
	let opspec = OpSpec(
		type: "_Send",
		name: (operationName ?? "Type"),
		input: [tensor],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Computes second-order gradients of the maxpooling function.
/// - Parameter origInput: The original input tensor.
/// - Parameter origOutput: The original output tensor.
/// - Parameter grad: 4-D.  Gradients of gradients w.r.t. the input of `max_pool`.
/// - Parameter ksize: The size of the window for each dimension of the input tensor.
/// - Parameter strides: The stride of the sliding window for each dimension of the
/// input tensor.
/// - Parameter padding: The type of padding algorithm to use.
/// - Parameter dataFormat: Specify the data format of the input and output data. With the
/// default format "NHWC", the data is stored in the order of:
///     [batch, in_height, in_width, in_channels].
/// Alternatively, the format could be "NCHW", the data storage order of:
///     [batch, in_channels, in_height, in_width].
/// - Returns: 
///	output: Gradients of gradients w.r.t. the input to `max_pool`.
public func maxPoolGradGradV2(operationName: String? = nil, origInput: Output, origOutput: Output, grad: Output, ksize: Output, strides: Output, padding: String, dataFormat: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["padding"] = padding
	attrs["data_format"] = dataFormat
	let opspec = OpSpec(
		type: "MaxPoolGradGradV2",
		name: (operationName ?? "Type"),
		input: [origInput, origOutput, grad, ksize, strides],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes second-order gradients of the maxpooling function.
/// - Parameter origInput: The original input tensor.
/// - Parameter origOutput: The original output tensor.
/// - Parameter grad: 4-D.  Gradients of gradients w.r.t. the input of `max_pool`.
/// - Parameter ksize: The size of the window for each dimension of the input tensor.
/// - Parameter strides: The stride of the sliding window for each dimension of the
/// input tensor.
/// - Parameter padding: The type of padding algorithm to use.
/// - Parameter dataFormat: Specify the data format of the input and output data. With the
/// default format "NHWC", the data is stored in the order of:
///     [batch, in_height, in_width, in_channels].
/// Alternatively, the format could be "NCHW", the data storage order of:
///     [batch, in_channels, in_height, in_width].
/// - Returns: 
///	output: Gradients of gradients w.r.t. the input to `max_pool`.
public func maxPoolGradGrad(operationName: String? = nil, origInput: Output, origOutput: Output, grad: Output, ksize: [Int64], strides: [Int64], padding: String, dataFormat: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["ksize"] = ksize
	attrs["strides"] = strides
	attrs["padding"] = padding
	attrs["data_format"] = dataFormat
	let opspec = OpSpec(
		type: "MaxPoolGradGrad",
		name: (operationName ?? "Type"),
		input: [origInput, origOutput, grad],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes gradients of the maxpooling function.
/// - Parameter origInput: The original input tensor.
/// - Parameter origOutput: The original output tensor.
/// - Parameter grad: 4-D.  Gradients w.r.t. the output of `max_pool`.
/// - Parameter ksize: The size of the window for each dimension of the input tensor.
/// - Parameter strides: The stride of the sliding window for each dimension of the
/// input tensor.
/// - Parameter padding: The type of padding algorithm to use.
/// - Parameter dataFormat: Specify the data format of the input and output data. With the
/// default format "NHWC", the data is stored in the order of:
///     [batch, in_height, in_width, in_channels].
/// Alternatively, the format could be "NCHW", the data storage order of:
///     [batch, in_channels, in_height, in_width].
/// - Returns: 
///	output: Gradients w.r.t. the input to `max_pool`.
public func maxPoolGradV2(operationName: String? = nil, origInput: Output, origOutput: Output, grad: Output, ksize: Output, strides: Output, padding: String, dataFormat: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["padding"] = padding
	attrs["data_format"] = dataFormat
	let opspec = OpSpec(
		type: "MaxPoolGradV2",
		name: (operationName ?? "Type"),
		input: [origInput, origOutput, grad, ksize, strides],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes gradients of the maxpooling function.
/// - Parameter origInput: The original input tensor.
/// - Parameter origOutput: The original output tensor.
/// - Parameter grad: 4-D.  Gradients w.r.t. the output of `max_pool`.
/// - Parameter ksize: The size of the window for each dimension of the input tensor.
/// - Parameter strides: The stride of the sliding window for each dimension of the
/// input tensor.
/// - Parameter padding: The type of padding algorithm to use.
/// - Parameter dataFormat: Specify the data format of the input and output data. With the
/// default format "NHWC", the data is stored in the order of:
///     [batch, in_height, in_width, in_channels].
/// Alternatively, the format could be "NCHW", the data storage order of:
///     [batch, in_channels, in_height, in_width].
/// - Returns: 
///	output: Gradients w.r.t. the input to `max_pool`.
public func maxPoolGrad(operationName: String? = nil, origInput: Output, origOutput: Output, grad: Output, ksize: [Int64], strides: [Int64], padding: String, dataFormat: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["ksize"] = ksize
	attrs["strides"] = strides
	attrs["padding"] = padding
	attrs["data_format"] = dataFormat
	let opspec = OpSpec(
		type: "MaxPoolGrad",
		name: (operationName ?? "Type"),
		input: [origInput, origOutput, grad],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Performs max pooling on the input.
/// - Parameter input: 4-D input to pool over.
/// - Parameter ksize: The size of the window for each dimension of the input tensor.
/// - Parameter strides: The stride of the sliding window for each dimension of the
/// input tensor.
/// - Parameter padding: The type of padding algorithm to use.
/// - Parameter dataFormat: Specify the data format of the input and output data. With the
/// default format "NHWC", the data is stored in the order of:
///     [batch, in_height, in_width, in_channels].
/// Alternatively, the format could be "NCHW", the data storage order of:
///     [batch, in_channels, in_height, in_width].
/// - Returns: 
///	output: The max pooled output tensor.
public func maxPoolV2(operationName: String? = nil, input: Output, ksize: Output, strides: Output, padding: String, dataFormat: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["padding"] = padding
	attrs["data_format"] = dataFormat
	let opspec = OpSpec(
		type: "MaxPoolV2",
		name: (operationName ?? "Type"),
		input: [input, ksize, strides],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Gradients for Local Response Normalization.
/// - Parameter inputGrads: 4-D with shape `[batch, height, width, channels]`.
/// - Parameter inputImage: 4-D with shape `[batch, height, width, channels]`.
/// - Parameter outputImage: 4-D with shape `[batch, height, width, channels]`.
/// - Parameter depthRadius: A depth radius.
/// - Parameter bias: An offset (usually > 0 to avoid dividing by 0).
/// - Parameter alpha: A scale factor, usually positive.
/// - Parameter beta: An exponent.
/// - Returns: 
///	output: The gradients for LRN.
public func lRNGrad(operationName: String? = nil, inputGrads: Output, inputImage: Output, outputImage: Output, depthRadius: UInt8, bias: Float, alpha: Float, beta: Float) throws -> Output { 
	var attrs = [String : Any]()
	attrs["depth_radius"] = depthRadius
	attrs["bias"] = bias
	attrs["alpha"] = alpha
	attrs["beta"] = beta
	let opspec = OpSpec(
		type: "LRNGrad",
		name: (operationName ?? "Type"),
		input: [inputGrads, inputImage, outputImage],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Receives the named tensor from send_device on recv_device.
///_HostRecv requires its input on host memory whereas _Recv requires its
/// input on device memory.
/// - Parameter tensorType: 
/// - Parameter tensorName: The name of the tensor to receive.
/// - Parameter sendDevice: The name of the device sending the tensor.
/// - Parameter sendDeviceIncarnation: The current incarnation of send_device.
/// - Parameter recvDevice: The name of the device receiving the tensor.
/// - Parameter clientTerminated: If set to true, this indicates that the node was added
/// to the graph as a result of a client-side feed or fetch of Tensor data,
/// in which case the corresponding send or recv is expected to be managed
/// locally by the caller.
/// - Returns: 
///	tensor: The tensor to receive.
public func hostRecv(operationName: String? = nil, tensorType: Any.Type, tensorName: String, sendDevice: String, sendDeviceIncarnation: UInt8, recvDevice: String, clientTerminated: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["tensor_type"] = tensorType
	attrs["tensor_name"] = tensorName
	attrs["send_device"] = sendDevice
	attrs["send_device_incarnation"] = sendDeviceIncarnation
	attrs["recv_device"] = recvDevice
	attrs["client_terminated"] = clientTerminated
	let opspec = OpSpec(
		type: "_HostRecv",
		name: (operationName ?? "Type"),
		input: [],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes second-order gradients of the maxpooling function.
/// - Parameter origInput: The original input tensor.
/// - Parameter origOutput: The original output tensor.
/// - Parameter grad: Output backprop of shape `[batch, depth, rows, cols, channels]`.
/// - Parameter ksize: 1-D tensor of length 5. The size of the window for each dimension of
/// the input tensor. Must have `ksize[0] = ksize[4] = 1`.
/// - Parameter strides: 1-D tensor of length 5. The stride of the sliding window for each
/// dimension of `input`. Must have `strides[0] = strides[4] = 1`.
/// - Parameter padding: The type of padding algorithm to use.
/// - Parameter dataFormat: The data format of the input and output data. With the
/// default format "NDHWC", the data is stored in the order of:
///     [batch, in_depth, in_height, in_width, in_channels].
/// Alternatively, the format could be "NCDHW", the data storage order is:
///     [batch, in_channels, in_depth, in_height, in_width].
/// - Returns: 
///	output: Gradients of gradients w.r.t. the input to `max_pool`.
public func maxPool3DGradGrad(operationName: String? = nil, origInput: Output, origOutput: Output, grad: Output, ksize: [Int64], strides: [Int64], padding: String, dataFormat: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["ksize"] = ksize
	attrs["strides"] = strides
	attrs["padding"] = padding
	attrs["data_format"] = dataFormat
	let opspec = OpSpec(
		type: "MaxPool3DGradGrad",
		name: (operationName ?? "Type"),
		input: [origInput, origOutput, grad],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes the gradients of 3-D convolution with respect to the filter.
/// - Parameter input: Shape `[batch, depth, rows, cols, in_channels]`.
/// - Parameter filter: Shape `[depth, rows, cols, in_channels, out_channels]`.
/// `in_channels` must match between `input` and `filter`.
/// - Parameter outBackprop: Backprop signal of shape `[batch, out_depth, out_rows, out_cols,
/// out_channels]`.
/// - Parameter strides: 1-D tensor of length 5. The stride of the sliding window for each
/// dimension of `input`. Must have `strides[0] = strides[4] = 1`.
/// - Parameter padding: The type of padding algorithm to use.
/// - Returns: 
///	output: 
public func conv3DBackpropFilter(operationName: String? = nil, input: Output, filter: Output, outBackprop: Output, strides: [Int64], padding: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["strides"] = strides
	attrs["padding"] = padding
	let opspec = OpSpec(
		type: "Conv3DBackpropFilter",
		name: (operationName ?? "Type"),
		input: [input, filter, outBackprop],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes a 3-D convolution given 5-D `input` and `filter` tensors.
///In signal processing, cross-correlation is a measure of similarity of
/// two waveforms as a function of a time-lag applied to one of them. This
/// is also known as a sliding dot product or sliding inner-product.
/// 
/// Our Conv3D implements a form of cross-correlation.
/// - Parameter input: Shape `[batch, in_depth, in_height, in_width, in_channels]`.
/// - Parameter filter: Shape `[filter_depth, filter_height, filter_width, in_channels,
/// out_channels]`. `in_channels` must match between `input` and `filter`.
/// - Parameter strides: 1-D tensor of length 5. The stride of the sliding window for each
/// dimension of `input`. Must have `strides[0] = strides[4] = 1`.
/// - Parameter padding: The type of padding algorithm to use.
/// - Parameter dataFormat: The data format of the input and output data. With the
/// default format "NDHWC", the data is stored in the order of:
///     [batch, in_depth, in_height, in_width, in_channels].
/// Alternatively, the format could be "NCDHW", the data storage order is:
///     [batch, in_channels, in_depth, in_height, in_width].
/// - Returns: 
///	output: 
public func conv3D(operationName: String? = nil, input: Output, filter: Output, strides: [Int64], padding: String, dataFormat: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["strides"] = strides
	attrs["padding"] = padding
	attrs["data_format"] = dataFormat
	let opspec = OpSpec(
		type: "Conv3D",
		name: (operationName ?? "Type"),
		input: [input, filter],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes the gradients of depthwise convolution with respect to the filter.
/// - Parameter input: 4-D with shape based on `data_format`.  For example, if
/// `data_format` is 'NHWC' then `input` is a 4-D `[batch, in_height,
/// in_width, in_channels]` tensor.
/// - Parameter filterSizes: An integer vector representing the tensor shape of `filter`,
/// where `filter` is a 4-D
/// `[filter_height, filter_width, in_channels, depthwise_multiplier]` tensor.
/// - Parameter outBackprop: 4-D with shape  based on `data_format`.
/// For example, if `data_format` is 'NHWC' then
/// out_backprop shape is `[batch, out_height, out_width, out_channels]`.
/// Gradients w.r.t. the output of the convolution.
/// - Parameter strides: The stride of the sliding window for each dimension of the input
/// of the convolution.
/// - Parameter padding: The type of padding algorithm to use.
/// - Parameter dataFormat: Specify the data format of the input and output data. With the
/// default format "NHWC", the data is stored in the order of:
///     [batch, height, width, channels].
/// Alternatively, the format could be "NCHW", the data storage order of:
///     [batch, channels, height, width].
/// - Returns: 
///	output: 4-D with shape
/// `[filter_height, filter_width, in_channels, out_channels]`.  Gradient w.r.t.
/// the `filter` input of the convolution.
public func depthwiseConv2dNativeBackpropFilter(operationName: String? = nil, input: Output, filterSizes: Output, outBackprop: Output, strides: [Int64], padding: String, dataFormat: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["strides"] = strides
	attrs["padding"] = padding
	attrs["data_format"] = dataFormat
	let opspec = OpSpec(
		type: "DepthwiseConv2dNativeBackpropFilter",
		name: (operationName ?? "Type"),
		input: [input, filterSizes, outBackprop],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes the gradients of convolution with respect to the filter.
/// - Parameter input: 4-D with shape `[batch, in_height, in_width, in_channels]`.
/// - Parameter filterSizes: An integer vector representing the tensor shape of `filter`,
/// where `filter` is a 4-D
/// `[filter_height, filter_width, in_channels, out_channels]` tensor.
/// - Parameter outBackprop: 4-D with shape `[batch, out_height, out_width, out_channels]`.
/// Gradients w.r.t. the output of the convolution.
/// - Parameter strides: The stride of the sliding window for each dimension of the input
/// of the convolution. Must be in the same order as the dimension specified with
/// format.
/// - Parameter useCudnnOnGpu: 
/// - Parameter padding: The type of padding algorithm to use.
/// - Parameter dataFormat: Specify the data format of the input and output data. With the
/// default format "NHWC", the data is stored in the order of:
///     [batch, in_height, in_width, in_channels].
/// Alternatively, the format could be "NCHW", the data storage order of:
///     [batch, in_channels, in_height, in_width].
/// - Returns: 
///	output: 4-D with shape
/// `[filter_height, filter_width, in_channels, out_channels]`.  Gradient w.r.t.
/// the `filter` input of the convolution.
public func conv2DBackpropFilter(operationName: String? = nil, input: Output, filterSizes: Output, outBackprop: Output, strides: [Int64], useCudnnOnGpu: Bool, padding: String, dataFormat: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["strides"] = strides
	attrs["use_cudnn_on_gpu"] = useCudnnOnGpu
	attrs["padding"] = padding
	attrs["data_format"] = dataFormat
	let opspec = OpSpec(
		type: "Conv2DBackpropFilter",
		name: (operationName ?? "Type"),
		input: [input, filterSizes, outBackprop],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes the gradients of convolution with respect to the input.
/// - Parameter inputSizes: An integer vector representing the shape of `input`,
/// where `input` is a 4-D `[batch, height, width, channels]` tensor.
/// - Parameter filter: 4-D with shape
/// `[filter_height, filter_width, in_channels, out_channels]`.
/// - Parameter outBackprop: 4-D with shape `[batch, out_height, out_width, out_channels]`.
/// Gradients w.r.t. the output of the convolution.
/// - Parameter strides: The stride of the sliding window for each dimension of the input
/// of the convolution. Must be in the same order as the dimension specified with
/// format.
/// - Parameter useCudnnOnGpu: 
/// - Parameter padding: The type of padding algorithm to use.
/// - Parameter dataFormat: Specify the data format of the input and output data. With the
/// default format "NHWC", the data is stored in the order of:
///     [batch, in_height, in_width, in_channels].
/// Alternatively, the format could be "NCHW", the data storage order of:
///     [batch, in_channels, in_height, in_width].
/// - Returns: 
///	output: 4-D with shape `[batch, in_height, in_width, in_channels]`.  Gradient
/// w.r.t. the input of the convolution.
public func conv2DBackpropInput(operationName: String? = nil, inputSizes: Output, filter: Output, outBackprop: Output, strides: [Int64], useCudnnOnGpu: Bool, padding: String, dataFormat: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["strides"] = strides
	attrs["use_cudnn_on_gpu"] = useCudnnOnGpu
	attrs["padding"] = padding
	attrs["data_format"] = dataFormat
	let opspec = OpSpec(
		type: "Conv2DBackpropInput",
		name: (operationName ?? "Type"),
		input: [inputSizes, filter, outBackprop],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Adds `bias` to `value`.
///This is a special case of `tf.add` where `bias` is restricted to be 1-D.
/// Broadcasting is supported, so `value` may have any number of dimensions.
/// - Parameter value: Any number of dimensions.
/// - Parameter bias: 1-D with size the last dimension of `value`.
/// - Parameter dataFormat: Specify the data format of the input and output data. With the
/// default format "NHWC", the bias tensor will be added to the last dimension
/// of the value tensor.
/// Alternatively, the format could be "NCHW", the data storage order of:
///     [batch, in_channels, in_height, in_width].
/// The tensor will be added to "in_channels", the third-to-the-last
///     dimension.
/// - Returns: 
///	output: Broadcasted sum of `value` and `bias`.
public func biasAdd(operationName: String? = nil, value: Output, bias: Output, dataFormat: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["data_format"] = dataFormat
	let opspec = OpSpec(
		type: "BiasAdd",
		name: (operationName ?? "Type"),
		input: [value, bias],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Batch normalization.
///Note that the size of 4D Tensors are defined by either "NHWC" or "NCHW".
/// The size of 1D Tensors matches the dimension C of the 4D Tensors.
/// - Parameter x: A 4D Tensor for input data.
/// - Parameter scale: A 1D Tensor for scaling factor, to scale the normalized x.
/// - Parameter offset: A 1D Tensor for offset, to shift to the normalized x.
/// - Parameter mean: A 1D Tensor for population mean. Used for inference only;
/// must be empty for training.
/// - Parameter variance: A 1D Tensor for population variance. Used for inference only;
/// must be empty for training.
/// - Parameter u: The data type for the scale, offset, mean, and variance.
/// - Parameter epsilon: A small float number added to the variance of x.
/// - Parameter dataFormat: The data format for x and y. Either "NHWC" (default) or "NCHW".
/// - Parameter isTraining: A bool value to indicate the operation is for training (default)
/// or inference.
/// - Returns: 
///	y: A 4D Tensor for output data.
///	batch_mean: A 1D Tensor for the computed batch mean, to be used by TensorFlow
/// to compute the running mean.
///	batch_variance: A 1D Tensor for the computed batch variance, to be used by
/// TensorFlow to compute the running variance.
///	reserve_space_1: A 1D Tensor for the computed batch mean, to be reused
/// in the gradient computation.
///	reserve_space_2: A 1D Tensor for the computed batch variance (inverted variance
/// in the cuDNN case), to be reused in the gradient computation.
public func fusedBatchNormV2(operationName: String? = nil, x: Output, scale: Output, offset: Output, mean: Output, variance: Output, u: Any.Type, epsilon: Float, dataFormat: String, isTraining: Bool) throws -> (y: Output, batchMean: Output, batchVariance: Output, reserveSpace1: Output, reserveSpace2: Output) { 
	var attrs = [String : Any]()
	attrs["U"] = u
	attrs["epsilon"] = epsilon
	attrs["data_format"] = dataFormat
	attrs["is_training"] = isTraining
	let opspec = OpSpec(
		type: "FusedBatchNormV2",
		name: (operationName ?? "Type"),
		input: [x, scale, offset, mean, variance],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (y: op.output(at: 0), batchMean: op.output(at: 1), batchVariance: op.output(at: 2), reserveSpace1: op.output(at: 3), reserveSpace2: op.output(at: 4))
} 

///Batch normalization.
///Note that the size of 4D Tensors are defined by either "NHWC" or "NCHW".
/// The size of 1D Tensors matches the dimension C of the 4D Tensors.
/// - Parameter x: A 4D Tensor for input data.
/// - Parameter scale: A 1D Tensor for scaling factor, to scale the normalized x.
/// - Parameter offset: A 1D Tensor for offset, to shift to the normalized x.
/// - Parameter mean: A 1D Tensor for population mean. Used for inference only;
/// must be empty for training.
/// - Parameter variance: A 1D Tensor for population variance. Used for inference only;
/// must be empty for training.
/// - Parameter epsilon: A small float number added to the variance of x.
/// - Parameter dataFormat: The data format for x and y. Either "NHWC" (default) or "NCHW".
/// - Parameter isTraining: A bool value to indicate the operation is for training (default)
/// or inference.
/// - Returns: 
///	y: A 4D Tensor for output data.
///	batch_mean: A 1D Tensor for the computed batch mean, to be used by TensorFlow
/// to compute the running mean.
///	batch_variance: A 1D Tensor for the computed batch variance, to be used by
/// TensorFlow to compute the running variance.
///	reserve_space_1: A 1D Tensor for the computed batch mean, to be reused
/// in the gradient computation.
///	reserve_space_2: A 1D Tensor for the computed batch variance (inverted variance
/// in the cuDNN case), to be reused in the gradient computation.
public func fusedBatchNorm(operationName: String? = nil, x: Output, scale: Output, offset: Output, mean: Output, variance: Output, epsilon: Float, dataFormat: String, isTraining: Bool) throws -> (y: Output, batchMean: Output, batchVariance: Output, reserveSpace1: Output, reserveSpace2: Output) { 
	var attrs = [String : Any]()
	attrs["epsilon"] = epsilon
	attrs["data_format"] = dataFormat
	attrs["is_training"] = isTraining
	let opspec = OpSpec(
		type: "FusedBatchNorm",
		name: (operationName ?? "Type"),
		input: [x, scale, offset, mean, variance],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (y: op.output(at: 0), batchMean: op.output(at: 1), batchVariance: op.output(at: 2), reserveSpace1: op.output(at: 3), reserveSpace2: op.output(at: 4))
} 

///Given a quantized tensor described by (input, input_min, input_max), outputs a
///range that covers the actual values present in that tensor.  This op is
/// typically used to produce the requested_output_min and requested_output_max for
/// Requantize.
/// - Parameter input: 
/// - Parameter inputMin: The float value that the minimum quantized input value represents.
/// - Parameter inputMax: The float value that the maximum quantized input value represents.
/// - Parameter tinput: The type of the input.
/// - Returns: 
///	output_min: The computed min output.
///	output_max: the computed max output.
public func requantizationRange(operationName: String? = nil, input: Output, inputMin: Output, inputMax: Output, tinput: Any.Type) throws -> (outputMin: Output, outputMax: Output) { 
	var attrs = [String : Any]()
	attrs["Tinput"] = tinput
	let opspec = OpSpec(
		type: "RequantizationRange",
		name: (operationName ?? "Type"),
		input: [input, inputMin, inputMax],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (outputMin: op.output(at: 0), outputMax: op.output(at: 1))
} 

///Convert the quantized 'input' tensor into a lower-precision 'output', using the
///actual distribution of the values to maximize the usage of the lower bit depth
/// and adjusting the output min and max ranges accordingly.
/// 
/// [input_min, input_max] are scalar floats that specify the range for the float
/// interpretation of the 'input' data. For example, if input_min is -1.0f and
/// input_max is 1.0f, and we are dealing with quint16 quantized data, then a 0
/// value in the 16-bit data should be interpreted as -1.0f, and a 65535 means 1.0f.
/// 
/// This operator tries to squeeze as much precision as possible into an output with
/// a lower bit depth by calculating the actual min and max values found in the
/// data. For example, maybe that quint16 input has no values lower than 16,384 and
/// none higher than 49,152. That means only half the range is actually needed, all
/// the float interpretations are between -0.5f and 0.5f, so if we want to compress
/// the data into a quint8 output, we can use that range rather than the theoretical
/// -1.0f to 1.0f that is suggested by the input min and max.
/// 
/// In practice, this is most useful for taking output from operations like
/// QuantizedMatMul that can produce higher bit-depth outputs than their inputs and
/// may have large potential output ranges, but in practice have a distribution of
/// input values that only uses a small fraction of the possible range. By feeding
/// that output into this operator, we can reduce it from 32 bits down to 8 with
/// minimal loss of accuracy.
/// - Parameter input: 
/// - Parameter inputMin: The float value that the minimum quantized input value represents.
/// - Parameter inputMax: The float value that the maximum quantized input value represents.
/// - Parameter tinput: The type of the input.
/// - Parameter outType: The type of the output. Should be a lower bit depth than Tinput.
/// - Returns: 
///	output: 
///	output_min: The float value that the minimum quantized output value represents.
///	output_max: The float value that the maximum quantized output value represents.
public func quantizeDownAndShrinkRange(operationName: String? = nil, input: Output, inputMin: Output, inputMax: Output, tinput: Any.Type, outType: Any.Type) throws -> (output: Output, outputMin: Output, outputMax: Output) { 
	var attrs = [String : Any]()
	attrs["Tinput"] = tinput
	attrs["out_type"] = outType
	let opspec = OpSpec(
		type: "QuantizeDownAndShrinkRange",
		name: (operationName ?? "Type"),
		input: [input, inputMin, inputMax],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (output: op.output(at: 0), outputMin: op.output(at: 1), outputMax: op.output(at: 2))
} 

///Perform a quantized matrix multiplication of  `a` by the matrix `b`.
///The inputs must be two-dimensional matrices and the inner dimension of
/// `a` (after being transposed if `transpose_a` is non-zero) must match the
/// outer dimension of `b` (after being transposed if `transposed_b` is
/// non-zero).
/// - Parameter a: Must be a two-dimensional tensor.
/// - Parameter b: Must be a two-dimensional tensor.
/// - Parameter minA: The float value that the lowest quantized `a` value represents.
/// - Parameter maxA: The float value that the highest quantized `a` value represents.
/// - Parameter minB: The float value that the lowest quantized `b` value represents.
/// - Parameter maxB: The float value that the highest quantized `b` value represents.
/// - Parameter t1: 
/// - Parameter t2: 
/// - Parameter toutput: 
/// - Parameter transposeA: If true, `a` is transposed before multiplication.
/// - Parameter transposeB: If true, `b` is transposed before multiplication.
/// - Parameter tactivation: The type of output produced by activation function
/// following this operation.
/// - Returns: 
///	out: 
///	min_out: The float value that the lowest quantized output value represents.
///	max_out: The float value that the highest quantized output value represents.
public func quantizedMatMul(operationName: String? = nil, a: Output, b: Output, minA: Output, maxA: Output, minB: Output, maxB: Output, t1: Any.Type, t2: Any.Type, toutput: Any.Type, transposeA: Bool, transposeB: Bool, tactivation: Any.Type) throws -> (out: Output, minOut: Output, maxOut: Output) { 
	var attrs = [String : Any]()
	attrs["T1"] = t1
	attrs["T2"] = t2
	attrs["Toutput"] = toutput
	attrs["transpose_a"] = transposeA
	attrs["transpose_b"] = transposeB
	attrs["Tactivation"] = tactivation
	let opspec = OpSpec(
		type: "QuantizedMatMul",
		name: (operationName ?? "Type"),
		input: [a, b, minA, maxA, minB, maxB],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (out: op.output(at: 0), minOut: op.output(at: 1), maxOut: op.output(at: 2))
} 

///Compute the cumulative sum of the tensor `x` along `axis`.
///By default, this op performs an inclusive cumsum, which means that the first
/// element of the input is identical to the first element of the output:
/// 
/// ```python
/// tf.cumsum([a, b, c])  # => [a, a + b, a + b + c]
/// ```
/// 
/// By setting the `exclusive` kwarg to `True`, an exclusive cumsum is
/// performed instead:
/// 
/// ```python
/// tf.cumsum([a, b, c], exclusive=True)  # => [0, a, a + b]
/// ```
/// 
/// By setting the `reverse` kwarg to `True`, the cumsum is performed in the
/// opposite direction:
/// 
/// ```python
/// tf.cumsum([a, b, c], reverse=True)  # => [a + b + c, b + c, c]
/// ```
/// 
/// This is more efficient than using separate `tf.reverse` ops.
/// 
/// The `reverse` and `exclusive` kwargs can also be combined:
/// 
/// ```python
/// tf.cumsum([a, b, c], exclusive=True, reverse=True)  # => [b + c, c, 0]
/// ```
/// - Parameter x: A `Tensor`. Must be one of the following types: `float32`, `float64`,
/// `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`,
/// `complex128`, `qint8`, `quint8`, `qint32`, `half`.
/// - Parameter axis: A `Tensor` of type `int32` (default: 0). Must be in the range
/// `[-rank(x), rank(x))`.
/// - Parameter exclusive: If `True`, perform exclusive cumsum.
/// - Parameter reverse: A `bool` (default: False).
/// - Parameter tidx: 
/// - Returns: 
///	out: 
public func cumsum(operationName: String? = nil, x: Output, axis: Output, exclusive: Bool, reverse: Bool, tidx: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["exclusive"] = exclusive
	attrs["reverse"] = reverse
	attrs["Tidx"] = tidx
	let opspec = OpSpec(
		type: "Cumsum",
		name: (operationName ?? "Type"),
		input: [x, axis],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Gradients for batch normalization.
///This op is deprecated. See `tf.nn.batch_normalization`.
/// - Parameter t: A 4D input Tensor.
/// - Parameter m: A 1D mean Tensor with size matching the last dimension of t.
/// This is the first output from tf.nn.moments,
/// or a saved moving average thereof.
/// - Parameter v: A 1D variance Tensor with size matching the last dimension of t.
/// This is the second output from tf.nn.moments,
/// or a saved moving average thereof.
/// - Parameter gamma: A 1D gamma Tensor with size matching the last dimension of t.
/// If "scale_after_normalization" is true, this Tensor will be multiplied
/// with the normalized Tensor.
/// - Parameter backprop: 4D backprop Tensor.
/// - Parameter varianceEpsilon: A small float number to avoid dividing by 0.
/// - Parameter scaleAfterNormalization: A bool indicating whether the resulted tensor
/// needs to be multiplied with gamma.
/// - Returns: 
///	dx: 4D backprop tensor for input.
///	dm: 1D backprop tensor for mean.
///	dv: 1D backprop tensor for variance.
///	db: 1D backprop tensor for beta.
///	dg: 1D backprop tensor for gamma.
public func batchNormWithGlobalNormalizationGrad(operationName: String? = nil, t: Output, m: Output, v: Output, gamma: Output, backprop: Output, varianceEpsilon: Float, scaleAfterNormalization: Bool) throws -> (dx: Output, dm: Output, dv: Output, db: Output, dg: Output) { 
	var attrs = [String : Any]()
	attrs["variance_epsilon"] = varianceEpsilon
	attrs["scale_after_normalization"] = scaleAfterNormalization
	let opspec = OpSpec(
		type: "BatchNormWithGlobalNormalizationGrad",
		name: (operationName ?? "Type"),
		input: [t, m, v, gamma, backprop],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (dx: op.output(at: 0), dm: op.output(at: 1), dv: op.output(at: 2), db: op.output(at: 3), dg: op.output(at: 4))
} 

///Counts the number of occurrences of each value in an integer array.
///Outputs a vector with length `size` and the same dtype as `weights`. If
/// `weights` are empty, then index `i` stores the number of times the value `i` is
/// counted in `arr`. If `weights` are non-empty, then index `i` stores the sum of
/// the value in `weights` at each index where the corresponding value in `arr` is
/// `i`.
/// 
/// Values in `arr` outside of the range [0, size) are ignored.
/// - Parameter arr: int32 `Tensor`.
/// - Parameter size: non-negative int32 scalar `Tensor`.
/// - Parameter weights: is an int32, int64, float32, or float64 `Tensor` with the same
/// shape as `arr`, or a length-0 `Tensor`, in which case it acts as all weights
/// equal to 1.
/// - Returns: 
///	bins: 1D `Tensor` with length equal to `size`. The counts or summed weights for
/// each value in the range [0, size).
public func bincount(operationName: String? = nil, arr: Output, size: Output, weights: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Bincount",
		name: (operationName ?? "Type"),
		input: [arr, size, weights],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Compute the pairwise cross product.
///`a` and `b` must be the same shape; they can either be simple 3-element vectors,
/// or any shape where the innermost dimension is 3. In the latter case, each pair
/// of corresponding 3-element vectors is cross-multiplied independently.
/// - Parameter a: A tensor containing 3-element vectors.
/// - Parameter b: Another tensor, of same type and shape as `a`.
/// - Returns: 
///	product: Pairwise cross product of the vectors in `a` and `b`.
public func cross(operationName: String? = nil, a: Output, b: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Cross",
		name: (operationName ?? "Type"),
		input: [a, b],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Returns the complex conjugate of a complex number.
///Given a tensor `input` of complex numbers, this operation returns a tensor of
/// complex numbers that are the complex conjugate of each element in `input`. The
/// complex numbers in `input` must be of the form \\(a + bj\\), where  * a *  is the
/// real part and  * b *  is the imaginary part.
/// 
/// The complex conjugate returned by this operation is of the form \\(a - bj\\).
/// 
/// For example:
/// 
/// ```
/// # tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
/// tf.conj(input) ==> [-2.25 - 4.75j, 3.25 - 5.75j]
/// ```
/// - Parameter input: 
/// - Returns: 
///	output: 
public func conj(operationName: String? = nil, input: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Conj",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Returns the real part of a complex number.
///Given a tensor `input` of complex numbers, this operation returns a tensor of
/// type `float` that is the real part of each element in `input`. All elements in
/// `input` must be complex numbers of the form \\(a + bj\\), where  * a *  is the real
///  part returned by this operation and  * b *  is the imaginary part.
/// 
/// For example:
/// 
/// ```
/// # tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
/// tf.real(input) ==> [-2.25, 3.25]
/// ```
/// - Parameter input: 
/// - Parameter tout: 
/// - Returns: 
///	output: 
public func real(operationName: String? = nil, input: Output, tout: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Tout"] = tout
	let opspec = OpSpec(
		type: "Real",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Converts two real numbers to a complex number.
///Given a tensor `real` representing the real part of a complex number, and a
/// tensor `imag` representing the imaginary part of a complex number, this
/// operation returns complex numbers elementwise of the form \\(a + bj\\), where
///  * a *  represents the `real` part and  * b *  represents the `imag` part.
/// 
/// The input tensors `real` and `imag` must have the same shape.
/// 
/// For example:
/// 
/// ```
/// # tensor 'real' is [2.25, 3.25]
/// # tensor `imag` is [4.75, 5.75]
/// tf.complex(real, imag) ==> [[2.25 + 4.75j], [3.25 + 5.75j]]
/// ```
/// - Parameter real: 
/// - Parameter imag: 
/// - Parameter tout: 
/// - Returns: 
///	out: 
public func complex(operationName: String? = nil, real: Output, imag: Output, tout: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Tout"] = tout
	let opspec = OpSpec(
		type: "Complex",
		name: (operationName ?? "Type"),
		input: [real, imag],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes the "logical or" of elements across dimensions of a tensor.
///Reduces `input` along the dimensions given in `reduction_indices`. Unless
/// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
/// `reduction_indices`. If `keep_dims` is true, the reduced dimensions are
/// retained with length 1.
/// - Parameter input: The tensor to reduce.
/// - Parameter reductionIndices: The dimensions to reduce. Must be in the range
/// `[-rank(input), rank(input))`.
/// - Parameter keepDims: If true, retain reduced dimensions with length 1.
/// - Parameter tidx: 
/// - Returns: 
///	output: The reduced tensor.
public func any(operationName: String? = nil, input: Output, reductionIndices: Output, keepDims: Bool, tidx: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["keep_dims"] = keepDims
	attrs["Tidx"] = tidx
	let opspec = OpSpec(
		type: "Any",
		name: (operationName ?? "Type"),
		input: [input, reductionIndices],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes the mean along sparse segments of a tensor.
///Read @{$math_ops#segmentation$the section on segmentation} for an explanation of
/// segments.
/// 
/// Like `SegmentMean`, but `segment_ids` can have rank less than `data`'s first
/// dimension, selecting a subset of dimension 0, specified by `indices`.
/// - Parameter data: 
/// - Parameter indices: A 1-D tensor. Has same rank as `segment_ids`.
/// - Parameter segmentIds: A 1-D tensor. Values should be sorted and can be repeated.
/// - Parameter tidx: 
/// - Returns: 
///	output: Has same shape as data, except for dimension 0 which
/// has size `k`, the number of segments.
public func sparseSegmentMean(operationName: String? = nil, data: Output, indices: Output, segmentIds: Output, tidx: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Tidx"] = tidx
	let opspec = OpSpec(
		type: "SparseSegmentMean",
		name: (operationName ?? "Type"),
		input: [data, indices, segmentIds],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes the sum along segments of a tensor.
///Read @{$math_ops#segmentation$the section on segmentation} for an explanation of
/// segments.
/// 
/// Computes a tensor such that
/// `(output[i] = sum_{j...} data[j...]` where the sum is over tuples `j...` such
/// that `segment_ids[j...] == i`.  Unlike `SegmentSum`, `segment_ids`
/// need not be sorted and need not cover all values in the full
/// range of valid values.
/// 
/// If the sum is empty for a given segment ID `i`, `output[i] = 0`.
/// 
/// `num_segments` should equal the number of distinct segment IDs.
/// 
/// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
/// <img style="width:100%" src="https://www.tensorflow.org/images/UnsortedSegmentSum.png" alt>
/// </div>
/// - Parameter data: 
/// - Parameter segmentIds: A tensor whose shape is a prefix of `data.shape`.
/// - Parameter numSegments: 
/// - Parameter tindices: 
/// - Returns: 
///	output: Has same shape as data, except for the first `segment_ids.rank`
/// dimensions, which are replaced with a single dimension which has size
/// `num_segments`.
public func unsortedSegmentSum(operationName: String? = nil, data: Output, segmentIds: Output, numSegments: Output, tindices: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Tindices"] = tindices
	let opspec = OpSpec(
		type: "UnsortedSegmentSum",
		name: (operationName ?? "Type"),
		input: [data, segmentIds, numSegments],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes the product along segments of a tensor.
///Read @{$math_ops#segmentation$the section on segmentation} for an explanation of
/// segments.
/// 
/// Computes a tensor such that
/// \\(output_i = \prod_j data_j\\) where the product is over `j` such
/// that `segment_ids[j] == i`.
/// 
/// If the product is empty for a given segment ID `i`, `output[i] = 1`.
/// 
/// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
/// <img style="width:100%" src="https://www.tensorflow.org/images/SegmentProd.png" alt>
/// </div>
/// - Parameter data: 
/// - Parameter segmentIds: A 1-D tensor whose rank is equal to the rank of `data`'s
/// first dimension.  Values should be sorted and can be repeated.
/// - Parameter tindices: 
/// - Returns: 
///	output: Has same shape as data, except for dimension 0 which
/// has size `k`, the number of segments.
public func segmentProd(operationName: String? = nil, data: Output, segmentIds: Output, tindices: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Tindices"] = tindices
	let opspec = OpSpec(
		type: "SegmentProd",
		name: (operationName ?? "Type"),
		input: [data, segmentIds],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes the maximum of elements across dimensions of a tensor.
///Reduces `input` along the dimensions given in `reduction_indices`. Unless
/// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
/// `reduction_indices`. If `keep_dims` is true, the reduced dimensions are
/// retained with length 1.
/// - Parameter input: The tensor to reduce.
/// - Parameter reductionIndices: The dimensions to reduce. Must be in the range
/// `[-rank(input), rank(input))`.
/// - Parameter keepDims: If true, retain reduced dimensions with length 1.
/// - Parameter tidx: 
/// - Returns: 
///	output: The reduced tensor.
public func max(operationName: String? = nil, input: Output, reductionIndices: Output, keepDims: Bool, tidx: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["keep_dims"] = keepDims
	attrs["Tidx"] = tidx
	let opspec = OpSpec(
		type: "Max",
		name: (operationName ?? "Type"),
		input: [input, reductionIndices],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes the minimum of elements across dimensions of a tensor.
///Reduces `input` along the dimensions given in `reduction_indices`. Unless
/// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
/// `reduction_indices`. If `keep_dims` is true, the reduced dimensions are
/// retained with length 1.
/// - Parameter input: The tensor to reduce.
/// - Parameter reductionIndices: The dimensions to reduce. Must be in the range
/// `[-rank(input), rank(input))`.
/// - Parameter keepDims: If true, retain reduced dimensions with length 1.
/// - Parameter tidx: 
/// - Returns: 
///	output: The reduced tensor.
public func min(operationName: String? = nil, input: Output, reductionIndices: Output, keepDims: Bool, tidx: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["keep_dims"] = keepDims
	attrs["Tidx"] = tidx
	let opspec = OpSpec(
		type: "Min",
		name: (operationName ?? "Type"),
		input: [input, reductionIndices],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes the product of elements across dimensions of a tensor.
///Reduces `input` along the dimensions given in `reduction_indices`. Unless
/// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
/// `reduction_indices`. If `keep_dims` is true, the reduced dimensions are
/// retained with length 1.
/// - Parameter input: The tensor to reduce.
/// - Parameter reductionIndices: The dimensions to reduce. Must be in the range
/// `[-rank(input), rank(input))`.
/// - Parameter keepDims: If true, retain reduced dimensions with length 1.
/// - Parameter tidx: 
/// - Returns: 
///	output: The reduced tensor.
public func prod(operationName: String? = nil, input: Output, reductionIndices: Output, keepDims: Bool, tidx: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["keep_dims"] = keepDims
	attrs["Tidx"] = tidx
	let opspec = OpSpec(
		type: "Prod",
		name: (operationName ?? "Type"),
		input: [input, reductionIndices],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes the sum of elements across dimensions of a tensor.
///Reduces `input` along the dimensions given in `reduction_indices`. Unless
/// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
/// `reduction_indices`. If `keep_dims` is true, the reduced dimensions are
/// retained with length 1.
/// - Parameter input: The tensor to reduce.
/// - Parameter reductionIndices: The dimensions to reduce. Must be in the range
/// `[-rank(input), rank(input))`.
/// - Parameter keepDims: If true, retain reduced dimensions with length 1.
/// - Parameter tidx: 
/// - Returns: 
///	output: The reduced tensor.
public func sum(operationName: String? = nil, input: Output, reductionIndices: Output, keepDims: Bool, tidx: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["keep_dims"] = keepDims
	attrs["Tidx"] = tidx
	let opspec = OpSpec(
		type: "Sum",
		name: (operationName ?? "Type"),
		input: [input, reductionIndices],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes gradients for the scaled exponential linear (Selu) operation.
/// - Parameter gradients: The backpropagated gradients to the corresponding Selu operation.
/// - Parameter outputs: The outputs of the corresponding Selu operation.
/// - Returns: 
///	backprops: The gradients: `gradients  *  (outputs + scale  *  alpha)`
/// if outputs < 0, `scale  *  gradients` otherwise.
public func seluGrad(operationName: String? = nil, gradients: Output, outputs: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "SeluGrad",
		name: (operationName ?? "Type"),
		input: [gradients, outputs],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Multiply matrix "a" by matrix "b".
///The inputs must be two-dimensional matrices and the inner dimension of "a" must
/// match the outer dimension of "b". This op is optimized for the case where at
/// least one of "a" or "b" is sparse. The breakeven for using this versus a dense
/// matrix multiply on one platform was 30% zero values in the sparse matrix.
/// 
/// The gradient computation of this operation will only take advantage of sparsity
/// in the input gradient when that gradient comes from a Relu.
/// - Parameter a: 
/// - Parameter b: 
/// - Parameter transposeA: 
/// - Parameter transposeB: 
/// - Parameter aIsSparse: 
/// - Parameter bIsSparse: 
/// - Parameter ta: 
/// - Parameter tb: 
/// - Returns: 
///	product: 
public func sparseMatMul(operationName: String? = nil, a: Output, b: Output, transposeA: Bool, transposeB: Bool, aIsSparse: Bool, bIsSparse: Bool, ta: Any.Type, tb: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["transpose_a"] = transposeA
	attrs["transpose_b"] = transposeB
	attrs["a_is_sparse"] = aIsSparse
	attrs["b_is_sparse"] = bIsSparse
	attrs["Ta"] = ta
	attrs["Tb"] = tb
	let opspec = OpSpec(
		type: "SparseMatMul",
		name: (operationName ?? "Type"),
		input: [a, b],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Multiply the matrix "a" by the matrix "b".
///The inputs must be two-dimensional matrices and the inner dimension of
/// "a" (after being transposed if transpose_a is true) must match the
/// outer dimension of "b" (after being transposed if transposed_b is
/// true).
/// 
///  * Note * : The default kernel implementation for MatMul on GPUs uses
/// cublas.
/// - Parameter a: 
/// - Parameter b: 
/// - Parameter transposeA: If true, "a" is transposed before multiplication.
/// - Parameter transposeB: If true, "b" is transposed before multiplication.
/// - Returns: 
///	product: 
public func matMul(operationName: String? = nil, a: Output, b: Output, transposeA: Bool, transposeB: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["transpose_a"] = transposeA
	attrs["transpose_b"] = transposeB
	let opspec = OpSpec(
		type: "MatMul",
		name: (operationName ?? "Type"),
		input: [a, b],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Returns the truth value of x AND y element-wise.
/// * NOTE * : `LogicalAnd` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
/// - Parameter x: 
/// - Parameter y: 
/// - Returns: 
///	z: 
public func logicalAnd(operationName: String? = nil, x: Output, y: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "LogicalAnd",
		name: (operationName ?? "Type"),
		input: [x, y],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Returns the truth value of abs(x-y) < tolerance element-wise.
/// - Parameter x: 
/// - Parameter y: 
/// - Parameter tolerance: 
/// - Returns: 
///	z: 
public func approximateEqual(operationName: String? = nil, x: Output, y: Output, tolerance: Float) throws -> Output { 
	var attrs = [String : Any]()
	attrs["tolerance"] = tolerance
	let opspec = OpSpec(
		type: "ApproximateEqual",
		name: (operationName ?? "Type"),
		input: [x, y],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Returns the truth value of (x >= y) element-wise.
/// * NOTE * : `GreaterEqual` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
/// - Parameter x: 
/// - Parameter y: 
/// - Returns: 
///	z: 
public func greaterEqual(operationName: String? = nil, x: Output, y: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "GreaterEqual",
		name: (operationName ?? "Type"),
		input: [x, y],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Returns the truth value of (x <= y) element-wise.
/// * NOTE * : `LessEqual` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
/// - Parameter x: 
/// - Parameter y: 
/// - Returns: 
///	z: 
public func lessEqual(operationName: String? = nil, x: Output, y: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "LessEqual",
		name: (operationName ?? "Type"),
		input: [x, y],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Compute the polygamma function \\(\psi// ^{(n)}(x)\\).
///The polygamma function is defined as:
/// 
/// 
/// \\(\psi// ^{(n)}(x) = \frac{d// ^n}{dx// ^n} \psi(x)\\)
/// 
/// where \\(\psi(x)\\) is the digamma function.
/// - Parameter a: 
/// - Parameter x: 
/// - Returns: 
///	z: 
public func polygamma(operationName: String? = nil, a: Output, x: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Polygamma",
		name: (operationName ?? "Type"),
		input: [a, x],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Compute the lower regularized incomplete Gamma function `Q(a, x)`.
///The lower regularized incomplete Gamma function is defined as:
/// 
/// 
/// \\(P(a, x) = gamma(a, x) / Gamma(a) = 1 - Q(a, x)\\)
/// 
/// where
/// 
/// \\(gamma(a, x) = int_{0}// ^{x} t// ^{a-1} exp(-t) dt\\)
/// 
/// is the lower incomplete Gamma function.
/// 
/// Note, above `Q(a, x)` (`Igammac`) is the upper regularized complete
/// Gamma function.
/// - Parameter a: 
/// - Parameter x: 
/// - Returns: 
///	z: 
public func igamma(operationName: String? = nil, a: Output, x: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Igamma",
		name: (operationName ?? "Type"),
		input: [a, x],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Compute the upper regularized incomplete Gamma function `Q(a, x)`.
///The upper regularized incomplete Gamma function is defined as:
/// 
/// \\(Q(a, x) = Gamma(a, x) / Gamma(a) = 1 - P(a, x)\\)
/// 
/// where
/// 
/// \\(Gamma(a, x) = int_{x}// ^{\infty} t// ^{a-1} exp(-t) dt\\)
/// 
/// is the upper incomplete Gama function.
/// 
/// Note, above `P(a, x)` (`Igamma`) is the lower regularized complete
/// Gamma function.
/// - Parameter a: 
/// - Parameter x: 
/// - Returns: 
///	z: 
public func igammac(operationName: String? = nil, a: Output, x: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Igammac",
		name: (operationName ?? "Type"),
		input: [a, x],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Returns element-wise remainder of division. This emulates C semantics in that
///the result here is consistent with a truncating divide. E.g. `truncate(x / y)  * 
/// y + truncate_mod(x, y) = x`.
/// 
///  * NOTE * : `Mod` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
/// - Parameter x: 
/// - Parameter y: 
/// - Returns: 
///	z: 
public func mod(operationName: String? = nil, x: Output, y: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Mod",
		name: (operationName ?? "Type"),
		input: [x, y],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Returns the max of x and y (i.e. x > y ? x : y) element-wise.
/// * NOTE * : `Maximum` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
/// - Parameter x: 
/// - Parameter y: 
/// - Returns: 
///	z: 
public func maximum(operationName: String? = nil, x: Output, y: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Maximum",
		name: (operationName ?? "Type"),
		input: [x, y],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Returns (x - y)(x - y) element-wise.
/// * NOTE * : `SquaredDifference` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
/// - Parameter x: 
/// - Parameter y: 
/// - Parameter mklX: 
/// - Parameter mklY: 
/// - Returns: 
///	z: 
///	mkl_z: 
public func mklSquaredDifference(operationName: String? = nil, x: Output, y: Output, mklX: Output, mklY: Output) throws -> (z: Output, mklZ: Output) { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "_MklSquaredDifference",
		name: (operationName ?? "Type"),
		input: [x, y, mklX, mklY],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (z: op.output(at: 0), mklZ: op.output(at: 1))
} 

///Returns (x - y)(x - y) element-wise.
/// * NOTE * : `SquaredDifference` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
/// - Parameter x: 
/// - Parameter y: 
/// - Returns: 
///	z: 
public func squaredDifference(operationName: String? = nil, x: Output, y: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "SquaredDifference",
		name: (operationName ?? "Type"),
		input: [x, y],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Returns x / y element-wise for real types.
///If `x` and `y` are reals, this will return the floating-point division.
/// 
///  * NOTE * : `Div` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
/// - Parameter x: 
/// - Parameter y: 
/// - Returns: 
///	z: 
public func realDiv(operationName: String? = nil, x: Output, y: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "RealDiv",
		name: (operationName ?? "Type"),
		input: [x, y],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Returns x / y element-wise for integer types.
///Truncation designates that negative numbers will round fractional quantities
/// toward zero. I.e. -7 / 5 = 1. This matches C semantics but it is different
/// than Python semantics. See `FloorDiv` for a division function that matches
/// Python Semantics.
/// 
///  * NOTE * : `TruncateDiv` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
/// - Parameter x: 
/// - Parameter y: 
/// - Returns: 
///	z: 
public func truncateDiv(operationName: String? = nil, x: Output, y: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "TruncateDiv",
		name: (operationName ?? "Type"),
		input: [x, y],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Returns x  *  y element-wise.
/// * NOTE * : `Mul` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
/// - Parameter x: 
/// - Parameter y: 
/// - Parameter mklX: 
/// - Parameter mklY: 
/// - Returns: 
///	z: 
///	mkl_z: 
public func mklMul(operationName: String? = nil, x: Output, y: Output, mklX: Output, mklY: Output) throws -> (z: Output, mklZ: Output) { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "_MklMul",
		name: (operationName ?? "Type"),
		input: [x, y, mklX, mklY],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (z: op.output(at: 0), mklZ: op.output(at: 1))
} 

///Returns x + y element-wise.
/// * NOTE * : `Add` supports broadcasting. `AddN` does not. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
/// - Parameter x: 
/// - Parameter y: 
/// - Returns: 
///	z: 
public func add(operationName: String? = nil, x: Output, y: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Add",
		name: (operationName ?? "Type"),
		input: [x, y],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Returns element-wise smallest integer in not less than x.
/// - Parameter x: 
/// - Returns: 
///	y: 
public func ceil(operationName: String? = nil, x: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Ceil",
		name: (operationName ?? "Type"),
		input: [x],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Returns which elements of x are finite.
///@compatibility(numpy)
/// Equivalent to np.isfinite
/// @end_compatibility
/// - Parameter x: 
/// - Returns: 
///	y: 
public func isFinite(operationName: String? = nil, x: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "IsFinite",
		name: (operationName ?? "Type"),
		input: [x],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Performs 3D max pooling on the input.
/// - Parameter input: Shape `[batch, depth, rows, cols, channels]` tensor to pool over.
/// - Parameter ksize: 1-D tensor of length 5. The size of the window for each dimension of
/// the input tensor. Must have `ksize[0] = ksize[4] = 1`.
/// - Parameter strides: 1-D tensor of length 5. The stride of the sliding window for each
/// dimension of `input`. Must have `strides[0] = strides[4] = 1`.
/// - Parameter padding: The type of padding algorithm to use.
/// - Parameter dataFormat: The data format of the input and output data. With the
/// default format "NDHWC", the data is stored in the order of:
///     [batch, in_depth, in_height, in_width, in_channels].
/// Alternatively, the format could be "NCDHW", the data storage order is:
///     [batch, in_channels, in_depth, in_height, in_width].
/// - Returns: 
///	output: The max pooled output tensor.
public func maxPool3D(operationName: String? = nil, input: Output, ksize: [Int64], strides: [Int64], padding: String, dataFormat: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["ksize"] = ksize
	attrs["strides"] = strides
	attrs["padding"] = padding
	attrs["data_format"] = dataFormat
	let opspec = OpSpec(
		type: "MaxPool3D",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Returns which elements of x are Inf.
///@compatibility(numpy)
/// Equivalent to np.isinf
/// @end_compatibility
/// - Parameter x: 
/// - Returns: 
///	y: 
public func isInf(operationName: String? = nil, x: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "IsInf",
		name: (operationName ?? "Type"),
		input: [x],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Finds values and indices of the `k` largest elements for the last dimension.
///If the input is a vector (rank-1), finds the `k` largest entries in the vector
/// and outputs their values and indices as vectors.  Thus `values[j]` is the
/// `j`-th largest entry in `input`, and its index is `indices[j]`.
/// 
/// For matrices (resp. higher rank input), computes the top `k` entries in each
/// row (resp. vector along the last dimension).  Thus,
/// 
///     values.shape = indices.shape = input.shape[:-1] + [k]
/// 
/// If two elements are equal, the lower-index element appears first.
/// - Parameter input: 1-D or higher with last dimension at least `k`.
/// - Parameter k: 0-D.  Number of top elements to look for along the last dimension (along each
/// row for matrices).
/// - Parameter sorted: If true the resulting `k` elements will be sorted by the values in
/// descending order.
/// - Returns: 
///	values: The `k` largest elements along each last dimensional slice.
///	indices: The indices of `values` within the last dimension of `input`.
public func topKV2(operationName: String? = nil, input: Output, k: Output, sorted: Bool) throws -> (values: Output, indices: Output) { 
	var attrs = [String : Any]()
	attrs["sorted"] = sorted
	let opspec = OpSpec(
		type: "TopKV2",
		name: (operationName ?? "Type"),
		input: [input, k],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (values: op.output(at: 0), indices: op.output(at: 1))
} 

///Computes cos of x element-wise.
/// - Parameter x: 
/// - Returns: 
///	y: 
public func cos(operationName: String? = nil, x: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Cos",
		name: (operationName ?? "Type"),
		input: [x],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes sin of x element-wise.
/// - Parameter x: 
/// - Returns: 
///	y: 
public func sin(operationName: String? = nil, x: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Sin",
		name: (operationName ?? "Type"),
		input: [x],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes the gradient of the sigmoid of `x` wrt its input.
///Specifically, `grad = dy  *  y  *  (1 - y)`, where `y = sigmoid(x)`, and
/// `dy` is the corresponding input gradient.
/// - Parameter y: 
/// - Parameter dy: 
/// - Returns: 
///	z: 
public func sigmoidGrad(operationName: String? = nil, y: Output, dy: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "SigmoidGrad",
		name: (operationName ?? "Type"),
		input: [y, dy],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes Psi, the derivative of Lgamma (the log of the absolute value of
///`Gamma(x)`), element-wise.
/// - Parameter x: 
/// - Returns: 
///	y: 
public func digamma(operationName: String? = nil, x: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Digamma",
		name: (operationName ?? "Type"),
		input: [x],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes the log of the absolute value of `Gamma(x)` element-wise.
/// - Parameter x: 
/// - Returns: 
///	y: 
public func lgamma(operationName: String? = nil, x: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Lgamma",
		name: (operationName ?? "Type"),
		input: [x],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes inverse hyperbolic cosine of x element-wise.
/// - Parameter x: 
/// - Returns: 
///	y: 
public func acosh(operationName: String? = nil, x: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Acosh",
		name: (operationName ?? "Type"),
		input: [x],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes inverse hyperbolic sine of x element-wise.
/// - Parameter x: 
/// - Returns: 
///	y: 
public func asinh(operationName: String? = nil, x: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Asinh",
		name: (operationName ?? "Type"),
		input: [x],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes asin of x element-wise.
/// - Parameter x: 
/// - Returns: 
///	y: 
public func asin(operationName: String? = nil, x: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Asin",
		name: (operationName ?? "Type"),
		input: [x],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes natural logarithm of (1 + x) element-wise.
///I.e., \\(y = \log_e (1 + x)\\).
/// - Parameter x: 
/// - Returns: 
///	y: 
public func log1p(operationName: String? = nil, x: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Log1p",
		name: (operationName ?? "Type"),
		input: [x],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Convert the quantized 'input' tensor into a lower-precision 'output', using the
///output range specified with 'requested_output_min' and 'requested_output_max'.
/// 
/// [input_min, input_max] are scalar floats that specify the range for the float
/// interpretation of the 'input' data. For example, if input_min is -1.0f and
/// input_max is 1.0f, and we are dealing with quint16 quantized data, then a 0
/// value in the 16-bit data should be interpreted as -1.0f, and a 65535 means 1.0f.
/// - Parameter input: 
/// - Parameter inputMin: The float value that the minimum quantized input value represents.
/// - Parameter inputMax: The float value that the maximum quantized input value represents.
/// - Parameter requestedOutputMin: The float value that the minimum quantized output value represents.
/// - Parameter requestedOutputMax: The float value that the maximum quantized output value represents.
/// - Parameter tinput: The type of the input.
/// - Parameter outType: The type of the output. Should be a lower bit depth than Tinput.
/// - Returns: 
///	output: 
///	output_min: The requested_output_min value is copied into this output.
///	output_max: The requested_output_max value is copied into this output.
public func requantize(operationName: String? = nil, input: Output, inputMin: Output, inputMax: Output, requestedOutputMin: Output, requestedOutputMax: Output, tinput: Any.Type, outType: Any.Type) throws -> (output: Output, outputMin: Output, outputMax: Output) { 
	var attrs = [String : Any]()
	attrs["Tinput"] = tinput
	attrs["out_type"] = outType
	let opspec = OpSpec(
		type: "Requantize",
		name: (operationName ?? "Type"),
		input: [input, inputMin, inputMax, requestedOutputMin, requestedOutputMax],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (output: op.output(at: 0), outputMin: op.output(at: 1), outputMax: op.output(at: 2))
} 

///Computes exponential of x - 1 element-wise.
///I.e., \\(y = (\exp x) - 1\\).
/// - Parameter x: 
/// - Returns: 
///	y: 
public func expm1(operationName: String? = nil, x: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Expm1",
		name: (operationName ?? "Type"),
		input: [x],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes exponential of x element-wise.  \\(y = e// ^x\\).
/// - Parameter x: 
/// - Returns: 
///	y: 
public func exp(operationName: String? = nil, x: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Exp",
		name: (operationName ?? "Type"),
		input: [x],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes the grayscale dilation of 4-D `input` and 3-D `filter` tensors.
///The `input` tensor has shape `[batch, in_height, in_width, depth]` and the
/// `filter` tensor has shape `[filter_height, filter_width, depth]`, i.e., each
/// input channel is processed independently of the others with its own structuring
/// function. The `output` tensor has shape
/// `[batch, out_height, out_width, depth]`. The spatial dimensions of the output
/// tensor depend on the `padding` algorithm. We currently only support the default
/// "NHWC" `data_format`.
/// 
/// In detail, the grayscale morphological 2-D dilation is the max-sum correlation
/// (for consistency with `conv2d`, we use unmirrored filters):
/// 
///     output[b, y, x, c] =
///        max_{dy, dx} input[b,
///                           strides[1]  *  y + rates[1]  *  dy,
///                           strides[2]  *  x + rates[2]  *  dx,
///                           c] +
///                     filter[dy, dx, c]
/// 
/// Max-pooling is a special case when the filter has size equal to the pooling
/// kernel size and contains all zeros.
/// 
/// Note on duality: The dilation of `input` by the `filter` is equal to the
/// negation of the erosion of `-input` by the reflected `filter`.
/// - Parameter input: 4-D with shape `[batch, in_height, in_width, depth]`.
/// - Parameter filter: 3-D with shape `[filter_height, filter_width, depth]`.
/// - Parameter strides: The stride of the sliding window for each dimension of the input
/// tensor. Must be: `[1, stride_height, stride_width, 1]`.
/// - Parameter rates: The input stride for atrous morphological dilation. Must be:
/// `[1, rate_height, rate_width, 1]`.
/// - Parameter padding: The type of padding algorithm to use.
/// - Returns: 
///	output: 4-D with shape `[batch, out_height, out_width, depth]`.
public func dilation2D(operationName: String? = nil, input: Output, filter: Output, strides: [Int64], rates: [Int64], padding: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["strides"] = strides
	attrs["rates"] = rates
	attrs["padding"] = padding
	let opspec = OpSpec(
		type: "Dilation2D",
		name: (operationName ?? "Type"),
		input: [input, filter],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes the gradient for the rsqrt of `x` wrt its input.
///Specifically, `grad = dy  *  -0.5  *  y// ^3`, where `y = rsqrt(x)`, and `dy`
/// is the corresponding input gradient.
/// - Parameter y: 
/// - Parameter dy: 
/// - Returns: 
///	z: 
public func rsqrtGrad(operationName: String? = nil, y: Output, dy: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "RsqrtGrad",
		name: (operationName ?? "Type"),
		input: [y, dy],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes reciprocal of square root of x element-wise.
///I.e., \\(y = 1 / \sqrt{x}\\).
/// - Parameter x: 
/// - Returns: 
///	y: 
public func rsqrt(operationName: String? = nil, x: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Rsqrt",
		name: (operationName ?? "Type"),
		input: [x],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes the gradient for the sqrt of `x` wrt its input.
///Specifically, `grad = dy  *  0.5 / y`, where `y = sqrt(x)`, and `dy`
/// is the corresponding input gradient.
/// - Parameter y: 
/// - Parameter dy: 
/// - Returns: 
///	z: 
public func sqrtGrad(operationName: String? = nil, y: Output, dy: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "SqrtGrad",
		name: (operationName ?? "Type"),
		input: [y, dy],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes the gradient for the inverse of `x` wrt its input.
///Specifically, `grad = -dy  *  y * y`, where `y = 1/x`, and `dy`
/// is the corresponding input gradient.
/// - Parameter y: 
/// - Parameter dy: 
/// - Returns: 
///	z: 
public func invGrad(operationName: String? = nil, y: Output, dy: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "InvGrad",
		name: (operationName ?? "Type"),
		input: [y, dy],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes the reciprocal of x element-wise.
///I.e., \\(y = 1 / x\\).
/// - Parameter x: 
/// - Returns: 
///	y: 
public func inv(operationName: String? = nil, x: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Inv",
		name: (operationName ?? "Type"),
		input: [x],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Cast x of type SrcT to y of DstT.
///_HostCast requires its input and produces its output in host memory.
/// - Parameter x: 
/// - Parameter srcT: 
/// - Parameter dstT: 
/// - Returns: 
///	y: 
public func hostCast(operationName: String? = nil, x: Output, srcT: Any.Type, dstT: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["SrcT"] = srcT
	attrs["DstT"] = dstT
	let opspec = OpSpec(
		type: "_HostCast",
		name: (operationName ?? "Type"),
		input: [x],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Multiplies slices of two tensors in batches.
///Multiplies all slices of `Tensor` `x` and `y` (each slice can be
/// viewed as an element of a batch), and arranges the individual results
/// in a single output tensor of the same batch size. Each of the
/// individual slices can optionally be adjointed (to adjoint a matrix
/// means to transpose and conjugate it) before multiplication by setting
/// the `adj_x` or `adj_y` flag to `True`, which are by default `False`.
/// 
/// The input tensors `x` and `y` are 2-D or higher with shape `[..., r_x, c_x]`
/// and `[..., r_y, c_y]`.
/// 
/// The output tensor is 2-D or higher with shape `[..., r_o, c_o]`, where:
/// 
///     r_o = c_x if adj_x else r_x
///     c_o = r_y if adj_y else c_y
/// 
/// It is computed as:
/// 
///     output[..., :, :] = matrix(x[..., :, :])  *  matrix(y[..., :, :])
/// - Parameter x: 2-D or higher with shape `[..., r_x, c_x]`.
/// - Parameter y: 2-D or higher with shape `[..., r_y, c_y]`.
/// - Parameter adjX: If `True`, adjoint the slices of `x`. Defaults to `False`.
/// - Parameter adjY: If `True`, adjoint the slices of `y`. Defaults to `False`.
/// - Returns: 
///	output: 3-D or higher with shape `[..., r_o, c_o]`
public func batchMatMul(operationName: String? = nil, x: Output, y: Output, adjX: Bool, adjY: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["adj_x"] = adjX
	attrs["adj_y"] = adjY
	let opspec = OpSpec(
		type: "BatchMatMul",
		name: (operationName ?? "Type"),
		input: [x, y],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Returns the element-wise sum of a list of tensors.
///`tf.accumulate_n_v2` performs the same operation as `tf.add_n`, but does not
/// wait for all of its inputs to be ready before beginning to sum. This can
/// save memory if inputs are ready at different times, since minimum temporary
/// storage is proportional to the output size rather than the inputs size.
/// 
/// Unlike the original `accumulate_n`, `accumulate_n_v2` is differentiable.
/// 
/// Returns a `Tensor` of same shape and type as the elements of `inputs`.
/// - Parameter inputs: A list of `Tensor` objects, each with same shape and type.
/// - Parameter n: 
/// - Parameter shape: Shape of elements of `inputs`.
/// - Returns: 
///	sum: 
public func accumulateNV2(operationName: String? = nil, inputs: [Output], n: UInt8, shape: Shape) throws -> Output { 
	var attrs = [String : Any]()
	attrs["N"] = n
	attrs["shape"] = shape
	let opspec = OpSpec(
		type: "AccumulateNV2",
		name: (operationName ?? "Type"),
		input: [inputs],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 


/// - Parameter input: 
/// - Parameter diagonal: 
/// - Returns: 
///	output: 
public func batchMatrixSetDiag(operationName: String? = nil, input: Output, diagonal: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "BatchMatrixSetDiag",
		name: (operationName ?? "Type"),
		input: [input, diagonal],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes the mean along segments of a tensor.
///Read @{$math_ops#segmentation$the section on segmentation} for an explanation of
/// segments.
/// 
/// Computes a tensor such that
/// \\(output_i = \frac{\sum_j data_j}{N}\\) where `mean` is
/// over `j` such that `segment_ids[j] == i` and `N` is the total number of
/// values summed.
/// 
/// If the mean is empty for a given segment ID `i`, `output[i] = 0`.
/// 
/// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
/// <img style="width:100%" src="https://www.tensorflow.org/images/SegmentMean.png" alt>
/// </div>
/// - Parameter data: 
/// - Parameter segmentIds: A 1-D tensor whose rank is equal to the rank of `data`'s
/// first dimension.  Values should be sorted and can be repeated.
/// - Parameter tindices: 
/// - Returns: 
///	output: Has same shape as data, except for dimension 0 which
/// has size `k`, the number of segments.
public func segmentMean(operationName: String? = nil, data: Output, segmentIds: Output, tindices: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Tindices"] = tindices
	let opspec = OpSpec(
		type: "SegmentMean",
		name: (operationName ?? "Type"),
		input: [data, segmentIds],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Quantized Instance normalization.
/// - Parameter x: A 4D input Tensor.
/// - Parameter xMin: The value represented by the lowest quantized input.
/// - Parameter xMax: The value represented by the highest quantized input.
/// - Parameter outputRangeGiven: If True, `given_y_min` and `given_y_min`
/// and `given_y_max` are used as the output range. Otherwise,
/// the implementation computes the output range.
/// - Parameter givenYMin: Output in `y_min` if `output_range_given` is True.
/// - Parameter givenYMax: Output in `y_max` if `output_range_given` is True.
/// - Parameter varianceEpsilon: A small float number to avoid dividing by 0.
/// - Parameter minSeparation: Minimum value of `y_max - y_min`
/// - Returns: 
///	y: A 4D Tensor.
///	y_min: The value represented by the lowest quantized output.
///	y_max: The value represented by the highest quantized output.
public func quantizedInstanceNorm(operationName: String? = nil, x: Output, xMin: Output, xMax: Output, outputRangeGiven: Bool, givenYMin: Float, givenYMax: Float, varianceEpsilon: Float, minSeparation: Float) throws -> (y: Output, yMin: Output, yMax: Output) { 
	var attrs = [String : Any]()
	attrs["output_range_given"] = outputRangeGiven
	attrs["given_y_min"] = givenYMin
	attrs["given_y_max"] = givenYMax
	attrs["variance_epsilon"] = varianceEpsilon
	attrs["min_separation"] = minSeparation
	let opspec = OpSpec(
		type: "QuantizedInstanceNorm",
		name: (operationName ?? "Type"),
		input: [x, xMin, xMax],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (y: op.output(at: 0), yMin: op.output(at: 1), yMax: op.output(at: 2))
} 

///Concatenates quantized tensors along one dimension.
/// - Parameter concatDim: 0-D.  The dimension along which to concatenate.  Must be in the
/// range [0, rank(values)).
/// - Parameter values: The `N` Tensors to concatenate. Their ranks and types must match,
/// and their sizes must match in all dimensions except `concat_dim`.
/// - Parameter inputMins: The minimum scalar values for each of the input tensors.
/// - Parameter inputMaxes: The maximum scalar values for each of the input tensors.
/// - Parameter n: 
/// - Returns: 
///	output: A `Tensor` with the concatenation of values stacked along the
/// `concat_dim` dimension.  This tensor's shape matches that of `values` except
/// in `concat_dim` where it has the sum of the sizes.
///	output_min: The float value that the minimum quantized output value represents.
///	output_max: The float value that the maximum quantized output value represents.
public func quantizedConcat(operationName: String? = nil, concatDim: Output, values: [Output], inputMins: [Output], inputMaxes: [Output], n: UInt8) throws -> (output: Output, outputMin: Output, outputMax: Output) { 
	var attrs = [String : Any]()
	attrs["N"] = n
	let opspec = OpSpec(
		type: "QuantizedConcat",
		name: (operationName ?? "Type"),
		input: [concatDim, values, inputMins, inputMaxes],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (output: op.output(at: 0), outputMin: op.output(at: 1), outputMax: op.output(at: 2))
} 

///Use QuantizeAndDequantizeV2 instead.
/// - Parameter input: 
/// - Parameter signedInput: 
/// - Parameter numBits: 
/// - Parameter rangeGiven: 
/// - Parameter inputMin: 
/// - Parameter inputMax: 
/// - Returns: 
///	output: 
public func quantizeAndDequantize(operationName: String? = nil, input: Output, signedInput: Bool, numBits: UInt8, rangeGiven: Bool, inputMin: Float, inputMax: Float) throws -> Output { 
	var attrs = [String : Any]()
	attrs["signed_input"] = signedInput
	attrs["num_bits"] = numBits
	attrs["range_given"] = rangeGiven
	attrs["input_min"] = inputMin
	attrs["input_max"] = inputMax
	let opspec = OpSpec(
		type: "QuantizeAndDequantize",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes the sum along sparse segments of a tensor divided by the sqrt of N.
///N is the size of the segment being reduced.
/// 
/// Read @{$math_ops#segmentation$the section on segmentation} for an explanation of
/// segments.
/// - Parameter data: 
/// - Parameter indices: A 1-D tensor. Has same rank as `segment_ids`.
/// - Parameter segmentIds: A 1-D tensor. Values should be sorted and can be repeated.
/// - Parameter tidx: 
/// - Returns: 
///	output: Has same shape as data, except for dimension 0 which
/// has size `k`, the number of segments.
public func sparseSegmentSqrtN(operationName: String? = nil, data: Output, indices: Output, segmentIds: Output, tidx: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Tidx"] = tidx
	let opspec = OpSpec(
		type: "SparseSegmentSqrtN",
		name: (operationName ?? "Type"),
		input: [data, indices, segmentIds],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///DepthToSpace for tensors of type T.
///Rearranges data from depth into blocks of spatial data.
/// This is the reverse transformation of SpaceToDepth. More specifically,
/// this op outputs a copy of the input tensor where values from the `depth`
/// dimension are moved in spatial blocks to the `height` and `width` dimensions.
/// The attr `block_size` indicates the input block size and how the data is moved.
/// 
///    *  Chunks of data of size `block_size  *  block_size` from depth are rearranged
///     into non-overlapping blocks of size `block_size x block_size`
///    *  The width the output tensor is `input_depth  *  block_size`, whereas the
///     height is `input_height  *  block_size`.
///    *  The Y, X coordinates within each block of the output image are determined
///     by the high order component of the input channel index.
///    *  The depth of the input tensor must be divisible by
///     `block_size  *  block_size`.
/// 
/// The `data_format` attr specifies the layout of the input and output tensors
/// with the following options:
///   "NHWC": `[ batch, height, width, channels ]`
///   "NCHW": `[ batch, channels, height, width ]`
///   "NCHW_VECT_C":
///       `qint8 [ batch, channels / 4, height, width, channels % 4 ]`
/// 
/// It is useful to consider the operation as transforming a 6-D Tensor.
/// e.g. for data_format = NHWC,
///      Each element in the input tensor can be specified via 6 coordinates,
///      ordered by decreasing memory layout significance as:
///      n,iY,iX,bY,bX,oC  (where n=batch index, iX, iY means X or Y coordinates
///                         within the input image, bX, bY means coordinates
///                         within the output block, oC means output channels).
///      The output would be the input transposed to the following layout:
///      n,iY,bY,iX,bX,oC
/// 
/// This operation is useful for resizing the activations between convolutions
/// (but keeping all data), e.g. instead of pooling. It is also useful for training
/// purely convolutional models.
/// 
/// For example, given an input of shape `[1, 1, 1, 4]`, data_format = "NHWC" and
/// block_size = 2:
/// 
/// ```
/// x = [[[[1, 2, 3, 4]]]]
/// 
/// ```
/// 
/// This operation will output a tensor of shape `[1, 2, 2, 1]`:
/// 
/// ```
///    [[[[1], [2]],
///      [[3], [4]]]]
/// ```
/// 
/// Here, the input has a batch of 1 and each batch element has shape `[1, 1, 4]`,
/// the corresponding output will have 2x2 elements and will have a depth of
/// 1 channel (1 = `4 / (block_size  *  block_size)`).
/// The output element shape is `[2, 2, 1]`.
/// 
/// For an input tensor with larger depth, here of shape `[1, 1, 1, 12]`, e.g.
/// 
/// ```
/// x = [[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]]]
/// ```
/// 
/// This operation, for block size of 2, will return the following tensor of shape
/// `[1, 2, 2, 3]`
/// 
/// ```
///    [[[[1, 2, 3], [4, 5, 6]],
///      [[7, 8, 9], [10, 11, 12]]]]
/// 
/// ```
/// 
/// Similarly, for the following input of shape `[1 2 2 4]`, and a block size of 2:
/// 
/// ```
/// x =  [[[[1, 2, 3, 4],
///        [5, 6, 7, 8]],
///       [[9, 10, 11, 12],
///        [13, 14, 15, 16]]]]
/// ```
/// 
/// the operator will return the following tensor of shape `[1 4 4 1]`:
/// 
/// ```
/// x = [[[ [1],   [2],  [5],  [6]],
///       [ [3],   [4],  [7],  [8]],
///       [ [9],  [10], [13],  [14]],
///       [ [11], [12], [15],  [16]]]]
/// 
/// ```
/// - Parameter input: 
/// - Parameter blockSize: The size of the spatial block, same as in Space2Depth.
/// - Parameter dataFormat: 
/// - Returns: 
///	output: 
public func depthToSpace(operationName: String? = nil, input: Output, blockSize: UInt8, dataFormat: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["block_size"] = blockSize
	attrs["data_format"] = dataFormat
	let opspec = OpSpec(
		type: "DepthToSpace",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///SpaceToDepth for tensors of type T.
///Rearranges blocks of spatial data, into depth. More specifically,
/// this op outputs a copy of the input tensor where values from the `height`
/// and `width` dimensions are moved to the `depth` dimension.
/// The attr `block_size` indicates the input block size.
/// 
///    *  Non-overlapping blocks of size `block_size x block size` are rearranged
///     into depth at each location.
///    *  The depth of the output tensor is `block_size  *  block_size  *  input_depth`.
///    *  The Y, X coordinates within each block of the input become the high order
///     component of the output channel index.
///    *  The input tensor's height and width must be divisible by block_size.
/// 
/// The `data_format` attr specifies the layout of the input and output tensors
/// with the following options:
///   "NHWC": `[ batch, height, width, channels ]`
///   "NCHW": `[ batch, channels, height, width ]`
///   "NCHW_VECT_C":
///       `qint8 [ batch, channels / 4, height, width, channels % 4 ]`
/// 
/// It is useful to consider the operation as transforming a 6-D Tensor.
/// e.g. for data_format = NHWC,
///      Each element in the input tensor can be specified via 6 coordinates,
///      ordered by decreasing memory layout significance as:
///      n,oY,bY,oX,bX,iC  (where n=batch index, oX, oY means X or Y coordinates
///                         within the output image, bX, bY means coordinates
///                         within the input block, iC means input channels).
///      The output would be a transpose to the following layout:
///      n,oY,oX,bY,bX,iC
/// 
/// This operation is useful for resizing the activations between convolutions
/// (but keeping all data), e.g. instead of pooling. It is also useful for training
/// purely convolutional models.
/// 
/// For example, given an input of shape `[1, 2, 2, 1]`, data_format = "NHWC" and
/// block_size = 2:
/// 
/// ```
/// x = [[[[1], [2]],
///       [[3], [4]]]]
/// ```
/// 
/// This operation will output a tensor of shape `[1, 1, 1, 4]`:
/// 
/// ```
/// [[[[1, 2, 3, 4]]]]
/// ```
/// 
/// Here, the input has a batch of 1 and each batch element has shape `[2, 2, 1]`,
/// the corresponding output will have a single element (i.e. width and height are
/// both 1) and will have a depth of 4 channels (1  *  block_size  *  block_size).
/// The output element shape is `[1, 1, 4]`.
/// 
/// For an input tensor with larger depth, here of shape `[1, 2, 2, 3]`, e.g.
/// 
/// ```
/// x = [[[[1, 2, 3], [4, 5, 6]],
///       [[7, 8, 9], [10, 11, 12]]]]
/// ```
/// 
/// This operation, for block_size of 2, will return the following tensor of shape
/// `[1, 1, 1, 12]`
/// 
/// ```
/// [[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]]]
/// ```
/// 
/// Similarly, for the following input of shape `[1 4 4 1]`, and a block size of 2:
/// 
/// ```
/// x = [[[[1],   [2],  [5],  [6]],
///       [[3],   [4],  [7],  [8]],
///       [[9],  [10], [13],  [14]],
///       [[11], [12], [15],  [16]]]]
/// ```
/// 
/// the operator will return the following tensor of shape `[1 2 2 4]`:
/// 
/// ```
/// x = [[[[1, 2, 3, 4],
///        [5, 6, 7, 8]],
///       [[9, 10, 11, 12],
///        [13, 14, 15, 16]]]]
/// ```
/// - Parameter input: 
/// - Parameter blockSize: The size of the spatial block.
/// - Parameter dataFormat: 
/// - Returns: 
///	output: 
public func spaceToDepth(operationName: String? = nil, input: Output, blockSize: UInt8, dataFormat: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["block_size"] = blockSize
	attrs["data_format"] = dataFormat
	let opspec = OpSpec(
		type: "SpaceToDepth",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes softplus gradients for a softplus operation.
/// - Parameter gradients: The backpropagated gradients to the corresponding softplus operation.
/// - Parameter features: The features passed as input to the corresponding softplus operation.
/// - Returns: 
///	backprops: The gradients: `gradients / (1 + exp(-features))`.
public func softplusGrad(operationName: String? = nil, gradients: Output, features: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "SoftplusGrad",
		name: (operationName ?? "Type"),
		input: [gradients, features],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Returns x  *  y element-wise.
/// * NOTE * : `Mul` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
/// - Parameter x: 
/// - Parameter y: 
/// - Returns: 
///	z: 
public func mul(operationName: String? = nil, x: Output, y: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Mul",
		name: (operationName ?? "Type"),
		input: [x, y],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///BatchToSpace for 4-D tensors of type T.
///This is a legacy version of the more general BatchToSpaceND.
/// 
/// Rearranges (permutes) data from batch into blocks of spatial data, followed by
/// cropping. This is the reverse transformation of SpaceToBatch. More specifically,
/// this op outputs a copy of the input tensor where values from the `batch`
/// dimension are moved in spatial blocks to the `height` and `width` dimensions,
/// followed by cropping along the `height` and `width` dimensions.
/// - Parameter input: 4-D tensor with shape
/// `[batch * block_size * block_size, height_pad/block_size, width_pad/block_size,
///   depth]`. Note that the batch size of the input tensor must be divisible by
/// `block_size  *  block_size`.
/// - Parameter crops: 2-D tensor of non-negative integers with shape `[2, 2]`. It specifies
/// how many elements to crop from the intermediate result across the spatial
/// dimensions as follows:
/// 
///     crops = [[crop_top, crop_bottom], [crop_left, crop_right]]
/// - Parameter blockSize: 
/// - Parameter tidx: 
/// - Returns: 
///	output: 4-D with shape `[batch, height, width, depth]`, where:
/// 
///       height = height_pad - crop_top - crop_bottom
///       width = width_pad - crop_left - crop_right
/// 
/// The attr `block_size` must be greater than one. It indicates the block size.
/// 
/// Some examples:
/// 
/// (1) For the following input of shape `[4, 1, 1, 1]` and block_size of 2:
/// 
/// ```
/// [[[[1]]], [[[2]]], [[[3]]], [[[4]]]]
/// ```
/// 
/// The output tensor has shape `[1, 2, 2, 1]` and value:
/// 
/// ```
/// x = [[[[1], [2]], [[3], [4]]]]
/// ```
/// 
/// (2) For the following input of shape `[4, 1, 1, 3]` and block_size of 2:
/// 
/// ```
/// [[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]], [[10, 11, 12]]]
/// ```
/// 
/// The output tensor has shape `[1, 2, 2, 3]` and value:
/// 
/// ```
/// x = [[[[1, 2, 3], [4, 5, 6]],
///       [[7, 8, 9], [10, 11, 12]]]]
/// ```
/// 
/// (3) For the following input of shape `[4, 2, 2, 1]` and block_size of 2:
/// 
/// ```
/// x = [[[[1], [3]], [[9], [11]]],
///      [[[2], [4]], [[10], [12]]],
///      [[[5], [7]], [[13], [15]]],
///      [[[6], [8]], [[14], [16]]]]
/// ```
/// 
/// The output tensor has shape `[1, 4, 4, 1]` and value:
/// 
/// ```
/// x = [[[1],   [2],  [3],  [4]],
///      [[5],   [6],  [7],  [8]],
///      [[9],  [10], [11],  [12]],
///      [[13], [14], [15],  [16]]]
/// ```
/// 
/// (4) For the following input of shape `[8, 1, 2, 1]` and block_size of 2:
/// 
/// ```
/// x = [[[[1], [3]]], [[[9], [11]]], [[[2], [4]]], [[[10], [12]]],
///      [[[5], [7]]], [[[13], [15]]], [[[6], [8]]], [[[14], [16]]]]
/// ```
/// 
/// The output tensor has shape `[2, 2, 4, 1]` and value:
/// 
/// ```
/// x = [[[[1], [3]], [[5], [7]]],
///      [[[2], [4]], [[10], [12]]],
///      [[[5], [7]], [[13], [15]]],
///      [[[6], [8]], [[14], [16]]]]
/// ```
public func batchToSpace(operationName: String? = nil, input: Output, crops: Output, blockSize: UInt8, tidx: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["block_size"] = blockSize
	attrs["Tidx"] = tidx
	let opspec = OpSpec(
		type: "BatchToSpace",
		name: (operationName ?? "Type"),
		input: [input, crops],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes arctangent of `y/x` element-wise, respecting signs of the arguments.
///This is the angle \( \theta \in [-\pi, \pi] \) such that
/// \[ x = r \cos(\theta) \]
/// and
/// \[ y = r \sin(\theta) \]
/// where \(r = \sqrt(x// ^2 + y// ^2) \).
/// - Parameter y: 
/// - Parameter x: 
/// - Returns: 
///	z: 
public func atan2(operationName: String? = nil, y: Output, x: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Atan2",
		name: (operationName ?? "Type"),
		input: [y, x],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///SpaceToBatch for 4-D tensors of type T.
///This is a legacy version of the more general SpaceToBatchND.
/// 
/// Zero-pads and then rearranges (permutes) blocks of spatial data into batch.
/// More specifically, this op outputs a copy of the input tensor where values from
/// the `height` and `width` dimensions are moved to the `batch` dimension. After
/// the zero-padding, both `height` and `width` of the input must be divisible by the
/// block size.
/// - Parameter input: 4-D with shape `[batch, height, width, depth]`.
/// - Parameter paddings: 2-D tensor of non-negative integers with shape `[2, 2]`. It specifies
///   the padding of the input with zeros across the spatial dimensions as follows:
/// 
///       paddings = [[pad_top, pad_bottom], [pad_left, pad_right]]
/// 
///   The effective spatial dimensions of the zero-padded input tensor will be:
/// 
///       height_pad = pad_top + height + pad_bottom
///       width_pad = pad_left + width + pad_right
/// 
/// The attr `block_size` must be greater than one. It indicates the block size.
/// 
///    *  Non-overlapping blocks of size `block_size x block size` in the height and
///     width dimensions are rearranged into the batch dimension at each location.
///    *  The batch of the output tensor is `batch  *  block_size  *  block_size`.
///    *  Both height_pad and width_pad must be divisible by block_size.
/// 
/// The shape of the output will be:
/// 
///     [batch * block_size * block_size, height_pad/block_size, width_pad/block_size,
///      depth]
/// 
/// Some examples:
/// 
/// (1) For the following input of shape `[1, 2, 2, 1]` and block_size of 2:
/// 
/// ```
/// x = [[[[1], [2]], [[3], [4]]]]
/// ```
/// 
/// The output tensor has shape `[4, 1, 1, 1]` and value:
/// 
/// ```
/// [[[[1]]], [[[2]]], [[[3]]], [[[4]]]]
/// ```
/// 
/// (2) For the following input of shape `[1, 2, 2, 3]` and block_size of 2:
/// 
/// ```
/// x = [[[[1, 2, 3], [4, 5, 6]],
///       [[7, 8, 9], [10, 11, 12]]]]
/// ```
/// 
/// The output tensor has shape `[4, 1, 1, 3]` and value:
/// 
/// ```
/// [[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]], [[10, 11, 12]]]
/// ```
/// 
/// (3) For the following input of shape `[1, 4, 4, 1]` and block_size of 2:
/// 
/// ```
/// x = [[[[1],   [2],  [3],  [4]],
///       [[5],   [6],  [7],  [8]],
///       [[9],  [10], [11],  [12]],
///       [[13], [14], [15],  [16]]]]
/// ```
/// 
/// The output tensor has shape `[4, 2, 2, 1]` and value:
/// 
/// ```
/// x = [[[[1], [3]], [[9], [11]]],
///      [[[2], [4]], [[10], [12]]],
///      [[[5], [7]], [[13], [15]]],
///      [[[6], [8]], [[14], [16]]]]
/// ```
/// 
/// (4) For the following input of shape `[2, 2, 4, 1]` and block_size of 2:
/// 
/// ```
/// x = [[[[1],   [2],  [3],  [4]],
///       [[5],   [6],  [7],  [8]]],
///      [[[9],  [10], [11],  [12]],
///       [[13], [14], [15],  [16]]]]
/// ```
/// 
/// The output tensor has shape `[8, 1, 2, 1]` and value:
/// 
/// ```
/// x = [[[[1], [3]]], [[[9], [11]]], [[[2], [4]]], [[[10], [12]]],
///      [[[5], [7]]], [[[13], [15]]], [[[6], [8]]], [[[14], [16]]]]
/// ```
/// 
/// Among others, this operation is useful for reducing atrous convolution into
/// regular convolution.
/// - Parameter tpaddings: 
/// - Parameter blockSize: 
/// - Returns: 
///	output: 
public func spaceToBatch(operationName: String? = nil, input: Output, paddings: Output, tpaddings: Any.Type, blockSize: UInt8) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Tpaddings"] = tpaddings
	attrs["block_size"] = blockSize
	let opspec = OpSpec(
		type: "SpaceToBatch",
		name: (operationName ?? "Type"),
		input: [input, paddings],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Removes dimensions of size 1 from the shape of a tensor.
///Given a tensor `input`, this operation returns a tensor of the same type with
/// all dimensions of size 1 removed. If you don't want to remove all size 1
/// dimensions, you can remove specific size 1 dimensions by specifying
/// `squeeze_dims`.
/// 
/// For example:
/// 
/// ```
/// # 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
/// shape(squeeze(t)) ==> [2, 3]
/// ```
/// 
/// Or, to remove specific size 1 dimensions:
/// 
/// ```
/// # 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
/// shape(squeeze(t, [2, 4])) ==> [1, 2, 3, 1]
/// ```
/// - Parameter input: The `input` to squeeze.
/// - Parameter squeezeDims: If specified, only squeezes the dimensions listed. The dimension
/// index starts at 0. It is an error to squeeze a dimension that is not 1. Must
/// be in the range `[-rank(input), rank(input))`.
/// - Returns: 
///	output: Contains the same data as `input`, but has one or more dimensions of
/// size 1 removed.
public func squeeze(operationName: String? = nil, input: Output, squeezeDims: [Int64]) throws -> Output { 
	var attrs = [String : Any]()
	attrs["squeeze_dims"] = squeezeDims
	let opspec = OpSpec(
		type: "Squeeze",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Inserts a dimension of 1 into a tensor's shape.
///Given a tensor `input`, this operation inserts a dimension of 1 at the
/// dimension index `dim` of `input`'s shape. The dimension index `dim` starts at
/// zero; if you specify a negative number for `dim` it is counted backward from
/// the end.
/// 
/// This operation is useful if you want to add a batch dimension to a single
/// element. For example, if you have a single image of shape `[height, width,
/// channels]`, you can make it a batch of 1 image with `expand_dims(image, 0)`,
/// which will make the shape `[1, height, width, channels]`.
/// 
/// Other examples:
/// 
/// ```
/// # 't' is a tensor of shape [2]
/// shape(expand_dims(t, 0)) ==> [1, 2]
/// shape(expand_dims(t, 1)) ==> [2, 1]
/// shape(expand_dims(t, -1)) ==> [2, 1]
/// 
/// # 't2' is a tensor of shape [2, 3, 5]
/// shape(expand_dims(t2, 0)) ==> [1, 2, 3, 5]
/// shape(expand_dims(t2, 2)) ==> [2, 3, 1, 5]
/// shape(expand_dims(t2, 3)) ==> [2, 3, 5, 1]
/// ```
/// 
/// This operation requires that:
/// 
/// `-1-input.dims() <= dim <= input.dims()`
/// 
/// This operation is related to `squeeze()`, which removes dimensions of
/// size 1.
/// - Parameter input: 
/// - Parameter dim: 0-D (scalar). Specifies the dimension index at which to
/// expand the shape of `input`. Must be in the range
/// `[-rank(input) - 1, rank(input)]`.
/// - Parameter tdim: 
/// - Returns: 
///	output: Contains the same data as `input`, but its shape has an additional
/// dimension of size 1 added.
public func expandDims(operationName: String? = nil, input: Output, dim: Output, tdim: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Tdim"] = tdim
	let opspec = OpSpec(
		type: "ExpandDims",
		name: (operationName ?? "Type"),
		input: [input, dim],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///A placeholder op that passes through `input` when its output is not fed.
/// - Parameter input: The default value to produce when `output` is not fed.
/// - Parameter dtype: The type of elements in the tensor.
/// - Parameter shape: The (possibly partial) shape of the tensor.
/// - Returns: 
///	output: A placeholder tensor that defaults to `input` if it is not fed.
public func placeholderWithDefault(operationName: String? = nil, input: Output, dtype: Any.Type, shape: Shape) throws -> Output { 
	var attrs = [String : Any]()
	attrs["dtype"] = dtype
	attrs["shape"] = shape
	let opspec = OpSpec(
		type: "PlaceholderWithDefault",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes acos of x element-wise.
/// - Parameter x: 
/// - Returns: 
///	y: 
public func acos(operationName: String? = nil, x: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Acos",
		name: (operationName ?? "Type"),
		input: [x],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///A placeholder op for a value that will be fed into the computation.
///N.B. This operation will fail with an error if it is executed. It is
/// intended as a way to represent a value that will always be fed, and to
/// provide attrs that enable the fed value to be checked at runtime.
/// - Parameter dtype: The type of elements in the tensor.
/// - Parameter shape: (Optional) The shape of the tensor. If the shape has 0 dimensions, the
/// shape is unconstrained.
/// - Returns: 
///	output: A placeholder tensor that must be replaced using the feed mechanism.
public func placeholder(operationName: String? = nil, dtype: Any.Type, shape: Shape) throws -> Output { 
	var attrs = [String : Any]()
	attrs["dtype"] = dtype
	attrs["shape"] = shape
	let opspec = OpSpec(
		type: "Placeholder",
		name: (operationName ?? "Type"),
		input: [],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Gradient op for `MirrorPad` op. This op folds a mirror-padded tensor.
///This operation folds the padded areas of `input` by `MirrorPad` according to the
/// `paddings` you specify. `paddings` must be the same as `paddings` argument
/// given to the corresponding `MirrorPad` op.
/// 
/// The folded size of each dimension D of the output is:
/// 
/// `input.dim_size(D) - paddings(D, 0) - paddings(D, 1)`
/// 
/// For example:
/// 
/// ```
/// # 't' is [[1, 2, 3], [4, 5, 6], [7, 8, 9]].
/// # 'paddings' is [[0, 1]], [0, 1]].
/// # 'mode' is SYMMETRIC.
/// # rank of 't' is 2.
/// pad(t, paddings) ==> [[ 1,  5]
///                       [11, 28]]
/// ```
/// - Parameter input: The input tensor to be folded.
/// - Parameter paddings: A two-column matrix specifying the padding sizes. The number of
/// rows must be the same as the rank of `input`.
/// - Parameter tpaddings: 
/// - Parameter mode: The mode used in the `MirrorPad` op.
/// - Returns: 
///	output: The folded tensor.
public func mirrorPadGrad(operationName: String? = nil, input: Output, paddings: Output, tpaddings: Any.Type, mode: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Tpaddings"] = tpaddings
	attrs["mode"] = mode
	let opspec = OpSpec(
		type: "MirrorPadGrad",
		name: (operationName ?? "Type"),
		input: [input, paddings],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Pads a tensor with mirrored values.
///This operation pads a `input` with mirrored values according to the `paddings`
/// you specify. `paddings` is an integer tensor with shape `[n, 2]`, where n is
/// the rank of `input`. For each dimension D of `input`, `paddings[D, 0]` indicates
/// how many values to add before the contents of `input` in that dimension, and
/// `paddings[D, 1]` indicates how many values to add after the contents of `input`
/// in that dimension. Both `paddings[D, 0]` and `paddings[D, 1]` must be no greater
/// than `input.dim_size(D)` (or `input.dim_size(D) - 1`) if `copy_border` is true
/// (if false, respectively).
/// 
/// The padded size of each dimension D of the output is:
/// 
/// `paddings(D, 0) + input.dim_size(D) + paddings(D, 1)`
/// 
/// For example:
/// 
/// ```
/// # 't' is [[1, 2, 3], [4, 5, 6]].
/// # 'paddings' is [[1, 1]], [2, 2]].
/// # 'mode' is SYMMETRIC.
/// # rank of 't' is 2.
/// pad(t, paddings) ==> [[2, 1, 1, 2, 3, 3, 2]
///                       [2, 1, 1, 2, 3, 3, 2]
///                       [5, 4, 4, 5, 6, 6, 5]
///                       [5, 4, 4, 5, 6, 6, 5]]
/// ```
/// - Parameter input: The input tensor to be padded.
/// - Parameter paddings: A two-column matrix specifying the padding sizes. The number of
/// rows must be the same as the rank of `input`.
/// - Parameter tpaddings: 
/// - Parameter mode: Either `REFLECT` or `SYMMETRIC`. In reflect mode the padded regions
/// do not include the borders, while in symmetric mode the padded regions
/// do include the borders. For example, if `input` is `[1, 2, 3]` and `paddings`
/// is `[0, 2]`, then the output is `[1, 2, 3, 2, 1]` in reflect mode, and
/// it is `[1, 2, 3, 3, 2]` in symmetric mode.
/// - Returns: 
///	output: The padded tensor.
public func mirrorPad(operationName: String? = nil, input: Output, paddings: Output, tpaddings: Any.Type, mode: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Tpaddings"] = tpaddings
	attrs["mode"] = mode
	let opspec = OpSpec(
		type: "MirrorPad",
		name: (operationName ?? "Type"),
		input: [input, paddings],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Pads a tensor with zeros.
///This operation pads a `input` with zeros according to the `paddings` you
/// specify. `paddings` is an integer tensor with shape `[Dn, 2]`, where n is the
/// rank of `input`. For each dimension D of `input`, `paddings[D, 0]` indicates
/// how many zeros to add before the contents of `input` in that dimension, and
/// `paddings[D, 1]` indicates how many zeros to add after the contents of `input`
/// in that dimension.
/// 
/// The padded size of each dimension D of the output is:
/// 
/// `paddings(D, 0) + input.dim_size(D) + paddings(D, 1)`
/// 
/// For example:
/// 
/// ```
/// # 't' is [[1, 1], [2, 2]]
/// # 'paddings' is [[1, 1], [2, 2]]
/// # rank of 't' is 2
/// pad(t, paddings) ==> [[0, 0, 0, 0, 0, 0]
///                       [0, 0, 1, 1, 0, 0]
///                       [0, 0, 2, 2, 0, 0]
///                       [0, 0, 0, 0, 0, 0]]
/// ```
/// - Parameter input: 
/// - Parameter paddings: 
/// - Parameter tpaddings: 
/// - Returns: 
///	output: 
public func pad(operationName: String? = nil, input: Output, paddings: Output, tpaddings: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Tpaddings"] = tpaddings
	let opspec = OpSpec(
		type: "Pad",
		name: (operationName ?? "Type"),
		input: [input, paddings],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes Quantized Rectified Linear: `max(features, 0)`
/// - Parameter features: 
/// - Parameter minFeatures: The float value that the lowest quantized value represents.
/// - Parameter maxFeatures: The float value that the highest quantized value represents.
/// - Parameter tinput: 
/// - Parameter outType: 
/// - Returns: 
///	activations: Has the same output shape as "features".
///	min_activations: The float value that the lowest quantized value represents.
///	max_activations: The float value that the highest quantized value represents.
public func quantizedRelu(operationName: String? = nil, features: Output, minFeatures: Output, maxFeatures: Output, tinput: Any.Type, outType: Any.Type) throws -> (activations: Output, minActivations: Output, maxActivations: Output) { 
	var attrs = [String : Any]()
	attrs["Tinput"] = tinput
	attrs["out_type"] = outType
	let opspec = OpSpec(
		type: "QuantizedRelu",
		name: (operationName ?? "Type"),
		input: [features, minFeatures, maxFeatures],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (activations: op.output(at: 0), minActivations: op.output(at: 1), maxActivations: op.output(at: 2))
} 

///Return the reduction indices for computing gradients of s0 op s1 with broadcast.
///This is typically used by gradient computations for a broadcasting operation.
/// - Parameter s0: 
/// - Parameter s1: 
/// - Returns: 
///	r0: 
///	r1: 
public func broadcastGradientArgs(operationName: String? = nil, s0: Output, s1: Output) throws -> (r0: Output, r1: Output) { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "BroadcastGradientArgs",
		name: (operationName ?? "Type"),
		input: [s0, s1],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (r0: op.output(at: 0), r1: op.output(at: 1))
} 

///Adds Tensor 'bias' to Tensor 'input' for Quantized types.
///Broadcasts the values of bias on dimensions 0..N-2 of 'input'.
/// - Parameter input: 
/// - Parameter bias: A 1D bias Tensor with size matching the last dimension of 'input'.
/// - Parameter minInput: The float value that the lowest quantized input value represents.
/// - Parameter maxInput: The float value that the highest quantized input value represents.
/// - Parameter minBias: The float value that the lowest quantized bias value represents.
/// - Parameter maxBias: The float value that the highest quantized bias value represents.
/// - Parameter t1: 
/// - Parameter t2: 
/// - Parameter outType: 
/// - Returns: 
///	output: 
///	min_out: The float value that the lowest quantized output value represents.
///	max_out: The float value that the highest quantized output value represents.
public func quantizedBiasAdd(operationName: String? = nil, input: Output, bias: Output, minInput: Output, maxInput: Output, minBias: Output, maxBias: Output, t1: Any.Type, t2: Any.Type, outType: Any.Type) throws -> (output: Output, minOut: Output, maxOut: Output) { 
	var attrs = [String : Any]()
	attrs["T1"] = t1
	attrs["T2"] = t2
	attrs["out_type"] = outType
	let opspec = OpSpec(
		type: "QuantizedBiasAdd",
		name: (operationName ?? "Type"),
		input: [input, bias, minInput, maxInput, minBias, maxBias],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (output: op.output(at: 0), minOut: op.output(at: 1), maxOut: op.output(at: 2))
} 

///Return the shape of s0 op s1 with broadcast.
///Given `s0` and `s1`, tensors that represent shapes, compute `r0`, the
/// broadcasted shape. `s0`, `s1` and `r0` are all integer vectors.
/// - Parameter s0: 
/// - Parameter s1: 
/// - Returns: 
///	r0: 
public func broadcastArgs(operationName: String? = nil, s0: Output, s1: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "BroadcastArgs",
		name: (operationName ?? "Type"),
		input: [s0, s1],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Assign `value` to the sliced l-value reference of `ref`.
///The values of `value` are assigned to the positions in the variable
/// `ref` that are selected by the slice parameters. The slice parameters
/// `begin, `end`, `strides`, etc. work exactly as in `StridedSlice`.
/// 
/// NOTE this op currently does not support broadcasting and so `value`'s
/// shape must be exactly the shape produced by the slice of `ref`.
/// - Parameter ref: 
/// - Parameter begin: 
/// - Parameter end: 
/// - Parameter strides: 
/// - Parameter value: 
/// - Parameter index: 
/// - Parameter beginMask: 
/// - Parameter endMask: 
/// - Parameter ellipsisMask: 
/// - Parameter newAxisMask: 
/// - Parameter shrinkAxisMask: 
public func resourceStridedSliceAssign(operationName: String? = nil, ref: Output, begin: Output, end: Output, strides: Output, value: Output, index: Any.Type, beginMask: UInt8, endMask: UInt8, ellipsisMask: UInt8, newAxisMask: UInt8, shrinkAxisMask: UInt8) throws -> Operation { 
	var attrs = [String : Any]()
	attrs["Index"] = index
	attrs["begin_mask"] = beginMask
	attrs["end_mask"] = endMask
	attrs["ellipsis_mask"] = ellipsisMask
	attrs["new_axis_mask"] = newAxisMask
	attrs["shrink_axis_mask"] = shrinkAxisMask
	let opspec = OpSpec(
		type: "ResourceStridedSliceAssign",
		name: (operationName ?? "Type"),
		input: [ref, begin, end, strides, value],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Returns element-wise remainder of division. This emulates C semantics in that
///the result here is consistent with a truncating divide. E.g. `truncate(x / y)  * 
/// y + truncate_mod(x, y) = x`.
/// 
///  * NOTE * : `TruncateMod` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
/// - Parameter x: 
/// - Parameter y: 
/// - Returns: 
///	z: 
public func truncateMod(operationName: String? = nil, x: Output, y: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "TruncateMod",
		name: (operationName ?? "Type"),
		input: [x, y],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Returns the gradient of `StridedSlice`.
///Since `StridedSlice` cuts out pieces of its `input` which is size
/// `shape`, its gradient will have the same shape (which is passed here
/// as `shape`). The gradient will be zero in any element that the slice
/// does not select.
/// 
/// Arguments are the same as StridedSliceGrad with the exception that
/// `dy` is the input gradient to be propagated and `shape` is the
/// shape of `StridedSlice`'s `input`.
/// - Parameter shape: 
/// - Parameter begin: 
/// - Parameter end: 
/// - Parameter strides: 
/// - Parameter dy: 
/// - Parameter index: 
/// - Parameter beginMask: 
/// - Parameter endMask: 
/// - Parameter ellipsisMask: 
/// - Parameter newAxisMask: 
/// - Parameter shrinkAxisMask: 
/// - Returns: 
///	output: 
public func stridedSliceGrad(operationName: String? = nil, shape: Output, begin: Output, end: Output, strides: Output, dy: Output, index: Any.Type, beginMask: UInt8, endMask: UInt8, ellipsisMask: UInt8, newAxisMask: UInt8, shrinkAxisMask: UInt8) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Index"] = index
	attrs["begin_mask"] = beginMask
	attrs["end_mask"] = endMask
	attrs["ellipsis_mask"] = ellipsisMask
	attrs["new_axis_mask"] = newAxisMask
	attrs["shrink_axis_mask"] = shrinkAxisMask
	let opspec = OpSpec(
		type: "StridedSliceGrad",
		name: (operationName ?? "Type"),
		input: [shape, begin, end, strides, dy],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Return a strided slice from `input`.
///Note, most python users will want to use the Python `Tensor.__getitem__`
/// or `Variable.__getitem__` rather than this op directly.
/// 
/// The goal of this op is to produce a new tensor with a subset of
/// the elements from the `n` dimensional `input` tensor. The subset is chosen using
/// a sequence of `m` sparse range specifications encoded into the arguments
/// of this function. Note, in some cases
/// `m` could be equal to `n`, but this need not be the case. Each
/// range specification entry can be one of the following:
/// 
/// - An ellipsis (...). Ellipses are used to imply zero or more
///   dimensions of full-dimension selection and are produced using
///   `ellipsis_mask`. For example, `foo[...]` is the identity slice.
/// 
/// - A new axis. This is used to insert a new shape=1 dimension and is
///   produced using `new_axis_mask`. For example, `foo[:, ...]` where
///   `foo` is shape `(3, 4)` produces a `(1, 3, 4)` tensor.
/// 
/// 
/// - A range `begin:end:stride`. This is used to specify how much to choose from
///   a given dimension. `stride` can be any integer but 0.  `begin` is an integer
///   which represents the index of the first value to select while `end` represents
///   the index of the last value to select. The number of values selected in each
///   dimension is `end - begin` if `stride > 0` and `begin - end` if `stride < 0`.
///   `begin` and `end` can be negative where `-1` is the last element, `-2` is
///   the second to last. `begin_mask` controls whether to replace the explicitly
///   given `begin` with an implicit effective value of `0` if `stride > 0` and
///   `-1` if `stride < 0`. `end_mask` is analogous but produces the number
///   required to create the largest open interval. For example, given a shape
///   `(3,)` tensor `foo[:]`, the effective `begin` and `end` are `0` and `3`. Do
///   not assume this is equivalent to `foo[0:-1]` which has an effective `begin`
///   and `end` of `0` and `2`. Another example is `foo[-2::-1]` which reverses the
///   first dimension of a tensor while dropping the last two (in the original
///   order elements). For example `foo = [1,2,3,4]; foo[-2::-1]` is `[4,3]`.
/// 
/// - A single index. This is used to keep only elements that have a given
///   index. For example (`foo[2, :]` on a shape `(5,6)` tensor produces a
///   shape `(6,)` tensor. This is encoded in `begin` and `end` and
///   `shrink_axis_mask`.
/// 
/// Each conceptual range specification is encoded in the op's argument. This
/// encoding is best understand by considering a non-trivial example. In
/// particular,
/// `foo[1, 2:4, None, ..., :-3:-1, :]` will be encoded as
/// 
/// ```
/// begin = [1, 2, x, x, 0, x] # x denotes don't care (usually 0)
/// end = [2, 4, x, x, -3, x]
/// strides = [1, 1, x, x, -1, 1]
/// begin_mask = 1<<4 | 1 << 5 = 48
/// end_mask = 1<<5 = 32
/// ellipsis_mask = 1<<3 = 8
/// new_axis_mask = 1<<2 4
/// shrink_axis_mask = 1<<0
/// ```
/// 
/// In this case if `foo.shape` is (5, 5, 5, 5, 5, 5) the final shape of
/// the slice becomes (2, 1, 5, 5, 2, 5).
/// Let us walk step by step through each argument specification.
/// 
/// 1.  The first argument in the example slice is turned into `begin = 1` and
/// `end = begin + 1 = 2`. To disambiguate from the original spec `2:4` we
/// also set the appropriate bit in `shrink_axis_mask`.
/// 
/// 2. `2:4` is contributes 2, 4, 1 to begin, end, and stride. All masks have
/// zero bits contributed.
/// 
/// 3. None is a synonym for `tf.newaxis`. This means insert a dimension of size 1
/// dimension in the final shape. Dummy values are contributed to begin,
/// end and stride, while the new_axis_mask bit is set.
/// 
/// 4. `...` grab the full ranges from as many dimensions as needed to
/// fully specify a slice for every dimension of the input shape.
/// 
/// 5. `:-3:-1` shows the use of negative indices. A negative index `i` associated
/// with a dimension that has shape `s` is converted to a positive index
/// `s + i`. So `-1` becomes `s-1` (i.e. the last element). This conversion
/// is done internally so begin, end and strides receive x, -3, and -1.
/// The appropriate begin_mask bit is set to indicate the start range is the
/// full range (ignoring the x).
/// 
/// 6. `:` indicates that the entire contents of the corresponding dimension
/// is selected. This is equivalent to `::` or `0::1`. begin, end, and strides
/// receive 0, 0, and 1, respectively. The appropriate bits in `begin_mask` and
/// `end_mask` are also set.
/// 
///  * Requirements * :
///   `0 != strides[i] for i in [0, m)`
///   `ellipsis_mask must be a power of two (only one ellipsis)`
/// - Parameter input: 
/// - Parameter begin: `begin[k]` specifies the offset into the `k`th range specification.
/// The exact dimension this corresponds to will be determined by context.
/// Out-of-bounds values will be silently clamped. If the `k`th bit of
/// `begin_mask` then `begin[k]` is ignored and the full range of the
/// appropriate dimension is used instead. Negative values causes indexing
/// to start from the highest element e.g. If `foo==[1,2,3]` then `foo[-1]==3`.
/// - Parameter end: `end[i]` is like `begin` with the exception that `end_mask` is
/// used to determine full ranges.
/// - Parameter strides: `strides[i]` specifies the increment in the `i`th specification
/// after extracting a given element. Negative indices will reverse
/// the original order. Out or range values are
/// clamped to `[0,dim[i]) if slice[i]>0` or `[-1,dim[i]-1] if slice[i] < 0`
/// - Parameter index: 
/// - Parameter beginMask: a bitmask where a bit i being 1 means to ignore the begin
/// value and instead use the largest interval possible. At runtime
/// begin[i] will be replaced with `[0, n-1) if `stride[i] > 0` or
/// `[-1, n-1]` if `stride[i] < 0`
/// - Parameter endMask: analogous to `begin_mask`
/// - Parameter ellipsisMask: a bitmask where bit `i` being 1 means the `i`th
/// position is actually an ellipsis. One bit at most can be 1.
/// If `ellipsis_mask == 0`, then an implicit ellipsis mask of `1 << (m+1)`
/// is provided. This means that `foo[3:5] == foo[3:5, ...]`. An ellipsis
/// implicitly creates as many range specifications as necessary to fully
/// specify the sliced range for every dimension. For example for a 4-dimensional
/// tensor `foo` the slice `foo[2, ..., 5:8]` implies `foo[2, :, :, 5:8]`.
/// - Parameter newAxisMask: a bitmask where bit `i` being 1 means the `i`th
/// specification creates a new shape 1 dimension. For example
/// `foo[:4, tf.newaxis, :2]` would produce a shape `(4, 1, 2)` tensor.
/// - Parameter shrinkAxisMask: a bitmask where bit `i` implies that the `i`th
/// specification should shrink the dimensionality. begin and end
/// must imply a slice of size 1 in the dimension. For example in
/// python one might do `foo[:, 3, :]` which would result in
/// `shrink_axis_mask` being 2.
/// - Returns: 
///	output: 
public func stridedSlice(operationName: String? = nil, input: Output, begin: Output, end: Output, strides: Output, index: Any.Type, beginMask: UInt8, endMask: UInt8, ellipsisMask: UInt8, newAxisMask: UInt8, shrinkAxisMask: UInt8) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Index"] = index
	attrs["begin_mask"] = beginMask
	attrs["end_mask"] = endMask
	attrs["ellipsis_mask"] = ellipsisMask
	attrs["new_axis_mask"] = newAxisMask
	attrs["shrink_axis_mask"] = shrinkAxisMask
	let opspec = OpSpec(
		type: "StridedSlice",
		name: (operationName ?? "Type"),
		input: [input, begin, end, strides],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Return a slice from 'input'.
///The output tensor is a tensor with dimensions described by 'size'
/// whose values are extracted from 'input' starting at the offsets in
/// 'begin'.
/// 
///  * Requirements * :
///   0 <= begin[i] <= begin[i] + size[i] <= Di  for i in [0, n)
/// - Parameter input: 
/// - Parameter begin: begin[i] specifies the offset into the 'i'th dimension of
/// 'input' to slice from.
/// - Parameter size: size[i] specifies the number of elements of the 'i'th dimension
/// of 'input' to slice. If size[i] is -1, all remaining elements in dimension
/// i are included in the slice (i.e. this is equivalent to setting
/// size[i] = input.dim_size(i) - begin[i]).
/// - Parameter index: 
/// - Returns: 
///	output: 
public func slice(operationName: String? = nil, input: Output, begin: Output, size: Output, index: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Index"] = index
	let opspec = OpSpec(
		type: "Slice",
		name: (operationName ?? "Type"),
		input: [input, begin, size],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Finds unique elements in a 1-D tensor.
///This operation returns a tensor `y` containing all of the unique elements of `x`
/// sorted in the same order that they occur in `x`. This operation also returns a
/// tensor `idx` the same size as `x` that contains the index of each value of `x`
/// in the unique output `y`. In other words:
/// 
/// `y[idx[i]] = x[i] for i in [0, 1,...,rank(x) - 1]`
/// 
/// For example:
/// 
/// ```
/// # tensor 'x' is [1, 1, 2, 4, 4, 4, 7, 8, 8]
/// y, idx = unique(x)
/// y ==> [1, 2, 4, 7, 8]
/// idx ==> [0, 0, 1, 2, 2, 2, 3, 4, 4]
/// ```
/// - Parameter x: 1-D.
/// - Parameter outIdx: 
/// - Returns: 
///	y: 1-D.
///	idx: 1-D.
public func unique(operationName: String? = nil, x: Output, outIdx: Any.Type) throws -> (y: Output, idx: Output) { 
	var attrs = [String : Any]()
	attrs["out_idx"] = outIdx
	let opspec = OpSpec(
		type: "Unique",
		name: (operationName ?? "Type"),
		input: [x],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (y: op.output(at: 0), idx: op.output(at: 1))
} 

///Reshapes a tensor.
///Given `tensor`, this operation returns a tensor that has the same values
/// as `tensor` with shape `shape`.
/// 
/// If one component of `shape` is the special value -1, the size of that dimension
/// is computed so that the total size remains constant.  In particular, a `shape`
/// of `[-1]` flattens into 1-D.  At most one component of `shape` can be -1.
/// 
/// If `shape` is 1-D or higher, then the operation returns a tensor with shape
/// `shape` filled with the values of `tensor`. In this case, the number of elements
/// implied by `shape` must be the same as the number of elements in `tensor`.
/// 
/// For example:
/// 
/// ```
/// # tensor 't' is [1, 2, 3, 4, 5, 6, 7, 8, 9]
/// # tensor 't' has shape [9]
/// reshape(t, [3, 3]) ==> [[1, 2, 3],
///                         [4, 5, 6],
///                         [7, 8, 9]]
/// 
/// # tensor 't' is [[[1, 1], [2, 2]],
/// #                [[3, 3], [4, 4]]]
/// # tensor 't' has shape [2, 2, 2]
/// reshape(t, [2, 4]) ==> [[1, 1, 2, 2],
///                         [3, 3, 4, 4]]
/// 
/// # tensor 't' is [[[1, 1, 1],
/// #                 [2, 2, 2]],
/// #                [[3, 3, 3],
/// #                 [4, 4, 4]],
/// #                [[5, 5, 5],
/// #                 [6, 6, 6]]]
/// # tensor 't' has shape [3, 2, 3]
/// # pass '[-1]' to flatten 't'
/// reshape(t, [-1]) ==> [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]
/// 
/// # -1 can also be used to infer the shape
/// 
/// # -1 is inferred to be 9:
/// reshape(t, [2, -1]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
///                          [4, 4, 4, 5, 5, 5, 6, 6, 6]]
/// # -1 is inferred to be 2:
/// reshape(t, [-1, 9]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
///                          [4, 4, 4, 5, 5, 5, 6, 6, 6]]
/// # -1 is inferred to be 3:
/// reshape(t, [ 2, -1, 3]) ==> [[[1, 1, 1],
///                               [2, 2, 2],
///                               [3, 3, 3]],
///                              [[4, 4, 4],
///                               [5, 5, 5],
///                               [6, 6, 6]]]
/// 
/// # tensor 't' is [7]
/// # shape `[]` reshapes to a scalar
/// reshape(t, []) ==> 7
/// ```
/// - Parameter tensor: 
/// - Parameter shape: Defines the shape of the output tensor.
/// - Parameter tshape: 
/// - Returns: 
///	output: 
public func reshape(operationName: String? = nil, tensor: Output, shape: Output, tshape: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Tshape"] = tshape
	let opspec = OpSpec(
		type: "Reshape",
		name: (operationName ?? "Type"),
		input: [tensor, shape],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Checks a tensor for NaN and Inf values.
///When run, reports an `InvalidArgument` error if `tensor` has any values
/// that are not a number (NaN) or infinity (Inf). Otherwise, passes `tensor` as-is.
/// - Parameter tensor: 
/// - Parameter message: Prefix of the error message.
/// - Returns: 
///	output: 
public func checkNumerics(operationName: String? = nil, tensor: Output, message: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["message"] = message
	let opspec = OpSpec(
		type: "CheckNumerics",
		name: (operationName ?? "Type"),
		input: [tensor],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Stops gradient computation.
///When executed in a graph, this op outputs its input tensor as-is.
/// 
/// When building ops to compute gradients, this op prevents the contribution of
/// its inputs to be taken into account.  Normally, the gradient generator adds ops
/// to a graph to compute the derivatives of a specified 'loss' by recursively
/// finding out inputs that contributed to its computation.  If you insert this op
/// in the graph it inputs are masked from the gradient generator.  They are not
/// taken into account for computing gradients.
/// 
/// This is useful any time you want to compute a value with TensorFlow but need
/// to pretend that the value was a constant. Some examples include:
/// 
///  *   The  * EM *  algorithm where the  * M-step *  should not involve backpropagation
///    through the output of the  * E-step * .
///  *   Contrastive divergence training of Boltzmann machines where, when
///    differentiating the energy function, the training must not backpropagate
///    through the graph that generated the samples from the model.
///  *   Adversarial training, where no backprop should happen through the adversarial
///    example generation process.
/// - Parameter input: 
/// - Returns: 
///	output: 
public func stopGradient(operationName: String? = nil, input: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "StopGradient",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Identity op for gradient debugging.
///This op is hidden from public in Python. It is used by TensorFlow Debugger to
/// register gradient tensors for gradient debugging.
/// - Parameter input: 
/// - Returns: 
///	output: 
public func debugGradientIdentity(operationName: String? = nil, input: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "DebugGradientIdentity",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Return the same ref tensor as the input ref tensor.
/// - Parameter input: 
/// - Returns: 
///	output: 
public func refIdentity(operationName: String? = nil, input: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "RefIdentity",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Rounds the values of a tensor to the nearest integer, element-wise.
///Rounds half to even.  Also known as bankers rounding. If you want to round
/// according to the current system rounding mode use std::cint.
/// - Parameter x: 
/// - Returns: 
///	y: 
public func round(operationName: String? = nil, x: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Round",
		name: (operationName ?? "Type"),
		input: [x],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Returns a list of tensors with the same shapes and contents as the input
///tensors.
/// 
/// This op can be used to override the gradient for complicated functions. For
/// example, suppose y = f(x) and we wish to apply a custom function g for backprop
/// such that dx = g(dy). In Python,
/// 
/// ```python
/// with tf.get_default_graph().gradient_override_map(
///     {'IdentityN': 'OverrideGradientWithG'}):
///   y, _ = identity_n([f(x), x])
/// 
/// @tf.RegisterGradient('OverrideGradientWithG')
/// def ApplyG(op, dy, _):
///   return [None, g(dy)]  # Do not backprop to f(x).
/// ```
/// - Parameter input: 
/// - Parameter t: 
/// - Returns: 
///	output: 
public func identityN(operationName: String? = nil, input: Output, t: [Any.Type]) throws -> Output { 
	var attrs = [String : Any]()
	attrs["T"] = t
	let opspec = OpSpec(
		type: "IdentityN",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Compute gradients for a FakeQuantWithMinMaxVars operation.
/// - Parameter gradients: Backpropagated gradients above the FakeQuantWithMinMaxVars operation.
/// - Parameter inputs: Values passed as inputs to the FakeQuantWithMinMaxVars operation.
/// min, max: Quantization interval, scalar floats.
/// - Parameter min: 
/// - Parameter max: 
/// - Parameter numBits: The bitwidth of the quantization; between 2 and 8, inclusive.
/// - Parameter narrowRange: Whether to quantize into 2// ^num_bits - 1 distinct values.
/// - Returns: 
///	backprops_wrt_input: Backpropagated gradients w.r.t. inputs:
/// `gradients  *  (inputs >= min && inputs <= max)`.
///	backprop_wrt_min: Backpropagated gradients w.r.t. min parameter:
/// `sum(gradients  *  (inputs < min))`.
///	backprop_wrt_max: Backpropagated gradients w.r.t. max parameter:
/// `sum(gradients  *  (inputs > max))`.
public func fakeQuantWithMinMaxVarsGradient(operationName: String? = nil, gradients: Output, inputs: Output, min: Output, max: Output, numBits: UInt8, narrowRange: Bool) throws -> (backpropsWrtInput: Output, backpropWrtMin: Output, backpropWrtMax: Output) { 
	var attrs = [String : Any]()
	attrs["num_bits"] = numBits
	attrs["narrow_range"] = narrowRange
	let opspec = OpSpec(
		type: "FakeQuantWithMinMaxVarsGradient",
		name: (operationName ?? "Type"),
		input: [gradients, inputs, min, max],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (backpropsWrtInput: op.output(at: 0), backpropWrtMin: op.output(at: 1), backpropWrtMax: op.output(at: 2))
} 

///Returns the size of a tensor.
///This operation returns an integer representing the number of elements in
/// `input`.
/// 
/// For example:
/// 
/// ```
/// # 't' is [[[1, 1,, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]]
/// size(t) ==> 12
/// ```
/// - Parameter input: 
/// - Parameter outType: 
/// - Returns: 
///	output: 
public func size(operationName: String? = nil, input: Output, outType: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["out_type"] = outType
	let opspec = OpSpec(
		type: "Size",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Creates an empty Tensor with shape `shape` and type `dtype`.
///The memory can optionally be initialized. This is usually useful in
/// conjunction with inplace operations.
/// - Parameter shape: 1-D `Tensor` indicating the shape of the output.
/// - Parameter dtype: The element type of the returned tensor.
/// - Returns: 
///	output: An empty Tensor of the specified type.
public func parallelConcatStart(operationName: String? = nil, shape: Shape, dtype: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["shape"] = shape
	attrs["dtype"] = dtype
	let opspec = OpSpec(
		type: "_ParallelConcatStart",
		name: (operationName ?? "Type"),
		input: [],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes softmax activations.
///For each batch `i` and class `j` we have
/// 
///     softmax[i, j] = exp(logits[i, j]) / sum_j(exp(logits[i, j]))
/// - Parameter logits: 2-D with shape `[batch_size, num_classes]`.
/// - Returns: 
///	softmax: Same shape as `logits`.
public func softmax(operationName: String? = nil, logits: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Softmax",
		name: (operationName ?? "Type"),
		input: [logits],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Return a tensor with the same shape and contents as the input tensor or value.
/// - Parameter input: 
/// - Returns: 
///	output: 
public func identity(operationName: String? = nil, input: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Identity",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Reverses specific dimensions of a tensor.
///NOTE `tf.reverse` has now changed behavior in preparation for 1.0.
/// `tf.reverse_v2` is currently an alias that will be deprecated before TF 1.0.
/// 
/// Given a `tensor`, and a `int32` tensor `axis` representing the set of
/// dimensions of `tensor` to reverse. This operation reverses each dimension
/// `i` for which there exists `j` s.t. `axis[j] == i`.
/// 
/// `tensor` can have up to 8 dimensions. The number of dimensions specified
/// in `axis` may be 0 or more entries. If an index is specified more than
/// once, a InvalidArgument error is raised.
/// 
/// For example:
/// 
/// ```
/// # tensor 't' is [[[[ 0,  1,  2,  3],
/// #                  [ 4,  5,  6,  7],
/// #                  [ 8,  9, 10, 11]],
/// #                 [[12, 13, 14, 15],
/// #                  [16, 17, 18, 19],
/// #                  [20, 21, 22, 23]]]]
/// # tensor 't' shape is [1, 2, 3, 4]
/// 
/// # 'dims' is [3] or 'dims' is -1
/// reverse(t, dims) ==> [[[[ 3,  2,  1,  0],
///                         [ 7,  6,  5,  4],
///                         [ 11, 10, 9, 8]],
///                        [[15, 14, 13, 12],
///                         [19, 18, 17, 16],
///                         [23, 22, 21, 20]]]]
/// 
/// # 'dims' is '[1]' (or 'dims' is '[-3]')
/// reverse(t, dims) ==> [[[[12, 13, 14, 15],
///                         [16, 17, 18, 19],
///                         [20, 21, 22, 23]
///                        [[ 0,  1,  2,  3],
///                         [ 4,  5,  6,  7],
///                         [ 8,  9, 10, 11]]]]
/// 
/// # 'dims' is '[2]' (or 'dims' is '[-2]')
/// reverse(t, dims) ==> [[[[8, 9, 10, 11],
///                         [4, 5, 6, 7],
///                         [0, 1, 2, 3]]
///                        [[20, 21, 22, 23],
///                         [16, 17, 18, 19],
///                         [12, 13, 14, 15]]]]
/// ```
/// - Parameter tensor: Up to 8-D.
/// - Parameter axis: 1-D. The indices of the dimensions to reverse. Must be in the range
/// `[-rank(tensor), rank(tensor))`.
/// - Parameter tidx: 
/// - Returns: 
///	output: The same shape as `tensor`.
public func reverseV2(operationName: String? = nil, tensor: Output, axis: Output, tidx: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Tidx"] = tidx
	let opspec = OpSpec(
		type: "ReverseV2",
		name: (operationName ?? "Type"),
		input: [tensor, axis],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Reverses specific dimensions of a tensor.
///Given a `tensor`, and a `bool` tensor `dims` representing the dimensions
/// of `tensor`, this operation reverses each dimension i of `tensor` where
/// `dims[i]` is `True`.
/// 
/// `tensor` can have up to 8 dimensions. The number of dimensions
/// of `tensor` must equal the number of elements in `dims`. In other words:
/// 
/// `rank(tensor) = size(dims)`
/// 
/// For example:
/// 
/// ```
/// # tensor 't' is [[[[ 0,  1,  2,  3],
/// #                  [ 4,  5,  6,  7],
/// #                  [ 8,  9, 10, 11]],
/// #                 [[12, 13, 14, 15],
/// #                  [16, 17, 18, 19],
/// #                  [20, 21, 22, 23]]]]
/// # tensor 't' shape is [1, 2, 3, 4]
/// 
/// # 'dims' is [False, False, False, True]
/// reverse(t, dims) ==> [[[[ 3,  2,  1,  0],
///                         [ 7,  6,  5,  4],
///                         [ 11, 10, 9, 8]],
///                        [[15, 14, 13, 12],
///                         [19, 18, 17, 16],
///                         [23, 22, 21, 20]]]]
/// 
/// # 'dims' is [False, True, False, False]
/// reverse(t, dims) ==> [[[[12, 13, 14, 15],
///                         [16, 17, 18, 19],
///                         [20, 21, 22, 23]
///                        [[ 0,  1,  2,  3],
///                         [ 4,  5,  6,  7],
///                         [ 8,  9, 10, 11]]]]
/// 
/// # 'dims' is [False, False, True, False]
/// reverse(t, dims) ==> [[[[8, 9, 10, 11],
///                         [4, 5, 6, 7],
///                         [0, 1, 2, 3]]
///                        [[20, 21, 22, 23],
///                         [16, 17, 18, 19],
///                         [12, 13, 14, 15]]]]
/// ```
/// - Parameter tensor: Up to 8-D.
/// - Parameter dims: 1-D. The dimensions to reverse.
/// - Returns: 
///	output: The same shape as `tensor`.
public func reverse(operationName: String? = nil, tensor: Output, dims: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Reverse",
		name: (operationName ?? "Type"),
		input: [tensor, dims],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Returns the batched diagonal part of a batched tensor.
///This operation returns a tensor with the `diagonal` part
/// of the batched `input`. The `diagonal` part is computed as follows:
/// 
/// Assume `input` has `k` dimensions `[I, J, K, ..., M, N]`, then the output is a
/// tensor of rank `k - 1` with dimensions `[I, J, K, ..., min(M, N)]` where:
/// 
/// `diagonal[i, j, k, ..., n] = input[i, j, k, ..., n, n]`.
/// 
/// The input must be at least a matrix.
/// 
/// For example:
/// 
/// ```
/// # 'input' is [[[1, 0, 0, 0]
///                [0, 2, 0, 0]
///                [0, 0, 3, 0]
///                [0, 0, 0, 4]],
///               [[5, 0, 0, 0]
///                [0, 6, 0, 0]
///                [0, 0, 7, 0]
///                [0, 0, 0, 8]]]
/// 
/// and input.shape = (2, 4, 4)
/// 
/// tf.matrix_diag_part(input) ==> [[1, 2, 3, 4], [5, 6, 7, 8]]
/// 
/// which has shape (2, 4)
/// ```
/// - Parameter input: Rank `k` tensor where `k >= 2`.
/// - Returns: 
///	diagonal: The extracted diagonal(s) having shape
/// `diagonal.shape = input.shape[:-2] + [min(input.shape[-2:])]`.
public func matrixDiagPart(operationName: String? = nil, input: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "MatrixDiagPart",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Returns a batched matrix tensor with new batched diagonal values.
///Given `input` and `diagonal`, this operation returns a tensor with the
/// same shape and values as `input`, except for the main diagonal of the
/// innermost matrices.  These will be overwritten by the values in `diagonal`.
/// 
/// The output is computed as follows:
/// 
/// Assume `input` has `k+1` dimensions `[I, J, K, ..., M, N]` and `diagonal` has
/// `k` dimensions `[I, J, K, ..., min(M, N)]`.  Then the output is a
/// tensor of rank `k+1` with dimensions `[I, J, K, ..., M, N]` where:
/// 
///    *  `output[i, j, k, ..., m, n] = diagonal[i, j, k, ..., n]` for `m == n`.
///    *  `output[i, j, k, ..., m, n] = input[i, j, k, ..., m, n]` for `m != n`.
/// - Parameter input: Rank `k+1`, where `k >= 1`.
/// - Parameter diagonal: Rank `k`, where `k >= 1`.
/// - Returns: 
///	output: Rank `k+1`, with `output.shape = input.shape`.
public func matrixSetDiag(operationName: String? = nil, input: Output, diagonal: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "MatrixSetDiag",
		name: (operationName ?? "Type"),
		input: [input, diagonal],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Returns a batched diagonal tensor with a given batched diagonal values.
///Given a `diagonal`, this operation returns a tensor with the `diagonal` and
/// everything else padded with zeros. The diagonal is computed as follows:
/// 
/// Assume `diagonal` has `k` dimensions `[I, J, K, ..., N]`, then the output is a
/// tensor of rank `k+1` with dimensions [I, J, K, ..., N, N]` where:
/// 
/// `output[i, j, k, ..., m, n] = 1{m=n}  *  diagonal[i, j, k, ..., n]`.
/// 
/// For example:
/// 
/// ```
/// # 'diagonal' is [[1, 2, 3, 4], [5, 6, 7, 8]]
/// 
/// and diagonal.shape = (2, 4)
/// 
/// tf.matrix_diag(diagonal) ==> [[[1, 0, 0, 0]
///                                      [0, 2, 0, 0]
///                                      [0, 0, 3, 0]
///                                      [0, 0, 0, 4]],
///                                     [[5, 0, 0, 0]
///                                      [0, 6, 0, 0]
///                                      [0, 0, 7, 0]
///                                      [0, 0, 0, 8]]]
/// 
/// which has shape (2, 4, 4)
/// ```
/// - Parameter diagonal: Rank `k`, where `k >= 1`.
/// - Returns: 
///	output: Rank `k+1`, with `output.shape = diagonal.shape + [diagonal.shape[-1]]`.
public func matrixDiag(operationName: String? = nil, diagonal: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "MatrixDiag",
		name: (operationName ?? "Type"),
		input: [diagonal],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///A placeholder op for a value that will be fed into the computation.
///N.B. This operation will fail with an error if it is executed. It is
/// intended as a way to represent a value that will always be fed, and to
/// provide attrs that enable the fed value to be checked at runtime.
/// - Parameter dtype: The type of elements in the tensor.
/// - Parameter shape: The shape of the tensor. The shape can be any partially-specified
/// shape.  To be unconstrained, pass in a shape with unknown rank.
/// - Returns: 
///	output: A placeholder tensor that must be replaced using the feed mechanism.
public func placeholderV2(operationName: String? = nil, dtype: Any.Type, shape: Shape) throws -> Output { 
	var attrs = [String : Any]()
	attrs["dtype"] = dtype
	attrs["shape"] = shape
	let opspec = OpSpec(
		type: "PlaceholderV2",
		name: (operationName ?? "Type"),
		input: [],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Returns the diagonal part of the tensor.
///This operation returns a tensor with the `diagonal` part
/// of the `input`. The `diagonal` part is computed as follows:
/// 
/// Assume `input` has dimensions `[D1,..., Dk, D1,..., Dk]`, then the output is a
/// tensor of rank `k` with dimensions `[D1,..., Dk]` where:
/// 
/// `diagonal[i1,..., ik] = input[i1, ..., ik, i1,..., ik]`.
/// 
/// For example:
/// 
/// ```
/// # 'input' is [[1, 0, 0, 0]
///               [0, 2, 0, 0]
///               [0, 0, 3, 0]
///               [0, 0, 0, 4]]
/// 
/// tf.diag_part(input) ==> [1, 2, 3, 4]
/// ```
/// - Parameter input: Rank k tensor where k is 2, 4, or 6.
/// - Returns: 
///	diagonal: The extracted diagonal.
public func diagPart(operationName: String? = nil, input: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "DiagPart",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Returns a diagonal tensor with a given diagonal values.
///Given a `diagonal`, this operation returns a tensor with the `diagonal` and
/// everything else padded with zeros. The diagonal is computed as follows:
/// 
/// Assume `diagonal` has dimensions [D1,..., Dk], then the output is a tensor of
/// rank 2k with dimensions [D1,..., Dk, D1,..., Dk] where:
/// 
/// `output[i1,..., ik, i1,..., ik] = diagonal[i1, ..., ik]` and 0 everywhere else.
/// 
/// For example:
/// 
/// ```
/// # 'diagonal' is [1, 2, 3, 4]
/// tf.diag(diagonal) ==> [[1, 0, 0, 0]
///                        [0, 2, 0, 0]
///                        [0, 0, 3, 0]
///                        [0, 0, 0, 4]]
/// ```
/// - Parameter diagonal: Rank k tensor where k is at most 3.
/// - Returns: 
///	output: 
public func diag(operationName: String? = nil, diagonal: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Diag",
		name: (operationName ?? "Type"),
		input: [diagonal],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Compute gradients for a FakeQuantWithMinMaxVarsPerChannel operation.
/// - Parameter gradients: Backpropagated gradients above the FakeQuantWithMinMaxVars operation,
/// shape one of: `[d]`, `[b, d]`,  `[b, h, w, d]`.
/// - Parameter inputs: Values passed as inputs to the FakeQuantWithMinMaxVars operation, shape
///   same as `gradients`.
/// min, max: Quantization interval, floats of shape `[d]`.
/// - Parameter min: 
/// - Parameter max: 
/// - Parameter numBits: The bitwidth of the quantization; between 2 and 8, inclusive.
/// - Parameter narrowRange: Whether to quantize into 2// ^num_bits - 1 distinct values.
/// - Returns: 
///	backprops_wrt_input: Backpropagated gradients w.r.t. inputs, shape same as
/// `inputs`:
///   `gradients  *  (inputs >= min && inputs <= max)`.
///	backprop_wrt_min: Backpropagated gradients w.r.t. min parameter, shape `[d]`:
/// `sum_per_d(gradients  *  (inputs < min))`.
///	backprop_wrt_max: Backpropagated gradients w.r.t. max parameter, shape `[d]`:
/// `sum_per_d(gradients  *  (inputs > max))`.
public func fakeQuantWithMinMaxVarsPerChannelGradient(operationName: String? = nil, gradients: Output, inputs: Output, min: Output, max: Output, numBits: UInt8, narrowRange: Bool) throws -> (backpropsWrtInput: Output, backpropWrtMin: Output, backpropWrtMax: Output) { 
	var attrs = [String : Any]()
	attrs["num_bits"] = numBits
	attrs["narrow_range"] = narrowRange
	let opspec = OpSpec(
		type: "FakeQuantWithMinMaxVarsPerChannelGradient",
		name: (operationName ?? "Type"),
		input: [gradients, inputs, min, max],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (backpropsWrtInput: op.output(at: 0), backpropWrtMin: op.output(at: 1), backpropWrtMax: op.output(at: 2))
} 

///Returns a tensor of ones with the same shape and type as x.
/// - Parameter x: a tensor of type T.
/// - Returns: 
///	y: a tensor of the same shape and type as x but filled with ones.
public func onesLike(operationName: String? = nil, x: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "OnesLike",
		name: (operationName ?? "Type"),
		input: [x],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Returns immutable tensor from memory region.
///The current implementation memmaps the tensor from a file.
/// - Parameter dtype: Type of the returned tensor.
/// - Parameter shape: Shape of the returned tensor.
/// - Parameter memoryRegionName: Name of readonly memory region used by the tensor, see
/// NewReadOnlyMemoryRegionFromFile in tensorflow::Env.
/// - Returns: 
///	tensor: 
public func immutableConst(operationName: String? = nil, dtype: Any.Type, shape: Shape, memoryRegionName: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["dtype"] = dtype
	attrs["shape"] = shape
	attrs["memory_region_name"] = memoryRegionName
	let opspec = OpSpec(
		type: "ImmutableConst",
		name: (operationName ?? "Type"),
		input: [],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Creates a tensor filled with a scalar value.
///This operation creates a tensor of shape `dims` and fills it with `value`.
/// 
/// For example:
/// 
/// ```
/// # Output tensor has shape [2, 3].
/// fill([2, 3], 9) ==> [[9, 9, 9]
///                      [9, 9, 9]]
/// ```
/// - Parameter dims: 1-D. Represents the shape of the output tensor.
/// - Parameter value: 0-D (scalar). Value to fill the returned tensor.
/// 
/// @compatibility(numpy)
/// Equivalent to np.full
/// @end_compatibility
/// - Returns: 
///	output: 
public func fill(operationName: String? = nil, dims: Output, value: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Fill",
		name: (operationName ?? "Type"),
		input: [dims, value],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Returns a constant tensor.
/// - Parameter value: Attr `value` is the tensor to return.
/// - Parameter dtype: 
/// - Returns: 
///	output: 
public func const(operationName: String? = nil, value: Tensor, dtype: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["value"] = value
	attrs["dtype"] = dtype
	let opspec = OpSpec(
		type: "Const",
		name: (operationName ?? "Type"),
		input: [],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Splits a tensor into `num_split` tensors along one dimension.
/// - Parameter value: The tensor to split.
/// - Parameter sizeSplits: list containing the sizes of each output tensor along the split
/// dimension. Must sum to the dimension of value along split_dim.
/// Can contain one -1 indicating that dimension is to be inferred.
/// - Parameter splitDim: 0-D.  The dimension along which to split.  Must be in the range
/// `[-rank(value), rank(value))`.
/// - Parameter numSplit: 
/// - Parameter tlen: 
/// - Returns: 
///	output: Tensors whose shape matches that of `value`
/// except along `split_dim`, where their sizes are
/// `size_splits[i]`.
public func splitV(operationName: String? = nil, value: Output, sizeSplits: Output, splitDim: Output, numSplit: UInt8, tlen: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["num_split"] = numSplit
	attrs["Tlen"] = tlen
	let opspec = OpSpec(
		type: "SplitV",
		name: (operationName ?? "Type"),
		input: [value, sizeSplits, splitDim],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Splits a tensor into `num_split` tensors along one dimension.
/// - Parameter splitDim: 0-D.  The dimension along which to split.  Must be in the range
/// `[-rank(value), rank(value))`.
/// - Parameter value: The tensor to split.
/// - Parameter numSplit: The number of ways to split.  Must evenly divide
/// `value.shape[split_dim]`.
/// - Returns: 
///	output: They are identically shaped tensors, whose shape matches that of `value`
/// except along `split_dim`, where their sizes are
/// `values.shape[split_dim] / num_split`.
public func split(operationName: String? = nil, splitDim: Output, value: Output, numSplit: UInt8) throws -> Output { 
	var attrs = [String : Any]()
	attrs["num_split"] = numSplit
	let opspec = OpSpec(
		type: "Split",
		name: (operationName ?? "Type"),
		input: [splitDim, value],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Concatenates tensors along one dimension.
/// - Parameter values: List of `N` Tensors to concatenate. Their ranks and types must match,
/// and their sizes must match in all dimensions except `concat_dim`.
/// - Parameter axis: 0-D.  The dimension along which to concatenate.  Must be in the
/// range [-rank(values), rank(values)).
/// - Parameter n: 
/// - Parameter tidx: 
/// - Returns: 
///	output: A `Tensor` with the concatenation of values stacked along the
/// `concat_dim` dimension.  This tensor's shape matches that of `values` except
/// in `concat_dim` where it has the sum of the sizes.
public func concatV2(operationName: String? = nil, values: [Output], axis: Output, n: UInt8, tidx: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["N"] = n
	attrs["Tidx"] = tidx
	let opspec = OpSpec(
		type: "ConcatV2",
		name: (operationName ?? "Type"),
		input: [values, axis],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Concatenates tensors along one dimension.
/// - Parameter concatDim: 0-D.  The dimension along which to concatenate.  Must be in the
/// range [0, rank(values)).
/// - Parameter values: The `N` Tensors to concatenate. Their ranks and types must match,
/// and their sizes must match in all dimensions except `concat_dim`.
/// - Parameter n: 
/// - Returns: 
///	output: A `Tensor` with the concatenation of values stacked along the
/// `concat_dim` dimension.  This tensor's shape matches that of `values` except
/// in `concat_dim` where it has the sum of the sizes.
public func concat(operationName: String? = nil, concatDim: Output, values: [Output], n: UInt8) throws -> Output { 
	var attrs = [String : Any]()
	attrs["N"] = n
	let opspec = OpSpec(
		type: "Concat",
		name: (operationName ?? "Type"),
		input: [concatDim, values],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Output a fact about factorials.
/// - Returns: 
///	fact: 
public func fact(operationName: String? = nil) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Fact",
		name: (operationName ?? "Type"),
		input: [],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Parses a text file and creates a batch of examples.
/// - Parameter filename: The corpus's text file name.
/// - Parameter batchSize: The size of produced batch.
/// - Parameter windowSize: The number of words to predict to the left and right of the target.
/// - Parameter minCount: The minimum number of word occurrences for it to be included in the
/// vocabulary.
/// - Parameter subsample: Threshold for word occurrence. Words that appear with higher
/// frequency will be randomly down-sampled. Set to 0 to disable.
/// - Returns: 
///	vocab_word: A vector of words in the corpus.
///	vocab_freq: Frequencies of words. Sorted in the non-ascending order.
///	words_per_epoch: Number of words per epoch in the data file.
///	current_epoch: The current epoch number.
///	total_words_processed: The total number of words processed so far.
///	examples: A vector of word ids.
///	labels: A vector of word ids.
public func skipgram(operationName: String? = nil, filename: String, batchSize: UInt8, windowSize: UInt8, minCount: UInt8, subsample: Float) throws -> (vocabWord: Output, vocabFreq: Output, wordsPerEpoch: Output, currentEpoch: Output, totalWordsProcessed: Output, examples: Output, labels: Output) { 
	var attrs = [String : Any]()
	attrs["filename"] = filename
	attrs["batch_size"] = batchSize
	attrs["window_size"] = windowSize
	attrs["min_count"] = minCount
	attrs["subsample"] = subsample
	let opspec = OpSpec(
		type: "Skipgram",
		name: (operationName ?? "Type"),
		input: [],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (vocabWord: op.output(at: 0), vocabFreq: op.output(at: 1), wordsPerEpoch: op.output(at: 2), currentEpoch: op.output(at: 3), totalWordsProcessed: op.output(at: 4), examples: op.output(at: 5), labels: op.output(at: 6))
} 

///Finds unique elements in a 1-D tensor.
///This operation returns a tensor `y` containing all of the unique elements of `x`
/// sorted in the same order that they occur in `x`. This operation also returns a
/// tensor `idx` the same size as `x` that contains the index of each value of `x`
/// in the unique output `y`. Finally, it returns a third tensor `count` that
/// contains the count of each element of `y` in `x`. In other words:
/// 
/// `y[idx[i]] = x[i] for i in [0, 1,...,rank(x) - 1]`
/// 
/// For example:
/// 
/// ```
/// # tensor 'x' is [1, 1, 2, 4, 4, 4, 7, 8, 8]
/// y, idx, count = unique_with_counts(x)
/// y ==> [1, 2, 4, 7, 8]
/// idx ==> [0, 0, 1, 2, 2, 2, 3, 4, 4]
/// count ==> [2, 1, 3, 1, 2]
/// ```
/// - Parameter x: 1-D.
/// - Parameter outIdx: 
/// - Returns: 
///	y: 1-D.
///	idx: 1-D.
///	count: 1-D.
public func uniqueWithCounts(operationName: String? = nil, x: Output, outIdx: Any.Type) throws -> (y: Output, idx: Output, count: Output) { 
	var attrs = [String : Any]()
	attrs["out_idx"] = outIdx
	let opspec = OpSpec(
		type: "UniqueWithCounts",
		name: (operationName ?? "Type"),
		input: [x],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (y: op.output(at: 0), idx: op.output(at: 1), count: op.output(at: 2))
} 

///Update ' * var' according to the centered RMSProp algorithm.
///The centered RMSProp algorithm uses an estimate of the centered second moment
/// (i.e., the variance) for normalization, as opposed to regular RMSProp, which
/// uses the (uncentered) second moment. This often helps with training, but is
/// slightly more expensive in terms of computation and memory.
/// 
/// Note that in dense implementation of this algorithm, mg, ms, and mom will
/// update even if the grad is zero, but in this sparse implementation, mg, ms,
/// and mom will not update in iterations during which the grad is zero.
/// 
/// mean_square = decay  *  mean_square + (1-decay)  *  gradient  *  *  2
/// mean_grad = decay  *  mean_grad + (1-decay)  *  gradient
/// 
/// Delta = learning_rate  *  gradient / sqrt(mean_square + epsilon - mean_grad  *  *  2)
/// 
/// mg <- rho  *  mg_{t-1} + (1-rho)  *  grad
/// ms <- rho  *  ms_{t-1} + (1-rho)  *  grad  *  grad
/// mom <- momentum  *  mom_{t-1} + lr  *  grad / sqrt(ms - mg  *  mg + epsilon)
/// var <- var - mom
/// - Parameter `var`: Should be from a Variable().
/// - Parameter mg: Should be from a Variable().
/// - Parameter ms: Should be from a Variable().
/// - Parameter mom: Should be from a Variable().
/// - Parameter lr: Scaling factor. Must be a scalar.
/// - Parameter rho: Decay rate. Must be a scalar.
/// - Parameter momentum: 
/// - Parameter epsilon: Ridge term. Must be a scalar.
/// - Parameter grad: The gradient.
/// - Parameter useLocking: If `True`, updating of the var, mg, ms, and mom tensors is
/// protected by a lock; otherwise the behavior is undefined, but may exhibit less
/// contention.
public func resourceApplyCenteredRMSProp(operationName: String? = nil, `var`: Output, mg: Output, ms: Output, mom: Output, lr: Output, rho: Output, momentum: Output, epsilon: Output, grad: Output, useLocking: Bool) throws -> Operation { 
	var attrs = [String : Any]()
	attrs["use_locking"] = useLocking
	let opspec = OpSpec(
		type: "ResourceApplyCenteredRMSProp",
		name: (operationName ?? "Type"),
		input: [`var`, mg, ms, mom, lr, rho, momentum, epsilon, grad],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Performs fractional max pooling on the input.
///Fractional max pooling is slightly different than regular max pooling.  In
/// regular max pooling, you downsize an input set by taking the maximum value of
/// smaller N x N subsections of the set (often 2x2), and try to reduce the set by
/// a factor of N, where N is an integer.  Fractional max pooling, as you might
/// expect from the word "fractional", means that the overall reduction ratio N
/// does not have to be an integer.
/// 
/// The sizes of the pooling regions are generated randomly but are fairly uniform.
/// For example, let's look at the height dimension, and the constraints on the
/// list of rows that will be pool boundaries.
/// 
/// First we define the following:
/// 
/// 1.  input_row_length : the number of rows from the input set
/// 2.  output_row_length : which will be smaller than the input
/// 3.  alpha = input_row_length / output_row_length : our reduction ratio
/// 4.  K = floor(alpha)
/// 5.  row_pooling_sequence : this is the result list of pool boundary rows
/// 
/// Then, row_pooling_sequence should satisfy:
/// 
/// 1.  a[0] = 0 : the first value of the sequence is 0
/// 2.  a[end] = input_row_length : the last value of the sequence is the size
/// 3.  K <= (a[i+1] - a[i]) <= K+1 : all intervals are K or K+1 size
/// 4.  length(row_pooling_sequence) = output_row_length+1
/// 
/// For more details on fractional max pooling, see this paper:
/// [Benjamin Graham, Fractional Max-Pooling](http://arxiv.org/abs/1412.6071)
/// - Parameter value: 4-D with shape `[batch, height, width, channels]`.
/// - Parameter poolingRatio: Pooling ratio for each dimension of `value`, currently only
/// supports row and col dimension and should be >= 1.0. For example, a valid
/// pooling ratio looks like [1.0, 1.44, 1.73, 1.0]. The first and last elements
/// must be 1.0 because we don't allow pooling on batch and channels
/// dimensions. 1.44 and 1.73 are pooling ratio on height and width dimensions
/// respectively.
/// - Parameter pseudoRandom: When set to True, generates the pooling sequence in a
/// pseudorandom fashion, otherwise, in a random fashion. Check paper [Benjamin
/// Graham, Fractional Max-Pooling](http://arxiv.org/abs/1412.6071) for
/// difference between pseudorandom and random.
/// - Parameter overlapping: When set to True, it means when pooling, the values at the boundary
/// of adjacent pooling cells are used by both cells. For example:
/// 
/// `index  0  1  2  3  4`
/// 
/// `value  20 5  16 3  7`
/// 
/// If the pooling sequence is [0, 2, 4], then 16, at index 2 will be used twice.
/// The result would be [20, 16] for fractional max pooling.
/// - Parameter deterministic: When set to True, a fixed pooling region will be used when
/// iterating over a FractionalMaxPool node in the computation graph. Mainly used
/// in unit test to make FractionalMaxPool deterministic.
/// - Parameter seed: If either seed or seed2 are set to be non-zero, the random number
/// generator is seeded by the given seed.  Otherwise, it is seeded by a
/// random seed.
/// - Parameter seed2: An second seed to avoid seed collision.
/// - Returns: 
///	output: output tensor after fractional max pooling.
///	row_pooling_sequence: row pooling sequence, needed to calculate gradient.
///	col_pooling_sequence: column pooling sequence, needed to calculate gradient.
public func fractionalMaxPool(operationName: String? = nil, value: Output, poolingRatio: [Float], pseudoRandom: Bool, overlapping: Bool, deterministic: Bool, seed: UInt8, seed2: UInt8) throws -> (output: Output, rowPoolingSequence: Output, colPoolingSequence: Output) { 
	var attrs = [String : Any]()
	attrs["pooling_ratio"] = poolingRatio
	attrs["pseudo_random"] = pseudoRandom
	attrs["overlapping"] = overlapping
	attrs["deterministic"] = deterministic
	attrs["seed"] = seed
	attrs["seed2"] = seed2
	let opspec = OpSpec(
		type: "FractionalMaxPool",
		name: (operationName ?? "Type"),
		input: [value],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (output: op.output(at: 0), rowPoolingSequence: op.output(at: 1), colPoolingSequence: op.output(at: 2))
} 

///Update ' * var' according to the RMSProp algorithm.
///Note that in dense implementation of this algorithm, ms and mom will
/// update even if the grad is zero, but in this sparse implementation, ms
/// and mom will not update in iterations during which the grad is zero.
/// 
/// mean_square = decay  *  mean_square + (1-decay)  *  gradient  *  *  2
/// Delta = learning_rate  *  gradient / sqrt(mean_square + epsilon)
/// 
/// ms <- rho  *  ms_{t-1} + (1-rho)  *  grad  *  grad
/// mom <- momentum  *  mom_{t-1} + lr  *  grad / sqrt(ms + epsilon)
/// var <- var - mom
/// - Parameter `var`: Should be from a Variable().
/// - Parameter ms: Should be from a Variable().
/// - Parameter mom: Should be from a Variable().
/// - Parameter lr: Scaling factor. Must be a scalar.
/// - Parameter rho: Decay rate. Must be a scalar.
/// - Parameter momentum: 
/// - Parameter epsilon: Ridge term. Must be a scalar.
/// - Parameter grad: The gradient.
/// - Parameter useLocking: If `True`, updating of the var, ms, and mom tensors is protected
/// by a lock; otherwise the behavior is undefined, but may exhibit less
/// contention.
public func resourceApplyRMSProp(operationName: String? = nil, `var`: Output, ms: Output, mom: Output, lr: Output, rho: Output, momentum: Output, epsilon: Output, grad: Output, useLocking: Bool) throws -> Operation { 
	var attrs = [String : Any]()
	attrs["use_locking"] = useLocking
	let opspec = OpSpec(
		type: "ResourceApplyRMSProp",
		name: (operationName ?? "Type"),
		input: [`var`, ms, mom, lr, rho, momentum, epsilon, grad],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Returns a tensor of zeros with the same shape and type as x.
/// - Parameter x: a tensor of type T.
/// - Returns: 
///	y: a tensor of the same shape and type as x but filled with zeros.
public func zerosLike(operationName: String? = nil, x: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "ZerosLike",
		name: (operationName ?? "Type"),
		input: [x],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Update ' * var' according to the centered RMSProp algorithm.
///The centered RMSProp algorithm uses an estimate of the centered second moment
/// (i.e., the variance) for normalization, as opposed to regular RMSProp, which
/// uses the (uncentered) second moment. This often helps with training, but is
/// slightly more expensive in terms of computation and memory.
/// 
/// Note that in dense implementation of this algorithm, mg, ms, and mom will
/// update even if the grad is zero, but in this sparse implementation, mg, ms,
/// and mom will not update in iterations during which the grad is zero.
/// 
/// mean_square = decay  *  mean_square + (1-decay)  *  gradient  *  *  2
/// mean_grad = decay  *  mean_grad + (1-decay)  *  gradient
/// 
/// Delta = learning_rate  *  gradient / sqrt(mean_square + epsilon - mean_grad  *  *  2)
/// 
/// mg <- rho  *  mg_{t-1} + (1-rho)  *  grad
/// ms <- rho  *  ms_{t-1} + (1-rho)  *  grad  *  grad
/// mom <- momentum  *  mom_{t-1} + lr  *  grad / sqrt(ms - mg  *  mg + epsilon)
/// var <- var - mom
/// - Parameter `var`: Should be from a Variable().
/// - Parameter mg: Should be from a Variable().
/// - Parameter ms: Should be from a Variable().
/// - Parameter mom: Should be from a Variable().
/// - Parameter lr: Scaling factor. Must be a scalar.
/// - Parameter rho: Decay rate. Must be a scalar.
/// - Parameter momentum: 
/// - Parameter epsilon: Ridge term. Must be a scalar.
/// - Parameter grad: The gradient.
/// - Parameter useLocking: If `True`, updating of the var, mg, ms, and mom tensors is
/// protected by a lock; otherwise the behavior is undefined, but may exhibit less
/// contention.
/// - Returns: 
///	out: Same as "var".
public func applyCenteredRMSProp(operationName: String? = nil, `var`: Output, mg: Output, ms: Output, mom: Output, lr: Output, rho: Output, momentum: Output, epsilon: Output, grad: Output, useLocking: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["use_locking"] = useLocking
	let opspec = OpSpec(
		type: "ApplyCenteredRMSProp",
		name: (operationName ?? "Type"),
		input: [`var`, mg, ms, mom, lr, rho, momentum, epsilon, grad],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes offsets of concat inputs within its output.
///For example:
/// 
/// ```
/// # 'x' is [2, 2, 7]
/// # 'y' is [2, 3, 7]
/// # 'z' is [2, 5, 7]
/// concat_offset(2, [x, y, z]) => [0, 0, 0], [0, 2, 0], [0, 5, 0]
/// ```
/// 
/// This is typically used by gradient computations for a concat operation.
/// - Parameter concatDim: The dimension along which to concatenate.
/// - Parameter shape: The `N` int32 vectors representing shape of tensors being concatenated.
/// - Parameter n: 
/// - Returns: 
///	offset: The `N` int32 vectors representing the starting offset
/// of input tensors within the concatenated output.
public func concatOffset(operationName: String? = nil, concatDim: Output, shape: [Output], n: UInt8) throws -> Output { 
	var attrs = [String : Any]()
	attrs["N"] = n
	let opspec = OpSpec(
		type: "ConcatOffset",
		name: (operationName ?? "Type"),
		input: [concatDim, shape],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Update ' * var' according to the Adam algorithm.
///lr_t <- learning_rate  *  sqrt(1 - beta2// ^t) / (1 - beta1// ^t)
/// m_t <- beta1  *  m_{t-1} + (1 - beta1)  *  g_t
/// v_t <- beta2  *  v_{t-1} + (1 - beta2)  *  g_t  *  g_t
/// variable <- variable - lr_t  *  m_t / (sqrt(v_t) + epsilon)
/// - Parameter `var`: Should be from a Variable().
/// - Parameter m: Should be from a Variable().
/// - Parameter v: Should be from a Variable().
/// - Parameter beta1Power: Must be a scalar.
/// - Parameter beta2Power: Must be a scalar.
/// - Parameter lr: Scaling factor. Must be a scalar.
/// - Parameter beta1: Momentum factor. Must be a scalar.
/// - Parameter beta2: Momentum factor. Must be a scalar.
/// - Parameter epsilon: Ridge term. Must be a scalar.
/// - Parameter grad: The gradient.
/// - Parameter useLocking: If `True`, updating of the var, m, and v tensors will be protected
/// by a lock; otherwise the behavior is undefined, but may exhibit less
/// contention.
/// - Parameter useNesterov: If `True`, uses the nesterov update.
public func resourceApplyAdam(operationName: String? = nil, `var`: Output, m: Output, v: Output, beta1Power: Output, beta2Power: Output, lr: Output, beta1: Output, beta2: Output, epsilon: Output, grad: Output, useLocking: Bool, useNesterov: Bool) throws -> Operation { 
	var attrs = [String : Any]()
	attrs["use_locking"] = useLocking
	attrs["use_nesterov"] = useNesterov
	let opspec = OpSpec(
		type: "ResourceApplyAdam",
		name: (operationName ?? "Type"),
		input: [`var`, m, v, beta1Power, beta2Power, lr, beta1, beta2, epsilon, grad],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Update relevant entries in ' * var' and ' * accum' according to the momentum scheme.
///Set use_nesterov = True if you want to use Nesterov momentum.
/// 
/// That is for rows we have grad for, we update var and accum as follows:
/// 
/// accum = accum  *  momentum + grad
/// var -= lr  *  accum
/// - Parameter `var`: Should be from a Variable().
/// - Parameter accum: Should be from a Variable().
/// - Parameter lr: Learning rate. Must be a scalar.
/// - Parameter grad: The gradient.
/// - Parameter indices: A vector of indices into the first dimension of var and accum.
/// - Parameter momentum: Momentum. Must be a scalar.
/// - Parameter tindices: 
/// - Parameter useLocking: If `True`, updating of the var and accum tensors will be protected
/// by a lock; otherwise the behavior is undefined, but may exhibit less
/// contention.
/// - Parameter useNesterov: If `True`, the tensor passed to compute grad will be
/// var - lr  *  momentum  *  accum, so in the end, the var you get is actually
/// var - lr  *  momentum  *  accum.
public func resourceSparseApplyMomentum(operationName: String? = nil, `var`: Output, accum: Output, lr: Output, grad: Output, indices: Output, momentum: Output, tindices: Any.Type, useLocking: Bool, useNesterov: Bool) throws -> Operation { 
	var attrs = [String : Any]()
	attrs["Tindices"] = tindices
	attrs["use_locking"] = useLocking
	attrs["use_nesterov"] = useNesterov
	let opspec = OpSpec(
		type: "ResourceSparseApplyMomentum",
		name: (operationName ?? "Type"),
		input: [`var`, accum, lr, grad, indices, momentum],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Update ' * var' according to the momentum scheme. Set use_nesterov = True if you
///want to use Nesterov momentum.
/// 
/// accum = accum  *  momentum + grad
/// var -= lr  *  accum
/// - Parameter `var`: Should be from a Variable().
/// - Parameter accum: Should be from a Variable().
/// - Parameter lr: Scaling factor. Must be a scalar.
/// - Parameter grad: The gradient.
/// - Parameter momentum: Momentum. Must be a scalar.
/// - Parameter useLocking: If `True`, updating of the var and accum tensors will be protected
/// by a lock; otherwise the behavior is undefined, but may exhibit less
/// contention.
/// - Parameter useNesterov: If `True`, the tensor passed to compute grad will be
/// var - lr  *  momentum  *  accum, so in the end, the var you get is actually
/// var - lr  *  momentum  *  accum.
public func resourceApplyMomentum(operationName: String? = nil, `var`: Output, accum: Output, lr: Output, grad: Output, momentum: Output, useLocking: Bool, useNesterov: Bool) throws -> Operation { 
	var attrs = [String : Any]()
	attrs["use_locking"] = useLocking
	attrs["use_nesterov"] = useNesterov
	let opspec = OpSpec(
		type: "ResourceApplyMomentum",
		name: (operationName ?? "Type"),
		input: [`var`, accum, lr, grad, momentum],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Update ' * var' according to the momentum scheme. Set use_nesterov = True if you
///want to use Nesterov momentum.
/// 
/// accum = accum  *  momentum + grad
/// var -= lr  *  accum
/// - Parameter `var`: Should be from a Variable().
/// - Parameter accum: Should be from a Variable().
/// - Parameter lr: Scaling factor. Must be a scalar.
/// - Parameter grad: The gradient.
/// - Parameter momentum: Momentum. Must be a scalar.
/// - Parameter useLocking: If `True`, updating of the var and accum tensors will be protected
/// by a lock; otherwise the behavior is undefined, but may exhibit less
/// contention.
/// - Parameter useNesterov: If `True`, the tensor passed to compute grad will be
/// var - lr  *  momentum  *  accum, so in the end, the var you get is actually
/// var - lr  *  momentum  *  accum.
/// - Returns: 
///	out: Same as "var".
public func applyMomentum(operationName: String? = nil, `var`: Output, accum: Output, lr: Output, grad: Output, momentum: Output, useLocking: Bool, useNesterov: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["use_locking"] = useLocking
	attrs["use_nesterov"] = useNesterov
	let opspec = OpSpec(
		type: "ApplyMomentum",
		name: (operationName ?? "Type"),
		input: [`var`, accum, lr, grad, momentum],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes the (possibly normalized) Levenshtein Edit Distance.
///The inputs are variable-length sequences provided by SparseTensors
///   (hypothesis_indices, hypothesis_values, hypothesis_shape)
/// and
///   (truth_indices, truth_values, truth_shape).
/// 
/// The inputs are:
/// - Parameter hypothesisIndices: The indices of the hypothesis list SparseTensor.
/// This is an N x R int64 matrix.
/// - Parameter hypothesisValues: The values of the hypothesis list SparseTensor.
/// This is an N-length vector.
/// - Parameter hypothesisShape: The shape of the hypothesis list SparseTensor.
/// This is an R-length vector.
/// - Parameter truthIndices: The indices of the truth list SparseTensor.
/// This is an M x R int64 matrix.
/// - Parameter truthValues: The values of the truth list SparseTensor.
/// This is an M-length vector.
/// - Parameter truthShape: truth indices, vector.
/// - Parameter normalize: boolean (if true, edit distances are normalized by length of truth).
/// 
/// The output is:
/// - Returns: 
///	output: A dense float tensor with rank R - 1.
/// 
/// For the example input:
/// 
///     // hypothesis represents a 2x1 matrix with variable-length values:
///     //   (0,0) = ["a"]
///     //   (1,0) = ["b"]
///     hypothesis_indices = [[0, 0, 0],
///                           [1, 0, 0]]
///     hypothesis_values = ["a", "b"]
///     hypothesis_shape = [2, 1, 1]
/// 
///     // truth represents a 2x2 matrix with variable-length values:
///     //   (0,0) = []
///     //   (0,1) = ["a"]
///     //   (1,0) = ["b", "c"]
///     //   (1,1) = ["a"]
///     truth_indices = [[0, 1, 0],
///                      [1, 0, 0],
///                      [1, 0, 1],
///                      [1, 1, 0]]
///     truth_values = ["a", "b", "c", "a"]
///     truth_shape = [2, 2, 2]
///     normalize = true
/// 
/// The output will be:
/// 
///     // output is a 2x2 matrix with edit distances normalized by truth lengths.
///     output = [[inf, 1.0],  // (0,0): no truth, (0,1): no hypothesis
///               [0.5, 1.0]]  // (1,0): addition, (1,1): no hypothesis
public func editDistance(operationName: String? = nil, hypothesisIndices: Output, hypothesisValues: Output, hypothesisShape: Output, truthIndices: Output, truthValues: Output, truthShape: Output, normalize: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["normalize"] = normalize
	let opspec = OpSpec(
		type: "EditDistance",
		name: (operationName ?? "Type"),
		input: [hypothesisIndices, hypothesisValues, hypothesisShape, truthIndices, truthValues, truthShape],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Update ' * var' according to the Ftrl-proximal scheme.
///grad_with_shrinkage = grad + 2  *  l2_shrinkage  *  var
/// accum_new = accum + grad_with_shrinkage  *  grad_with_shrinkage
/// linear += grad_with_shrinkage +
///     (accum_new// ^(-lr_power) - accum// ^(-lr_power)) / lr  *  var
/// quadratic = 1.0 / (accum_new// ^(lr_power)  *  lr) + 2  *  l2
/// var = (sign(linear)  *  l1 - linear) / quadratic if |linear| > l1 else 0.0
/// accum = accum_new
/// - Parameter `var`: Should be from a Variable().
/// - Parameter accum: Should be from a Variable().
/// - Parameter linear: Should be from a Variable().
/// - Parameter grad: The gradient.
/// - Parameter lr: Scaling factor. Must be a scalar.
/// - Parameter l1: L1 regulariation. Must be a scalar.
/// - Parameter l2: L2 shrinkage regulariation. Must be a scalar.
/// - Parameter l2Shrinkage: 
/// - Parameter lrPower: Scaling factor. Must be a scalar.
/// - Parameter useLocking: If `True`, updating of the var and accum tensors will be protected
/// by a lock; otherwise the behavior is undefined, but may exhibit less
/// contention.
public func resourceApplyFtrlV2(operationName: String? = nil, `var`: Output, accum: Output, linear: Output, grad: Output, lr: Output, l1: Output, l2: Output, l2Shrinkage: Output, lrPower: Output, useLocking: Bool) throws -> Operation { 
	var attrs = [String : Any]()
	attrs["use_locking"] = useLocking
	let opspec = OpSpec(
		type: "ResourceApplyFtrlV2",
		name: (operationName ?? "Type"),
		input: [`var`, accum, linear, grad, lr, l1, l2, l2Shrinkage, lrPower],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Update relevant entries in ' * var' according to the Ftrl-proximal scheme.
///That is for rows we have grad for, we update var, accum and linear as follows:
/// grad_with_shrinkage = grad + 2  *  l2_shrinkage  *  var
/// accum_new = accum + grad_with_shrinkage  *  grad_with_shrinkage
/// linear += grad_with_shrinkage +
///     (accum_new// ^(-lr_power) - accum// ^(-lr_power)) / lr  *  var
/// quadratic = 1.0 / (accum_new// ^(lr_power)  *  lr) + 2  *  l2
/// var = (sign(linear)  *  l1 - linear) / quadratic if |linear| > l1 else 0.0
/// accum = accum_new
/// - Parameter `var`: Should be from a Variable().
/// - Parameter accum: Should be from a Variable().
/// - Parameter linear: Should be from a Variable().
/// - Parameter grad: The gradient.
/// - Parameter indices: A vector of indices into the first dimension of var and accum.
/// - Parameter lr: Scaling factor. Must be a scalar.
/// - Parameter l1: L1 regularization. Must be a scalar.
/// - Parameter l2: L2 shrinkage regulariation. Must be a scalar.
/// - Parameter l2Shrinkage: 
/// - Parameter lrPower: Scaling factor. Must be a scalar.
/// - Parameter tindices: 
/// - Parameter useLocking: If `True`, updating of the var and accum tensors will be protected
/// by a lock; otherwise the behavior is undefined, but may exhibit less
/// contention.
/// - Returns: 
///	out: Same as "var".
public func sparseApplyFtrlV2(operationName: String? = nil, `var`: Output, accum: Output, linear: Output, grad: Output, indices: Output, lr: Output, l1: Output, l2: Output, l2Shrinkage: Output, lrPower: Output, tindices: Any.Type, useLocking: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Tindices"] = tindices
	attrs["use_locking"] = useLocking
	let opspec = OpSpec(
		type: "SparseApplyFtrlV2",
		name: (operationName ?? "Type"),
		input: [`var`, accum, linear, grad, indices, lr, l1, l2, l2Shrinkage, lrPower],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Update relevant entries in ' * var' according to the Ftrl-proximal scheme.
///That is for rows we have grad for, we update var, accum and linear as follows:
/// accum_new = accum + grad  *  grad
/// linear += grad + (accum_new// ^(-lr_power) - accum// ^(-lr_power)) / lr  *  var
/// quadratic = 1.0 / (accum_new// ^(lr_power)  *  lr) + 2  *  l2
/// var = (sign(linear)  *  l1 - linear) / quadratic if |linear| > l1 else 0.0
/// accum = accum_new
/// - Parameter `var`: Should be from a Variable().
/// - Parameter accum: Should be from a Variable().
/// - Parameter linear: Should be from a Variable().
/// - Parameter grad: The gradient.
/// - Parameter indices: A vector of indices into the first dimension of var and accum.
/// - Parameter lr: Scaling factor. Must be a scalar.
/// - Parameter l1: L1 regularization. Must be a scalar.
/// - Parameter l2: L2 regularization. Must be a scalar.
/// - Parameter lrPower: Scaling factor. Must be a scalar.
/// - Parameter tindices: 
/// - Parameter useLocking: If `True`, updating of the var and accum tensors will be protected
/// by a lock; otherwise the behavior is undefined, but may exhibit less
/// contention.
public func resourceSparseApplyFtrl(operationName: String? = nil, `var`: Output, accum: Output, linear: Output, grad: Output, indices: Output, lr: Output, l1: Output, l2: Output, lrPower: Output, tindices: Any.Type, useLocking: Bool) throws -> Operation { 
	var attrs = [String : Any]()
	attrs["Tindices"] = tindices
	attrs["use_locking"] = useLocking
	let opspec = OpSpec(
		type: "ResourceSparseApplyFtrl",
		name: (operationName ?? "Type"),
		input: [`var`, accum, linear, grad, indices, lr, l1, l2, lrPower],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Returns an element-wise indication of the sign of a number.
///`y = sign(x) = -1` if `x < 0`; 0 if `x == 0`; 1 if `x > 0`.
/// 
/// For complex numbers, `y = sign(x) = x / |x|` if `x != 0`, otherwise `y = 0`.
/// - Parameter x: 
/// - Returns: 
///	y: 
public func sign(operationName: String? = nil, x: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Sign",
		name: (operationName ?? "Type"),
		input: [x],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Sparse update entries in ' * var' and ' * accum' according to FOBOS algorithm.
///That is for rows we have grad for, we update var and accum as follows:
/// accum += grad  *  grad
/// prox_v = var
/// prox_v -= lr  *  grad  *  (1 / sqrt(accum))
/// var = sign(prox_v)/(1+lr * l2)  *  max{|prox_v|-lr * l1,0}
/// - Parameter `var`: Should be from a Variable().
/// - Parameter accum: Should be from a Variable().
/// - Parameter lr: Learning rate. Must be a scalar.
/// - Parameter l1: L1 regularization. Must be a scalar.
/// - Parameter l2: L2 regularization. Must be a scalar.
/// - Parameter grad: The gradient.
/// - Parameter indices: A vector of indices into the first dimension of var and accum.
/// - Parameter tindices: 
/// - Parameter useLocking: If True, updating of the var and accum tensors will be protected by
/// a lock; otherwise the behavior is undefined, but may exhibit less contention.
public func resourceSparseApplyProximalAdagrad(operationName: String? = nil, `var`: Output, accum: Output, lr: Output, l1: Output, l2: Output, grad: Output, indices: Output, tindices: Any.Type, useLocking: Bool) throws -> Operation { 
	var attrs = [String : Any]()
	attrs["Tindices"] = tindices
	attrs["use_locking"] = useLocking
	let opspec = OpSpec(
		type: "ResourceSparseApplyProximalAdagrad",
		name: (operationName ?? "Type"),
		input: [`var`, accum, lr, l1, l2, grad, indices],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Update ' * var' according to the proximal adagrad scheme.
/// - Parameter `var`: Should be from a Variable().
/// - Parameter gradientAccumulator: Should be from a Variable().
/// - Parameter gradientSquaredAccumulator: Should be from a Variable().
/// - Parameter grad: The gradient.
/// - Parameter lr: Scaling factor. Must be a scalar.
/// - Parameter l1: L1 regularization. Must be a scalar.
/// - Parameter l2: L2 regularization. Must be a scalar.
/// - Parameter globalStep: Training step number. Must be a scalar.
/// - Parameter useLocking: If True, updating of the var and accum tensors will be protected by
/// a lock; otherwise the behavior is undefined, but may exhibit less contention.
public func resourceApplyAdagradDA(operationName: String? = nil, `var`: Output, gradientAccumulator: Output, gradientSquaredAccumulator: Output, grad: Output, lr: Output, l1: Output, l2: Output, globalStep: Output, useLocking: Bool) throws -> Operation { 
	var attrs = [String : Any]()
	attrs["use_locking"] = useLocking
	let opspec = OpSpec(
		type: "ResourceApplyAdagradDA",
		name: (operationName ?? "Type"),
		input: [`var`, gradientAccumulator, gradientSquaredAccumulator, grad, lr, l1, l2, globalStep],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Update entries in ' * var' and ' * accum' according to the proximal adagrad scheme.
/// - Parameter `var`: Should be from a Variable().
/// - Parameter gradientAccumulator: Should be from a Variable().
/// - Parameter gradientSquaredAccumulator: Should be from a Variable().
/// - Parameter grad: The gradient.
/// - Parameter indices: A vector of indices into the first dimension of var and accum.
/// - Parameter lr: Learning rate. Must be a scalar.
/// - Parameter l1: L1 regularization. Must be a scalar.
/// - Parameter l2: L2 regularization. Must be a scalar.
/// - Parameter globalStep: Training step number. Must be a scalar.
/// - Parameter tindices: 
/// - Parameter useLocking: If True, updating of the var and accum tensors will be protected by
/// a lock; otherwise the behavior is undefined, but may exhibit less contention.
/// - Returns: 
///	out: Same as "var".
public func sparseApplyAdagradDA(operationName: String? = nil, `var`: Output, gradientAccumulator: Output, gradientSquaredAccumulator: Output, grad: Output, indices: Output, lr: Output, l1: Output, l2: Output, globalStep: Output, tindices: Any.Type, useLocking: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Tindices"] = tindices
	attrs["use_locking"] = useLocking
	let opspec = OpSpec(
		type: "SparseApplyAdagradDA",
		name: (operationName ?? "Type"),
		input: [`var`, gradientAccumulator, gradientSquaredAccumulator, grad, indices, lr, l1, l2, globalStep],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes scaled exponential linear: `scale  *  alpha  *  (exp(features) - 1)`
///if < 0, `scale  *  features` otherwise.
/// 
/// See [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
/// - Parameter features: 
/// - Returns: 
///	activations: 
public func selu(operationName: String? = nil, features: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Selu",
		name: (operationName ?? "Type"),
		input: [features],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Update relevant entries in ' * var' and ' * accum' according to the adagrad scheme.
///That is for rows we have grad for, we update var and accum as follows:
/// accum += grad  *  grad
/// var -= lr  *  grad  *  (1 / sqrt(accum))
/// - Parameter `var`: Should be from a Variable().
/// - Parameter accum: Should be from a Variable().
/// - Parameter lr: Learning rate. Must be a scalar.
/// - Parameter grad: The gradient.
/// - Parameter indices: A vector of indices into the first dimension of var and accum.
/// - Parameter tindices: 
/// - Parameter useLocking: If `True`, updating of the var and accum tensors will be protected
/// by a lock; otherwise the behavior is undefined, but may exhibit less
/// contention.
public func resourceSparseApplyAdagrad(operationName: String? = nil, `var`: Output, accum: Output, lr: Output, grad: Output, indices: Output, tindices: Any.Type, useLocking: Bool) throws -> Operation { 
	var attrs = [String : Any]()
	attrs["Tindices"] = tindices
	attrs["use_locking"] = useLocking
	let opspec = OpSpec(
		type: "ResourceSparseApplyAdagrad",
		name: (operationName ?? "Type"),
		input: [`var`, accum, lr, grad, indices],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Update ' * var' and ' * accum' according to FOBOS with Adagrad learning rate.
///accum += grad  *  grad
/// prox_v = var - lr  *  grad  *  (1 / sqrt(accum))
/// var = sign(prox_v)/(1+lr * l2)  *  max{|prox_v|-lr * l1,0}
/// - Parameter `var`: Should be from a Variable().
/// - Parameter accum: Should be from a Variable().
/// - Parameter lr: Scaling factor. Must be a scalar.
/// - Parameter l1: L1 regularization. Must be a scalar.
/// - Parameter l2: L2 regularization. Must be a scalar.
/// - Parameter grad: The gradient.
/// - Parameter useLocking: If True, updating of the var and accum tensors will be protected by
/// a lock; otherwise the behavior is undefined, but may exhibit less contention.
public func resourceApplyProximalAdagrad(operationName: String? = nil, `var`: Output, accum: Output, lr: Output, l1: Output, l2: Output, grad: Output, useLocking: Bool) throws -> Operation { 
	var attrs = [String : Any]()
	attrs["use_locking"] = useLocking
	let opspec = OpSpec(
		type: "ResourceApplyProximalAdagrad",
		name: (operationName ?? "Type"),
		input: [`var`, accum, lr, l1, l2, grad],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Produces the max pool of the input tensor for quantized types.
/// - Parameter input: The 4D (batch x rows x cols x depth) Tensor to MaxReduce over.
/// - Parameter minInput: The float value that the lowest quantized input value represents.
/// - Parameter maxInput: The float value that the highest quantized input value represents.
/// - Parameter ksize: The size of the window for each dimension of the input tensor.
/// The length must be 4 to match the number of dimensions of the input.
/// - Parameter strides: The stride of the sliding window for each dimension of the input
/// tensor. The length must be 4 to match the number of dimensions of the input.
/// - Parameter padding: The type of padding algorithm to use.
/// - Returns: 
///	output: 
///	min_output: The float value that the lowest quantized output value represents.
///	max_output: The float value that the highest quantized output value represents.
public func quantizedMaxPool(operationName: String? = nil, input: Output, minInput: Output, maxInput: Output, ksize: [Int64], strides: [Int64], padding: String) throws -> (output: Output, minOutput: Output, maxOutput: Output) { 
	var attrs = [String : Any]()
	attrs["ksize"] = ksize
	attrs["strides"] = strides
	attrs["padding"] = padding
	let opspec = OpSpec(
		type: "QuantizedMaxPool",
		name: (operationName ?? "Type"),
		input: [input, minInput, maxInput],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (output: op.output(at: 0), minOutput: op.output(at: 1), maxOutput: op.output(at: 2))
} 

///Returns the max of x and y (i.e. x > y ? x : y) element-wise.
/// * NOTE * : `Maximum` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
/// - Parameter x: 
/// - Parameter y: 
/// - Parameter mklX: 
/// - Parameter mklY: 
/// - Returns: 
///	z: 
///	mkl_z: 
public func mklMaximum(operationName: String? = nil, x: Output, y: Output, mklX: Output, mklY: Output) throws -> (z: Output, mklZ: Output) { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "_MklMaximum",
		name: (operationName ?? "Type"),
		input: [x, y, mklX, mklY],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (z: op.output(at: 0), mklZ: op.output(at: 1))
} 

///Computes square root of x element-wise.
///I.e., \\(y = \sqrt{x} = x// ^{1/2}\\).
/// - Parameter x: 
/// - Returns: 
///	y: 
public func sqrt(operationName: String? = nil, x: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Sqrt",
		name: (operationName ?? "Type"),
		input: [x],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Update ' * var' according to the adagrad scheme.
///accum += grad  *  grad
/// var -= lr  *  grad  *  (1 / sqrt(accum))
/// - Parameter `var`: Should be from a Variable().
/// - Parameter accum: Should be from a Variable().
/// - Parameter lr: Scaling factor. Must be a scalar.
/// - Parameter grad: The gradient.
/// - Parameter useLocking: If `True`, updating of the var and accum tensors will be protected
/// by a lock; otherwise the behavior is undefined, but may exhibit less
/// contention.
public func resourceApplyAdagrad(operationName: String? = nil, `var`: Output, accum: Output, lr: Output, grad: Output, useLocking: Bool) throws -> Operation { 
	var attrs = [String : Any]()
	attrs["use_locking"] = useLocking
	let opspec = OpSpec(
		type: "ResourceApplyAdagrad",
		name: (operationName ?? "Type"),
		input: [`var`, accum, lr, grad],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Says whether the targets are in the top `K` predictions.
///This outputs a `batch_size` bool array, an entry `out[i]` is `true` if the
/// prediction for the target class is among the top `k` predictions among
/// all predictions for example `i`. Note that the behavior of `InTopK` differs
/// from the `TopK` op in its handling of ties; if multiple classes have the
/// same prediction value and straddle the top-`k` boundary, all of those
/// classes are considered to be in the top `k`.
/// 
/// More formally, let
/// 
///   \\(predictions_i\\) be the predictions for all classes for example `i`,
///   \\(targets_i\\) be the target class for example `i`,
///   \\(out_i\\) be the output for example `i`,
/// 
/// $$out_i = predictions_{i, targets_i} \in TopKIncludingTies(predictions_i)$$
/// - Parameter predictions: A `batch_size` x `classes` tensor.
/// - Parameter targets: A `batch_size` vector of class ids.
/// - Parameter k: Number of top elements to look at for computing precision.
/// - Returns: 
///	precision: Computed precision at `k` as a `bool Tensor`.
public func inTopKV2(operationName: String? = nil, predictions: Output, targets: Output, k: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "InTopKV2",
		name: (operationName ?? "Type"),
		input: [predictions, targets, k],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Update ' * var' according to the adadelta scheme.
///accum = rho()  *  accum + (1 - rho())  *  grad.square();
/// update = (update_accum + epsilon).sqrt()  *  (accum + epsilon()).rsqrt()  *  grad;
/// update_accum = rho()  *  update_accum + (1 - rho())  *  update.square();
/// var -= update;
/// - Parameter `var`: Should be from a Variable().
/// - Parameter accum: Should be from a Variable().
/// - Parameter accumUpdate: Should be from a Variable().
/// - Parameter lr: Scaling factor. Must be a scalar.
/// - Parameter rho: Decay factor. Must be a scalar.
/// - Parameter epsilon: Constant factor. Must be a scalar.
/// - Parameter grad: The gradient.
/// - Parameter useLocking: If True, updating of the var, accum and update_accum tensors will be protected by
/// a lock; otherwise the behavior is undefined, but may exhibit less contention.
/// - Returns: 
///	out: Same as "var".
public func applyAdadelta(operationName: String? = nil, `var`: Output, accum: Output, accumUpdate: Output, lr: Output, rho: Output, epsilon: Output, grad: Output, useLocking: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["use_locking"] = useLocking
	let opspec = OpSpec(
		type: "ApplyAdadelta",
		name: (operationName ?? "Type"),
		input: [`var`, accum, accumUpdate, lr, rho, epsilon, grad],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes softmax cross entropy cost and gradients to backpropagate.
///Unlike `SoftmaxCrossEntropyWithLogits`, this operation does not accept
/// a matrix of label probabilities, but rather a single label per row
/// of features.  This label is considered to have probability 1.0 for the
/// given row.
/// 
/// Inputs are the logits, not probabilities.
/// - Parameter features: batch_size x num_classes matrix
/// - Parameter labels: batch_size vector with values in [0, num_classes).
/// This is the label for the given minibatch entry.
/// - Parameter tlabels: 
/// - Returns: 
///	loss: Per example loss (batch_size vector).
///	backprop: backpropagated gradients (batch_size x num_classes matrix).
public func sparseSoftmaxCrossEntropyWithLogits(operationName: String? = nil, features: Output, labels: Output, tlabels: Any.Type) throws -> (loss: Output, backprop: Output) { 
	var attrs = [String : Any]()
	attrs["Tlabels"] = tlabels
	let opspec = OpSpec(
		type: "SparseSoftmaxCrossEntropyWithLogits",
		name: (operationName ?? "Type"),
		input: [features, labels],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (loss: op.output(at: 0), backprop: op.output(at: 1))
} 

///Update ' * var' as FOBOS algorithm with fixed learning rate.
///prox_v = var - alpha  *  delta
/// var = sign(prox_v)/(1+alpha * l2)  *  max{|prox_v|-alpha * l1,0}
/// - Parameter `var`: Should be from a Variable().
/// - Parameter alpha: Scaling factor. Must be a scalar.
/// - Parameter l1: L1 regularization. Must be a scalar.
/// - Parameter l2: L2 regularization. Must be a scalar.
/// - Parameter delta: The change.
/// - Parameter useLocking: If True, the subtraction will be protected by a lock;
/// otherwise the behavior is undefined, but may exhibit less contention.
public func resourceApplyProximalGradientDescent(operationName: String? = nil, `var`: Output, alpha: Output, l1: Output, l2: Output, delta: Output, useLocking: Bool) throws -> Operation { 
	var attrs = [String : Any]()
	attrs["use_locking"] = useLocking
	let opspec = OpSpec(
		type: "ResourceApplyProximalGradientDescent",
		name: (operationName ?? "Type"),
		input: [`var`, alpha, l1, l2, delta],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Sparse update ' * var' as FOBOS algorithm with fixed learning rate.
///That is for rows we have grad for, we update var as follows:
/// prox_v = var - alpha  *  grad
/// var = sign(prox_v)/(1+alpha * l2)  *  max{|prox_v|-alpha * l1,0}
/// - Parameter `var`: Should be from a Variable().
/// - Parameter alpha: Scaling factor. Must be a scalar.
/// - Parameter l1: L1 regularization. Must be a scalar.
/// - Parameter l2: L2 regularization. Must be a scalar.
/// - Parameter grad: The gradient.
/// - Parameter indices: A vector of indices into the first dimension of var and accum.
/// - Parameter tindices: 
/// - Parameter useLocking: If True, the subtraction will be protected by a lock;
/// otherwise the behavior is undefined, but may exhibit less contention.
/// - Returns: 
///	out: Same as "var".
public func sparseApplyProximalGradientDescent(operationName: String? = nil, `var`: Output, alpha: Output, l1: Output, l2: Output, grad: Output, indices: Output, tindices: Any.Type, useLocking: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Tindices"] = tindices
	attrs["use_locking"] = useLocking
	let opspec = OpSpec(
		type: "SparseApplyProximalGradientDescent",
		name: (operationName ?? "Type"),
		input: [`var`, alpha, l1, l2, grad, indices],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Returns x - y element-wise.
/// * NOTE * : `Sub` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
/// - Parameter x: 
/// - Parameter y: 
/// - Parameter mklX: 
/// - Parameter mklY: 
/// - Returns: 
///	z: 
///	mkl_z: 
public func mklSub(operationName: String? = nil, x: Output, y: Output, mklX: Output, mklY: Output) throws -> (z: Output, mklZ: Output) { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "_MklSub",
		name: (operationName ?? "Type"),
		input: [x, y, mklX, mklY],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (z: op.output(at: 0), mklZ: op.output(at: 1))
} 

///Update ' * var' as FOBOS algorithm with fixed learning rate.
///prox_v = var - alpha  *  delta
/// var = sign(prox_v)/(1+alpha * l2)  *  max{|prox_v|-alpha * l1,0}
/// - Parameter `var`: Should be from a Variable().
/// - Parameter alpha: Scaling factor. Must be a scalar.
/// - Parameter l1: L1 regularization. Must be a scalar.
/// - Parameter l2: L2 regularization. Must be a scalar.
/// - Parameter delta: The change.
/// - Parameter useLocking: If True, the subtraction will be protected by a lock;
/// otherwise the behavior is undefined, but may exhibit less contention.
/// - Returns: 
///	out: Same as "var".
public func applyProximalGradientDescent(operationName: String? = nil, `var`: Output, alpha: Output, l1: Output, l2: Output, delta: Output, useLocking: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["use_locking"] = useLocking
	let opspec = OpSpec(
		type: "ApplyProximalGradientDescent",
		name: (operationName ?? "Type"),
		input: [`var`, alpha, l1, l2, delta],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Update ' * var' by subtracting 'alpha'  *  'delta' from it.
/// - Parameter `var`: Should be from a Variable().
/// - Parameter alpha: Scaling factor. Must be a scalar.
/// - Parameter delta: The change.
/// - Parameter useLocking: If `True`, the subtraction will be protected by a lock;
/// otherwise the behavior is undefined, but may exhibit less contention.
public func resourceApplyGradientDescent(operationName: String? = nil, `var`: Output, alpha: Output, delta: Output, useLocking: Bool) throws -> Operation { 
	var attrs = [String : Any]()
	attrs["use_locking"] = useLocking
	let opspec = OpSpec(
		type: "ResourceApplyGradientDescent",
		name: (operationName ?? "Type"),
		input: [`var`, alpha, delta],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Computes hyperbolic cosine of x element-wise.
/// - Parameter x: 
/// - Returns: 
///	y: 
public func cosh(operationName: String? = nil, x: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Cosh",
		name: (operationName ?? "Type"),
		input: [x],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Update ' * var' by subtracting 'alpha'  *  'delta' from it.
/// - Parameter `var`: Should be from a Variable().
/// - Parameter alpha: Scaling factor. Must be a scalar.
/// - Parameter delta: The change.
/// - Parameter useLocking: If `True`, the subtraction will be protected by a lock;
/// otherwise the behavior is undefined, but may exhibit less contention.
/// - Returns: 
///	out: Same as "var".
public func applyGradientDescent(operationName: String? = nil, `var`: Output, alpha: Output, delta: Output, useLocking: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["use_locking"] = useLocking
	let opspec = OpSpec(
		type: "ApplyGradientDescent",
		name: (operationName ?? "Type"),
		input: [`var`, alpha, delta],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///L2 Loss.
///Computes half the L2 norm of a tensor without the `sqrt`:
/// 
///     output = sum(t  *  *  2) / 2
/// - Parameter t: Typically 2-D, but may have any dimensions.
/// - Returns: 
///	output: 0-D.
public func l2Loss(operationName: String? = nil, t: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "L2Loss",
		name: (operationName ?? "Type"),
		input: [t],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes the maximum along segments of a tensor.
///Read @{$math_ops#segmentation$the section on segmentation} for an explanation of
/// segments.
/// 
/// Computes a tensor such that
/// \\(output_i = \max_j(data_j)\\) where `max` is over `j` such
/// that `segment_ids[j] == i`.
/// 
/// If the max is empty for a given segment ID `i`, `output[i] = 0`.
/// 
/// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
/// <img style="width:100%" src="https://www.tensorflow.org/images/SegmentMax.png" alt>
/// </div>
/// - Parameter data: 
/// - Parameter segmentIds: A 1-D tensor whose rank is equal to the rank of `data`'s
/// first dimension.  Values should be sorted and can be repeated.
/// - Parameter tindices: 
/// - Returns: 
///	output: Has same shape as data, except for dimension 0 which
/// has size `k`, the number of segments.
public func segmentMax(operationName: String? = nil, data: Output, segmentIds: Output, tindices: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Tindices"] = tindices
	let opspec = OpSpec(
		type: "SegmentMax",
		name: (operationName ?? "Type"),
		input: [data, segmentIds],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Increments 'ref' until it reaches 'limit'.
/// - Parameter ref: Should be from a scalar `Variable` node.
/// - Parameter limit: If incrementing ref would bring it above limit, instead generates an
/// 'OutOfRange' error.
/// - Returns: 
///	output: A copy of the input before increment. If nothing else modifies the
/// input, the values produced will all be distinct.
public func countUpTo(operationName: String? = nil, ref: Output, limit: UInt8) throws -> Output { 
	var attrs = [String : Any]()
	attrs["limit"] = limit
	let opspec = OpSpec(
		type: "CountUpTo",
		name: (operationName ?? "Type"),
		input: [ref],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Fake-quantize the 'inputs' tensor, type float to 'outputs' tensor of same type.
///Attributes `[min; max]` define the clamping range for the `inputs` data.
/// `inputs` values are quantized into the quantization range (`[0; 2// ^num_bits - 1]`
/// when `narrow_range` is false and `[1; 2// ^num_bits - 1]` when it is true) and
/// then de-quantized and output as floats in `[min; max]` interval.
/// `num_bits` is the bitwidth of the quantization; between 2 and 8, inclusive.
/// 
/// Quantization is called fake since the output is still in floating point.
/// - Parameter inputs: 
/// - Parameter min: 
/// - Parameter max: 
/// - Parameter numBits: 
/// - Parameter narrowRange: 
/// - Returns: 
///	outputs: 
public func fakeQuantWithMinMaxArgs(operationName: String? = nil, inputs: Output, min: Float, max: Float, numBits: UInt8, narrowRange: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["min"] = min
	attrs["max"] = max
	attrs["num_bits"] = numBits
	attrs["narrow_range"] = narrowRange
	let opspec = OpSpec(
		type: "FakeQuantWithMinMaxArgs",
		name: (operationName ?? "Type"),
		input: [inputs],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Applies sparse addition between `updates` and individual values or slices
///within a given variable according to `indices`.
/// 
/// `ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.
/// 
/// `indices` must be integer tensor, containing indices into `ref`.
/// It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.
/// 
/// The innermost dimension of `indices` (with length `K`) corresponds to
/// indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
/// dimension of `ref`.
/// 
/// `updates` is `Tensor` of rank `Q-1+P-K` with shape:
/// 
/// ```
/// [d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]].
/// ```
/// 
/// For example, say we want to add 4 scattered elements to a rank-1 tensor to 8
/// elements. In Python, that addition would look like this:
/// 
///     ref = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8])
///     indices = tf.constant([[4], [3], [1], [7]])
///     updates = tf.constant([9, 10, 11, 12])
///     add = tf.scatter_nd_add(ref, indices, updates)
///     with tf.Session() as sess:
///       print sess.run(add)
/// 
/// The resulting update to ref would look like this:
/// 
///     [1, 13, 3, 14, 14, 6, 7, 20]
/// 
/// See @{tf.scatter_nd} for more details about how to make updates to
/// slices.
/// - Parameter ref: A mutable Tensor. Should be from a Variable node.
/// - Parameter indices: A Tensor. Must be one of the following types: int32, int64.
/// A tensor of indices into ref.
/// - Parameter updates: A Tensor. Must have the same type as ref. A tensor of updated values
/// to add to ref.
/// - Parameter tindices: 
/// - Parameter useLocking: An optional bool. Defaults to True. If True, the assignment will
/// be protected by a lock; otherwise the behavior is undefined,
/// but may exhibit less contention.
/// - Returns: 
///	output_ref: Same as ref. Returned as a convenience for operations that want
/// to use the updated values after the update is done.
public func scatterNdAdd(operationName: String? = nil, ref: Output, indices: Output, updates: Output, tindices: Any.Type, useLocking: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Tindices"] = tindices
	attrs["use_locking"] = useLocking
	let opspec = OpSpec(
		type: "ScatterNdAdd",
		name: (operationName ?? "Type"),
		input: [ref, indices, updates],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Applies sparse `updates` to individual values or slices within a given
///variable according to `indices`.
/// 
/// `ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.
/// 
/// `indices` must be integer tensor, containing indices into `ref`.
/// It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.
/// 
/// The innermost dimension of `indices` (with length `K`) corresponds to
/// indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
/// dimension of `ref`.
/// 
/// `updates` is `Tensor` of rank `Q-1+P-K` with shape:
/// 
/// ```
/// [d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]].
/// ```
/// 
/// For example, say we want to update 4 scattered elements to a rank-1 tensor to
/// 8 elements. In Python, that update would look like this:
/// 
/// ```python
///     ref = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8])
///     indices = tf.constant([[4], [3], [1] ,[7]])
///     updates = tf.constant([9, 10, 11, 12])
///     update = tf.scatter_nd_update(ref, indices, updates)
///     with tf.Session() as sess:
///       print sess.run(update)
/// ```
/// 
/// The resulting update to ref would look like this:
/// 
///     [1, 11, 3, 10, 9, 6, 7, 12]
/// 
/// See @{tf.scatter_nd} for more details about how to make updates to
/// slices.
/// - Parameter ref: A mutable Tensor. Should be from a Variable node.
/// - Parameter indices: A Tensor. Must be one of the following types: int32, int64.
/// A tensor of indices into ref.
/// - Parameter updates: A Tensor. Must have the same type as ref. A tensor of updated
/// values to add to ref.
/// - Parameter tindices: 
/// - Parameter useLocking: An optional bool. Defaults to True. If True, the assignment will
/// be protected by a lock; otherwise the behavior is undefined,
/// but may exhibit less contention.
/// - Returns: 
///	output_ref: Same as ref. Returned as a convenience for operations that want to
/// use the updated values after the update is done.
public func scatterNdUpdate(operationName: String? = nil, ref: Output, indices: Output, updates: Output, tindices: Any.Type, useLocking: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Tindices"] = tindices
	attrs["use_locking"] = useLocking
	let opspec = OpSpec(
		type: "ScatterNdUpdate",
		name: (operationName ?? "Type"),
		input: [ref, indices, updates],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Multiplies sparse updates into a variable reference.
///This operation computes
/// 
/// ```python
///     # Scalar indices
///     ref[indices, ...]  * = updates[...]
/// 
///     # Vector indices (for each i)
///     ref[indices[i], ...]  * = updates[i, ...]
/// 
///     # High rank indices (for each i, ..., j)
///     ref[indices[i, ..., j], ...]  * = updates[i, ..., j, ...]
/// ```
/// 
/// This operation outputs `ref` after the update is done.
/// This makes it easier to chain operations that need to use the reset value.
/// 
/// Duplicate entries are handled correctly: if multiple `indices` reference
/// the same location, their contributions multiply.
/// 
/// Requires `updates.shape = indices.shape + ref.shape[1:]`.
/// - Parameter ref: Should be from a `Variable` node.
/// - Parameter indices: A tensor of indices into the first dimension of `ref`.
/// - Parameter updates: A tensor of updated values to multiply to `ref`.
/// - Parameter tindices: 
/// - Parameter useLocking: If True, the operation will be protected by a lock;
/// otherwise the behavior is undefined, but may exhibit less contention.
/// - Returns: 
///	output_ref: = Same as `ref`.  Returned as a convenience for operations that want
/// to use the updated values after the update is done.
public func scatterMul(operationName: String? = nil, ref: Output, indices: Output, updates: Output, tindices: Any.Type, useLocking: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Tindices"] = tindices
	attrs["use_locking"] = useLocking
	let opspec = OpSpec(
		type: "ScatterMul",
		name: (operationName ?? "Type"),
		input: [ref, indices, updates],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Subtracts sparse updates to a variable reference.
///```python
///     # Scalar indices
///     ref[indices, ...] -= updates[...]
/// 
///     # Vector indices (for each i)
///     ref[indices[i], ...] -= updates[i, ...]
/// 
///     # High rank indices (for each i, ..., j)
///     ref[indices[i, ..., j], ...] -= updates[i, ..., j, ...]
/// ```
/// 
/// This operation outputs `ref` after the update is done.
/// This makes it easier to chain operations that need to use the reset value.
/// 
/// Duplicate entries are handled correctly: if multiple `indices` reference
/// the same location, their (negated) contributions add.
/// 
/// Requires `updates.shape = indices.shape + ref.shape[1:]`.
/// 
/// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
/// <img style="width:100%" src="https://www.tensorflow.org/images/ScatterSub.png" alt>
/// </div>
/// - Parameter ref: Should be from a `Variable` node.
/// - Parameter indices: A tensor of indices into the first dimension of `ref`.
/// - Parameter updates: A tensor of updated values to subtract from `ref`.
/// - Parameter tindices: 
/// - Parameter useLocking: If True, the subtraction will be protected by a lock;
/// otherwise the behavior is undefined, but may exhibit less contention.
/// - Returns: 
///	output_ref: = Same as `ref`.  Returned as a convenience for operations that want
/// to use the updated values after the update is done.
public func scatterSub(operationName: String? = nil, ref: Output, indices: Output, updates: Output, tindices: Any.Type, useLocking: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Tindices"] = tindices
	attrs["use_locking"] = useLocking
	let opspec = OpSpec(
		type: "ScatterSub",
		name: (operationName ?? "Type"),
		input: [ref, indices, updates],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes the mean of elements across dimensions of a tensor.
///Reduces `input` along the dimensions given in `reduction_indices`. Unless
/// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
/// `reduction_indices`. If `keep_dims` is true, the reduced dimensions are
/// retained with length 1.
/// - Parameter input: The tensor to reduce.
/// - Parameter reductionIndices: The dimensions to reduce. Must be in the range
/// `[-rank(input), rank(input))`.
/// - Parameter keepDims: If true, retain reduced dimensions with length 1.
/// - Parameter tidx: 
/// - Returns: 
///	output: The reduced tensor.
public func mean(operationName: String? = nil, input: Output, reductionIndices: Output, keepDims: Bool, tidx: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["keep_dims"] = keepDims
	attrs["Tidx"] = tidx
	let opspec = OpSpec(
		type: "Mean",
		name: (operationName ?? "Type"),
		input: [input, reductionIndices],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Adds sparse updates to a variable reference.
///This operation computes
/// 
///     # Scalar indices
///     ref[indices, ...] += updates[...]
/// 
///     # Vector indices (for each i)
///     ref[indices[i], ...] += updates[i, ...]
/// 
///     # High rank indices (for each i, ..., j)
///     ref[indices[i, ..., j], ...] += updates[i, ..., j, ...]
/// 
/// This operation outputs `ref` after the update is done.
/// This makes it easier to chain operations that need to use the reset value.
/// 
/// Duplicate entries are handled correctly: if multiple `indices` reference
/// the same location, their contributions add.
/// 
/// Requires `updates.shape = indices.shape + ref.shape[1:]`.
/// 
/// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
/// <img style="width:100%" src="https://www.tensorflow.org/images/ScatterAdd.png" alt>
/// </div>
/// - Parameter ref: Should be from a `Variable` node.
/// - Parameter indices: A tensor of indices into the first dimension of `ref`.
/// - Parameter updates: A tensor of updated values to add to `ref`.
/// - Parameter tindices: 
/// - Parameter useLocking: If True, the addition will be protected by a lock;
/// otherwise the behavior is undefined, but may exhibit less contention.
/// - Returns: 
///	output_ref: = Same as `ref`.  Returned as a convenience for operations that want
/// to use the updated values after the update is done.
public func scatterAdd(operationName: String? = nil, ref: Output, indices: Output, updates: Output, tindices: Any.Type, useLocking: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Tindices"] = tindices
	attrs["use_locking"] = useLocking
	let opspec = OpSpec(
		type: "ScatterAdd",
		name: (operationName ?? "Type"),
		input: [ref, indices, updates],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Applies sparse updates to a variable reference.
///This operation computes
/// 
/// ```python
///     # Scalar indices
///     ref[indices, ...] = updates[...]
/// 
///     # Vector indices (for each i)
///     ref[indices[i], ...] = updates[i, ...]
/// 
///     # High rank indices (for each i, ..., j)
///     ref[indices[i, ..., j], ...] = updates[i, ..., j, ...]
/// ```
/// 
/// This operation outputs `ref` after the update is done.
/// This makes it easier to chain operations that need to use the reset value.
/// 
/// If values in `ref` is to be updated more than once, because there are
/// duplicate entries in `indices`, the order at which the updates happen
/// for each value is undefined.
/// 
/// Requires `updates.shape = indices.shape + ref.shape[1:]`.
/// 
/// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
/// <img style="width:100%" src="https://www.tensorflow.org/images/ScatterUpdate.png" alt>
/// </div>
/// - Parameter ref: Should be from a `Variable` node.
/// - Parameter indices: A tensor of indices into the first dimension of `ref`.
/// - Parameter updates: A tensor of updated values to store in `ref`.
/// - Parameter tindices: 
/// - Parameter useLocking: If True, the assignment will be protected by a lock;
/// otherwise the behavior is undefined, but may exhibit less contention.
/// - Returns: 
///	output_ref: = Same as `ref`.  Returned as a convenience for operations that want
/// to use the updated values after the update is done.
public func scatterUpdate(operationName: String? = nil, ref: Output, indices: Output, updates: Output, tindices: Any.Type, useLocking: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Tindices"] = tindices
	attrs["use_locking"] = useLocking
	let opspec = OpSpec(
		type: "ScatterUpdate",
		name: (operationName ?? "Type"),
		input: [ref, indices, updates],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Update 'ref' by subtracting 'value' from it.
///This operation outputs "ref" after the update is done.
/// This makes it easier to chain operations that need to use the reset value.
/// - Parameter ref: Should be from a `Variable` node.
/// - Parameter value: The value to be subtracted to the variable.
/// - Parameter useLocking: If True, the subtraction will be protected by a lock;
/// otherwise the behavior is undefined, but may exhibit less contention.
/// - Returns: 
///	output_ref: = Same as "ref".  Returned as a convenience for operations that want
/// to use the new value after the variable has been updated.
public func assignSub(operationName: String? = nil, ref: Output, value: Output, useLocking: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["use_locking"] = useLocking
	let opspec = OpSpec(
		type: "AssignSub",
		name: (operationName ?? "Type"),
		input: [ref, value],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Update 'ref' by adding 'value' to it.
///This operation outputs "ref" after the update is done.
/// This makes it easier to chain operations that need to use the reset value.
/// - Parameter ref: Should be from a `Variable` node.
/// - Parameter value: The value to be added to the variable.
/// - Parameter useLocking: If True, the addition will be protected by a lock;
/// otherwise the behavior is undefined, but may exhibit less contention.
/// - Returns: 
///	output_ref: = Same as "ref".  Returned as a convenience for operations that want
/// to use the new value after the variable has been updated.
public func assignAdd(operationName: String? = nil, ref: Output, value: Output, useLocking: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["use_locking"] = useLocking
	let opspec = OpSpec(
		type: "AssignAdd",
		name: (operationName ?? "Type"),
		input: [ref, value],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Compute the regularized incomplete beta integral \\(I_x(a, b)\\).
///The regularized incomplete beta integral is defined as:
/// 
/// 
/// \\(I_x(a, b) = \frac{B(x; a, b)}{B(a, b)}\\)
/// 
/// where
/// 
/// 
/// \\(B(x; a, b) = \int_0// ^x t// ^{a-1} (1 - t)// ^{b-1} dt\\)
/// 
/// 
/// is the incomplete beta function and \\(B(a, b)\\) is the  * complete * 
/// beta function.
/// - Parameter a: 
/// - Parameter b: 
/// - Parameter x: 
/// - Returns: 
///	z: 
public func betainc(operationName: String? = nil, a: Output, b: Output, x: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Betainc",
		name: (operationName ?? "Type"),
		input: [a, b, x],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Update 'ref' by assigning 'value' to it.
///This operation outputs "ref" after the assignment is done.
/// This makes it easier to chain operations that need to use the reset value.
/// - Parameter ref: Should be from a `Variable` node. May be uninitialized.
/// - Parameter value: The value to be assigned to the variable.
/// - Parameter validateShape: If true, the operation will validate that the shape
/// of 'value' matches the shape of the Tensor being assigned to.  If false,
/// 'ref' will take on the shape of 'value'.
/// - Parameter useLocking: If True, the assignment will be protected by a lock;
/// otherwise the behavior is undefined, but may exhibit less contention.
/// - Returns: 
///	output_ref: = Same as "ref".  Returned as a convenience for operations that want
/// to use the new value after the variable has been reset.
public func assign(operationName: String? = nil, ref: Output, value: Output, validateShape: Bool, useLocking: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["validate_shape"] = validateShape
	attrs["use_locking"] = useLocking
	let opspec = OpSpec(
		type: "Assign",
		name: (operationName ?? "Type"),
		input: [ref, value],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Checks whether a tensor has been initialized.
///Outputs boolean scalar indicating whether the tensor has been initialized.
/// - Parameter ref: Should be from a `Variable` node. May be uninitialized.
/// - Parameter dtype: The type of elements in the variable tensor.
/// - Returns: 
///	is_initialized: 
public func isVariableInitialized(operationName: String? = nil, ref: Output, dtype: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["dtype"] = dtype
	let opspec = OpSpec(
		type: "IsVariableInitialized",
		name: (operationName ?? "Type"),
		input: [ref],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Use VariableV2 instead.
/// - Parameter shape: 
/// - Parameter dtype: 
/// - Parameter container: 
/// - Parameter sharedName: 
/// - Returns: 
///	ref: 
public func variable(operationName: String? = nil, shape: Shape, dtype: Any.Type, container: String, sharedName: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["shape"] = shape
	attrs["dtype"] = dtype
	attrs["container"] = container
	attrs["shared_name"] = sharedName
	let opspec = OpSpec(
		type: "Variable",
		name: (operationName ?? "Type"),
		input: [],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Updates input `value` at `loc` with `update`.
///If you use this function you will almost certainly want to add
/// a control dependency as done in the implementation of parallel_stack to
/// avoid race conditions.
/// - Parameter value: A `Tensor` object that will be updated in-place.
/// - Parameter update: A `Tensor` of rank one less than `value` if `loc` is a scalar,
/// otherwise of rank equal to `value` that contains the new values
/// for `value`.
/// - Parameter loc: A scalar indicating the index of the first dimension such that
/// value[loc, :] is updated.
/// - Returns: 
///	output: `value` that has been updated accordingly.
public func parallelConcatUpdate(operationName: String? = nil, value: Output, update: Output, loc: UInt8) throws -> Output { 
	var attrs = [String : Any]()
	attrs["loc"] = loc
	let opspec = OpSpec(
		type: "_ParallelConcatUpdate",
		name: (operationName ?? "Type"),
		input: [value, update],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Holds state in the form of a tensor that persists across steps.
///Outputs a ref to the tensor state so it may be read or modified.
/// TODO(zhifengc/mrry): Adds a pointer to a more detail document
/// about sharing states in tensorflow.
/// - Parameter shape: The shape of the variable tensor.
/// - Parameter dtype: The type of elements in the variable tensor.
/// - Parameter container: If non-empty, this variable is placed in the given container.
/// Otherwise, a default container is used.
/// - Parameter sharedName: If non-empty, this variable is named in the given bucket
/// with this shared_name. Otherwise, the node name is used instead.
/// - Returns: 
///	ref: A reference to the variable tensor.
public func variableV2(operationName: String? = nil, shape: Shape, dtype: Any.Type, container: String, sharedName: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["shape"] = shape
	attrs["dtype"] = dtype
	attrs["container"] = container
	attrs["shared_name"] = sharedName
	let opspec = OpSpec(
		type: "VariableV2",
		name: (operationName ?? "Type"),
		input: [],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Writes a `Summary` protocol buffer with audio.
///The summary has up to `max_outputs` summary values containing audio. The
/// audio is built from `tensor` which must be 3-D with shape `[batch_size,
/// frames, channels]` or 2-D with shape `[batch_size, frames]`. The values are
/// assumed to be in the range of `[-1.0, 1.0]` with a sample rate of `sample_rate`.
/// 
/// The `tag` argument is a scalar `Tensor` of type `string`.  It is used to
/// build the `tag` of the summary values:
/// 
///  *   If `max_outputs` is 1, the summary value tag is ' * tag * /audio'.
///  *   If `max_outputs` is greater than 1, the summary value tags are
///    generated sequentially as ' * tag * /audio/0', ' * tag * /audio/1', etc.
/// - Parameter writer: A handle to a summary writer.
/// - Parameter globalStep: The step to write the summary for.
/// - Parameter tag: Scalar. Used to build the `tag` attribute of the summary values.
/// - Parameter tensor: 2-D of shape `[batch_size, frames]`.
/// - Parameter sampleRate: The sample rate of the signal in hertz.
/// - Parameter maxOutputs: Max number of batch elements to generate audio for.
public func writeAudioSummary(operationName: String? = nil, writer: Output, globalStep: Output, tag: Output, tensor: Output, sampleRate: Output, maxOutputs: UInt8) throws -> Operation { 
	var attrs = [String : Any]()
	attrs["max_outputs"] = maxOutputs
	let opspec = OpSpec(
		type: "WriteAudioSummary",
		name: (operationName ?? "Type"),
		input: [writer, globalStep, tag, tensor, sampleRate],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Copy a tensor setting everything outside a central band in each innermost matrix
///to zero.
/// 
/// The `band` part is computed as follows:
/// Assume `input` has `k` dimensions `[I, J, K, ..., M, N]`, then the output is a
/// tensor with the same shape where
/// 
/// `band[i, j, k, ..., m, n] = in_band(m, n)  *  input[i, j, k, ..., m, n]`.
/// 
/// The indicator function
/// 
/// `in_band(m, n) = (num_lower < 0 || (m-n) <= num_lower)) &&
///                  (num_upper < 0 || (n-m) <= num_upper)`.
/// 
/// For example:
/// 
/// ```
/// # if 'input' is [[ 0,  1,  2, 3]
///                  [-1,  0,  1, 2]
///                  [-2, -1,  0, 1]
///                  [-3, -2, -1, 0]],
/// 
/// tf.matrix_band_part(input, 1, -1) ==> [[ 0,  1,  2, 3]
///                                        [-1,  0,  1, 2]
///                                        [ 0, -1,  0, 1]
///                                        [ 0,  0, -1, 0]],
/// 
/// tf.matrix_band_part(input, 2, 1) ==> [[ 0,  1,  0, 0]
///                                       [-1,  0,  1, 0]
///                                       [-2, -1,  0, 1]
///                                       [ 0, -2, -1, 0]]
/// ```
/// 
/// Useful special cases:
/// 
/// ```
///  tf.matrix_band_part(input, 0, -1) ==> Upper triangular part.
///  tf.matrix_band_part(input, -1, 0) ==> Lower triangular part.
///  tf.matrix_band_part(input, 0, 0) ==> Diagonal.
/// ```
/// - Parameter input: Rank `k` tensor.
/// - Parameter numLower: 0-D tensor. Number of subdiagonals to keep. If negative, keep entire
/// lower triangle.
/// - Parameter numUpper: 0-D tensor. Number of superdiagonals to keep. If negative, keep
/// entire upper triangle.
/// - Returns: 
///	band: Rank `k` tensor of the same shape as input. The extracted banded tensor.
public func matrixBandPart(operationName: String? = nil, input: Output, numLower: Output, numUpper: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "MatrixBandPart",
		name: (operationName ?? "Type"),
		input: [input, numLower, numUpper],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Writes a `Summary` protocol buffer with images.
///The summary has up to `max_images` summary values containing images. The
/// images are built from `tensor` which must be 4-D with shape `[batch_size,
/// height, width, channels]` and where `channels` can be:
/// 
///  *   1: `tensor` is interpreted as Grayscale.
///  *   3: `tensor` is interpreted as RGB.
///  *   4: `tensor` is interpreted as RGBA.
/// 
/// The images have the same number of channels as the input tensor. For float
/// input, the values are normalized one image at a time to fit in the range
/// `[0, 255]`.  `uint8` values are unchanged.  The op uses two different
/// normalization algorithms:
/// 
///  *   If the input values are all positive, they are rescaled so the largest one
///    is 255.
/// 
///  *   If any input value is negative, the values are shifted so input value 0.0
///    is at 127.  They are then rescaled so that either the smallest value is 0,
///    or the largest one is 255.
/// 
/// The `tag` argument is a scalar `Tensor` of type `string`.  It is used to
/// build the `tag` of the summary values:
/// 
///  *   If `max_images` is 1, the summary value tag is ' * tag * /image'.
///  *   If `max_images` is greater than 1, the summary value tags are
///    generated sequentially as ' * tag * /image/0', ' * tag * /image/1', etc.
/// 
/// The `bad_color` argument is the color to use in the generated images for
/// non-finite input values.  It is a `unit8` 1-D tensor of length `channels`.
/// Each element must be in the range `[0, 255]` (It represents the value of a
/// pixel in the output image).  Non-finite values in the input tensor are
/// replaced by this tensor in the output image.  The default value is the color
/// red.
/// - Parameter writer: A handle to a summary writer.
/// - Parameter globalStep: The step to write the summary for.
/// - Parameter tag: Scalar. Used to build the `tag` attribute of the summary values.
/// - Parameter tensor: 4-D of shape `[batch_size, height, width, channels]` where
/// `channels` is 1, 3, or 4.
/// - Parameter badColor: Color to use for pixels with non-finite values.
/// - Parameter maxImages: Max number of batch elements to generate images for.
public func writeImageSummary(operationName: String? = nil, writer: Output, globalStep: Output, tag: Output, tensor: Output, badColor: Output, maxImages: UInt8) throws -> Operation { 
	var attrs = [String : Any]()
	attrs["max_images"] = maxImages
	let opspec = OpSpec(
		type: "WriteImageSummary",
		name: (operationName ?? "Type"),
		input: [writer, globalStep, tag, tensor, badColor],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Update ' * var' according to the Ftrl-proximal scheme.
///accum_new = accum + grad  *  grad
/// linear += grad - (accum_new// ^(-lr_power) - accum// ^(-lr_power)) / lr  *  var
/// quadratic = 1.0 / (accum_new// ^(lr_power)  *  lr) + 2  *  l2
/// var = (sign(linear)  *  l1 - linear) / quadratic if |linear| > l1 else 0.0
/// accum = accum_new
/// - Parameter `var`: Should be from a Variable().
/// - Parameter accum: Should be from a Variable().
/// - Parameter linear: Should be from a Variable().
/// - Parameter grad: The gradient.
/// - Parameter lr: Scaling factor. Must be a scalar.
/// - Parameter l1: L1 regulariation. Must be a scalar.
/// - Parameter l2: L2 regulariation. Must be a scalar.
/// - Parameter lrPower: Scaling factor. Must be a scalar.
/// - Parameter useLocking: If `True`, updating of the var and accum tensors will be protected
/// by a lock; otherwise the behavior is undefined, but may exhibit less
/// contention.
public func resourceApplyFtrl(operationName: String? = nil, `var`: Output, accum: Output, linear: Output, grad: Output, lr: Output, l1: Output, l2: Output, lrPower: Output, useLocking: Bool) throws -> Operation { 
	var attrs = [String : Any]()
	attrs["use_locking"] = useLocking
	let opspec = OpSpec(
		type: "ResourceApplyFtrl",
		name: (operationName ?? "Type"),
		input: [`var`, accum, linear, grad, lr, l1, l2, lrPower],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Writes a `Summary` protocol buffer with a histogram.
///The generated
/// [`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
/// has one summary value containing a histogram for `values`.
/// 
/// This op reports an `InvalidArgument` error if any value is not finite.
/// - Parameter writer: A handle to a summary writer.
/// - Parameter globalStep: The step to write the summary for.
/// - Parameter tag: Scalar.  Tag to use for the `Summary.Value`.
/// - Parameter values: Any shape. Values to use to build the histogram.
public func writeHistogramSummary(operationName: String? = nil, writer: Output, globalStep: Output, tag: Output, values: Output) throws -> Operation { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "WriteHistogramSummary",
		name: (operationName ?? "Type"),
		input: [writer, globalStep, tag, values],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///var: Should be from a Variable().
/// - Parameter `var`: 
/// - Parameter accum: Should be from a Variable().
/// - Parameter accumUpdate: : Should be from a Variable().
/// - Parameter lr: Learning rate. Must be a scalar.
/// - Parameter rho: Decay factor. Must be a scalar.
/// - Parameter epsilon: Constant factor. Must be a scalar.
/// - Parameter grad: The gradient.
/// - Parameter indices: A vector of indices into the first dimension of var and accum.
/// - Parameter tindices: 
/// - Parameter useLocking: If True, updating of the var and accum tensors will be protected by
/// a lock; otherwise the behavior is undefined, but may exhibit less contention.
public func resourceSparseApplyAdadelta(operationName: String? = nil, `var`: Output, accum: Output, accumUpdate: Output, lr: Output, rho: Output, epsilon: Output, grad: Output, indices: Output, tindices: Any.Type, useLocking: Bool) throws -> Operation { 
	var attrs = [String : Any]()
	attrs["Tindices"] = tindices
	attrs["use_locking"] = useLocking
	let opspec = OpSpec(
		type: "ResourceSparseApplyAdadelta",
		name: (operationName ?? "Type"),
		input: [`var`, accum, accumUpdate, lr, rho, epsilon, grad, indices],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Outputs a `Summary` protocol buffer with a tensor.
/// - Parameter writer: A handle to a summary writer.
/// - Parameter globalStep: The step to write the summary for.
/// - Parameter tensor: A tensor to serialize.
/// - Parameter tag: The summary's tag.
/// - Parameter summaryMetadata: Serialized SummaryMetadata protocol buffer containing
/// plugin-related metadata for this summary.
public func writeSummary(operationName: String? = nil, writer: Output, globalStep: Output, tensor: Output, tag: Output, summaryMetadata: Output) throws -> Operation { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "WriteSummary",
		name: (operationName ?? "Type"),
		input: [writer, globalStep, tensor, tag, summaryMetadata],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Flushes the writer's unwritten events.
/// - Parameter writer: A handle to the summary writer resource.
public func flushSummaryWriter(operationName: String? = nil, writer: Output) throws -> Operation { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "FlushSummaryWriter",
		name: (operationName ?? "Type"),
		input: [writer],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Update ' * var' according to the RMSProp algorithm.
///Note that in dense implementation of this algorithm, ms and mom will
/// update even if the grad is zero, but in this sparse implementation, ms
/// and mom will not update in iterations during which the grad is zero.
/// 
/// mean_square = decay  *  mean_square + (1-decay)  *  gradient  *  *  2
/// Delta = learning_rate  *  gradient / sqrt(mean_square + epsilon)
/// 
/// ms <- rho  *  ms_{t-1} + (1-rho)  *  grad  *  grad
/// mom <- momentum  *  mom_{t-1} + lr  *  grad / sqrt(ms + epsilon)
/// var <- var - mom
/// - Parameter `var`: Should be from a Variable().
/// - Parameter ms: Should be from a Variable().
/// - Parameter mom: Should be from a Variable().
/// - Parameter lr: Scaling factor. Must be a scalar.
/// - Parameter rho: Decay rate. Must be a scalar.
/// - Parameter momentum: 
/// - Parameter epsilon: Ridge term. Must be a scalar.
/// - Parameter grad: The gradient.
/// - Parameter indices: A vector of indices into the first dimension of var, ms and mom.
/// - Parameter tindices: 
/// - Parameter useLocking: If `True`, updating of the var, ms, and mom tensors is protected
/// by a lock; otherwise the behavior is undefined, but may exhibit less
/// contention.
/// - Returns: 
///	out: Same as "var".
public func sparseApplyRMSProp(operationName: String? = nil, `var`: Output, ms: Output, mom: Output, lr: Output, rho: Output, momentum: Output, epsilon: Output, grad: Output, indices: Output, tindices: Any.Type, useLocking: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Tindices"] = tindices
	attrs["use_locking"] = useLocking
	let opspec = OpSpec(
		type: "SparseApplyRMSProp",
		name: (operationName ?? "Type"),
		input: [`var`, ms, mom, lr, rho, momentum, epsilon, grad, indices],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Returns a handle to be used to access a summary writer.
///The summary writer is an in-graph resource which can be used by ops to write
/// summaries to event files.
/// - Parameter sharedName: 
/// - Parameter container: 
/// - Returns: 
///	writer: the summary writer resource. Scalar handle.
public func summaryWriter(operationName: String? = nil, sharedName: String, container: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["shared_name"] = sharedName
	attrs["container"] = container
	let opspec = OpSpec(
		type: "SummaryWriter",
		name: (operationName ?? "Type"),
		input: [],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes a 2D convolution given quantized 4D input and filter tensors.
///The inputs are quantized tensors where the lowest value represents the real
/// number of the associated minimum, and the highest represents the maximum.
/// This means that you can only interpret the quantized output in the same way, by
/// taking the returned minimum and maximum values into account.
/// - Parameter input: 
/// - Parameter filter: filter's input_depth dimension must match input's depth dimensions.
/// - Parameter minInput: The float value that the lowest quantized input value represents.
/// - Parameter maxInput: The float value that the highest quantized input value represents.
/// - Parameter minFilter: The float value that the lowest quantized filter value represents.
/// - Parameter maxFilter: The float value that the highest quantized filter value represents.
/// - Parameter tinput: 
/// - Parameter tfilter: 
/// - Parameter outType: 
/// - Parameter strides: The stride of the sliding window for each dimension of the input
/// tensor.
/// - Parameter padding: The type of padding algorithm to use.
/// - Returns: 
///	output: 
///	min_output: The float value that the lowest quantized output value represents.
///	max_output: The float value that the highest quantized output value represents.
public func quantizedConv2D(operationName: String? = nil, input: Output, filter: Output, minInput: Output, maxInput: Output, minFilter: Output, maxFilter: Output, tinput: Any.Type, tfilter: Any.Type, outType: Any.Type, strides: [Int64], padding: String) throws -> (output: Output, minOutput: Output, maxOutput: Output) { 
	var attrs = [String : Any]()
	attrs["Tinput"] = tinput
	attrs["Tfilter"] = tfilter
	attrs["out_type"] = outType
	attrs["strides"] = strides
	attrs["padding"] = padding
	let opspec = OpSpec(
		type: "QuantizedConv2D",
		name: (operationName ?? "Type"),
		input: [input, filter, minInput, maxInput, minFilter, maxFilter],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (output: op.output(at: 0), minOutput: op.output(at: 1), maxOutput: op.output(at: 2))
} 

///Computes rectified linear 6 gradients for a Relu6 operation.
/// - Parameter gradients: The backpropagated gradients to the corresponding Relu6 operation.
/// - Parameter features: The features passed as input to the corresponding Relu6 operation.
/// - Returns: 
///	backprops: The gradients:
/// `gradients  *  (features > 0)  *  (features < 6)`.
public func relu6Grad(operationName: String? = nil, gradients: Output, features: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Relu6Grad",
		name: (operationName ?? "Type"),
		input: [gradients, features],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes gradients of the average pooling function.
/// - Parameter origInputShape: 1-D.  Shape of the original input to `avg_pool`.
/// - Parameter grad: 4-D with shape `[batch, height, width, channels]`.  Gradients w.r.t.
/// the output of `avg_pool`.
/// - Parameter ksize: The size of the sliding window for each dimension of the input.
/// - Parameter strides: The stride of the sliding window for each dimension of the input.
/// - Parameter padding: The type of padding algorithm to use.
/// - Parameter dataFormat: Specify the data format of the input and output data. With the
/// default format "NHWC", the data is stored in the order of:
///     [batch, in_height, in_width, in_channels].
/// Alternatively, the format could be "NCHW", the data storage order of:
///     [batch, in_channels, in_height, in_width].
/// - Returns: 
///	output: 4-D.  Gradients w.r.t. the input of `avg_pool`.
public func avgPoolGrad(operationName: String? = nil, origInputShape: Output, grad: Output, ksize: [Int64], strides: [Int64], padding: String, dataFormat: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["ksize"] = ksize
	attrs["strides"] = strides
	attrs["padding"] = padding
	attrs["data_format"] = dataFormat
	let opspec = OpSpec(
		type: "AvgPoolGrad",
		name: (operationName ?? "Type"),
		input: [origInputShape, grad],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Returns the rank of a tensor.
///This operation returns an integer representing the rank of `input`.
/// 
/// For example:
/// 
/// ```
/// # 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
/// # shape of tensor 't' is [2, 2, 3]
/// rank(t) ==> 3
/// ```
/// 
///  *  * Note *  * : The rank of a tensor is not the same as the rank of a matrix. The rank
/// of a tensor is the number of indices required to uniquely select each element
/// of the tensor. Rank is also known as "order", "degree", or "ndims."
/// - Parameter input: 
/// - Returns: 
///	output: 
public func rank(operationName: String? = nil, input: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Rank",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Split elements of `input` based on `delimiter` into a `SparseTensor`.
///Let N be the size of source (typically N will be the batch size). Split each
/// element of `input` based on `delimiter` and return a `SparseTensor`
/// containing the splitted tokens. Empty tokens are ignored.
/// 
/// `delimiter` can be empty, or a string of split characters. If `delimiter` is an
///  empty string, each element of `input` is split into individual single-byte
///  character strings, including splitting of UTF-8 multibyte sequences. Otherwise
///  every character of `delimiter` is a potential split point.
/// 
/// For example:
///   N = 2, input[0] is 'hello world' and input[1] is 'a b c', then the output
///   will be
/// 
///   indices = [0, 0;
///              0, 1;
///              1, 0;
///              1, 1;
///              1, 2]
///   shape = [2, 3]
///   values = ['hello', 'world', 'a', 'b', 'c']
/// - Parameter input: 1-D. Strings to split.
/// - Parameter delimiter: 0-D. Delimiter characters (bytes), or empty string.
/// - Parameter skipEmpty: A `bool`. If `True`, skip the empty strings from the result.
/// - Returns: 
///	indices: A dense matrix of int64 representing the indices of the sparse tensor.
///	values: A vector of strings corresponding to the splited values.
///	shape: a length-2 vector of int64 representing the shape of the sparse
/// tensor, where the first value is N and the second value is the maximum number
/// of tokens in a single input entry.
public func stringSplit(operationName: String? = nil, input: Output, delimiter: Output, skipEmpty: Bool) throws -> (indices: Output, values: Output, shape: Output) { 
	var attrs = [String : Any]()
	attrs["skip_empty"] = skipEmpty
	let opspec = OpSpec(
		type: "StringSplit",
		name: (operationName ?? "Type"),
		input: [input, delimiter],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (indices: op.output(at: 0), values: op.output(at: 1), shape: op.output(at: 2))
} 

///Joins the strings in the given list of string tensors into one tensor;
///with the given separator (default is an empty separator).
/// - Parameter inputs: A list of string tensors.  The tensors must all have the same shape,
/// or be scalars.  Scalars may be mixed in; these will be broadcast to the shape
/// of non-scalar inputs.
/// - Parameter n: 
/// - Parameter separator: string, an optional join separator.
/// - Returns: 
///	output: 
public func stringJoin(operationName: String? = nil, inputs: [Output], n: UInt8, separator: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["N"] = n
	attrs["separator"] = separator
	let opspec = OpSpec(
		type: "StringJoin",
		name: (operationName ?? "Type"),
		input: [inputs],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Converts each entry in the given tensor to strings.  Supports many numeric
///types and boolean.
/// - Parameter input: 
/// - Parameter precision: The post-decimal precision to use for floating point numbers.
/// Only used if precision > -1.
/// - Parameter scientific: Use scientific notation for floating point numbers.
/// - Parameter shortest: Use shortest representation (either scientific or standard) for
/// floating point numbers.
/// - Parameter width: Pad pre-decimal numbers to this width.
/// Applies to both floating point and integer numbers.
/// Only used if width > -1.
/// - Parameter fill: The value to pad if width > -1.  If empty, pads with spaces.
/// Another typical value is '0'.  String cannot be longer than 1 character.
/// - Returns: 
///	output: 
public func asString(operationName: String? = nil, input: Output, precision: UInt8, scientific: Bool, shortest: Bool, width: UInt8, fill: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["precision"] = precision
	attrs["scientific"] = scientific
	attrs["shortest"] = shortest
	attrs["width"] = width
	attrs["fill"] = fill
	let opspec = OpSpec(
		type: "AsString",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Shuffle dimensions of x according to a permutation.
///The output `y` has the same rank as `x`. The shapes of `x` and `y` satisfy:
///   `y.shape[i] == x.shape[perm[i]] for i in [0, 1, ..., rank(x) - 1]`
/// - Parameter x: 
/// - Parameter perm: 
/// - Parameter tperm: 
/// - Returns: 
///	y: 
public func transpose(operationName: String? = nil, x: Output, perm: Output, tperm: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Tperm"] = tperm
	let opspec = OpSpec(
		type: "Transpose",
		name: (operationName ?? "Type"),
		input: [x, perm],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Writes a `Summary` protocol buffer with scalar values.
///The input `tag` and `value` must have the scalars.
/// - Parameter writer: A handle to a summary writer.
/// - Parameter globalStep: The step to write the summary for.
/// - Parameter tag: Tag for the summary.
/// - Parameter value: Value for the summary.
public func writeScalarSummary(operationName: String? = nil, writer: Output, globalStep: Output, tag: Output, value: Output) throws -> Operation { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "WriteScalarSummary",
		name: (operationName ?? "Type"),
		input: [writer, globalStep, tag, value],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Concatenates a list of `SparseTensor` along the specified dimension.
///Concatenation is with respect to the dense versions of these sparse tensors.
/// It is assumed that each input is a `SparseTensor` whose elements are ordered
/// along increasing dimension number.
/// 
/// All inputs' shapes must match, except for the concat dimension.  The
/// `indices`, `values`, and `shapes` lists must have the same length.
/// 
/// The output shape is identical to the inputs', except along the concat
/// dimension, where it is the sum of the inputs' sizes along that dimension.
/// 
/// The output elements will be resorted to preserve the sort order along
/// increasing dimension number.
/// 
/// This op runs in `O(M log M)` time, where `M` is the total number of non-empty
/// values across all inputs. This is due to the need for an internal sort in
/// order to concatenate efficiently across an arbitrary dimension.
/// 
/// For example, if `concat_dim = 1` and the inputs are
/// 
///     sp_inputs[0]: shape = [2, 3]
///     [0, 2]: "a"
///     [1, 0]: "b"
///     [1, 1]: "c"
/// 
///     sp_inputs[1]: shape = [2, 4]
///     [0, 1]: "d"
///     [0, 2]: "e"
/// 
/// then the output will be
/// 
///     shape = [2, 7]
///     [0, 2]: "a"
///     [0, 4]: "d"
///     [0, 5]: "e"
///     [1, 0]: "b"
///     [1, 1]: "c"
/// 
/// Graphically this is equivalent to doing
/// 
///     [    a] concat [  d e  ] = [    a   d e  ]
///     [b c  ]        [       ]   [b c          ]
/// - Parameter indices: 2-D.  Indices of each input `SparseTensor`.
/// - Parameter values: 1-D.  Non-empty values of each `SparseTensor`.
/// - Parameter shapes: 1-D.  Shapes of each `SparseTensor`.
/// - Parameter concatDim: Dimension to concatenate along. Must be in range [-rank, rank),
/// where rank is the number of dimensions in each input `SparseTensor`.
/// - Parameter n: 
/// - Returns: 
///	output_indices: 2-D.  Indices of the concatenated `SparseTensor`.
///	output_values: 1-D.  Non-empty values of the concatenated `SparseTensor`.
///	output_shape: 1-D.  Shape of the concatenated `SparseTensor`.
public func sparseConcat(operationName: String? = nil, indices: [Output], values: [Output], shapes: [Output], concatDim: UInt8, n: UInt8) throws -> (outputIndices: Output, outputValues: Output, outputShape: Output) { 
	var attrs = [String : Any]()
	attrs["concat_dim"] = concatDim
	attrs["N"] = n
	let opspec = OpSpec(
		type: "SparseConcat",
		name: (operationName ?? "Type"),
		input: [indices, values, shapes],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (outputIndices: op.output(at: 0), outputValues: op.output(at: 1), outputShape: op.output(at: 2))
} 

///Generate a glob pattern matching all sharded file names.
/// - Parameter basename: 
/// - Parameter numShards: 
/// - Returns: 
///	filename: 
public func shardedFilespec(operationName: String? = nil, basename: Output, numShards: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "ShardedFilespec",
		name: (operationName ?? "Type"),
		input: [basename, numShards],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Inverse 2D fast Fourier transform.
///Computes the inverse 2-dimensional discrete Fourier transform over the
/// inner-most 2 dimensions of `input`.
/// - Parameter input: A complex64 tensor.
/// - Returns: 
///	output: A complex64 tensor of the same shape as `input`. The inner-most 2
///   dimensions of `input` are replaced with their inverse 2D Fourier transform.
/// 
/// @compatibility(numpy)
/// Equivalent to np.fft.ifft2
/// @end_compatibility
public func ifft2D(operationName: String? = nil, input: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "IFFT2D",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Joins a string Tensor across the given dimensions.
///Computes the string join across dimensions in the given string Tensor of shape
/// `[d_0, d_1, ..., d_n-1]`.  Returns a new Tensor created by joining the input
/// strings with the given separator (default: empty string).  Negative indices are
/// counted backwards from the end, with `-1` being equivalent to `n - 1`.
/// 
/// For example:
/// 
/// ```python
/// # tensor `a` is [["a", "b"], ["c", "d"]]
/// tf.reduce_join(a, 0) ==> ["ac", "bd"]
/// tf.reduce_join(a, 1) ==> ["ab", "cd"]
/// tf.reduce_join(a, -2) = tf.reduce_join(a, 0) ==> ["ac", "bd"]
/// tf.reduce_join(a, -1) = tf.reduce_join(a, 1) ==> ["ab", "cd"]
/// tf.reduce_join(a, 0, keep_dims=True) ==> [["ac", "bd"]]
/// tf.reduce_join(a, 1, keep_dims=True) ==> [["ab"], ["cd"]]
/// tf.reduce_join(a, 0, separator=".") ==> ["a.c", "b.d"]
/// tf.reduce_join(a, [0, 1]) ==> ["acbd"]
/// tf.reduce_join(a, [1, 0]) ==> ["abcd"]
/// tf.reduce_join(a, []) ==> ["abcd"]
/// ```
/// - Parameter inputs: The input to be joined.  All reduced indices must have non-zero size.
/// - Parameter reductionIndices: The dimensions to reduce over.  Dimensions are reduced in the
/// order specified.  Omitting `reduction_indices` is equivalent to passing
/// `[n-1, n-2, ..., 0]`.  Negative indices from `-n` to `-1` are supported.
/// - Parameter keepDims: If `True`, retain reduced dimensions with length `1`.
/// - Parameter separator: The separator to use when joining.
/// - Returns: 
///	output: Has shape equal to that of the input with reduced dimensions removed or
/// set to `1` depending on `keep_dims`.
public func reduceJoin(operationName: String? = nil, inputs: Output, reductionIndices: Output, keepDims: Bool, separator: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["keep_dims"] = keepDims
	attrs["separator"] = separator
	let opspec = OpSpec(
		type: "ReduceJoin",
		name: (operationName ?? "Type"),
		input: [inputs, reductionIndices],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Converts each string in the input Tensor to its hash mod by a number of buckets.
///The hash function is deterministic on the content of the string within the
/// process.
/// 
/// Note that the hash function may change from time to time.
/// This functionality will be deprecated and it's recommended to use
/// `tf.string_to_hash_bucket_fast()` or `tf.string_to_hash_bucket_strong()`.
/// - Parameter stringTensor: 
/// - Parameter numBuckets: The number of buckets.
/// - Returns: 
///	output: A Tensor of the same shape as the input `string_tensor`.
public func stringToHashBucket(operationName: String? = nil, stringTensor: Output, numBuckets: UInt8) throws -> Output { 
	var attrs = [String : Any]()
	attrs["num_buckets"] = numBuckets
	let opspec = OpSpec(
		type: "StringToHashBucket",
		name: (operationName ?? "Type"),
		input: [stringTensor],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Outputs deterministic pseudorandom values from a truncated normal distribution.
///The generated values follow a normal distribution with mean 0 and standard
/// deviation 1, except that values whose magnitude is more than 2 standard
/// deviations from the mean are dropped and re-picked.
/// 
/// The outputs are a deterministic function of `shape` and `seed`.
/// - Parameter shape: The shape of the output tensor.
/// - Parameter seed: 2 seeds (shape [2]).
/// - Parameter dtype: The type of the output.
/// - Returns: 
///	output: Random values with specified shape.
public func statelessTruncatedNormal(operationName: String? = nil, shape: Output, seed: Output, dtype: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["dtype"] = dtype
	let opspec = OpSpec(
		type: "StatelessTruncatedNormal",
		name: (operationName ?? "Type"),
		input: [shape, seed],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Outputs deterministic pseudorandom random values from a uniform distribution.
///The generated values follow a uniform distribution in the range `[0, 1)`. The
/// lower bound 0 is included in the range, while the upper bound 1 is excluded.
/// 
/// The outputs are a deterministic function of `shape` and `seed`.
/// - Parameter shape: The shape of the output tensor.
/// - Parameter seed: 2 seeds (shape [2]).
/// - Parameter dtype: The type of the output.
/// - Returns: 
///	output: Random values with specified shape.
public func statelessRandomUniform(operationName: String? = nil, shape: Output, seed: Output, dtype: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["dtype"] = dtype
	let opspec = OpSpec(
		type: "StatelessRandomUniform",
		name: (operationName ?? "Type"),
		input: [shape, seed],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Outputs random values from the Gamma distribution(s) described by alpha.
///This op uses the algorithm by Marsaglia et al. to acquire samples via
/// transformation-rejection from pairs of uniform and normal random variables.
/// See http://dl.acm.org/citation.cfm?id=358414
/// - Parameter shape: 1-D integer tensor. Shape of independent samples to draw from each
/// distribution described by the shape parameters given in alpha.
/// - Parameter alpha: A tensor in which each scalar is a "shape" parameter describing the
/// associated gamma distribution.
/// - Parameter seed: If either `seed` or `seed2` are set to be non-zero, the random number
/// generator is seeded by the given seed.  Otherwise, it is seeded by a
/// random seed.
/// - Parameter seed2: A second seed to avoid seed collision.
/// - Parameter s: 
/// - Returns: 
///	output: A tensor with shape `shape + shape(alpha)`. Each slice
/// `[:, ..., :, i0, i1, ...iN]` contains the samples drawn for
/// `alpha[i0, i1, ...iN]`. The dtype of the output matches the dtype of alpha.
public func randomGamma(operationName: String? = nil, shape: Output, alpha: Output, seed: UInt8, seed2: UInt8, s: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["seed"] = seed
	attrs["seed2"] = seed2
	attrs["S"] = s
	let opspec = OpSpec(
		type: "RandomGamma",
		name: (operationName ?? "Type"),
		input: [shape, alpha],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Outputs random values from a uniform distribution.
///The generated values follow a uniform distribution in the range `[0, 1)`. The
/// lower bound 0 is included in the range, while the upper bound 1 is excluded.
/// - Parameter shape: The shape of the output tensor.
/// - Parameter seed: If either `seed` or `seed2` are set to be non-zero, the random number
/// generator is seeded by the given seed.  Otherwise, it is seeded by a
/// random seed.
/// - Parameter seed2: A second seed to avoid seed collision.
/// - Parameter dtype: The type of the output.
/// - Returns: 
///	output: A tensor of the specified shape filled with uniform random values.
public func randomUniform(operationName: String? = nil, shape: Output, seed: UInt8, seed2: UInt8, dtype: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["seed"] = seed
	attrs["seed2"] = seed2
	attrs["dtype"] = dtype
	let opspec = OpSpec(
		type: "RandomUniform",
		name: (operationName ?? "Type"),
		input: [shape],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Applies sparse subtraction between `updates` and individual values or slices
///within a given variable according to `indices`.
/// 
/// `ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.
/// 
/// `indices` must be integer tensor, containing indices into `ref`.
/// It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.
/// 
/// The innermost dimension of `indices` (with length `K`) corresponds to
/// indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
/// dimension of `ref`.
/// 
/// `updates` is `Tensor` of rank `Q-1+P-K` with shape:
/// 
/// ```
/// [d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]].
/// ```
/// 
/// For example, say we want to subtract 4 scattered elements from a rank-1 tensor
/// with 8 elements. In Python, that subtraction would look like this:
/// 
///     ref = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8])
///     indices = tf.constant([[4], [3], [1], [7]])
///     updates = tf.constant([9, 10, 11, 12])
///     sub = tf.scatter_nd_sub(ref, indices, updates)
///     with tf.Session() as sess:
///       print sess.run(sub)
/// 
/// The resulting update to ref would look like this:
/// 
///     [1, -9, 3, -6, -4, 6, 7, -4]
/// 
/// See @{tf.scatter_nd} for more details about how to make updates to
/// slices.
/// - Parameter ref: A mutable Tensor. Should be from a Variable node.
/// - Parameter indices: A Tensor. Must be one of the following types: int32, int64.
/// A tensor of indices into ref.
/// - Parameter updates: A Tensor. Must have the same type as ref. A tensor of updated values
/// to subtract from ref.
/// - Parameter tindices: 
/// - Parameter useLocking: An optional bool. Defaults to True. If True, the assignment will
/// be protected by a lock; otherwise the behavior is undefined,
/// but may exhibit less contention.
/// - Returns: 
///	output_ref: Same as ref. Returned as a convenience for operations that want
/// to use the updated values after the update is done.
public func scatterNdSub(operationName: String? = nil, ref: Output, indices: Output, updates: Output, tindices: Any.Type, useLocking: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Tindices"] = tindices
	attrs["use_locking"] = useLocking
	let opspec = OpSpec(
		type: "ScatterNdSub",
		name: (operationName ?? "Type"),
		input: [ref, indices, updates],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Fills empty rows in the input 2-D `SparseTensor` with a default value.
///The input `SparseTensor` is represented via the tuple of inputs
/// (`indices`, `values`, `dense_shape`).  The output `SparseTensor` has the
/// same `dense_shape` but with indices `output_indices` and values
/// `output_values`.
/// 
/// This op inserts a single entry for every row that doesn't have any values.
/// The index is created as `[row, 0, ..., 0]` and the inserted value
/// is `default_value`.
/// 
/// For example, suppose `sp_input` has shape `[5, 6]` and non-empty values:
/// 
///     [0, 1]: a
///     [0, 3]: b
///     [2, 0]: c
///     [3, 1]: d
/// 
/// Rows 1 and 4 are empty, so the output will be of shape `[5, 6]` with values:
/// 
///     [0, 1]: a
///     [0, 3]: b
///     [1, 0]: default_value
///     [2, 0]: c
///     [3, 1]: d
///     [4, 0]: default_value
/// 
/// The output `SparseTensor` will be in row-major order and will have the
/// same shape as the input.
/// 
/// This op also returns an indicator vector shaped `[dense_shape[0]]` such that
/// 
///     empty_row_indicator[i] = True iff row i was an empty row.
/// 
/// And a reverse index map vector shaped `[indices.shape[0]]` that is used during
/// backpropagation,
/// 
///     reverse_index_map[j] = out_j s.t. indices[j, :] == output_indices[out_j, :]
/// - Parameter indices: 2-D. the indices of the sparse tensor.
/// - Parameter values: 1-D. the values of the sparse tensor.
/// - Parameter denseShape: 1-D. the shape of the sparse tensor.
/// - Parameter defaultValue: 0-D. default value to insert into location `[row, 0, ..., 0]`
///   for rows missing from the input sparse tensor.
/// output indices: 2-D. the indices of the filled sparse tensor.
/// - Returns: 
///	output_indices: 
///	output_values: 1-D. the values of the filled sparse tensor.
///	empty_row_indicator: 1-D. whether the dense row was missing in the
/// input sparse tensor.
///	reverse_index_map: 1-D. a map from the input indices to the output indices.
public func sparseFillEmptyRows(operationName: String? = nil, indices: Output, values: Output, denseShape: Output, defaultValue: Output) throws -> (outputIndices: Output, outputValues: Output, emptyRowIndicator: Output, reverseIndexMap: Output) { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "SparseFillEmptyRows",
		name: (operationName ?? "Type"),
		input: [indices, values, denseShape, defaultValue],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (outputIndices: op.output(at: 0), outputValues: op.output(at: 1), emptyRowIndicator: op.output(at: 2), reverseIndexMap: op.output(at: 3))
} 

///A Reader that outputs the lines of a file delimited by '\n'.
/// - Parameter skipHeaderLines: Number of lines to skip from the beginning of every file.
/// - Parameter container: If non-empty, this reader is placed in the given container.
/// Otherwise, a default container is used.
/// - Parameter sharedName: If non-empty, this reader is named in the given bucket
/// with this shared_name. Otherwise, the node name is used instead.
/// - Returns: 
///	reader_handle: The handle to reference the Reader.
public func textLineReaderV2(operationName: String? = nil, skipHeaderLines: UInt8, container: String, sharedName: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["skip_header_lines"] = skipHeaderLines
	attrs["container"] = container
	attrs["shared_name"] = sharedName
	let opspec = OpSpec(
		type: "TextLineReaderV2",
		name: (operationName ?? "Type"),
		input: [],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///A Reader that outputs the queued work as both the key and value.
///To use, enqueue strings in a Queue.  ReaderRead will take the front
/// work string and output (work, work).
/// - Parameter container: If non-empty, this reader is placed in the given container.
/// Otherwise, a default container is used.
/// - Parameter sharedName: If non-empty, this reader is named in the given bucket
/// with this shared_name. Otherwise, the node name is used instead.
/// - Returns: 
///	reader_handle: The handle to reference the Reader.
public func identityReaderV2(operationName: String? = nil, container: String, sharedName: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["container"] = container
	attrs["shared_name"] = sharedName
	let opspec = OpSpec(
		type: "IdentityReaderV2",
		name: (operationName ?? "Type"),
		input: [],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Inverse 3D real-valued fast Fourier transform.
///Computes the inverse 3-dimensional discrete Fourier transform of a real-valued
/// signal over the inner-most 3 dimensions of `input`.
/// 
/// The inner-most 3 dimensions of `input` are assumed to be the result of `RFFT3D`:
/// The inner-most dimension contains the `fft_length / 2 + 1` unique components of
/// the DFT of a real-valued signal. If `fft_length` is not provided, it is computed
/// from the size of the inner-most 3 dimensions of `input`. If the FFT length used
/// to compute `input` is odd, it should be provided since it cannot be inferred
/// properly.
/// 
/// Along each axis `IRFFT3D` is computed on, if `fft_length` (or
/// `fft_length / 2 + 1` for the inner-most dimension) is smaller than the
/// corresponding dimension of `input`, the dimension is cropped. If it is larger,
/// the dimension is padded with zeros.
/// - Parameter input: A complex64 tensor.
/// - Parameter fftLength: An int32 tensor of shape [3]. The FFT length for each dimension.
/// - Returns: 
///	output: A float32 tensor of the same rank as `input`. The inner-most 3
///   dimensions of `input` are replaced with the `fft_length` samples of their
///   inverse 3D real Fourier transform.
/// 
/// @compatibility(numpy)
/// Equivalent to np.irfftn with 3 dimensions.
/// @end_compatibility
public func irfft3D(operationName: String? = nil, input: Output, fftLength: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "IRFFT3D",
		name: (operationName ?? "Type"),
		input: [input, fftLength],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Returns the element-wise min of two SparseTensors.
///Assumes the two SparseTensors have the same shape, i.e., no broadcasting.
/// - Parameter aIndices: 2-D.  `N x R` matrix with the indices of non-empty values in a
/// SparseTensor, in the canonical lexicographic ordering.
/// - Parameter aValues: 1-D.  `N` non-empty values corresponding to `a_indices`.
/// - Parameter aShape: 1-D.  Shape of the input SparseTensor.
/// - Parameter bIndices: counterpart to `a_indices` for the other operand.
/// - Parameter bValues: counterpart to `a_values` for the other operand; must be of the same dtype.
/// - Parameter bShape: counterpart to `a_shape` for the other operand; the two shapes must be equal.
/// - Returns: 
///	output_indices: 2-D.  The indices of the output SparseTensor.
///	output_values: 1-D.  The values of the output SparseTensor.
public func sparseSparseMinimum(operationName: String? = nil, aIndices: Output, aValues: Output, aShape: Output, bIndices: Output, bValues: Output, bShape: Output) throws -> (outputIndices: Output, outputValues: Output) { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "SparseSparseMinimum",
		name: (operationName ?? "Type"),
		input: [aIndices, aValues, aShape, bIndices, bValues, bShape],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (outputIndices: op.output(at: 0), outputValues: op.output(at: 1))
} 

///An identity op that triggers an error if a gradient is requested.
///When executed in a graph, this op outputs its input tensor as-is.
/// 
/// When building ops to compute gradients, the TensorFlow gradient system
/// will return an error when trying to lookup the gradient of this op,
/// because no gradient must ever be registered for this function.  This
/// op exists to prevent subtle bugs from silently returning unimplemented
/// gradients in some corner cases.
/// - Parameter input: any tensor.
/// - Parameter message: Will be printed in the error when anyone tries to differentiate
/// this operation.
/// - Returns: 
///	output: the same input tensor.
public func preventGradient(operationName: String? = nil, input: Output, message: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["message"] = message
	let opspec = OpSpec(
		type: "PreventGradient",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Applies softmax to a batched N-D `SparseTensor`.
///The inputs represent an N-D SparseTensor  with logical shape `[..., B, C]`
/// (where `N >= 2`), and with indices sorted in the canonical lexicographic order.
/// 
/// This op is equivalent to applying the normal `tf.nn.softmax()` to each innermost
/// logical submatrix with shape `[B, C]`, but with the catch that  * the implicitly
/// zero elements do not participate * .  Specifically, the algorithm is equivalent
/// to the following:
/// 
///   (1) Applies `tf.nn.softmax()` to a densified view of each innermost submatrix
///       with shape `[B, C]`, along the size-C dimension;
///   (2) Masks out the original implicitly-zero locations;
///   (3) Renormalizes the remaining elements.
/// 
/// Hence, the `SparseTensor` result has exactly the same non-zero indices and
/// shape.
/// - Parameter spIndices: 2-D.  `NNZ x R` matrix with the indices of non-empty values in a
/// SparseTensor, in canonical ordering.
/// - Parameter spValues: 1-D.  `NNZ` non-empty values corresponding to `sp_indices`.
/// - Parameter spShape: 1-D.  Shape of the input SparseTensor.
/// - Returns: 
///	output: 1-D.  The `NNZ` values for the result `SparseTensor`.
public func sparseSoftmax(operationName: String? = nil, spIndices: Output, spValues: Output, spShape: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "SparseSoftmax",
		name: (operationName ?? "Type"),
		input: [spIndices, spValues, spShape],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Adds up a SparseTensor and a dense Tensor, using these special rules:
///(1) Broadcasts the dense side to have the same shape as the sparse side, if
///     eligible;
/// (2) Then, only the dense values pointed to by the indices of the SparseTensor
///     participate in the cwise addition.
/// 
/// By these rules, the result is a logical SparseTensor with exactly the same
/// indices and shape, but possibly with different non-zero values.  The output of
/// this Op is the resultant non-zero values.
/// - Parameter spIndices: 2-D.  `N x R` matrix with the indices of non-empty values in a
/// SparseTensor, possibly not in canonical ordering.
/// - Parameter spValues: 1-D.  `N` non-empty values corresponding to `sp_indices`.
/// - Parameter spShape: 1-D.  Shape of the input SparseTensor.
/// - Parameter dense: `R`-D.  The dense Tensor operand.
/// - Returns: 
///	output: 1-D.  The `N` values that are operated on.
public func sparseDenseCwiseAdd(operationName: String? = nil, spIndices: Output, spValues: Output, spShape: Output, dense: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "SparseDenseCwiseAdd",
		name: (operationName ?? "Type"),
		input: [spIndices, spValues, spShape, dense],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Update ' * var' according to the adagrad scheme.
///accum += grad  *  grad
/// var -= lr  *  grad  *  (1 / sqrt(accum))
/// - Parameter `var`: Should be from a Variable().
/// - Parameter accum: Should be from a Variable().
/// - Parameter lr: Scaling factor. Must be a scalar.
/// - Parameter grad: The gradient.
/// - Parameter useLocking: If `True`, updating of the var and accum tensors will be protected
/// by a lock; otherwise the behavior is undefined, but may exhibit less
/// contention.
/// - Returns: 
///	out: Same as "var".
public func applyAdagrad(operationName: String? = nil, `var`: Output, accum: Output, lr: Output, grad: Output, useLocking: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["use_locking"] = useLocking
	let opspec = OpSpec(
		type: "ApplyAdagrad",
		name: (operationName ?? "Type"),
		input: [`var`, accum, lr, grad],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Outputs deterministic pseudorandom values from a normal distribution.
///The generated values will have mean 0 and standard deviation 1.
/// 
/// The outputs are a deterministic function of `shape` and `seed`.
/// - Parameter shape: The shape of the output tensor.
/// - Parameter seed: 2 seeds (shape [2]).
/// - Parameter dtype: The type of the output.
/// - Returns: 
///	output: Random values with specified shape.
public func statelessRandomNormal(operationName: String? = nil, shape: Output, seed: Output, dtype: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["dtype"] = dtype
	let opspec = OpSpec(
		type: "StatelessRandomNormal",
		name: (operationName ?? "Type"),
		input: [shape, seed],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Adds up a `SparseTensor` and a dense `Tensor`, producing a dense `Tensor`.
///This Op does not require `a_indices` be sorted in standard lexicographic order.
/// - Parameter aIndices: 2-D.  The `indices` of the `SparseTensor`, with shape `[nnz, ndims]`.
/// - Parameter aValues: 1-D.  The `values` of the `SparseTensor`, with shape `[nnz]`.
/// - Parameter aShape: 1-D.  The `shape` of the `SparseTensor`, with shape `[ndims]`.
/// - Parameter b: `ndims`-D Tensor.  With shape `a_shape`.
/// - Parameter tindices: 
/// - Returns: 
///	output: 
public func sparseTensorDenseAdd(operationName: String? = nil, aIndices: Output, aValues: Output, aShape: Output, b: Output, tindices: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Tindices"] = tindices
	let opspec = OpSpec(
		type: "SparseTensorDenseAdd",
		name: (operationName ?? "Type"),
		input: [aIndices, aValues, aShape, b],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Get the value of the tensor specified by its handle.
/// - Parameter handle: The handle for a tensor stored in the session state.
/// - Parameter dtype: The type of the output value.
/// - Returns: 
///	value: The tensor for the given handle.
public func getSessionTensor(operationName: String? = nil, handle: Output, dtype: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["dtype"] = dtype
	let opspec = OpSpec(
		type: "GetSessionTensor",
		name: (operationName ?? "Type"),
		input: [handle],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Reorders a SparseTensor into the canonical, row-major ordering.
///Note that by convention, all sparse ops preserve the canonical ordering along
/// increasing dimension number. The only time ordering can be violated is during
/// manual manipulation of the indices and values vectors to add entries.
/// 
/// Reordering does not affect the shape of the SparseTensor.
/// 
/// If the tensor has rank `R` and `N` non-empty values, `input_indices` has
/// shape `[N, R]`, input_values has length `N`, and input_shape has length `R`.
/// - Parameter inputIndices: 2-D.  `N x R` matrix with the indices of non-empty values in a
/// SparseTensor, possibly not in canonical ordering.
/// - Parameter inputValues: 1-D.  `N` non-empty values corresponding to `input_indices`.
/// - Parameter inputShape: 1-D.  Shape of the input SparseTensor.
/// - Returns: 
///	output_indices: 2-D.  `N x R` matrix with the same indices as input_indices, but
/// in canonical row-major ordering.
///	output_values: 1-D.  `N` non-empty values corresponding to `output_indices`.
public func sparseReorder(operationName: String? = nil, inputIndices: Output, inputValues: Output, inputShape: Output) throws -> (outputIndices: Output, outputValues: Output) { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "SparseReorder",
		name: (operationName ?? "Type"),
		input: [inputIndices, inputValues, inputShape],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (outputIndices: op.output(at: 0), outputValues: op.output(at: 1))
} 

///Split a `SparseTensor` into `num_split` tensors along one dimension.
///If the `shape[split_dim]` is not an integer multiple of `num_split`. Slices
/// `[0 : shape[split_dim] % num_split]` gets one extra dimension.
/// For example, if `split_dim = 1` and `num_split = 2` and the input is
/// 
///     input_tensor = shape = [2, 7]
///     [    a   d e  ]
///     [b c          ]
/// 
/// Graphically the output tensors are:
/// 
///     output_tensor[0] = shape = [2, 4]
///     [    a  ]
///     [b c    ]
/// 
///     output_tensor[1] = shape = [2, 3]
///     [ d e  ]
///     [      ]
/// - Parameter splitDim: 0-D.  The dimension along which to split.  Must be in the range
/// `[0, rank(shape))`.
/// - Parameter indices: 2-D tensor represents the indices of the sparse tensor.
/// - Parameter values: 1-D tensor represents the values of the sparse tensor.
/// - Parameter shape: 1-D. tensor represents the shape of the sparse tensor.
/// output indices: A list of 1-D tensors represents the indices of the output
/// sparse tensors.
/// - Parameter numSplit: The number of ways to split.
/// - Returns: 
///	output_indices: 
///	output_values: A list of 1-D tensors represents the values of the output sparse
/// tensors.
///	output_shape: A list of 1-D tensors represents the shape of the output sparse
/// tensors.
public func sparseSplit(operationName: String? = nil, splitDim: Output, indices: Output, values: Output, shape: Output, numSplit: UInt8) throws -> (outputIndices: Output, outputValues: Output, outputShape: Output) { 
	var attrs = [String : Any]()
	attrs["num_split"] = numSplit
	let opspec = OpSpec(
		type: "SparseSplit",
		name: (operationName ?? "Type"),
		input: [splitDim, indices, values, shape],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (outputIndices: op.output(at: 0), outputValues: op.output(at: 1), outputShape: op.output(at: 2))
} 

///Converts a sparse representation into a dense tensor.
///Builds an array `dense` with shape `output_shape` such that
/// 
/// ```
/// # If sparse_indices is scalar
/// dense[i] = (i == sparse_indices ? sparse_values : default_value)
/// 
/// # If sparse_indices is a vector, then for each i
/// dense[sparse_indices[i]] = sparse_values[i]
/// 
/// # If sparse_indices is an n by d matrix, then for each i in [0, n)
/// dense[sparse_indices[i][0], ..., sparse_indices[i][d-1]] = sparse_values[i]
/// ```
/// 
/// All other values in `dense` are set to `default_value`.  If `sparse_values` is a
/// scalar, all sparse indices are set to this single value.
/// 
/// Indices should be sorted in lexicographic order, and indices must not
/// contain any repeats. If `validate_indices` is true, these properties
/// are checked during execution.
/// - Parameter sparseIndices: 0-D, 1-D, or 2-D.  `sparse_indices[i]` contains the complete
/// index where `sparse_values[i]` will be placed.
/// - Parameter outputShape: 1-D.  Shape of the dense output tensor.
/// - Parameter sparseValues: 1-D.  Values corresponding to each row of `sparse_indices`,
/// or a scalar value to be used for all sparse indices.
/// - Parameter defaultValue: Scalar value to set for indices not specified in
/// `sparse_indices`.
/// - Parameter validateIndices: If true, indices are checked to make sure they are sorted in
/// lexicographic order and that there are no repeats.
/// - Parameter tindices: 
/// - Returns: 
///	dense: Dense output tensor of shape `output_shape`.
public func sparseToDense(operationName: String? = nil, sparseIndices: Output, outputShape: Output, sparseValues: Output, defaultValue: Output, validateIndices: Bool, tindices: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["validate_indices"] = validateIndices
	attrs["Tindices"] = tindices
	let opspec = OpSpec(
		type: "SparseToDense",
		name: (operationName ?? "Type"),
		input: [sparseIndices, outputShape, sparseValues, defaultValue],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Deprecated. Use TensorArrayGradV3
/// - Parameter handle: 
/// - Parameter index: 
/// - Parameter value: 
/// - Parameter flowIn: 
/// - Returns: 
///	flow_out: 
public func tensorArrayWriteV2(operationName: String? = nil, handle: Output, index: Output, value: Output, flowIn: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "TensorArrayWriteV2",
		name: (operationName ?? "Type"),
		input: [handle, index, value, flowIn],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Elementwise computes the bitwise XOR of `x` and `y`.
///The result will have those bits set, that are different in `x` and `y`. The
/// computation is performed on the underlying representations of `x` and `y`.
/// - Parameter x: 
/// - Parameter y: 
/// - Returns: 
///	z: 
public func bitwiseXor(operationName: String? = nil, x: Output, y: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "BitwiseXor",
		name: (operationName ?? "Type"),
		input: [x, y],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes element-wise population count (a.k.a. popcount, bitsum, bitcount).
///For each entry in `x`, calculates the number of `1` (on) bits in the binary
/// representation of that entry.
/// 
///  *  * NOTE *  * : It is more efficient to first `tf.bitcast` your tensors into
/// `int32` or `int64` and perform the bitcount on the result, than to feed in
/// 8- or 16-bit inputs and then aggregate the resulting counts.
/// - Parameter x: 
/// - Returns: 
///	y: 
public func populationCount(operationName: String? = nil, x: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "PopulationCount",
		name: (operationName ?? "Type"),
		input: [x],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///A container for an iterator resource.
/// - Parameter sharedName: 
/// - Parameter container: 
/// - Parameter outputTypes: 
/// - Parameter outputShapes: 
/// - Returns: 
///	handle: A handle to the iterator that can be passed to a "MakeIterator"
/// or "IteratorGetNext" op.
public func iterator(operationName: String? = nil, sharedName: String, container: String, outputTypes: [Any.Type], outputShapes: [Shape]) throws -> Output { 
	var attrs = [String : Any]()
	attrs["shared_name"] = sharedName
	attrs["container"] = container
	attrs["output_types"] = outputTypes
	attrs["output_shapes"] = outputShapes
	let opspec = OpSpec(
		type: "Iterator",
		name: (operationName ?? "Type"),
		input: [],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Applies set operation along last dimension of `Tensor` and `SparseTensor`.
///See SetOperationOp::SetOperationFromContext for values of `set_operation`.
/// 
/// Input `set2` is a `SparseTensor` represented by `set2_indices`, `set2_values`,
/// and `set2_shape`. For `set2` ranked `n`, 1st `n-1` dimensions must be the same
/// as `set1`. Dimension `n` contains values in a set, duplicates are allowed but
/// ignored.
/// 
/// If `validate_indices` is `True`, this op validates the order and range of `set2`
/// indices.
/// 
/// Output `result` is a `SparseTensor` represented by `result_indices`,
/// `result_values`, and `result_shape`. For `set1` and `set2` ranked `n`, this
/// has rank `n` and the same 1st `n-1` dimensions as `set1` and `set2`. The `nth`
/// dimension contains the result of `set_operation` applied to the corresponding
/// `[0...n-1]` dimension of `set`.
/// - Parameter set1: `Tensor` with rank `n`. 1st `n-1` dimensions must be the same as `set2`.
/// Dimension `n` contains values in a set, duplicates are allowed but ignored.
/// - Parameter set2Indices: 2D `Tensor`, indices of a `SparseTensor`. Must be in row-major
/// order.
/// - Parameter set2Values: 1D `Tensor`, values of a `SparseTensor`. Must be in row-major
/// order.
/// - Parameter set2Shape: 1D `Tensor`, shape of a `SparseTensor`. `set2_shape[0...n-1]` must
/// be the same as the 1st `n-1` dimensions of `set1`, `result_shape[n]` is the
/// max set size across `n-1` dimensions.
/// - Parameter setOperation: 
/// - Parameter validateIndices: 
/// - Returns: 
///	result_indices: 2D indices of a `SparseTensor`.
///	result_values: 1D values of a `SparseTensor`.
///	result_shape: 1D `Tensor` shape of a `SparseTensor`. `result_shape[0...n-1]` is
/// the same as the 1st `n-1` dimensions of `set1` and `set2`, `result_shape[n]`
/// is the max result set size across all `0...n-1` dimensions.
public func denseToSparseSetOperation(operationName: String? = nil, set1: Output, set2Indices: Output, set2Values: Output, set2Shape: Output, setOperation: String, validateIndices: Bool) throws -> (resultIndices: Output, resultValues: Output, resultShape: Output) { 
	var attrs = [String : Any]()
	attrs["set_operation"] = setOperation
	attrs["validate_indices"] = validateIndices
	let opspec = OpSpec(
		type: "DenseToSparseSetOperation",
		name: (operationName ?? "Type"),
		input: [set1, set2Indices, set2Values, set2Shape],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (resultIndices: op.output(at: 0), resultValues: op.output(at: 1), resultShape: op.output(at: 2))
} 

///Returns x + y element-wise.
/// * NOTE * : `Add` supports broadcasting. `AddN` does not. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
/// - Parameter x: 
/// - Parameter y: 
/// - Parameter mklX: 
/// - Parameter mklY: 
/// - Returns: 
///	z: 
///	mkl_z: 
public func mklAdd(operationName: String? = nil, x: Output, y: Output, mklX: Output, mklY: Output) throws -> (z: Output, mklZ: Output) { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "_MklAdd",
		name: (operationName ?? "Type"),
		input: [x, y, mklX, mklY],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (z: op.output(at: 0), mklZ: op.output(at: 1))
} 

///Applies L1 regularization shrink step on the parameters.
/// - Parameter weights: a list of vectors where each value is the weight associated with a
/// feature group.
/// - Parameter numFeatures: Number of feature groups to apply shrinking step.
/// - Parameter l1: Symmetric l1 regularization strength.
/// - Parameter l2: Symmetric l2 regularization strength. Should be a positive float.
public func sdcaShrinkL1(operationName: String? = nil, weights: Output, numFeatures: UInt8, l1: Float, l2: Float) throws -> Operation { 
	var attrs = [String : Any]()
	attrs["num_features"] = numFeatures
	attrs["l1"] = l1
	attrs["l2"] = l2
	let opspec = OpSpec(
		type: "SdcaShrinkL1",
		name: (operationName ?? "Type"),
		input: [weights],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 


/// - Parameter l: 
/// - Parameter grad: 
/// - Returns: 
///	output: 
public func batchCholeskyGrad(operationName: String? = nil, l: Output, grad: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "BatchCholeskyGrad",
		name: (operationName ?? "Type"),
		input: [l, grad],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Adds sparse updates to the variable referenced by `resource`.
///This operation computes
/// 
///     # Scalar indices
///     ref[indices, ...] += updates[...]
/// 
///     # Vector indices (for each i)
///     ref[indices[i], ...] += updates[i, ...]
/// 
///     # High rank indices (for each i, ..., j)
///     ref[indices[i, ..., j], ...] += updates[i, ..., j, ...]
/// 
/// Duplicate entries are handled correctly: if multiple `indices` reference
/// the same location, their contributions add.
/// 
/// Requires `updates.shape = indices.shape + ref.shape[1:]`.
/// 
/// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
/// <img style="width:100%" src="https://www.tensorflow.org/images/ScatterAdd.png" alt>
/// </div>
/// - Parameter resource: Should be from a `Variable` node.
/// - Parameter indices: A tensor of indices into the first dimension of `ref`.
/// - Parameter updates: A tensor of updated values to add to `ref`.
/// - Parameter dtype: 
/// - Parameter tindices: 
public func resourceScatterAdd(operationName: String? = nil, resource: Output, indices: Output, updates: Output, dtype: Any.Type, tindices: Any.Type) throws -> Operation { 
	var attrs = [String : Any]()
	attrs["dtype"] = dtype
	attrs["Tindices"] = tindices
	let opspec = OpSpec(
		type: "ResourceScatterAdd",
		name: (operationName ?? "Type"),
		input: [resource, indices, updates],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Multiply SparseTensor (of rank 2) "A" by dense matrix "B".
///No validity checking is performed on the indices of A.  However, the following
/// input format is recommended for optimal behavior:
/// 
/// if adjoint_a == false:
///   A should be sorted in lexicographically increasing order.  Use SparseReorder
///   if you're not sure.
/// if adjoint_a == true:
///   A should be sorted in order of increasing dimension 1 (i.e., "column major"
///   order instead of "row major" order).
/// - Parameter aIndices: 2-D.  The `indices` of the `SparseTensor`, size `[nnz, 2]` Matrix.
/// - Parameter aValues: 1-D.  The `values` of the `SparseTensor`, size `[nnz]` Vector.
/// - Parameter aShape: 1-D.  The `shape` of the `SparseTensor`, size `[2]` Vector.
/// - Parameter b: 2-D.  A dense Matrix.
/// - Parameter tindices: 
/// - Parameter adjointA: Use the adjoint of A in the matrix multiply.  If A is complex, this
/// is transpose(conj(A)).  Otherwise it's transpose(A).
/// - Parameter adjointB: Use the adjoint of B in the matrix multiply.  If B is complex, this
/// is transpose(conj(B)).  Otherwise it's transpose(B).
/// - Returns: 
///	product: 
public func sparseTensorDenseMatMul(operationName: String? = nil, aIndices: Output, aValues: Output, aShape: Output, b: Output, tindices: Any.Type, adjointA: Bool, adjointB: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Tindices"] = tindices
	attrs["adjoint_a"] = adjointA
	attrs["adjoint_b"] = adjointB
	let opspec = OpSpec(
		type: "SparseTensorDenseMatMul",
		name: (operationName ?? "Type"),
		input: [aIndices, aValues, aShape, b],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Deletes the resource specified by the handle.
///All subsequent operations using the resource will result in a NotFound
/// error status.
/// - Parameter resource: handle to the resource to delete.
/// - Parameter ignoreLookupError: whether to ignore the error when the resource
/// doesn't exist.
public func destroyResourceOp(operationName: String? = nil, resource: Output, ignoreLookupError: Bool) throws -> Operation { 
	var attrs = [String : Any]()
	attrs["ignore_lookup_error"] = ignoreLookupError
	let opspec = OpSpec(
		type: "DestroyResourceOp",
		name: (operationName ?? "Type"),
		input: [resource],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Reads the value of a variable.
///The tensor returned by this operation is immutable.
/// 
/// The value returned by this operation is guaranteed to be influenced by all the
/// writes on which this operation depends directly or indirectly, and to not be
/// influenced by any of the writes which depend directly or indirectly on this
/// operation.
/// - Parameter resource: handle to the resource in which to store the variable.
/// - Parameter dtype: the dtype of the value.
/// - Returns: 
///	value: 
public func readVariableOp(operationName: String? = nil, resource: Output, dtype: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["dtype"] = dtype
	let opspec = OpSpec(
		type: "ReadVariableOp",
		name: (operationName ?? "Type"),
		input: [resource],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes the minimum along segments of a tensor.
///Read @{$math_ops#segmentation$the section on segmentation} for an explanation of
/// segments.
/// 
/// Computes a tensor such that
/// \\(output_i = \min_j(data_j)\\) where `min` is over `j` such
/// that `segment_ids[j] == i`.
/// 
/// If the min is empty for a given segment ID `i`, `output[i] = 0`.
/// 
/// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
/// <img style="width:100%" src="https://www.tensorflow.org/images/SegmentMin.png" alt>
/// </div>
/// - Parameter data: 
/// - Parameter segmentIds: A 1-D tensor whose rank is equal to the rank of `data`'s
/// first dimension.  Values should be sorted and can be repeated.
/// - Parameter tindices: 
/// - Returns: 
///	output: Has same shape as data, except for dimension 0 which
/// has size `k`, the number of segments.
public func segmentMin(operationName: String? = nil, data: Output, segmentIds: Output, tindices: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Tindices"] = tindices
	let opspec = OpSpec(
		type: "SegmentMin",
		name: (operationName ?? "Type"),
		input: [data, segmentIds],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Execute a sub graph on a remote processor.
///The graph specifications(such as graph itself, input tensors and output names)
/// are stored as a serialized protocol buffer of RemoteFusedGraphExecuteInfo
/// as serialized_remote_fused_graph_execute_info.
/// The specifications will be passed to a dedicated registered
/// remote fused graph executor.  The executor will send the graph specifications
/// to a remote processor and execute that graph.  The execution results
/// will be passed to consumer nodes as outputs of this node.
/// - Parameter inputs: Arbitrary number of tensors with arbitrary data types
/// - Parameter tinputs: 
/// - Parameter toutputs: 
/// - Parameter serializedRemoteFusedGraphExecuteInfo: Serialized protocol buffer
/// of RemoteFusedGraphExecuteInfo which contains graph specifications.
/// - Returns: 
///	outputs: Arbitrary number of tensors with arbitrary data types
public func remoteFusedGraphExecute(operationName: String? = nil, inputs: Output, tinputs: [Any.Type], toutputs: [Any.Type], serializedRemoteFusedGraphExecuteInfo: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Tinputs"] = tinputs
	attrs["Toutputs"] = toutputs
	attrs["serialized_remote_fused_graph_execute_info"] = serializedRemoteFusedGraphExecuteInfo
	let opspec = OpSpec(
		type: "RemoteFusedGraphExecute",
		name: (operationName ?? "Type"),
		input: [inputs],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Update ' * var' according to the RMSProp algorithm.
///Note that in dense implementation of this algorithm, ms and mom will
/// update even if the grad is zero, but in this sparse implementation, ms
/// and mom will not update in iterations during which the grad is zero.
/// 
/// mean_square = decay  *  mean_square + (1-decay)  *  gradient  *  *  2
/// Delta = learning_rate  *  gradient / sqrt(mean_square + epsilon)
/// 
/// ms <- rho  *  ms_{t-1} + (1-rho)  *  grad  *  grad
/// mom <- momentum  *  mom_{t-1} + lr  *  grad / sqrt(ms + epsilon)
/// var <- var - mom
/// - Parameter `var`: Should be from a Variable().
/// - Parameter ms: Should be from a Variable().
/// - Parameter mom: Should be from a Variable().
/// - Parameter lr: Scaling factor. Must be a scalar.
/// - Parameter rho: Decay rate. Must be a scalar.
/// - Parameter momentum: 
/// - Parameter epsilon: Ridge term. Must be a scalar.
/// - Parameter grad: The gradient.
/// - Parameter indices: A vector of indices into the first dimension of var, ms and mom.
/// - Parameter tindices: 
/// - Parameter useLocking: If `True`, updating of the var, ms, and mom tensors is protected
/// by a lock; otherwise the behavior is undefined, but may exhibit less
/// contention.
public func resourceSparseApplyRMSProp(operationName: String? = nil, `var`: Output, ms: Output, mom: Output, lr: Output, rho: Output, momentum: Output, epsilon: Output, grad: Output, indices: Output, tindices: Any.Type, useLocking: Bool) throws -> Operation { 
	var attrs = [String : Any]()
	attrs["Tindices"] = tindices
	attrs["use_locking"] = useLocking
	let opspec = OpSpec(
		type: "ResourceSparseApplyRMSProp",
		name: (operationName ?? "Type"),
		input: [`var`, ms, mom, lr, rho, momentum, epsilon, grad, indices],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Converts each string in the input Tensor to the specified numeric type.
///(Note that int32 overflow results in an error while float overflow
/// results in a rounded value.)
/// - Parameter stringTensor: 
/// - Parameter outType: The numeric type to interpret each string in `string_tensor` as.
/// - Returns: 
///	output: A Tensor of the same shape as the input `string_tensor`.
public func stringToNumber(operationName: String? = nil, stringTensor: Output, outType: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["out_type"] = outType
	let opspec = OpSpec(
		type: "StringToNumber",
		name: (operationName ?? "Type"),
		input: [stringTensor],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Convert JSON-encoded Example records to binary protocol buffer strings.
///This op translates a tensor containing Example records, encoded using
/// the [standard JSON
/// mapping](https://developers.google.com/protocol-buffers/docs/proto3#json),
/// into a tensor containing the same records encoded as binary protocol
/// buffers. The resulting tensor can then be fed to any of the other
/// Example-parsing ops.
/// - Parameter jsonExamples: Each string is a JSON object serialized according to the JSON
/// mapping of the Example proto.
/// - Returns: 
///	binary_examples: Each string is a binary Example protocol buffer corresponding
/// to the respective element of `json_examples`.
public func decodeJSONExample(operationName: String? = nil, jsonExamples: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "DecodeJSONExample",
		name: (operationName ?? "Type"),
		input: [jsonExamples],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Divides a variable reference by sparse updates.
///This operation computes
/// 
/// ```python
///     # Scalar indices
///     ref[indices, ...] /= updates[...]
/// 
///     # Vector indices (for each i)
///     ref[indices[i], ...] /= updates[i, ...]
/// 
///     # High rank indices (for each i, ..., j)
///     ref[indices[i, ..., j], ...] /= updates[i, ..., j, ...]
/// ```
/// 
/// This operation outputs `ref` after the update is done.
/// This makes it easier to chain operations that need to use the reset value.
/// 
/// Duplicate entries are handled correctly: if multiple `indices` reference
/// the same location, their contributions divide.
/// 
/// Requires `updates.shape = indices.shape + ref.shape[1:]`.
/// - Parameter ref: Should be from a `Variable` node.
/// - Parameter indices: A tensor of indices into the first dimension of `ref`.
/// - Parameter updates: A tensor of values that `ref` is divided by.
/// - Parameter tindices: 
/// - Parameter useLocking: If True, the operation will be protected by a lock;
/// otherwise the behavior is undefined, but may exhibit less contention.
/// - Returns: 
///	output_ref: = Same as `ref`.  Returned as a convenience for operations that want
/// to use the updated values after the update is done.
public func scatterDiv(operationName: String? = nil, ref: Output, indices: Output, updates: Output, tindices: Any.Type, useLocking: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Tindices"] = tindices
	attrs["use_locking"] = useLocking
	let opspec = OpSpec(
		type: "ScatterDiv",
		name: (operationName ?? "Type"),
		input: [ref, indices, updates],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Transforms a Tensor into a serialized TensorProto proto.
/// - Parameter tensor: A Tensor of type `T`.
/// - Returns: 
///	serialized: A serialized TensorProto proto of the input tensor.
public func serializeTensor(operationName: String? = nil, tensor: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "SerializeTensor",
		name: (operationName ?? "Type"),
		input: [tensor],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Performs beam search decoding on the logits given in input.
///A note about the attribute merge_repeated: For the beam search decoder,
/// this means that if consecutive entries in a beam are the same, only
/// the first of these is emitted.  That is, when the top path is "A B B B B",
/// "A B" is returned if merge_repeated = True but "A B B B B" is
/// returned if merge_repeated = False.
/// - Parameter inputs: 3-D, shape: `(max_time x batch_size x num_classes)`, the logits.
/// - Parameter sequenceLength: A vector containing sequence lengths, size `(batch)`.
/// - Parameter beamWidth: A scalar >= 0 (beam search beam width).
/// - Parameter topPaths: A scalar >= 0, <= beam_width (controls output size).
/// - Parameter mergeRepeated: If true, merge repeated classes in output.
/// - Returns: 
///	decoded_indices: A list (length: top_paths) of indices matrices.  Matrix j,
/// size `(total_decoded_outputs[j] x 2)`, has indices of a
/// `SparseTensor<int64, 2>`.  The rows store: [batch, time].
///	decoded_values: A list (length: top_paths) of values vectors.  Vector j,
/// size `(length total_decoded_outputs[j])`, has the values of a
/// `SparseTensor<int64, 2>`.  The vector stores the decoded classes for beam j.
///	decoded_shape: A list (length: top_paths) of shape vector.  Vector j,
/// size `(2)`, stores the shape of the decoded `SparseTensor[j]`.
/// Its values are: `[batch_size, max_decoded_length[j]]`.
///	log_probability: A matrix, shaped: `(batch_size x top_paths)`.  The
/// sequence log-probabilities.
public func cTCBeamSearchDecoder(operationName: String? = nil, inputs: Output, sequenceLength: Output, beamWidth: UInt8, topPaths: UInt8, mergeRepeated: Bool) throws -> (decodedIndices: Output, decodedValues: Output, decodedShape: Output, logProbability: Output) { 
	var attrs = [String : Any]()
	attrs["beam_width"] = beamWidth
	attrs["top_paths"] = topPaths
	attrs["merge_repeated"] = mergeRepeated
	let opspec = OpSpec(
		type: "CTCBeamSearchDecoder",
		name: (operationName ?? "Type"),
		input: [inputs, sequenceLength],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (decodedIndices: op.output(at: 0), decodedValues: op.output(at: 1), decodedShape: op.output(at: 2), logProbability: op.output(at: 3))
} 

///Transforms a serialized tensorflow.TensorProto proto into a Tensor.
/// - Parameter serialized: A scalar string containing a serialized TensorProto proto.
/// - Parameter outType: The type of the serialized tensor.  The provided type must match the
/// type of the serialized tensor and no implicit conversion will take place.
/// - Returns: 
///	output: A Tensor of type `out_type`.
public func parseTensor(operationName: String? = nil, serialized: Output, outType: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["out_type"] = outType
	let opspec = OpSpec(
		type: "ParseTensor",
		name: (operationName ?? "Type"),
		input: [serialized],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes fingerprints of the input strings.
/// - Parameter input: vector of strings to compute fingerprints on.
/// - Returns: 
///	output: a (N,2) shaped matrix where N is the number of elements in the input
/// vector. Each row contains the low and high parts of the fingerprint.
public func sdcaFprint(operationName: String? = nil, input: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "SdcaFprint",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Reinterpret the bytes of a string as a vector of numbers.
/// - Parameter bytes: All the elements must have the same length.
/// - Parameter outType: 
/// - Parameter littleEndian: Whether the input `bytes` are in little-endian order.
/// Ignored for `out_type` values that are stored in a single byte like
/// `uint8`.
/// - Returns: 
///	output: A Tensor with one more dimension than the input `bytes`.  The
/// added dimension will have size equal to the length of the elements
/// of `bytes` divided by the number of bytes to represent `out_type`.
public func decodeRaw(operationName: String? = nil, bytes: Output, outType: Any.Type, littleEndian: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["out_type"] = outType
	attrs["little_endian"] = littleEndian
	let opspec = OpSpec(
		type: "DecodeRaw",
		name: (operationName ?? "Type"),
		input: [bytes],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Saves input tensors slices to disk.
///This is like `Save` except that tensors can be listed in the saved file as being
/// a slice of a larger tensor.  `shapes_and_slices` specifies the shape of the
/// larger tensor and the slice that this tensor covers. `shapes_and_slices` must
/// have as many elements as `tensor_names`.
/// 
/// Elements of the `shapes_and_slices` input must either be:
/// 
///  *   The empty string, in which case the corresponding tensor is
///    saved normally.
///  *   A string of the form `dim0 dim1 ... dimN-1 slice-spec` where the
///    `dimI` are the dimensions of the larger tensor and `slice-spec`
///    specifies what part is covered by the tensor to save.
/// 
/// `slice-spec` itself is a `:`-separated list: `slice0:slice1:...:sliceN-1`
/// where each `sliceI` is either:
/// 
///  *   The string `-` meaning that the slice covers all indices of this dimension
///  *   `start,length` where `start` and `length` are integers.  In that
///    case the slice covers `length` indices starting at `start`.
/// 
/// See also `Save`.
/// - Parameter filename: Must have a single element. The name of the file to which we write the
/// tensor.
/// - Parameter tensorNames: Shape `[N]`. The names of the tensors to be saved.
/// - Parameter shapesAndSlices: Shape `[N]`.  The shapes and slice specifications to use when
/// saving the tensors.
/// - Parameter data: `N` tensors to save.
/// - Parameter t: 
public func saveSlices(operationName: String? = nil, filename: Output, tensorNames: Output, shapesAndSlices: Output, data: Output, t: [Any.Type]) throws -> Operation { 
	var attrs = [String : Any]()
	attrs["T"] = t
	let opspec = OpSpec(
		type: "SaveSlices",
		name: (operationName ?? "Type"),
		input: [filename, tensorNames, shapesAndSlices, data],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 


/// - Parameter input: 
/// - Returns: 
///	output: 
public func batchIFFT3D(operationName: String? = nil, input: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "BatchIFFT3D",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 


/// - Parameter input: 
/// - Returns: 
///	output: 
public func batchFFT3D(operationName: String? = nil, input: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "BatchFFT3D",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Real-valued fast Fourier transform.
///Computes the 1-dimensional discrete Fourier transform of a real-valued signal
/// over the inner-most dimension of `input`.
/// 
/// Since the DFT of a real signal is Hermitian-symmetric, `RFFT` only returns the
/// `fft_length / 2 + 1` unique components of the FFT: the zero-frequency term,
/// followed by the `fft_length / 2` positive-frequency terms.
/// 
/// Along the axis `RFFT` is computed on, if `fft_length` is smaller than the
/// corresponding dimension of `input`, the dimension is cropped. If it is larger,
/// the dimension is padded with zeros.
/// - Parameter input: A float32 tensor.
/// - Parameter fftLength: An int32 tensor of shape [1]. The FFT length.
/// - Returns: 
///	output: A complex64 tensor of the same rank as `input`. The inner-most
///   dimension of `input` is replaced with the `fft_length / 2 + 1` unique
///   frequency components of its 1D Fourier transform.
/// 
/// @compatibility(numpy)
/// Equivalent to np.fft.rfft
/// @end_compatibility
public func rfft(operationName: String? = nil, input: Output, fftLength: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "RFFT",
		name: (operationName ?? "Type"),
		input: [input, fftLength],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Inverse 3D fast Fourier transform.
///Computes the inverse 3-dimensional discrete Fourier transform over the
/// inner-most 3 dimensions of `input`.
/// - Parameter input: A complex64 tensor.
/// - Returns: 
///	output: A complex64 tensor of the same shape as `input`. The inner-most 3
///   dimensions of `input` are replaced with their inverse 3D Fourier transform.
/// 
/// @compatibility(numpy)
/// Equivalent to np.fft.ifftn with 3 dimensions.
/// @end_compatibility
public func ifft3D(operationName: String? = nil, input: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "IFFT3D",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///3D fast Fourier transform.
///Computes the 3-dimensional discrete Fourier transform over the inner-most 3
/// dimensions of `input`.
/// - Parameter input: A complex64 tensor.
/// - Returns: 
///	output: A complex64 tensor of the same shape as `input`. The inner-most 3
///   dimensions of `input` are replaced with their 3D Fourier transform.
/// 
/// @compatibility(numpy)
/// Equivalent to np.fft.fftn with 3 dimensions.
/// @end_compatibility
public func fft3D(operationName: String? = nil, input: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "FFT3D",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes gradients of the maxpooling function.
/// - Parameter input: The original input.
/// - Parameter grad: 4-D with shape `[batch, height, width, channels]`.  Gradients w.r.t. the
/// output of `max_pool`.
/// - Parameter argmax: The indices of the maximum values chosen for each output of `max_pool`.
/// - Parameter ksize: The size of the window for each dimension of the input tensor.
/// - Parameter strides: The stride of the sliding window for each dimension of the
/// input tensor.
/// - Parameter padding: The type of padding algorithm to use.
/// - Parameter targmax: 
/// - Returns: 
///	output: Gradients w.r.t. the input of `max_pool`.
public func maxPoolGradWithArgmax(operationName: String? = nil, input: Output, grad: Output, argmax: Output, ksize: [Int64], strides: [Int64], padding: String, targmax: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["ksize"] = ksize
	attrs["strides"] = strides
	attrs["padding"] = padding
	attrs["Targmax"] = targmax
	let opspec = OpSpec(
		type: "MaxPoolGradWithArgmax",
		name: (operationName ?? "Type"),
		input: [input, grad, argmax],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///2D fast Fourier transform.
///Computes the 2-dimensional discrete Fourier transform over the inner-most
/// 2 dimensions of `input`.
/// - Parameter input: A complex64 tensor.
/// - Returns: 
///	output: A complex64 tensor of the same shape as `input`. The inner-most 2
///   dimensions of `input` are replaced with their 2D Fourier transform.
/// 
/// @compatibility(numpy)
/// Equivalent to np.fft.fft2
/// @end_compatibility
public func fft2D(operationName: String? = nil, input: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "FFT2D",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///The gradient of SparseFillEmptyRows.
///Takes vectors reverse_index_map, shaped `[N]`, and grad_values,
/// shaped `[N_full]`, where `N_full >= N` and copies data into either
/// `d_values` or `d_default_value`.  Here `d_values` is shaped `[N]` and
/// `d_default_value` is a scalar.
/// 
///   d_values[j] = grad_values[reverse_index_map[j]]
///   d_default_value = sum_{k : 0 .. N_full - 1} (
///      grad_values[k]  *  1{k not in reverse_index_map})
/// - Parameter reverseIndexMap: 1-D.  The reverse index map from SparseFillEmptyRows.
/// - Parameter gradValues: 1-D.  The gradients from backprop.
/// - Returns: 
///	d_values: 1-D.  The backprop into values.
///	d_default_value: 0-D.  The backprop into default_value.
public func sparseFillEmptyRowsGrad(operationName: String? = nil, reverseIndexMap: Output, gradValues: Output) throws -> (dValues: Output, dDefaultValue: Output) { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "SparseFillEmptyRowsGrad",
		name: (operationName ?? "Type"),
		input: [reverseIndexMap, gradValues],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (dValues: op.output(at: 0), dDefaultValue: op.output(at: 1))
} 

///Update ' * var' according to the Adam algorithm.
///lr_t <- learning_rate  *  sqrt(1 - beta2// ^t) / (1 - beta1// ^t)
/// m_t <- beta1  *  m_{t-1} + (1 - beta1)  *  g_t
/// v_t <- beta2  *  v_{t-1} + (1 - beta2)  *  g_t  *  g_t
/// variable <- variable - lr_t  *  m_t / (sqrt(v_t) + epsilon)
/// - Parameter `var`: Should be from a Variable().
/// - Parameter m: Should be from a Variable().
/// - Parameter v: Should be from a Variable().
/// - Parameter beta1Power: Must be a scalar.
/// - Parameter beta2Power: Must be a scalar.
/// - Parameter lr: Scaling factor. Must be a scalar.
/// - Parameter beta1: Momentum factor. Must be a scalar.
/// - Parameter beta2: Momentum factor. Must be a scalar.
/// - Parameter epsilon: Ridge term. Must be a scalar.
/// - Parameter grad: The gradient.
/// - Parameter useLocking: If `True`, updating of the var, m, and v tensors will be protected
/// by a lock; otherwise the behavior is undefined, but may exhibit less
/// contention.
/// - Parameter useNesterov: If `True`, uses the nesterov update.
/// - Returns: 
///	out: Same as "var".
public func applyAdam(operationName: String? = nil, `var`: Output, m: Output, v: Output, beta1Power: Output, beta2Power: Output, lr: Output, beta1: Output, beta2: Output, epsilon: Output, grad: Output, useLocking: Bool, useNesterov: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["use_locking"] = useLocking
	attrs["use_nesterov"] = useNesterov
	let opspec = OpSpec(
		type: "ApplyAdam",
		name: (operationName ?? "Type"),
		input: [`var`, m, v, beta1Power, beta2Power, lr, beta1, beta2, epsilon, grad],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Adds a value to the current value of a variable.
///Any ReadVariableOp which depends directly or indirectly on this assign is
/// guaranteed to see the incremented value or a subsequent newer one.
/// 
/// Outputs the incremented value, which can be used to totally order the
/// increments to this variable.
/// - Parameter resource: handle to the resource in which to store the variable.
/// - Parameter value: the value by which the variable will be incremented.
/// - Parameter dtype: the dtype of the value.
public func assignAddVariableOp(operationName: String? = nil, resource: Output, value: Output, dtype: Any.Type) throws -> Operation { 
	var attrs = [String : Any]()
	attrs["dtype"] = dtype
	let opspec = OpSpec(
		type: "AssignAddVariableOp",
		name: (operationName ?? "Type"),
		input: [resource, value],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Merges summaries.
///This op creates a
/// [`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
/// protocol buffer that contains the union of all the values in the input
/// summaries.
/// 
/// When the Op is run, it reports an `InvalidArgument` error if multiple values
/// in the summaries to merge use the same tag.
/// - Parameter inputs: Can be of any shape.  Each must contain serialized `Summary` protocol
/// buffers.
/// - Parameter n: 
/// - Returns: 
///	summary: Scalar. Serialized `Summary` protocol buffer.
public func mergeSummary(operationName: String? = nil, inputs: [Output], n: UInt8) throws -> Output { 
	var attrs = [String : Any]()
	attrs["N"] = n
	let opspec = OpSpec(
		type: "MergeSummary",
		name: (operationName ?? "Type"),
		input: [inputs],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Creates a dataset that batches and pads `batch_size` elements from the input.
/// - Parameter inputDataset: 
/// - Parameter batchSize: A scalar representing the number of elements to accumulate in a
/// batch.
/// - Parameter paddedShapes: A list of int64 tensors representing the desired padded shapes
/// of the corresponding output components. These shapes may be partially
/// specified, using `-1` to indicate that a particular dimension should be
/// padded to the maximum size of all batch elements.
/// - Parameter paddingValues: A list of scalars containing the padding value to use for
/// each of the outputs.
/// - Parameter toutputTypes: 
/// - Parameter outputShapes: 
/// - Parameter n: 
/// - Returns: 
///	handle: 
public func paddedBatchDataset(operationName: String? = nil, inputDataset: Output, batchSize: Output, paddedShapes: [Output], paddingValues: Output, toutputTypes: [Any.Type], outputShapes: [Shape], n: UInt8) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Toutput_types"] = toutputTypes
	attrs["output_shapes"] = outputShapes
	attrs["N"] = n
	let opspec = OpSpec(
		type: "PaddedBatchDataset",
		name: (operationName ?? "Type"),
		input: [inputDataset, batchSize, paddedShapes, paddingValues],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///A stack that produces elements in first-in last-out order.
/// - Parameter maxSize: The maximum size of the stack if non-negative. If negative, the stack
/// size is unlimited.
/// - Parameter elemType: The type of the elements on the stack.
/// - Parameter stackName: Overrides the name used for the temporary stack resource. Default
/// value is the name of the 'Stack' op (which is guaranteed unique).
/// - Returns: 
///	handle: The handle to the stack.
public func stackV2(operationName: String? = nil, maxSize: Output, elemType: Any.Type, stackName: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["elem_type"] = elemType
	attrs["stack_name"] = stackName
	let opspec = OpSpec(
		type: "StackV2",
		name: (operationName ?? "Type"),
		input: [maxSize],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Outputs a `Summary` protocol buffer with audio.
///The summary has up to `max_outputs` summary values containing audio. The
/// audio is built from `tensor` which must be 3-D with shape `[batch_size,
/// frames, channels]` or 2-D with shape `[batch_size, frames]`. The values are
/// assumed to be in the range of `[-1.0, 1.0]` with a sample rate of `sample_rate`.
/// 
/// The `tag` argument is a scalar `Tensor` of type `string`.  It is used to
/// build the `tag` of the summary values:
/// 
///  *   If `max_outputs` is 1, the summary value tag is ' * tag * /audio'.
///  *   If `max_outputs` is greater than 1, the summary value tags are
///    generated sequentially as ' * tag * /audio/0', ' * tag * /audio/1', etc.
/// - Parameter tag: Scalar. Used to build the `tag` attribute of the summary values.
/// - Parameter tensor: 2-D of shape `[batch_size, frames]`.
/// - Parameter sampleRate: The sample rate of the signal in hertz.
/// - Parameter maxOutputs: Max number of batch elements to generate audio for.
/// - Returns: 
///	summary: Scalar. Serialized `Summary` protocol buffer.
public func audioSummary(operationName: String? = nil, tag: Output, tensor: Output, sampleRate: Float, maxOutputs: UInt8) throws -> Output { 
	var attrs = [String : Any]()
	attrs["sample_rate"] = sampleRate
	attrs["max_outputs"] = maxOutputs
	let opspec = OpSpec(
		type: "AudioSummary",
		name: (operationName ?? "Type"),
		input: [tag, tensor],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes the complementary error function of `x` element-wise.
/// - Parameter x: 
/// - Returns: 
///	y: 
public func erfc(operationName: String? = nil, x: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Erfc",
		name: (operationName ?? "Type"),
		input: [x],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Outputs random integers from a uniform distribution.
///The generated values are uniform integers in the range `[minval, maxval)`.
/// The lower bound `minval` is included in the range, while the upper bound
/// `maxval` is excluded.
/// 
/// The random integers are slightly biased unless `maxval - minval` is an exact
/// power of two.  The bias is small for values of `maxval - minval` significantly
/// smaller than the range of the output (either `2// ^32` or `2// ^64`).
/// - Parameter shape: The shape of the output tensor.
/// - Parameter minval: 0-D.  Inclusive lower bound on the generated integers.
/// - Parameter maxval: 0-D.  Exclusive upper bound on the generated integers.
/// - Parameter seed: If either `seed` or `seed2` are set to be non-zero, the random number
/// generator is seeded by the given seed.  Otherwise, it is seeded by a
/// random seed.
/// - Parameter seed2: A second seed to avoid seed collision.
/// - Parameter tout: 
/// - Returns: 
///	output: A tensor of the specified shape filled with uniform random integers.
public func randomUniformInt(operationName: String? = nil, shape: Output, minval: Output, maxval: Output, seed: UInt8, seed2: UInt8, tout: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["seed"] = seed
	attrs["seed2"] = seed2
	attrs["Tout"] = tout
	let opspec = OpSpec(
		type: "RandomUniformInt",
		name: (operationName ?? "Type"),
		input: [shape, minval, maxval],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Op removes and returns the values associated with the key
///from the underlying container.   If the underlying container
/// does not contain this key, the op will block until it does.
/// - Parameter key: 
/// - Parameter indices: 
/// - Parameter capacity: 
/// - Parameter memoryLimit: 
/// - Parameter dtypes: 
/// - Parameter container: 
/// - Parameter sharedName: 
/// - Returns: 
///	values: 
public func mapUnstage(operationName: String? = nil, key: Output, indices: Output, capacity: UInt8, memoryLimit: UInt8, dtypes: [Any.Type], container: String, sharedName: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["capacity"] = capacity
	attrs["memory_limit"] = memoryLimit
	attrs["dtypes"] = dtypes
	attrs["container"] = container
	attrs["shared_name"] = sharedName
	let opspec = OpSpec(
		type: "MapUnstage",
		name: (operationName ?? "Type"),
		input: [key, indices],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Outputs a `Summary` protocol buffer with a tensor and per-plugin data.
/// - Parameter tag: A string attached to this summary. Used for organization in TensorBoard.
/// - Parameter tensor: A tensor to serialize.
/// - Parameter serializedSummaryMetadata: A serialized SummaryMetadata proto. Contains plugin
/// data.
/// - Returns: 
///	summary: 
public func tensorSummaryV2(operationName: String? = nil, tag: Output, tensor: Output, serializedSummaryMetadata: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "TensorSummaryV2",
		name: (operationName ?? "Type"),
		input: [tag, tensor, serializedSummaryMetadata],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Quantizes then dequantizes a tensor.
///This op simulates the precision loss from the quantized forward pass by:
/// 1. Quantizing the tensor to fixed point numbers, which should match the target
///    quantization method when it is used in inference.
/// 2. Dequantizing it back to floating point numbers for the following ops, most
///    likely matmul.
/// 
/// There are different ways to quantize. This version does not use the full range
/// of the output type, choosing to elide the lowest possible value for symmetry
/// (e.g., output range is -127 to 127, not -128 to 127 for signed 8 bit
/// quantization), so that 0.0 maps to 0.
/// 
/// To perform this op, we first find the range of values in our tensor. The range
/// we use is always centered on 0, so we find m such that
/// 
/// 1. m = max(abs(input_min), abs(input_max)) if range_given is true,
/// 2. m = max(abs(min_elem(input)), abs(max_elem(input))) otherwise.
/// 
/// Our input tensor range is then [-m, m].
/// 
/// Next, we choose our fixed-point quantization buckets, [min_fixed, max_fixed].
/// If signed_input is true, this is
/// 
///   [min_fixed, max_fixed ] =
///       [-(1 << (num_bits - 1) - 1), (1 << (num_bits - 1)) - 1].
/// 
/// Otherwise, if signed_input is false, the fixed-point range is
/// 
///   [min_fixed, max_fixed] = [0, (1 << num_bits) - 1].
/// 
/// From this we compute our scaling factor, s:
/// 
///   s = (max_fixed - min_fixed) / (2  *  m).
/// 
/// Now we can quantize and dequantize the elements of our tensor.  An element e
/// is transformed into e':
/// 
///   e' = (e  *  s).round_to_nearest() / s.
/// 
/// Note that we have a different number of buckets in the signed vs. unsigned
/// cases.  For example, if num_bits == 8, we get 254 buckets in the signed case
/// vs. 255 in the unsigned case.
/// 
/// For example, suppose num_bits = 8 and m = 1.  Then
/// 
///   [min_fixed, max_fixed] = [-127, 127], and
///   s = (127 + 127) / 2 = 127.
/// 
/// Given the vector {-1, -0.5, 0, 0.3}, this is quantized to
/// {-127, -63, 0, 38}, and dequantized to {-1, -63.0/127, 0, 38.0/127}.
/// - Parameter input: Tensor to quantize and then dequantize.
/// - Parameter inputMin: If range_given, this is the min of the range, otherwise this input
/// will be ignored.
/// - Parameter inputMax: If range_given, this is the max of the range, otherwise this input
/// will be ignored.
/// - Parameter signedInput: If the quantization is signed or unsigned.
/// - Parameter numBits: The bitwidth of the quantization.
/// - Parameter rangeGiven: If the range is given or should be computed from the tensor.
/// - Returns: 
///	output: 
public func quantizeAndDequantizeV2(operationName: String? = nil, input: Output, inputMin: Output, inputMax: Output, signedInput: Bool, numBits: UInt8, rangeGiven: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["signed_input"] = signedInput
	attrs["num_bits"] = numBits
	attrs["range_given"] = rangeGiven
	let opspec = OpSpec(
		type: "QuantizeAndDequantizeV2",
		name: (operationName ?? "Type"),
		input: [input, inputMin, inputMax],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Prints a list of tensors.
///Passes `input` through to `output` and prints `data` when evaluating.
/// - Parameter input: The tensor passed to `output`
/// - Parameter data: A list of tensors to print out when op is evaluated.
/// - Parameter u: 
/// - Parameter message: A string, prefix of the error message.
/// - Parameter firstN: Only log `first_n` number of times. -1 disables logging.
/// - Parameter summarize: Only print this many entries of each tensor.
/// - Returns: 
///	output: = The unmodified `input` tensor
public func print(operationName: String? = nil, input: Output, data: Output, u: [Any.Type], message: String, firstN: UInt8, summarize: UInt8) throws -> Output { 
	var attrs = [String : Any]()
	attrs["U"] = u
	attrs["message"] = message
	attrs["first_n"] = firstN
	attrs["summarize"] = summarize
	let opspec = OpSpec(
		type: "Print",
		name: (operationName ?? "Type"),
		input: [input, data],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 


/// - Parameter handle: 
/// - Parameter index: 
/// - Parameter value: 
/// - Parameter flowIn: 
/// - Returns: 
///	flow_out: 
public func tensorArrayWrite(operationName: String? = nil, handle: Output, index: Output, value: Output, flowIn: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "TensorArrayWrite",
		name: (operationName ?? "Type"),
		input: [handle, index, value, flowIn],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Asserts that the given condition is true.
///If `condition` evaluates to false, print the list of tensors in `data`.
/// `summarize` determines how many entries of the tensors to print.
/// - Parameter condition: The condition to evaluate.
/// - Parameter data: The tensors to print out when condition is false.
/// - Parameter t: 
/// - Parameter summarize: Print this many entries of each tensor.
public func assert(operationName: String? = nil, condition: Output, data: Output, t: [Any.Type], summarize: UInt8) throws -> Operation { 
	var attrs = [String : Any]()
	attrs["T"] = t
	attrs["summarize"] = summarize
	let opspec = OpSpec(
		type: "Assert",
		name: (operationName ?? "Type"),
		input: [condition, data],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Interleave the values from the `data` tensors into a single tensor.
///Builds a merged tensor such that
/// 
/// ```python
///     merged[indices[m][i, ..., j], ...] = data[m][i, ..., j, ...]
/// ```
/// 
/// For example, if each `indices[m]` is scalar or vector, we have
/// 
/// ```python
///     # Scalar indices:
///     merged[indices[m], ...] = data[m][...]
/// 
///     # Vector indices:
///     merged[indices[m][i], ...] = data[m][i, ...]
/// ```
/// 
/// Each `data[i].shape` must start with the corresponding `indices[i].shape`,
/// and the rest of `data[i].shape` must be constant w.r.t. `i`.  That is, we
/// must have `data[i].shape = indices[i].shape + constant`.  In terms of this
/// `constant`, the output shape is
/// 
///     merged.shape = [max(indices)] + constant
/// 
/// Values may be merged in parallel, so if an index appears in both `indices[m][i]`
/// and `indices[n][j]`, the result may be invalid. This differs from the normal
/// DynamicStitch operator that defines the behavior in that case.
/// 
/// For example:
/// 
/// ```python
///     indices[0] = 6
///     indices[1] = [4, 1]
///     indices[2] = [[5, 2], [0, 3]]
///     data[0] = [61, 62]
///     data[1] = [[41, 42], [11, 12]]
///     data[2] = [[[51, 52], [21, 22]], [[1, 2], [31, 32]]]
///     merged = [[1, 2], [11, 12], [21, 22], [31, 32], [41, 42],
///               [51, 52], [61, 62]]
/// ```
/// 
/// This method can be used to merge partitions created by `dynamic_partition`
/// as illustrated on the following example:
/// 
/// ```python
///     # Apply function (increments x_i) on elements for which a certain condition
///     # apply (x_i != -1 in this example).
///     x=tf.constant([0.1, -1., 5.2, 4.3, -1., 7.4])
///     condition_mask=tf.not_equal(x,tf.constant(-1.))
///     partitioned_data = tf.dynamic_partition(
///         x, tf.cast(condition_mask, tf.int32) , 2)
///     partitioned_data[1] = partitioned_data[1] + 1.0
///     condition_indices = tf.dynamic_partition(
///         tf.range(tf.shape(x)[0]), tf.cast(condition_mask, tf.int32) , 2)
///     x = tf.dynamic_stitch(condition_indices, partitioned_data)
///     # Here x=[1.1, -1., 6.2, 5.3, -1, 8.4], the -1. values remain
///     # unchanged.
/// ```
/// 
/// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
/// <img style="width:100%" src="https://www.tensorflow.org/images/DynamicStitch.png" alt>
/// </div>
/// - Parameter indices: 
/// - Parameter data: 
/// - Parameter n: 
/// - Returns: 
///	merged: 
public func parallelDynamicStitch(operationName: String? = nil, indices: [Output], data: [Output], n: UInt8) throws -> Output { 
	var attrs = [String : Any]()
	attrs["N"] = n
	let opspec = OpSpec(
		type: "ParallelDynamicStitch",
		name: (operationName ?? "Type"),
		input: [indices, data],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Decode a PNG-encoded image to a uint8 or uint16 tensor.
///The attr `channels` indicates the desired number of color channels for the
/// decoded image.
/// 
/// Accepted values are:
/// 
///  *    0: Use the number of channels in the PNG-encoded image.
///  *    1: output a grayscale image.
///  *    3: output an RGB image.
///  *    4: output an RGBA image.
/// 
/// If needed, the PNG-encoded image is transformed to match the requested number
/// of color channels.
/// 
/// This op also supports decoding JPEGs and non-animated GIFs since the interface
/// is the same, though it is cleaner to use `tf.image.decode_image`.
/// - Parameter contents: 0-D.  The PNG-encoded image.
/// - Parameter channels: Number of color channels for the decoded image.
/// - Parameter dtype: 
/// - Returns: 
///	image: 3-D with shape `[height, width, channels]`.
public func decodePng(operationName: String? = nil, contents: Output, channels: UInt8, dtype: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["channels"] = channels
	attrs["dtype"] = dtype
	let opspec = OpSpec(
		type: "DecodePng",
		name: (operationName ?? "Type"),
		input: [contents],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Initializes a table from a text file.
///It inserts one key-value pair into the table for each line of the file.
/// The key and value is extracted from the whole line content, elements from the
/// split line based on `delimiter` or the line number (starting from zero).
/// Where to extract the key and value from a line is specified by `key_index` and
/// `value_index`.
/// 
/// - A value of -1 means use the line number(starting from zero), expects `int64`.
/// - A value of -2 means use the whole line content, expects `string`.
/// - A value >= 0 means use the index (starting at zero) of the split line based
///   on `delimiter`.
/// - Parameter tableHandle: Handle to a table which will be initialized.
/// - Parameter filename: Filename of a vocabulary text file.
/// - Parameter keyIndex: Column index in a line to get the table `key` values from.
/// - Parameter valueIndex: Column index that represents information of a line to get the table
/// `value` values from.
/// - Parameter vocabSize: Number of elements of the file, use -1 if unknown.
/// - Parameter delimiter: Delimiter to separate fields in a line.
public func initializeTableFromTextFile(operationName: String? = nil, tableHandle: Output, filename: Output, keyIndex: UInt8, valueIndex: UInt8, vocabSize: UInt8, delimiter: String) throws -> Operation { 
	var attrs = [String : Any]()
	attrs["key_index"] = keyIndex
	attrs["value_index"] = valueIndex
	attrs["vocab_size"] = vocabSize
	attrs["delimiter"] = delimiter
	let opspec = OpSpec(
		type: "InitializeTableFromTextFile",
		name: (operationName ?? "Type"),
		input: [tableHandle, filename],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Makes its input available to the next iteration.
/// - Parameter data: The tensor to be made available to the next iteration.
/// - Returns: 
///	output: The same tensor as `data`.
public func nextIteration(operationName: String? = nil, data: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "NextIteration",
		name: (operationName ?? "Type"),
		input: [data],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Table initializer that takes two tensors for keys and values respectively.
/// - Parameter tableHandle: Handle to a table which will be initialized.
/// - Parameter keys: Keys of type Tkey.
/// - Parameter values: Values of type Tval.
/// - Parameter tkey: 
/// - Parameter tval: 
public func initializeTableV2(operationName: String? = nil, tableHandle: Output, keys: Output, values: Output, tkey: Any.Type, tval: Any.Type) throws -> Operation { 
	var attrs = [String : Any]()
	attrs["Tkey"] = tkey
	attrs["Tval"] = tval
	let opspec = OpSpec(
		type: "InitializeTableV2",
		name: (operationName ?? "Type"),
		input: [tableHandle, keys, values],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Table initializer that takes two tensors for keys and values respectively.
/// - Parameter tableHandle: Handle to a table which will be initialized.
/// - Parameter keys: Keys of type Tkey.
/// - Parameter values: Values of type Tval.
/// - Parameter tkey: 
/// - Parameter tval: 
public func initializeTable(operationName: String? = nil, tableHandle: Output, keys: Output, values: Output, tkey: Any.Type, tval: Any.Type) throws -> Operation { 
	var attrs = [String : Any]()
	attrs["Tkey"] = tkey
	attrs["Tval"] = tval
	let opspec = OpSpec(
		type: "InitializeTable",
		name: (operationName ?? "Type"),
		input: [tableHandle, keys, values],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Returns the imaginary part of a complex number.
///Given a tensor `input` of complex numbers, this operation returns a tensor of
/// type `float` that is the imaginary part of each element in `input`. All
/// elements in `input` must be complex numbers of the form \\(a + bj\\), where  * a * 
/// is the real part and  * b *  is the imaginary part returned by this operation.
/// 
/// For example:
/// 
/// ```
/// # tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
/// tf.imag(input) ==> [4.75, 5.75]
/// ```
/// - Parameter input: 
/// - Parameter tout: 
/// - Returns: 
///	output: 
public func imag(operationName: String? = nil, input: Output, tout: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Tout"] = tout
	let opspec = OpSpec(
		type: "Imag",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 


/// - Parameter handle: 
/// - Parameter flowIn: 
/// - Parameter source: 
/// - Returns: 
///	grad_handle: 
public func tensorArrayGrad(operationName: String? = nil, handle: Output, flowIn: Output, source: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["source"] = source
	let opspec = OpSpec(
		type: "TensorArrayGrad",
		name: (operationName ?? "Type"),
		input: [handle, flowIn],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Creates an empty hash table that uses tensors as the backing store.
///It uses "open addressing" with quadratic reprobing to resolve
/// collisions.
/// 
/// This op creates a mutable hash table, specifying the type of its keys and
/// values. Each value must be a scalar. Data can be inserted into the table using
/// the insert operations. It does not support the initialization operation.
/// - Parameter emptyKey: The key used to represent empty key buckets internally. Must not
/// be used in insert or lookup operations.
/// - Parameter container: If non-empty, this table is placed in the given container.
/// Otherwise, a default container is used.
/// - Parameter sharedName: If non-empty, this table is shared under the given name across
/// multiple sessions.
/// - Parameter useNodeNameSharing: 
/// - Parameter keyDtype: Type of the table keys.
/// - Parameter valueDtype: Type of the table values.
/// - Parameter valueShape: The shape of each value.
/// - Parameter initialNumBuckets: The initial number of hash table buckets. Must be a power
/// to 2.
/// - Parameter maxLoadFactor: The maximum ratio between number of entries and number of
/// buckets before growing the table. Must be between 0 and 1.
/// - Returns: 
///	table_handle: Handle to a table.
public func mutableDenseHashTable(operationName: String? = nil, emptyKey: Output, container: String, sharedName: String, useNodeNameSharing: Bool, keyDtype: Any.Type, valueDtype: Any.Type, valueShape: Shape, initialNumBuckets: UInt8, maxLoadFactor: Float) throws -> Output { 
	var attrs = [String : Any]()
	attrs["container"] = container
	attrs["shared_name"] = sharedName
	attrs["use_node_name_sharing"] = useNodeNameSharing
	attrs["key_dtype"] = keyDtype
	attrs["value_dtype"] = valueDtype
	attrs["value_shape"] = valueShape
	attrs["initial_num_buckets"] = initialNumBuckets
	attrs["max_load_factor"] = maxLoadFactor
	let opspec = OpSpec(
		type: "MutableDenseHashTable",
		name: (operationName ?? "Type"),
		input: [emptyKey],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Returns a one-hot tensor.
///The locations represented by indices in `indices` take value `on_value`,
/// while all other locations take value `off_value`.
/// 
/// If the input `indices` is rank `N`, the output will have rank `N+1`,
/// The new axis is created at dimension `axis` (default: the new axis is
/// appended at the end).
/// 
/// If `indices` is a scalar the output shape will be a vector of length `depth`.
/// 
/// If `indices` is a vector of length `features`, the output shape will be:
/// ```
///   features x depth if axis == -1
///   depth x features if axis == 0
/// ```
/// 
/// If `indices` is a matrix (batch) with shape `[batch, features]`,
/// the output shape will be:
/// ```
///   batch x features x depth if axis == -1
///   batch x depth x features if axis == 1
///   depth x batch x features if axis == 0
/// ```
/// 
/// 
/// Examples
/// =========
/// 
/// Suppose that
/// 
/// ```
///   indices = [0, 2, -1, 1]
///   depth = 3
///   on_value = 5.0
///   off_value = 0.0
///   axis = -1
/// ```
/// 
/// Then output is `[4 x 3]`:
/// 
///     ```output =
///       [5.0 0.0 0.0]  // one_hot(0)
///       [0.0 0.0 5.0]  // one_hot(2)
///       [0.0 0.0 0.0]  // one_hot(-1)
///       [0.0 5.0 0.0]  // one_hot(1)
///     ```
/// 
/// Suppose that
/// 
/// ```
///   indices = [0, 2, -1, 1]
///   depth = 3
///   on_value = 0.0
///   off_value = 3.0
///   axis = 0
/// ```
/// 
/// Then output is `[3 x 4]`:
/// 
///     ```output =
///       [0.0 3.0 3.0 3.0]
///       [3.0 3.0 3.0 0.0]
///       [3.0 3.0 3.0 3.0]
///       [3.0 0.0 3.0 3.0]
///     //  // ^                one_hot(0)
///     //      // ^            one_hot(2)
///     //          // ^        one_hot(-1)
///     //              // ^    one_hot(1)
///     ```
/// Suppose that
/// 
/// ```
///   indices = [[0, 2], [1, -1]]
///   depth = 3
///   on_value = 1.0
///   off_value = 0.0
///   axis = -1
/// ```
/// 
/// Then output is `[2 x 2 x 3]`:
/// 
///     ```output =
///       [
///         [1.0, 0.0, 0.0]  // one_hot(0)
///         [0.0, 0.0, 1.0]  // one_hot(2)
///       ][
///         [0.0, 1.0, 0.0]  // one_hot(1)
///         [0.0, 0.0, 0.0]  // one_hot(-1)
///       ]```
/// - Parameter indices: A tensor of indices.
/// - Parameter depth: A scalar defining the depth of the one hot dimension.
/// - Parameter onValue: A scalar defining the value to fill in output when `indices[j] = i`.
/// - Parameter offValue: A scalar defining the value to fill in output when `indices[j] != i`.
/// - Parameter axis: The axis to fill (default: -1, a new inner-most axis).
/// - Parameter ti: 
/// - Returns: 
///	output: The one-hot tensor.
public func oneHot(operationName: String? = nil, indices: Output, depth: Output, onValue: Output, offValue: Output, axis: UInt8, ti: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["axis"] = axis
	attrs["TI"] = ti
	let opspec = OpSpec(
		type: "OneHot",
		name: (operationName ?? "Type"),
		input: [indices, depth, onValue, offValue],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Creates an empty hash table.
///This op creates a mutable hash table, specifying the type of its keys and
/// values. Each value must be a vector. Data can be inserted into the table using
/// the insert operations. It does not support the initialization operation.
/// - Parameter container: If non-empty, this table is placed in the given container.
/// Otherwise, a default container is used.
/// - Parameter sharedName: If non-empty, this table is shared under the given name across
/// multiple sessions.
/// - Parameter useNodeNameSharing: 
/// - Parameter keyDtype: Type of the table keys.
/// - Parameter valueDtype: Type of the table values.
/// - Parameter valueShape: 
/// - Returns: 
///	table_handle: Handle to a table.
public func mutableHashTableOfTensorsV2(operationName: String? = nil, container: String, sharedName: String, useNodeNameSharing: Bool, keyDtype: Any.Type, valueDtype: Any.Type, valueShape: Shape) throws -> Output { 
	var attrs = [String : Any]()
	attrs["container"] = container
	attrs["shared_name"] = sharedName
	attrs["use_node_name_sharing"] = useNodeNameSharing
	attrs["key_dtype"] = keyDtype
	attrs["value_dtype"] = valueDtype
	attrs["value_shape"] = valueShape
	let opspec = OpSpec(
		type: "MutableHashTableOfTensorsV2",
		name: (operationName ?? "Type"),
		input: [],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Creates an empty hash table.
///This op creates a mutable hash table, specifying the type of its keys and
/// values. Each value must be a scalar. Data can be inserted into the table using
/// the insert operations. It does not support the initialization operation.
/// - Parameter container: If non-empty, this table is placed in the given container.
/// Otherwise, a default container is used.
/// - Parameter sharedName: If non-empty, this table is shared under the given name across
/// multiple sessions.
/// - Parameter useNodeNameSharing: If true and shared_name is empty, the table is shared
/// using the node name.
/// - Parameter keyDtype: Type of the table keys.
/// - Parameter valueDtype: Type of the table values.
/// - Returns: 
///	table_handle: Handle to a table.
public func mutableHashTableV2(operationName: String? = nil, container: String, sharedName: String, useNodeNameSharing: Bool, keyDtype: Any.Type, valueDtype: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["container"] = container
	attrs["shared_name"] = sharedName
	attrs["use_node_name_sharing"] = useNodeNameSharing
	attrs["key_dtype"] = keyDtype
	attrs["value_dtype"] = valueDtype
	let opspec = OpSpec(
		type: "MutableHashTableV2",
		name: (operationName ?? "Type"),
		input: [],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Creates a non-initialized hash table.
///This op creates a hash table, specifying the type of its keys and values.
/// Before using the table you will have to initialize it.  After initialization the
/// table will be immutable.
/// - Parameter container: If non-empty, this table is placed in the given container.
/// Otherwise, a default container is used.
/// - Parameter sharedName: If non-empty, this table is shared under the given name across
/// multiple sessions.
/// - Parameter useNodeNameSharing: If true and shared_name is empty, the table is shared
/// using the node name.
/// - Parameter keyDtype: Type of the table keys.
/// - Parameter valueDtype: Type of the table values.
/// - Returns: 
///	table_handle: Handle to a table.
public func hashTableV2(operationName: String? = nil, container: String, sharedName: String, useNodeNameSharing: Bool, keyDtype: Any.Type, valueDtype: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["container"] = container
	attrs["shared_name"] = sharedName
	attrs["use_node_name_sharing"] = useNodeNameSharing
	attrs["key_dtype"] = keyDtype
	attrs["value_dtype"] = valueDtype
	let opspec = OpSpec(
		type: "HashTableV2",
		name: (operationName ?? "Type"),
		input: [],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Creates a non-initialized hash table.
///This op creates a hash table, specifying the type of its keys and values.
/// Before using the table you will have to initialize it.  After initialization the
/// table will be immutable.
/// - Parameter container: If non-empty, this table is placed in the given container.
/// Otherwise, a default container is used.
/// - Parameter sharedName: If non-empty, this table is shared under the given name across
/// multiple sessions.
/// - Parameter useNodeNameSharing: If true and shared_name is empty, the table is shared
/// using the node name.
/// - Parameter keyDtype: Type of the table keys.
/// - Parameter valueDtype: Type of the table values.
/// - Returns: 
///	table_handle: Handle to a table.
public func hashTable(operationName: String? = nil, container: String, sharedName: String, useNodeNameSharing: Bool, keyDtype: Any.Type, valueDtype: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["container"] = container
	attrs["shared_name"] = sharedName
	attrs["use_node_name_sharing"] = useNodeNameSharing
	attrs["key_dtype"] = keyDtype
	attrs["value_dtype"] = valueDtype
	let opspec = OpSpec(
		type: "HashTable",
		name: (operationName ?? "Type"),
		input: [],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Component-wise divides a SparseTensor by a dense Tensor.
/// * Limitation * : this Op only broadcasts the dense side to the sparse side, but not
/// the other direction.
/// - Parameter spIndices: 2-D.  `N x R` matrix with the indices of non-empty values in a
/// SparseTensor, possibly not in canonical ordering.
/// - Parameter spValues: 1-D.  `N` non-empty values corresponding to `sp_indices`.
/// - Parameter spShape: 1-D.  Shape of the input SparseTensor.
/// - Parameter dense: `R`-D.  The dense Tensor operand.
/// - Returns: 
///	output: 1-D.  The `N` values that are operated on.
public func sparseDenseCwiseDiv(operationName: String? = nil, spIndices: Output, spValues: Output, spShape: Output, dense: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "SparseDenseCwiseDiv",
		name: (operationName ?? "Type"),
		input: [spIndices, spValues, spShape, dense],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Replaces the contents of the table with the specified keys and values.
///The tensor `keys` must be of the same type as the keys of the table.
/// The tensor `values` must be of the type of the table values.
/// - Parameter tableHandle: Handle to the table.
/// - Parameter keys: Any shape.  Keys to look up.
/// - Parameter values: Values to associate with keys.
/// - Parameter tin: 
/// - Parameter tout: 
public func lookupTableImport(operationName: String? = nil, tableHandle: Output, keys: Output, values: Output, tin: Any.Type, tout: Any.Type) throws -> Operation { 
	var attrs = [String : Any]()
	attrs["Tin"] = tin
	attrs["Tout"] = tout
	let opspec = OpSpec(
		type: "LookupTableImport",
		name: (operationName ?? "Type"),
		input: [tableHandle, keys, values],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Outputs all keys and values in the table.
/// - Parameter tableHandle: Handle to the table.
/// - Parameter tkeys: 
/// - Parameter tvalues: 
/// - Returns: 
///	keys: Vector of all keys present in the table.
///	values: Tensor of all values in the table. Indexed in parallel with `keys`.
public func lookupTableExportV2(operationName: String? = nil, tableHandle: Output, tkeys: Any.Type, tvalues: Any.Type) throws -> (keys: Output, values: Output) { 
	var attrs = [String : Any]()
	attrs["Tkeys"] = tkeys
	attrs["Tvalues"] = tvalues
	let opspec = OpSpec(
		type: "LookupTableExportV2",
		name: (operationName ?? "Type"),
		input: [tableHandle],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (keys: op.output(at: 0), values: op.output(at: 1))
} 

///Computes the number of elements in the given table.
/// - Parameter tableHandle: Handle to the table.
/// - Returns: 
///	size: Scalar that contains number of elements in the table.
public func lookupTableSizeV2(operationName: String? = nil, tableHandle: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "LookupTableSizeV2",
		name: (operationName ?? "Type"),
		input: [tableHandle],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes the number of elements in the given table.
/// - Parameter tableHandle: Handle to the table.
/// - Returns: 
///	size: Scalar that contains number of elements in the table.
public func lookupTableSize(operationName: String? = nil, tableHandle: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "LookupTableSize",
		name: (operationName ?? "Type"),
		input: [tableHandle],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Updates the table to associates keys with values.
///The tensor `keys` must be of the same type as the keys of the table.
/// The tensor `values` must be of the type of the table values.
/// - Parameter tableHandle: Handle to the table.
/// - Parameter keys: Any shape.  Keys to look up.
/// - Parameter values: Values to associate with keys.
/// - Parameter tin: 
/// - Parameter tout: 
public func lookupTableInsert(operationName: String? = nil, tableHandle: Output, keys: Output, values: Output, tin: Any.Type, tout: Any.Type) throws -> Operation { 
	var attrs = [String : Any]()
	attrs["Tin"] = tin
	attrs["Tout"] = tout
	let opspec = OpSpec(
		type: "LookupTableInsert",
		name: (operationName ?? "Type"),
		input: [tableHandle, keys, values],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Computes the Cholesky decomposition of one or more square matrices.
///The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
/// form square matrices.
/// 
/// The input has to be symmetric and positive definite. Only the lower-triangular
/// part of the input will be used for this operation. The upper-triangular part
/// will not be read.
/// 
/// The output is a tensor of the same shape as the input
/// containing the Cholesky decompositions for all input submatrices `[..., :, :]`.
/// 
///  *  * Note *  * : The gradient computation on GPU is faster for large matrices but
/// not for large batch dimensions when the submatrices are small. In this
/// case it might be faster to use the CPU.
/// - Parameter input: Shape is `[..., M, M]`.
/// - Returns: 
///	output: Shape is `[..., M, M]`.
public func cholesky(operationName: String? = nil, input: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Cholesky",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 


/// - Parameter matrix: 
/// - Parameter rhs: 
/// - Parameter l2Regularizer: 
/// - Parameter fast: 
/// - Returns: 
///	output: 
public func batchMatrixSolveLs(operationName: String? = nil, matrix: Output, rhs: Output, l2Regularizer: Output, fast: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["fast"] = fast
	let opspec = OpSpec(
		type: "BatchMatrixSolveLs",
		name: (operationName ?? "Type"),
		input: [matrix, rhs, l2Regularizer],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Outputs all keys and values in the table.
/// - Parameter tableHandle: Handle to the table.
/// - Parameter tkeys: 
/// - Parameter tvalues: 
/// - Returns: 
///	keys: Vector of all keys present in the table.
///	values: Tensor of all values in the table. Indexed in parallel with `keys`.
public func lookupTableExport(operationName: String? = nil, tableHandle: Output, tkeys: Any.Type, tvalues: Any.Type) throws -> (keys: Output, values: Output) { 
	var attrs = [String : Any]()
	attrs["Tkeys"] = tkeys
	attrs["Tvalues"] = tvalues
	let opspec = OpSpec(
		type: "LookupTableExport",
		name: (operationName ?? "Type"),
		input: [tableHandle],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (keys: op.output(at: 0), values: op.output(at: 1))
} 

///Gather slices from `params` axis `axis` according to `indices`.
///`indices` must be an integer tensor of any dimension (usually 0-D or 1-D).
/// Produces an output tensor with shape `params.shape[:axis] + indices.shape +
/// params.shape[axis + 1:]` where:
/// 
/// ```python
///     # Scalar indices (output is rank(params) - 1).
///     output[a_0, ..., a_n, b_0, ..., b_n] =
///       params[a_0, ..., a_n, indices, b_0, ..., b_n]
/// 
///     # Vector indices (output is rank(params)).
///     output[a_0, ..., a_n, i, b_0, ..., b_n] =
///       params[a_0, ..., a_n, indices[i], b_0, ..., b_n]
/// 
///     # Higher rank indices (output is rank(params) + rank(indices) - 1).
///     output[a_0, ..., a_n, i, ..., j, b_0, ... b_n] =
///       params[a_0, ..., a_n, indices[i, ..., j], b_0, ..., b_n]
/// ```
/// 
/// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
/// <img style="width:100%" src="https://www.tensorflow.org/images/Gather.png" alt>
/// </div>
/// - Parameter params: The tensor from which to gather values. Must be at least rank
/// `axis + 1`.
/// - Parameter indices: Index tensor. Must be in range `[0, params.shape[axis])`.
/// - Parameter axis: The axis in `params` to gather `indices` from. Defaults to the first
/// dimension. Supports negative indexes.
/// - Parameter tparams: 
/// - Parameter tindices: 
/// - Parameter taxis: 
/// - Returns: 
///	output: Values from `params` gathered from indices given by `indices`, with
/// shape `params.shape[:axis] + indices.shape + params.shape[axis + 1:]`.
public func gatherV2(operationName: String? = nil, params: Output, indices: Output, axis: Output, tparams: Any.Type, tindices: Any.Type, taxis: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Tparams"] = tparams
	attrs["Tindices"] = tindices
	attrs["Taxis"] = taxis
	let opspec = OpSpec(
		type: "GatherV2",
		name: (operationName ?? "Type"),
		input: [params, indices, axis],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 


/// - Parameter input: 
/// - Parameter computeUv: 
/// - Parameter fullMatrices: 
/// - Returns: 
///	s: 
///	u: 
///	v: 
public func batchSvd(operationName: String? = nil, input: Output, computeUv: Bool, fullMatrices: Bool) throws -> (s: Output, u: Output, v: Output) { 
	var attrs = [String : Any]()
	attrs["compute_uv"] = computeUv
	attrs["full_matrices"] = fullMatrices
	let opspec = OpSpec(
		type: "BatchSvd",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (s: op.output(at: 0), u: op.output(at: 1), v: op.output(at: 2))
} 


/// - Parameter matrix: 
/// - Parameter rhs: 
/// - Parameter adjoint: 
/// - Returns: 
///	output: 
public func batchMatrixSolve(operationName: String? = nil, matrix: Output, rhs: Output, adjoint: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["adjoint"] = adjoint
	let opspec = OpSpec(
		type: "BatchMatrixSolve",
		name: (operationName ?? "Type"),
		input: [matrix, rhs],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 


/// - Parameter input: 
/// - Returns: 
///	output: 
public func batchIFFT2D(operationName: String? = nil, input: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "BatchIFFT2D",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Creates a summary file writer accessible by the given resource handle.
/// - Parameter writer: A handle to the summary writer resource
/// - Parameter logdir: Directory where the event file will be written.
/// - Parameter maxQueue: Size of the queue of pending events and summaries.
/// - Parameter flushMillis: How often, in milliseconds, to flush the pending events and
/// summaries to disk.
/// - Parameter filenameSuffix: Every event file's name is suffixed with this suffix.
public func createSummaryFileWriter(operationName: String? = nil, writer: Output, logdir: Output, maxQueue: Output, flushMillis: Output, filenameSuffix: Output) throws -> Operation { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "CreateSummaryFileWriter",
		name: (operationName ?? "Type"),
		input: [writer, logdir, maxQueue, flushMillis, filenameSuffix],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 


/// - Parameter handle: 
/// - Parameter indices: 
/// - Parameter flowIn: 
/// - Parameter dtype: 
/// - Parameter elementShape: 
/// - Returns: 
///	value: 
public func tensorArrayGather(operationName: String? = nil, handle: Output, indices: Output, flowIn: Output, dtype: Any.Type, elementShape: Shape) throws -> Output { 
	var attrs = [String : Any]()
	attrs["dtype"] = dtype
	attrs["element_shape"] = elementShape
	let opspec = OpSpec(
		type: "TensorArrayGather",
		name: (operationName ?? "Type"),
		input: [handle, indices, flowIn],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Restore a reader to a previously saved state.
///Not all Readers support being restored, so this can produce an
/// Unimplemented error.
/// - Parameter readerHandle: Handle to a Reader.
/// - Parameter state: Result of a ReaderSerializeState of a Reader with type
/// matching reader_handle.
public func readerRestoreState(operationName: String? = nil, readerHandle: Output, state: Output) throws -> Operation { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "ReaderRestoreState",
		name: (operationName ?? "Type"),
		input: [readerHandle, state],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 


/// - Parameter input: 
/// - Returns: 
///	output: 
public func batchCholesky(operationName: String? = nil, input: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "BatchCholesky",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes the singular value decompositions of one or more matrices.
///Computes the SVD of each inner matrix in `input` such that
/// `input[..., :, :] = u[..., :, :]  *  diag(s[..., :, :])  *  transpose(v[..., :, :])`
/// 
/// ```python
/// # a is a tensor containing a batch of matrices.
/// # s is a tensor of singular values for each matrix.
/// # u is the tensor containing of left singular vectors for each matrix.
/// # v is the tensor containing of right singular vectors for each matrix.
/// s, u, v = svd(a)
/// s, _, _ = svd(a, compute_uv=False)
/// ```
/// - Parameter input: A tensor of shape `[..., M, N]` whose inner-most 2 dimensions
/// form matrices of size `[M, N]`. Let `P` be the minimum of `M` and `N`.
/// - Parameter computeUv: If true, left and right singular vectors will be
/// computed and returned in `u` and `v`, respectively.
/// If false, `u` and `v` are not set and should never referenced.
/// - Parameter fullMatrices: If true, compute full-sized `u` and `v`. If false
/// (the default), compute only the leading `P` singular vectors.
/// Ignored if `compute_uv` is `False`.
/// - Returns: 
///	s: Singular values. Shape is `[..., P]`.
///	u: Left singular vectors. If `full_matrices` is `False` then shape is
/// `[..., M, P]`; if `full_matrices` is `True` then shape is
/// `[..., M, M]`. Undefined if `compute_uv` is `False`.
///	v: Left singular vectors. If `full_matrices` is `False` then shape is
/// `[..., N, P]`. If `full_matrices` is `True` then shape is `[..., N, N]`.
/// Undefined if `compute_uv` is false.
public func svd(operationName: String? = nil, input: Output, computeUv: Bool, fullMatrices: Bool) throws -> (s: Output, u: Output, v: Output) { 
	var attrs = [String : Any]()
	attrs["compute_uv"] = computeUv
	attrs["full_matrices"] = fullMatrices
	let opspec = OpSpec(
		type: "Svd",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (s: op.output(at: 0), u: op.output(at: 1), v: op.output(at: 2))
} 

///Computes the QR decompositions of one or more matrices.
///Computes the QR decomposition of each inner matrix in `tensor` such that
/// `tensor[..., :, :] = q[..., :, :]  *  r[..., :,:])`
/// 
/// ```python
/// # a is a tensor.
/// # q is a tensor of orthonormal matrices.
/// # r is a tensor of upper triangular matrices.
/// q, r = qr(a)
/// q_full, r_full = qr(a, full_matrices=True)
/// ```
/// - Parameter input: A tensor of shape `[..., M, N]` whose inner-most 2 dimensions
/// form matrices of size `[M, N]`. Let `P` be the minimum of `M` and `N`.
/// - Parameter fullMatrices: If true, compute full-sized `q` and `r`. If false
/// (the default), compute only the leading `P` columns of `q`.
/// - Returns: 
///	q: Orthonormal basis for range of `a`. If `full_matrices` is `False` then
/// shape is `[..., M, P]`; if `full_matrices` is `True` then shape is
/// `[..., M, M]`.
///	r: Triangular factor. If `full_matrices` is `False` then shape is
/// `[..., P, N]`. If `full_matrices` is `True` then shape is `[..., M, N]`.
public func qr(operationName: String? = nil, input: Output, fullMatrices: Bool) throws -> (q: Output, r: Output) { 
	var attrs = [String : Any]()
	attrs["full_matrices"] = fullMatrices
	let opspec = OpSpec(
		type: "Qr",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (q: op.output(at: 0), r: op.output(at: 1))
} 

///Generates sparse cross from a list of sparse and dense tensors.
///The op takes two lists, one of 2D `SparseTensor` and one of 2D `Tensor`, each
/// representing features of one feature column. It outputs a 2D `SparseTensor` with
/// the batchwise crosses of these features.
/// 
/// For example, if the inputs are
/// 
///     inputs[0]: SparseTensor with shape = [2, 2]
///     [0, 0]: "a"
///     [1, 0]: "b"
///     [1, 1]: "c"
/// 
///     inputs[1]: SparseTensor with shape = [2, 1]
///     [0, 0]: "d"
///     [1, 0]: "e"
/// 
///     inputs[2]: Tensor [["f"], ["g"]]
/// 
/// then the output will be
/// 
///     shape = [2, 2]
///     [0, 0]: "a_X_d_X_f"
///     [1, 0]: "b_X_e_X_g"
///     [1, 1]: "c_X_e_X_g"
/// 
/// if hashed_output=true then the output will be
/// 
///     shape = [2, 2]
///     [0, 0]: FingerprintCat64(
///                 Fingerprint64("f"), FingerprintCat64(
///                     Fingerprint64("d"), Fingerprint64("a")))
///     [1, 0]: FingerprintCat64(
///                 Fingerprint64("g"), FingerprintCat64(
///                     Fingerprint64("e"), Fingerprint64("b")))
///     [1, 1]: FingerprintCat64(
///                 Fingerprint64("g"), FingerprintCat64(
///                     Fingerprint64("e"), Fingerprint64("c")))
/// - Parameter indices: 2-D.  Indices of each input `SparseTensor`.
/// - Parameter values: 1-D.   values of each `SparseTensor`.
/// - Parameter shapes: 1-D.   Shapes of each `SparseTensor`.
/// - Parameter denseInputs: 2-D.    Columns represented by dense `Tensor`.
/// - Parameter n: 
/// - Parameter hashedOutput: If true, returns the hash of the cross instead of the string.
/// This will allow us avoiding string manipulations.
/// - Parameter numBuckets: It is used if hashed_output is true.
/// output = hashed_value%num_buckets if num_buckets > 0 else hashed_value.
/// - Parameter hashKey: Specify the hash_key that will be used by the `FingerprintCat64`
/// function to combine the crosses fingerprints.
/// - Parameter sparseTypes: 
/// - Parameter denseTypes: 
/// - Parameter outType: 
/// - Parameter internalType: 
/// - Returns: 
///	output_indices: 2-D.  Indices of the concatenated `SparseTensor`.
///	output_values: 1-D.  Non-empty values of the concatenated or hashed
/// `SparseTensor`.
///	output_shape: 1-D.  Shape of the concatenated `SparseTensor`.
public func sparseCross(operationName: String? = nil, indices: [Output], values: Output, shapes: [Output], denseInputs: Output, n: UInt8, hashedOutput: Bool, numBuckets: UInt8, hashKey: UInt8, sparseTypes: [Any.Type], denseTypes: [Any.Type], outType: Any.Type, internalType: Any.Type) throws -> (outputIndices: Output, outputValues: Output, outputShape: Output) { 
	var attrs = [String : Any]()
	attrs["N"] = n
	attrs["hashed_output"] = hashedOutput
	attrs["num_buckets"] = numBuckets
	attrs["hash_key"] = hashKey
	attrs["sparse_types"] = sparseTypes
	attrs["dense_types"] = denseTypes
	attrs["out_type"] = outType
	attrs["internal_type"] = internalType
	let opspec = OpSpec(
		type: "SparseCross",
		name: (operationName ?? "Type"),
		input: [indices, values, shapes, denseInputs],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (outputIndices: op.output(at: 0), outputValues: op.output(at: 1), outputShape: op.output(at: 2))
} 

///Solves one or more linear least-squares problems.
///`matrix` is a tensor of shape `[..., M, N]` whose inner-most 2 dimensions
/// form real or complex matrices of size `[M, N]`. `Rhs` is a tensor of the same
/// type as `matrix` and shape `[..., M, K]`.
/// The output is a tensor shape `[..., N, K]` where each output matrix solves
/// each of the equations
/// `matrix[..., :, :]`  *  `output[..., :, :]` = `rhs[..., :, :]`
/// in the least squares sense.
/// 
/// We use the following notation for (complex) matrix and right-hand sides
/// in the batch:
/// 
/// `matrix`=\\(A \in \mathbb{C}// ^{m \times n}\\),
/// `rhs`=\\(B  \in \mathbb{C}// ^{m \times k}\\),
/// `output`=\\(X  \in \mathbb{C}// ^{n \times k}\\),
/// `l2_regularizer`=\\(\lambda \in \mathbb{R}\\).
/// 
/// If `fast` is `True`, then the solution is computed by solving the normal
/// equations using Cholesky decomposition. Specifically, if \\(m \ge n\\) then
/// \\(X = (A// ^H A + \lambda I)// ^{-1} A// ^H B\\), which solves the least-squares
/// problem \\(X = \mathrm{argmin}_{Z \in \Re// ^{n \times k} } ||A Z - B||_F// ^2 +
/// \lambda ||Z||_F// ^2\\). If \\(m \lt n\\) then `output` is computed as
/// \\(X = A// ^H (A A// ^H + \lambda I)// ^{-1} B\\), which (for \\(\lambda = 0\\)) is the
/// minimum-norm solution to the under-determined linear system, i.e.
/// \\(X = \mathrm{argmin}_{Z \in \mathbb{C}// ^{n \times k} } ||Z||_F// ^2 \\),
/// subject to \\(A Z = B\\). Notice that the fast path is only numerically stable
/// when \\(A\\) is numerically full rank and has a condition number
/// \\(\mathrm{cond}(A) \lt \frac{1}{\sqrt{\epsilon_{mach} } }\\) or\\(\lambda\\) is
/// sufficiently large.
/// 
/// If `fast` is `False` an algorithm based on the numerically robust complete
/// orthogonal decomposition is used. This computes the minimum-norm
/// least-squares solution, even when \\(A\\) is rank deficient. This path is
/// typically 6-7 times slower than the fast path. If `fast` is `False` then
/// `l2_regularizer` is ignored.
/// - Parameter matrix: Shape is `[..., M, N]`.
/// - Parameter rhs: Shape is `[..., M, K]`.
/// - Parameter l2Regularizer: Scalar tensor.
/// 
/// @compatibility(numpy)
/// Equivalent to np.linalg.lstsq
/// @end_compatibility
/// - Parameter fast: 
/// - Returns: 
///	output: Shape is `[..., N, K]`.
public func matrixSolveLs(operationName: String? = nil, matrix: Output, rhs: Output, l2Regularizer: Output, fast: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["fast"] = fast
	let opspec = OpSpec(
		type: "MatrixSolveLs",
		name: (operationName ?? "Type"),
		input: [matrix, rhs, l2Regularizer],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Packs a list of `N` rank-`R` tensors into one rank-`(R+1)` tensor.
///Packs the `N` tensors in `values` into a tensor with rank one higher than each
/// tensor in `values`, by packing them along the `axis` dimension.
/// Given a list of tensors of shape `(A, B, C)`;
/// 
/// if `axis == 0` then the `output` tensor will have the shape `(N, A, B, C)`.
/// if `axis == 1` then the `output` tensor will have the shape `(A, N, B, C)`.
/// Etc.
/// 
/// For example:
/// 
/// ```
/// # 'x' is [1, 4]
/// # 'y' is [2, 5]
/// # 'z' is [3, 6]
/// pack([x, y, z]) => [[1, 4], [2, 5], [3, 6]]  # Pack along first dim.
/// pack([x, y, z], axis=1) => [[1, 2, 3], [4, 5, 6]]
/// ```
/// 
/// This is the opposite of `unpack`.
/// - Parameter values: Must be of same shape and type.
/// - Parameter n: 
/// - Parameter axis: Dimension along which to pack.  Negative values wrap around, so the
/// valid range is `[-(R+1), R+1)`.
/// - Returns: 
///	output: The packed tensor.
public func pack(operationName: String? = nil, values: [Output], n: UInt8, axis: UInt8) throws -> Output { 
	var attrs = [String : Any]()
	attrs["N"] = n
	attrs["axis"] = axis
	let opspec = OpSpec(
		type: "Pack",
		name: (operationName ?? "Type"),
		input: [values],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Closes the given barrier.
///This operation signals that no more new elements will be inserted in the
/// given barrier. Subsequent InsertMany that try to introduce a new key will fail.
/// Subsequent InsertMany operations that just add missing components to already
/// existing elements will continue to succeed. Subsequent TakeMany operations will
/// continue to succeed if sufficient completed elements remain in the barrier.
/// Subsequent TakeMany operations that would block will fail immediately.
/// - Parameter handle: The handle to a barrier.
/// - Parameter cancelPendingEnqueues: If true, all pending enqueue requests that are
/// blocked on the barrier's queue will be canceled. InsertMany will fail, even
/// if no new key is introduced.
public func barrierClose(operationName: String? = nil, handle: Output, cancelPendingEnqueues: Bool) throws -> Operation { 
	var attrs = [String : Any]()
	attrs["cancel_pending_enqueues"] = cancelPendingEnqueues
	let opspec = OpSpec(
		type: "BarrierClose",
		name: (operationName ?? "Type"),
		input: [handle],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Computes the eigen decomposition of one or more square self-adjoint matrices.
///Computes the eigenvalues and (optionally) eigenvectors of each inner matrix in
/// `input` such that `input[..., :, :] = v[..., :, :]  *  diag(e[..., :])`.
/// 
/// ```python
/// # a is a tensor.
/// # e is a tensor of eigenvalues.
/// # v is a tensor of eigenvectors.
/// e, v = self_adjoint_eig(a)
/// e = self_adjoint_eig(a, compute_v=False)
/// ```
/// - Parameter input: `Tensor` input of shape `[N, N]`.
/// - Parameter computeV: If `True` then eigenvectors will be computed and returned in `v`.
/// Otherwise, only the eigenvalues will be computed.
/// - Returns: 
///	e: Eigenvalues. Shape is `[N]`.
///	v: Eigenvectors. Shape is `[N, N]`.
public func selfAdjointEigV2(operationName: String? = nil, input: Output, computeV: Bool) throws -> (e: Output, v: Output) { 
	var attrs = [String : Any]()
	attrs["compute_v"] = computeV
	let opspec = OpSpec(
		type: "SelfAdjointEigV2",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (e: op.output(at: 0), v: op.output(at: 1))
} 

///Returns the index with the largest value across dimensions of a tensor.
///Note that in case of ties the identity of the return value is not guaranteed.
/// - Parameter input: 
/// - Parameter dimension: int32 or int64, must be in the range `[-rank(input), rank(input))`.
/// Describes which dimension of the input Tensor to reduce across. For vectors,
/// use dimension = 0.
/// - Parameter tidx: 
/// - Parameter outputType: 
/// - Returns: 
///	output: 
public func argMax(operationName: String? = nil, input: Output, dimension: Output, tidx: Any.Type, outputType: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Tidx"] = tidx
	attrs["output_type"] = outputType
	let opspec = OpSpec(
		type: "ArgMax",
		name: (operationName ?? "Type"),
		input: [input, dimension],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes the reverse mode backpropagated gradient of the Cholesky algorithm.
///For an explanation see "Differentiation of the Cholesky algorithm" by
/// Iain Murray http://arxiv.org/abs/1602.07527.
/// - Parameter l: Output of batch Cholesky algorithm l = cholesky(A). Shape is `[..., M, M]`.
/// Algorithm depends only on lower triangular part of the innermost matrices of
/// this tensor.
/// - Parameter grad: df/dl where f is some scalar function. Shape is `[..., M, M]`.
/// Algorithm depends only on lower triangular part of the innermost matrices of
/// this tensor.
/// - Returns: 
///	output: Symmetrized version of df/dA . Shape is `[..., M, M]`
public func choleskyGrad(operationName: String? = nil, l: Output, grad: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "CholeskyGrad",
		name: (operationName ?? "Type"),
		input: [l, grad],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes the determinant of one or more square matrices.
///The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
/// form square matrices. The output is a tensor containing the determinants
/// for all input submatrices `[..., :, :]`.
/// - Parameter input: Shape is `[..., M, M]`.
/// - Returns: 
///	output: Shape is `[...]`.
public func matrixDeterminant(operationName: String? = nil, input: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "MatrixDeterminant",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Returns the shape of a tensor.
///This operation returns a 1-D integer tensor representing the shape of `input`.
/// 
/// For example:
/// 
/// ```
/// # 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
/// shape(t) ==> [2, 2, 3]
/// ```
/// - Parameter input: 
/// - Parameter outType: 
/// - Returns: 
///	output: 
public func shape(operationName: String? = nil, input: Output, outType: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["out_type"] = outType
	let opspec = OpSpec(
		type: "Shape",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Looks up keys in a table, outputs the corresponding values.
///The tensor `keys` must of the same type as the keys of the table.
/// The output `values` is of the type of the table values.
/// 
/// The scalar `default_value` is the value output for keys not present in the
/// table. It must also be of the same type as the table values.
/// - Parameter tableHandle: Handle to the table.
/// - Parameter keys: Any shape.  Keys to look up.
/// - Parameter defaultValue: 
/// - Parameter tin: 
/// - Parameter tout: 
/// - Returns: 
///	values: Same shape as `keys`.  Values found in the table, or `default_values`
/// for missing keys.
public func lookupTableFindV2(operationName: String? = nil, tableHandle: Output, keys: Output, defaultValue: Output, tin: Any.Type, tout: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Tin"] = tin
	attrs["Tout"] = tout
	let opspec = OpSpec(
		type: "LookupTableFindV2",
		name: (operationName ?? "Type"),
		input: [tableHandle, keys, defaultValue],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Update ' * var' according to the Ftrl-proximal scheme.
///grad_with_shrinkage = grad + 2  *  l2_shrinkage  *  var
/// accum_new = accum + grad_with_shrinkage  *  grad_with_shrinkage
/// linear += grad_with_shrinkage +
///     (accum_new// ^(-lr_power) - accum// ^(-lr_power)) / lr  *  var
/// quadratic = 1.0 / (accum_new// ^(lr_power)  *  lr) + 2  *  l2
/// var = (sign(linear)  *  l1 - linear) / quadratic if |linear| > l1 else 0.0
/// accum = accum_new
/// - Parameter `var`: Should be from a Variable().
/// - Parameter accum: Should be from a Variable().
/// - Parameter linear: Should be from a Variable().
/// - Parameter grad: The gradient.
/// - Parameter lr: Scaling factor. Must be a scalar.
/// - Parameter l1: L1 regulariation. Must be a scalar.
/// - Parameter l2: L2 shrinkage regulariation. Must be a scalar.
/// - Parameter l2Shrinkage: 
/// - Parameter lrPower: Scaling factor. Must be a scalar.
/// - Parameter useLocking: If `True`, updating of the var and accum tensors will be protected
/// by a lock; otherwise the behavior is undefined, but may exhibit less
/// contention.
/// - Returns: 
///	out: Same as "var".
public func applyFtrlV2(operationName: String? = nil, `var`: Output, accum: Output, linear: Output, grad: Output, lr: Output, l1: Output, l2: Output, l2Shrinkage: Output, lrPower: Output, useLocking: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["use_locking"] = useLocking
	let opspec = OpSpec(
		type: "ApplyFtrlV2",
		name: (operationName ?? "Type"),
		input: [`var`, accum, linear, grad, lr, l1, l2, l2Shrinkage, lrPower],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Writes contents to the file at input filename. Creates file and recursively
///creates directory if not existing.
/// - Parameter filename: scalar. The name of the file to which we write the contents.
/// - Parameter contents: scalar. The content to be written to the output file.
public func writeFile(operationName: String? = nil, filename: Output, contents: Output) throws -> Operation { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "WriteFile",
		name: (operationName ?? "Type"),
		input: [filename, contents],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Computes gradients of average pooling function.
/// - Parameter origInputShape: The original input dimensions.
/// - Parameter grad: Output backprop of shape `[batch, depth, rows, cols, channels]`.
/// - Parameter ksize: 1-D tensor of length 5. The size of the window for each dimension of
/// the input tensor. Must have `ksize[0] = ksize[4] = 1`.
/// - Parameter strides: 1-D tensor of length 5. The stride of the sliding window for each
/// dimension of `input`. Must have `strides[0] = strides[4] = 1`.
/// - Parameter padding: The type of padding algorithm to use.
/// - Parameter dataFormat: The data format of the input and output data. With the
/// default format "NDHWC", the data is stored in the order of:
///     [batch, in_depth, in_height, in_width, in_channels].
/// Alternatively, the format could be "NCDHW", the data storage order is:
///     [batch, in_channels, in_depth, in_height, in_width].
/// - Returns: 
///	output: The backprop for input.
public func avgPool3DGrad(operationName: String? = nil, origInputShape: Output, grad: Output, ksize: [Int64], strides: [Int64], padding: String, dataFormat: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["ksize"] = ksize
	attrs["strides"] = strides
	attrs["padding"] = padding
	attrs["data_format"] = dataFormat
	let opspec = OpSpec(
		type: "AvgPool3DGrad",
		name: (operationName ?? "Type"),
		input: [origInputShape, grad],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Returns the gradient of `Tile`.
///Since `Tile` takes an input and repeats the input `multiples` times
/// along each dimension, `TileGrad` takes in `multiples` and aggregates
/// each repeated tile of `input` into `output`.
/// - Parameter input: 
/// - Parameter multiples: 
/// - Returns: 
///	output: 
public func tileGrad(operationName: String? = nil, input: Output, multiples: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "TileGrad",
		name: (operationName ?? "Type"),
		input: [input, multiples],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Restore a Reader to its initial clean state.
/// - Parameter readerHandle: Handle to a Reader.
public func readerReset(operationName: String? = nil, readerHandle: Output) throws -> Operation { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "ReaderReset",
		name: (operationName ?? "Type"),
		input: [readerHandle],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Slice a `SparseTensor` based on the `start` and `size`.
///For example, if the input is
/// 
///     input_tensor = shape = [2, 7]
///     [    a   d e  ]
///     [b c          ]
/// 
/// Graphically the output tensors are:
/// 
///     sparse_slice([0, 0], [2, 4]) = shape = [2, 4]
///     [    a  ]
///     [b c    ]
/// 
///     sparse_slice([0, 4], [2, 3]) = shape = [2, 3]
///     [ d e  ]
///     [      ]
/// - Parameter indices: 2-D tensor represents the indices of the sparse tensor.
/// - Parameter values: 1-D tensor represents the values of the sparse tensor.
/// - Parameter shape: 1-D. tensor represents the shape of the sparse tensor.
/// - Parameter start: 1-D. tensor represents the start of the slice.
/// - Parameter size: 1-D. tensor represents the size of the slice.
/// output indices: A list of 1-D tensors represents the indices of the output
/// sparse tensors.
/// - Returns: 
///	output_indices: 
///	output_values: A list of 1-D tensors represents the values of the output sparse
/// tensors.
///	output_shape: A list of 1-D tensors represents the shape of the output sparse
/// tensors.
public func sparseSlice(operationName: String? = nil, indices: Output, values: Output, shape: Output, start: Output, size: Output) throws -> (outputIndices: Output, outputValues: Output, outputShape: Output) { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "SparseSlice",
		name: (operationName ?? "Type"),
		input: [indices, values, shape, start, size],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (outputIndices: op.output(at: 0), outputValues: op.output(at: 1), outputShape: op.output(at: 2))
} 

///Outputs a `Summary` protocol buffer with audio.
///The summary has up to `max_outputs` summary values containing audio. The
/// audio is built from `tensor` which must be 3-D with shape `[batch_size,
/// frames, channels]` or 2-D with shape `[batch_size, frames]`. The values are
/// assumed to be in the range of `[-1.0, 1.0]` with a sample rate of `sample_rate`.
/// 
/// The `tag` argument is a scalar `Tensor` of type `string`.  It is used to
/// build the `tag` of the summary values:
/// 
///  *   If `max_outputs` is 1, the summary value tag is ' * tag * /audio'.
///  *   If `max_outputs` is greater than 1, the summary value tags are
///    generated sequentially as ' * tag * /audio/0', ' * tag * /audio/1', etc.
/// - Parameter tag: Scalar. Used to build the `tag` attribute of the summary values.
/// - Parameter tensor: 2-D of shape `[batch_size, frames]`.
/// - Parameter sampleRate: The sample rate of the signal in hertz.
/// - Parameter maxOutputs: Max number of batch elements to generate audio for.
/// - Returns: 
///	summary: Scalar. Serialized `Summary` protocol buffer.
public func audioSummaryV2(operationName: String? = nil, tag: Output, tensor: Output, sampleRate: Output, maxOutputs: UInt8) throws -> Output { 
	var attrs = [String : Any]()
	attrs["max_outputs"] = maxOutputs
	let opspec = OpSpec(
		type: "AudioSummaryV2",
		name: (operationName ?? "Type"),
		input: [tag, tensor, sampleRate],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Deprecated. Use TensorArrayReadV3
/// - Parameter handle: 
/// - Parameter index: 
/// - Parameter flowIn: 
/// - Parameter dtype: 
/// - Returns: 
///	value: 
public func tensorArrayReadV2(operationName: String? = nil, handle: Output, index: Output, flowIn: Output, dtype: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["dtype"] = dtype
	let opspec = OpSpec(
		type: "TensorArrayReadV2",
		name: (operationName ?? "Type"),
		input: [handle, index, flowIn],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Op peeks at the values at the specified index.  If the
///underlying container does not contain sufficient elements
/// this op will block until it does.   This Op is optimized for
/// performance.
/// - Parameter index: 
/// - Parameter capacity: 
/// - Parameter memoryLimit: 
/// - Parameter dtypes: 
/// - Parameter container: 
/// - Parameter sharedName: 
/// - Returns: 
///	values: 
public func stagePeek(operationName: String? = nil, index: Output, capacity: UInt8, memoryLimit: UInt8, dtypes: [Any.Type], container: String, sharedName: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["capacity"] = capacity
	attrs["memory_limit"] = memoryLimit
	attrs["dtypes"] = dtypes
	attrs["container"] = container
	attrs["shared_name"] = sharedName
	let opspec = OpSpec(
		type: "StagePeek",
		name: (operationName ?? "Type"),
		input: [index],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Restore a reader to a previously saved state.
///Not all Readers support being restored, so this can produce an
/// Unimplemented error.
/// - Parameter readerHandle: Handle to a Reader.
/// - Parameter state: Result of a ReaderSerializeState of a Reader with type
/// matching reader_handle.
public func readerRestoreStateV2(operationName: String? = nil, readerHandle: Output, state: Output) throws -> Operation { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "ReaderRestoreStateV2",
		name: (operationName ?? "Type"),
		input: [readerHandle, state],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Number of unique elements along last dimension of input `set`.
///Input `set` is a `SparseTensor` represented by `set_indices`, `set_values`,
/// and `set_shape`. The last dimension contains values in a set, duplicates are
/// allowed but ignored.
/// 
/// If `validate_indices` is `True`, this op validates the order and range of `set`
/// indices.
/// - Parameter setIndices: 2D `Tensor`, indices of a `SparseTensor`.
/// - Parameter setValues: 1D `Tensor`, values of a `SparseTensor`.
/// - Parameter setShape: 1D `Tensor`, shape of a `SparseTensor`.
/// - Parameter validateIndices: 
/// - Returns: 
///	size: For `set` ranked `n`, this is a `Tensor` with rank `n-1`, and the same 1st
/// `n-1` dimensions as `set`. Each value is the number of unique elements in
/// the corresponding `[0...n-1]` dimension of `set`.
public func setSize(operationName: String? = nil, setIndices: Output, setValues: Output, setShape: Output, validateIndices: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["validate_indices"] = validateIndices
	let opspec = OpSpec(
		type: "SetSize",
		name: (operationName ?? "Type"),
		input: [setIndices, setValues, setShape],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Produce a string tensor that encodes the state of a Reader.
///Not all Readers support being serialized, so this can produce an
/// Unimplemented error.
/// - Parameter readerHandle: Handle to a Reader.
/// - Returns: 
///	state: 
public func readerSerializeStateV2(operationName: String? = nil, readerHandle: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "ReaderSerializeStateV2",
		name: (operationName ?? "Type"),
		input: [readerHandle],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Produce a string tensor that encodes the state of a Reader.
///Not all Readers support being serialized, so this can produce an
/// Unimplemented error.
/// - Parameter readerHandle: Handle to a Reader.
/// - Returns: 
///	state: 
public func readerSerializeState(operationName: String? = nil, readerHandle: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "ReaderSerializeState",
		name: (operationName ?? "Type"),
		input: [readerHandle],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Returns up to `num_records` (key, value) pairs produced by a Reader.
///Will dequeue from the input queue if necessary (e.g. when the
/// Reader needs to start reading from a new file since it has finished
/// with the previous file).
/// It may return less than `num_records` even before the last batch.
/// - Parameter readerHandle: Handle to a `Reader`.
/// - Parameter queueHandle: Handle to a `Queue`, with string work items.
/// - Parameter numRecords: number of records to read from `Reader`.
/// - Returns: 
///	keys: A 1-D tensor.
///	values: A 1-D tensor.
public func readerReadUpToV2(operationName: String? = nil, readerHandle: Output, queueHandle: Output, numRecords: Output) throws -> (keys: Output, values: Output) { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "ReaderReadUpToV2",
		name: (operationName ?? "Type"),
		input: [readerHandle, queueHandle, numRecords],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (keys: op.output(at: 0), values: op.output(at: 1))
} 

///Creates a dataset that concatenates `input_dataset` with `another_dataset`.
/// - Parameter inputDataset: 
/// - Parameter anotherDataset: 
/// - Parameter outputTypes: 
/// - Parameter outputShapes: 
/// - Returns: 
///	handle: 
public func concatenateDataset(operationName: String? = nil, inputDataset: Output, anotherDataset: Output, outputTypes: [Any.Type], outputShapes: [Shape]) throws -> Output { 
	var attrs = [String : Any]()
	attrs["output_types"] = outputTypes
	attrs["output_shapes"] = outputShapes
	let opspec = OpSpec(
		type: "ConcatenateDataset",
		name: (operationName ?? "Type"),
		input: [inputDataset, anotherDataset],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Returns up to `num_records` (key, value) pairs produced by a Reader.
///Will dequeue from the input queue if necessary (e.g. when the
/// Reader needs to start reading from a new file since it has finished
/// with the previous file).
/// It may return less than `num_records` even before the last batch.
/// - Parameter readerHandle: Handle to a `Reader`.
/// - Parameter queueHandle: Handle to a `Queue`, with string work items.
/// - Parameter numRecords: number of records to read from `Reader`.
/// - Returns: 
///	keys: A 1-D tensor.
///	values: A 1-D tensor.
public func readerReadUpTo(operationName: String? = nil, readerHandle: Output, queueHandle: Output, numRecords: Output) throws -> (keys: Output, values: Output) { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "ReaderReadUpTo",
		name: (operationName ?? "Type"),
		input: [readerHandle, queueHandle, numRecords],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (keys: op.output(at: 0), values: op.output(at: 1))
} 

///Computes the inverse permutation of a tensor.
///This operation computes the inverse of an index permutation. It takes a 1-D
/// integer tensor `x`, which represents the indices of a zero-based array, and
/// swaps each value with its index position. In other words, for an output tensor
/// `y` and an input tensor `x`, this operation computes the following:
/// 
/// `y[x[i]] = i for i in [0, 1, ..., len(x) - 1]`
/// 
/// The values must include 0. There can be no duplicate values or negative values.
/// 
/// For example:
/// 
/// ```
/// # tensor `x` is [3, 4, 0, 2, 1]
/// invert_permutation(x) ==> [2, 4, 3, 0, 1]
/// ```
/// - Parameter x: 1-D.
/// - Returns: 
///	y: 1-D.
public func invertPermutation(operationName: String? = nil, x: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "InvertPermutation",
		name: (operationName ?? "Type"),
		input: [x],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Outputs random values from a truncated normal distribution.
///The generated values follow a normal distribution with mean 0 and standard
/// deviation 1, except that values whose magnitude is more than 2 standard
/// deviations from the mean are dropped and re-picked.
/// - Parameter shape: The shape of the output tensor.
/// - Parameter seed: If either `seed` or `seed2` are set to be non-zero, the random number
/// generator is seeded by the given seed.  Otherwise, it is seeded by a
/// random seed.
/// - Parameter seed2: A second seed to avoid seed collision.
/// - Parameter dtype: The type of the output.
/// - Returns: 
///	output: A tensor of the specified shape filled with random truncated normal
/// values.
public func truncatedNormal(operationName: String? = nil, shape: Output, seed: UInt8, seed2: UInt8, dtype: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["seed"] = seed
	attrs["seed2"] = seed2
	attrs["dtype"] = dtype
	let opspec = OpSpec(
		type: "TruncatedNormal",
		name: (operationName ?? "Type"),
		input: [shape],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Solves systems of linear equations with upper or lower triangular matrices by
///backsubstitution.
/// 
/// `matrix` is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions form
/// square matrices. If `lower` is `True` then the strictly upper triangular part
/// of each inner-most matrix is assumed to be zero and not accessed.
/// If `lower` is False then the strictly lower triangular part of each inner-most
/// matrix is assumed to be zero and not accessed.
/// `rhs` is a tensor of shape `[..., M, K]`.
/// 
/// The output is a tensor of shape `[..., M, K]`. If `adjoint` is
/// `True` then the innermost matrices in `output` satisfy matrix equations
/// `matrix[..., :, :]  *  output[..., :, :] = rhs[..., :, :]`.
/// If `adjoint` is `False` then the strictly then the  innermost matrices in
/// `output` satisfy matrix equations
/// `adjoint(matrix[..., i, k])  *  output[..., k, j] = rhs[..., i, j]`.
/// - Parameter matrix: Shape is `[..., M, M]`.
/// - Parameter rhs: Shape is `[..., M, K]`.
/// - Parameter lower: Boolean indicating whether the innermost matrices in `matrix` are
/// lower or upper triangular.
/// - Parameter adjoint: Boolean indicating whether to solve with `matrix` or its (block-wise)
///          adjoint.
/// 
/// @compatibility(numpy)
/// Equivalent to np.linalg.triangular_solve
/// @end_compatibility
/// - Returns: 
///	output: Shape is `[..., M, K]`.
public func matrixTriangularSolve(operationName: String? = nil, matrix: Output, rhs: Output, lower: Bool, adjoint: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["lower"] = lower
	attrs["adjoint"] = adjoint
	let opspec = OpSpec(
		type: "MatrixTriangularSolve",
		name: (operationName ?? "Type"),
		input: [matrix, rhs],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Returns the next record (key, value pair) produced by a Reader.
///Will dequeue from the input queue if necessary (e.g. when the
/// Reader needs to start reading from a new file since it has finished
/// with the previous file).
/// - Parameter readerHandle: Handle to a Reader.
/// - Parameter queueHandle: Handle to a Queue, with string work items.
/// - Returns: 
///	key: A scalar.
///	value: A scalar.
public func readerRead(operationName: String? = nil, readerHandle: Output, queueHandle: Output) throws -> (key: Output, value: Output) { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "ReaderRead",
		name: (operationName ?? "Type"),
		input: [readerHandle, queueHandle],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (key: op.output(at: 0), value: op.output(at: 1))
} 

///Randomly shuffles a tensor along its first dimension.
///  The tensor is shuffled along dimension 0, such that each `value[j]` is mapped
///   to one and only one `output[i]`. For example, a mapping that might occur for a
///   3x2 tensor is:
/// 
/// ```
/// [[1, 2],       [[5, 6],
///  [3, 4],  ==>   [1, 2],
///  [5, 6]]        [3, 4]]
/// ```
/// - Parameter value: The tensor to be shuffled.
/// - Parameter seed: If either `seed` or `seed2` are set to be non-zero, the random number
/// generator is seeded by the given seed.  Otherwise, it is seeded by a
/// random seed.
/// - Parameter seed2: A second seed to avoid seed collision.
/// - Returns: 
///	output: A tensor of same shape and type as `value`, shuffled along its first
/// dimension.
public func randomShuffle(operationName: String? = nil, value: Output, seed: UInt8, seed2: UInt8) throws -> Output { 
	var attrs = [String : Any]()
	attrs["seed"] = seed
	attrs["seed2"] = seed2
	let opspec = OpSpec(
		type: "RandomShuffle",
		name: (operationName ?? "Type"),
		input: [value],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Selects elements from `t` or `e`, depending on `condition`.
///The `t`, and `e` tensors must all have the same shape, and the
/// output will also have that shape.
/// 
/// The `condition` tensor must be a scalar if `t` and `e` are scalars.
/// If `t` and `e` are vectors or higher rank, then `condition` must be either a
/// scalar, a vector with size matching the first dimension of `t`, or must have
/// the same shape as `t`.
/// 
/// The `condition` tensor acts as a mask that chooses, based on the value at each
/// element, whether the corresponding element / row in the output should be
/// taken from `t` (if true) or `e` (if false).
/// 
/// If `condition` is a vector and `t` and `e` are higher rank matrices, then
/// it chooses which row (outer dimension) to copy from `t` and `e`.
/// If `condition` has the same shape as `t` and `e`, then it chooses which
/// element to copy from `t` and `e`.
/// 
/// For example:
/// 
/// ```python
/// # 'condition' tensor is [[True,  False]
/// #                        [False, True]]
/// # 't' is [[1, 2],
/// #         [3, 4]]
/// # 'e' is [[5, 6],
/// #         [7, 8]]
/// select(condition, t, e)  # => [[1, 6], [7, 4]]
/// 
/// 
/// # 'condition' tensor is [True, False]
/// # 't' is [[1, 2],
/// #         [3, 4]]
/// # 'e' is [[5, 6],
/// #         [7, 8]]
/// select(condition, t, e) ==> [[1, 2],
///                              [7, 8]]
/// 
/// ```
/// - Parameter condition: 
/// - Parameter t: = A `Tensor` which may have the same shape as `condition`.
/// If `condition` is rank 1, `t` may have higher rank,
/// but its first dimension must match the size of `condition`.
/// - Parameter e: = A `Tensor` with the same type and shape as `t`.
/// - Returns: 
///	output: = A `Tensor` with the same type and shape as `t` and `e`.
public func select(operationName: String? = nil, condition: Output, t: Output, e: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Select",
		name: (operationName ?? "Type"),
		input: [condition, t, e],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///The gradient operator for the SparseAdd op.
///The SparseAdd op calculates A + B, where A, B, and the sum are all represented
/// as `SparseTensor` objects.  This op takes in the upstream gradient w.r.t.
/// non-empty values of the sum, and outputs the gradients w.r.t. the non-empty
/// values of A and B.
/// - Parameter backpropValGrad: 1-D with shape `[nnz(sum)]`.  The gradient with respect to
/// the non-empty values of the sum.
/// - Parameter aIndices: 2-D.  The `indices` of the `SparseTensor` A, size `[nnz(A), ndims]`.
/// - Parameter bIndices: 2-D.  The `indices` of the `SparseTensor` B, size `[nnz(B), ndims]`.
/// - Parameter sumIndices: 2-D.  The `indices` of the sum `SparseTensor`, size
/// `[nnz(sum), ndims]`.
/// - Returns: 
///	a_val_grad: 1-D with shape `[nnz(A)]`. The gradient with respect to the
/// non-empty values of A.
///	b_val_grad: 1-D with shape `[nnz(B)]`. The gradient with respect to the
/// non-empty values of B.
public func sparseAddGrad(operationName: String? = nil, backpropValGrad: Output, aIndices: Output, bIndices: Output, sumIndices: Output) throws -> (aValGrad: Output, bValGrad: Output) { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "SparseAddGrad",
		name: (operationName ?? "Type"),
		input: [backpropValGrad, aIndices, bIndices, sumIndices],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (aValGrad: op.output(at: 0), bValGrad: op.output(at: 1))
} 

///A Reader that outputs the records from a LMDB file.
/// - Parameter container: If non-empty, this reader is placed in the given container.
/// Otherwise, a default container is used.
/// - Parameter sharedName: If non-empty, this reader is named in the given bucket
/// with this shared_name. Otherwise, the node name is used instead.
/// - Returns: 
///	reader_handle: The handle to reference the Reader.
public func lMDBReader(operationName: String? = nil, container: String, sharedName: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["container"] = container
	attrs["shared_name"] = sharedName
	let opspec = OpSpec(
		type: "LMDBReader",
		name: (operationName ?? "Type"),
		input: [],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes natural logarithm of x element-wise.
///I.e., \\(y = \log_e x\\).
/// - Parameter x: 
/// - Returns: 
///	y: 
public func log(operationName: String? = nil, x: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Log",
		name: (operationName ?? "Type"),
		input: [x],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Inverse 2D real-valued fast Fourier transform.
///Computes the inverse 2-dimensional discrete Fourier transform of a real-valued
/// signal over the inner-most 2 dimensions of `input`.
/// 
/// The inner-most 2 dimensions of `input` are assumed to be the result of `RFFT2D`:
/// The inner-most dimension contains the `fft_length / 2 + 1` unique components of
/// the DFT of a real-valued signal. If `fft_length` is not provided, it is computed
/// from the size of the inner-most 2 dimensions of `input`. If the FFT length used
/// to compute `input` is odd, it should be provided since it cannot be inferred
/// properly.
/// 
/// Along each axis `IRFFT2D` is computed on, if `fft_length` (or
/// `fft_length / 2 + 1` for the inner-most dimension) is smaller than the
/// corresponding dimension of `input`, the dimension is cropped. If it is larger,
/// the dimension is padded with zeros.
/// - Parameter input: A complex64 tensor.
/// - Parameter fftLength: An int32 tensor of shape [2]. The FFT length for each dimension.
/// - Returns: 
///	output: A float32 tensor of the same rank as `input`. The inner-most 2
///   dimensions of `input` are replaced with the `fft_length` samples of their
///   inverse 2D Fourier transform.
/// 
/// @compatibility(numpy)
/// Equivalent to np.fft.irfft2
/// @end_compatibility
public func irfft2D(operationName: String? = nil, input: Output, fftLength: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "IRFFT2D",
		name: (operationName ?? "Type"),
		input: [input, fftLength],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes gradient of the FractionalAvgPool function.
///Unlike FractionalMaxPoolGrad, we don't need to find arg_max for
/// FractionalAvgPoolGrad, we just need to evenly back-propagate each element of
/// out_backprop to those indices that form the same pooling cell. Therefore, we
/// just need to know the shape of original input tensor, instead of the whole
/// tensor.
/// - Parameter origInputTensorShape: Original input tensor shape for `fractional_avg_pool`
/// - Parameter outBackprop: 4-D with shape `[batch, height, width, channels]`.  Gradients
/// w.r.t. the output of `fractional_avg_pool`.
/// - Parameter rowPoolingSequence: row pooling sequence, form pooling region with
/// col_pooling_sequence.
/// - Parameter colPoolingSequence: column pooling sequence, form pooling region with
/// row_pooling sequence.
/// - Parameter overlapping: When set to True, it means when pooling, the values at the boundary
/// of adjacent pooling cells are used by both cells. For example:
/// 
/// `index  0  1  2  3  4`
/// 
/// `value  20 5  16 3  7`
/// 
/// If the pooling sequence is [0, 2, 4], then 16, at index 2 will be used twice.
/// The result would be [41/3, 26/3] for fractional avg pooling.
/// - Returns: 
///	output: 4-D.  Gradients w.r.t. the input of `fractional_avg_pool`.
public func fractionalAvgPoolGrad(operationName: String? = nil, origInputTensorShape: Output, outBackprop: Output, rowPoolingSequence: Output, colPoolingSequence: Output, overlapping: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["overlapping"] = overlapping
	let opspec = OpSpec(
		type: "FractionalAvgPoolGrad",
		name: (operationName ?? "Type"),
		input: [origInputTensorShape, outBackprop, rowPoolingSequence, colPoolingSequence],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Creates an empty hash table that uses tensors as the backing store.
///It uses "open addressing" with quadratic reprobing to resolve
/// collisions.
/// 
/// This op creates a mutable hash table, specifying the type of its keys and
/// values. Each value must be a scalar. Data can be inserted into the table using
/// the insert operations. It does not support the initialization operation.
/// - Parameter emptyKey: The key used to represent empty key buckets internally. Must not
/// be used in insert or lookup operations.
/// - Parameter container: If non-empty, this table is placed in the given container.
/// Otherwise, a default container is used.
/// - Parameter sharedName: If non-empty, this table is shared under the given name across
/// multiple sessions.
/// - Parameter useNodeNameSharing: 
/// - Parameter keyDtype: Type of the table keys.
/// - Parameter valueDtype: Type of the table values.
/// - Parameter valueShape: The shape of each value.
/// - Parameter initialNumBuckets: The initial number of hash table buckets. Must be a power
/// to 2.
/// - Parameter maxLoadFactor: The maximum ratio between number of entries and number of
/// buckets before growing the table. Must be between 0 and 1.
/// - Returns: 
///	table_handle: Handle to a table.
public func mutableDenseHashTableV2(operationName: String? = nil, emptyKey: Output, container: String, sharedName: String, useNodeNameSharing: Bool, keyDtype: Any.Type, valueDtype: Any.Type, valueShape: Shape, initialNumBuckets: UInt8, maxLoadFactor: Float) throws -> Output { 
	var attrs = [String : Any]()
	attrs["container"] = container
	attrs["shared_name"] = sharedName
	attrs["use_node_name_sharing"] = useNodeNameSharing
	attrs["key_dtype"] = keyDtype
	attrs["value_dtype"] = valueDtype
	attrs["value_shape"] = valueShape
	attrs["initial_num_buckets"] = initialNumBuckets
	attrs["max_load_factor"] = maxLoadFactor
	let opspec = OpSpec(
		type: "MutableDenseHashTableV2",
		name: (operationName ?? "Type"),
		input: [emptyKey],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///A Reader that outputs fixed-length records from a file.
/// - Parameter headerBytes: Number of bytes in the header, defaults to 0.
/// - Parameter recordBytes: Number of bytes in the record.
/// - Parameter footerBytes: Number of bytes in the footer, defaults to 0.
/// - Parameter hopBytes: Number of bytes to hop before each read. Default of 0 means using
/// record_bytes.
/// - Parameter container: If non-empty, this reader is placed in the given container.
/// Otherwise, a default container is used.
/// - Parameter sharedName: If non-empty, this reader is named in the given bucket
/// with this shared_name. Otherwise, the node name is used instead.
/// - Parameter encoding: The type of encoding for the file. Currently ZLIB and GZIP
/// are supported. Defaults to none.
/// - Returns: 
///	reader_handle: The handle to reference the Reader.
public func fixedLengthRecordReaderV2(operationName: String? = nil, headerBytes: UInt8, recordBytes: UInt8, footerBytes: UInt8, hopBytes: UInt8, container: String, sharedName: String, encoding: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["header_bytes"] = headerBytes
	attrs["record_bytes"] = recordBytes
	attrs["footer_bytes"] = footerBytes
	attrs["hop_bytes"] = hopBytes
	attrs["container"] = container
	attrs["shared_name"] = sharedName
	attrs["encoding"] = encoding
	let opspec = OpSpec(
		type: "FixedLengthRecordReaderV2",
		name: (operationName ?? "Type"),
		input: [],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///A Reader that outputs fixed-length records from a file.
/// - Parameter headerBytes: Number of bytes in the header, defaults to 0.
/// - Parameter recordBytes: Number of bytes in the record.
/// - Parameter footerBytes: Number of bytes in the footer, defaults to 0.
/// - Parameter hopBytes: Number of bytes to hop before each read. Default of 0 means using
/// record_bytes.
/// - Parameter container: If non-empty, this reader is placed in the given container.
/// Otherwise, a default container is used.
/// - Parameter sharedName: If non-empty, this reader is named in the given bucket
/// with this shared_name. Otherwise, the node name is used instead.
/// - Returns: 
///	reader_handle: The handle to reference the Reader.
public func fixedLengthRecordReader(operationName: String? = nil, headerBytes: UInt8, recordBytes: UInt8, footerBytes: UInt8, hopBytes: UInt8, container: String, sharedName: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["header_bytes"] = headerBytes
	attrs["record_bytes"] = recordBytes
	attrs["footer_bytes"] = footerBytes
	attrs["hop_bytes"] = hopBytes
	attrs["container"] = container
	attrs["shared_name"] = sharedName
	let opspec = OpSpec(
		type: "FixedLengthRecordReader",
		name: (operationName ?? "Type"),
		input: [],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Emits randomized records.
/// - Parameter filePattern: Glob pattern for the data files.
/// - Parameter fileRandomSeed: Random seeds used to produce randomized records.
/// - Parameter fileShuffleShiftRatio: Shifts the list of files after the list is randomly
/// shuffled.
/// - Parameter fileBufferSize: The randomization shuffling buffer.
/// - Parameter fileParallelism: How many sstables are opened and concurrently iterated over.
/// - Parameter batchSize: The batch size.
/// - Returns: 
///	records: A tensor of shape [batch_size].
public func recordInput(operationName: String? = nil, filePattern: String, fileRandomSeed: UInt8, fileShuffleShiftRatio: Float, fileBufferSize: UInt8, fileParallelism: UInt8, batchSize: UInt8) throws -> Output { 
	var attrs = [String : Any]()
	attrs["file_pattern"] = filePattern
	attrs["file_random_seed"] = fileRandomSeed
	attrs["file_shuffle_shift_ratio"] = fileShuffleShiftRatio
	attrs["file_buffer_size"] = fileBufferSize
	attrs["file_parallelism"] = fileParallelism
	attrs["batch_size"] = batchSize
	let opspec = OpSpec(
		type: "RecordInput",
		name: (operationName ?? "Type"),
		input: [],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///A Reader that outputs the lines of a file delimited by '\n'.
/// - Parameter skipHeaderLines: Number of lines to skip from the beginning of every file.
/// - Parameter container: If non-empty, this reader is placed in the given container.
/// Otherwise, a default container is used.
/// - Parameter sharedName: If non-empty, this reader is named in the given bucket
/// with this shared_name. Otherwise, the node name is used instead.
/// - Returns: 
///	reader_handle: The handle to reference the Reader.
public func textLineReader(operationName: String? = nil, skipHeaderLines: UInt8, container: String, sharedName: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["skip_header_lines"] = skipHeaderLines
	attrs["container"] = container
	attrs["shared_name"] = sharedName
	let opspec = OpSpec(
		type: "TextLineReader",
		name: (operationName ?? "Type"),
		input: [],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Restores a tensor from checkpoint files.
///This is like `Restore` except that restored tensor can be listed as filling
/// only a slice of a larger tensor.  `shape_and_slice` specifies the shape of the
/// larger tensor and the slice that the restored tensor covers.
/// 
/// The `shape_and_slice` input has the same format as the
/// elements of the `shapes_and_slices` input of the `SaveSlices` op.
/// - Parameter filePattern: Must have a single element. The pattern of the files from
/// which we read the tensor.
/// - Parameter tensorName: Must have a single element. The name of the tensor to be
/// restored.
/// - Parameter shapeAndSlice: Scalar. The shapes and slice specifications to use when
/// restoring a tensors.
/// - Parameter dt: The type of the tensor to be restored.
/// - Parameter preferredShard: Index of file to open first if multiple files match
/// `file_pattern`. See the documentation for `Restore`.
/// - Returns: 
///	tensor: The restored tensor.
public func restoreSlice(operationName: String? = nil, filePattern: Output, tensorName: Output, shapeAndSlice: Output, dt: Any.Type, preferredShard: UInt8) throws -> Output { 
	var attrs = [String : Any]()
	attrs["dt"] = dt
	attrs["preferred_shard"] = preferredShard
	let opspec = OpSpec(
		type: "RestoreSlice",
		name: (operationName ?? "Type"),
		input: [filePattern, tensorName, shapeAndSlice],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Saves the input tensors to disk.
///The size of `tensor_names` must match the number of tensors in `data`. `data[i]`
/// is written to `filename` with name `tensor_names[i]`.
/// 
/// See also `SaveSlices`.
/// - Parameter filename: Must have a single element. The name of the file to which we write
/// the tensor.
/// - Parameter tensorNames: Shape `[N]`. The names of the tensors to be saved.
/// - Parameter data: `N` tensors to save.
/// - Parameter t: 
public func save(operationName: String? = nil, filename: Output, tensorNames: Output, data: Output, t: [Any.Type]) throws -> Operation { 
	var attrs = [String : Any]()
	attrs["T"] = t
	let opspec = OpSpec(
		type: "Save",
		name: (operationName ?? "Type"),
		input: [filename, tensorNames, data],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Stage (key, values) in the underlying container which behaves like a ordered
///associative container.   Elements are ordered by key.
/// - Parameter key: int64
/// - Parameter indices: 
/// - Parameter values: a list of tensors
/// dtypes A list of data types that inserted values should adhere to.
/// - Parameter capacity: Maximum number of elements in the Staging Area. If > 0, inserts
/// on the container will block when the capacity is reached.
/// - Parameter memoryLimit: 
/// - Parameter dtypes: 
/// - Parameter fakeDtypes: 
/// - Parameter container: If non-empty, this queue is placed in the given container. Otherwise,
/// a default container is used.
/// - Parameter sharedName: It is necessary to match this name to the matching Unstage Op.
public func orderedMapStage(operationName: String? = nil, key: Output, indices: Output, values: Output, capacity: UInt8, memoryLimit: UInt8, dtypes: [Any.Type], fakeDtypes: [Any.Type], container: String, sharedName: String) throws -> Operation { 
	var attrs = [String : Any]()
	attrs["capacity"] = capacity
	attrs["memory_limit"] = memoryLimit
	attrs["dtypes"] = dtypes
	attrs["fake_dtypes"] = fakeDtypes
	attrs["container"] = container
	attrs["shared_name"] = sharedName
	let opspec = OpSpec(
		type: "OrderedMapStage",
		name: (operationName ?? "Type"),
		input: [key, indices, values],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Saves tensors in V2 checkpoint format.
///By default, saves the named tensors in full.  If the caller wishes to save
/// specific slices of full tensors, "shape_and_slices" should be non-empty strings
/// and correspondingly well-formed.
/// - Parameter prefix: Must have a single element. The prefix of the V2 checkpoint to which we
/// write the tensors.
/// - Parameter tensorNames: shape {N}. The names of the tensors to be saved.
/// - Parameter shapeAndSlices: shape {N}.  The slice specs of the tensors to be saved.
/// Empty strings indicate that they are non-partitioned tensors.
/// - Parameter tensors: `N` tensors to save.
/// - Parameter dtypes: 
public func saveV2(operationName: String? = nil, prefix: Output, tensorNames: Output, shapeAndSlices: Output, tensors: Output, dtypes: [Any.Type]) throws -> Operation { 
	var attrs = [String : Any]()
	attrs["dtypes"] = dtypes
	let opspec = OpSpec(
		type: "SaveV2",
		name: (operationName ?? "Type"),
		input: [prefix, tensorNames, shapeAndSlices, tensors],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Returns the truth value of (x != y) element-wise.
/// * NOTE * : `NotEqual` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
/// - Parameter x: 
/// - Parameter y: 
/// - Returns: 
///	z: 
public func notEqual(operationName: String? = nil, x: Output, y: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "NotEqual",
		name: (operationName ?? "Type"),
		input: [x, y],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Greedily selects a subset of bounding boxes in descending order of score,
///pruning away boxes that have high intersection-over-union (IOU) overlap
/// with previously selected boxes.  Bounding boxes are supplied as
/// [y1, x1, y2, x2], where (y1, x1) and (y2, x2) are the coordinates of any
/// diagonal pair of box corners and the coordinates can be provided as normalized
/// (i.e., lying in the interval [0, 1]) or absolute.  Note that this algorithm
/// is agnostic to where the origin is in the coordinate system.  Note that this
/// algorithm is invariant to orthogonal transformations and translations
/// of the coordinate system; thus translating or reflections of the coordinate
/// system result in the same boxes being selected by the algorithm.
/// The output of this operation is a set of integers indexing into the input
/// collection of bounding boxes representing the selected boxes.  The bounding
/// box coordinates corresponding to the selected indices can then be obtained
/// using the `tf.gather operation`.  For example:
///   selected_indices = tf.image.non_max_suppression(
///       boxes, scores, max_output_size, iou_threshold)
///   selected_boxes = tf.gather(boxes, selected_indices)
/// - Parameter boxes: A 2-D float tensor of shape `[num_boxes, 4]`.
/// - Parameter scores: A 1-D float tensor of shape `[num_boxes]` representing a single
/// score corresponding to each box (each row of boxes).
/// - Parameter maxOutputSize: A scalar integer tensor representing the maximum number of
/// boxes to be selected by non max suppression.
/// - Parameter iouThreshold: A float representing the threshold for deciding whether boxes
/// overlap too much with respect to IOU.
/// - Returns: 
///	selected_indices: A 1-D integer tensor of shape `[M]` representing the selected
/// indices from the boxes tensor, where `M <= max_output_size`.
public func nonMaxSuppression(operationName: String? = nil, boxes: Output, scores: Output, maxOutputSize: Output, iouThreshold: Float) throws -> Output { 
	var attrs = [String : Any]()
	attrs["iou_threshold"] = iouThreshold
	let opspec = OpSpec(
		type: "NonMaxSuppression",
		name: (operationName ?? "Type"),
		input: [boxes, scores, maxOutputSize],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///BatchToSpace for N-D tensors of type T.
///This operation reshapes the "batch" dimension 0 into `M + 1` dimensions of shape
/// `block_shape + [batch]`, interleaves these blocks back into the grid defined by
/// the spatial dimensions `[1, ..., M]`, to obtain a result with the same rank as
/// the input.  The spatial dimensions of this intermediate result are then
/// optionally cropped according to `crops` to produce the output.  This is the
/// reverse of SpaceToBatch.  See below for a precise description.
/// - Parameter input: N-D with shape `input_shape = [batch] + spatial_shape + remaining_shape`,
/// where spatial_shape has M dimensions.
/// - Parameter blockShape: 1-D with shape `[M]`, all values must be >= 1.
/// - Parameter crops: 2-D with shape `[M, 2]`, all values must be >= 0.
///   `crops[i] = [crop_start, crop_end]` specifies the amount to crop from input
///   dimension `i + 1`, which corresponds to spatial dimension `i`.  It is
///   required that
///   `crop_start[i] + crop_end[i] <= block_shape[i]  *  input_shape[i + 1]`.
/// 
/// This operation is equivalent to the following steps:
/// 
/// 1. Reshape `input` to `reshaped` of shape:
///      [block_shape[0], ..., block_shape[M-1],
///       batch / prod(block_shape),
///       input_shape[1], ..., input_shape[N-1]]
/// 
/// 2. Permute dimensions of `reshaped` to produce `permuted` of shape
///      [batch / prod(block_shape),
/// 
///       input_shape[1], block_shape[0],
///       ...,
///       input_shape[M], block_shape[M-1],
/// 
///       input_shape[M+1], ..., input_shape[N-1]]
/// 
/// 3. Reshape `permuted` to produce `reshaped_permuted` of shape
///      [batch / prod(block_shape),
/// 
///       input_shape[1]  *  block_shape[0],
///       ...,
///       input_shape[M]  *  block_shape[M-1],
/// 
///       input_shape[M+1],
///       ...,
///       input_shape[N-1]]
/// 
/// 4. Crop the start and end of dimensions `[1, ..., M]` of
///    `reshaped_permuted` according to `crops` to produce the output of shape:
///      [batch / prod(block_shape),
/// 
///       input_shape[1]  *  block_shape[0] - crops[0,0] - crops[0,1],
///       ...,
///       input_shape[M]  *  block_shape[M-1] - crops[M-1,0] - crops[M-1,1],
/// 
///       input_shape[M+1], ..., input_shape[N-1]]
/// 
/// Some examples:
/// 
/// (1) For the following input of shape `[4, 1, 1, 1]`, `block_shape = [2, 2]`, and
///     `crops = [[0, 0], [0, 0]]`:
/// 
/// ```
/// [[[[1]]], [[[2]]], [[[3]]], [[[4]]]]
/// ```
/// 
/// The output tensor has shape `[1, 2, 2, 1]` and value:
/// 
/// ```
/// x = [[[[1], [2]], [[3], [4]]]]
/// ```
/// 
/// (2) For the following input of shape `[4, 1, 1, 3]`, `block_shape = [2, 2]`, and
///     `crops = [[0, 0], [0, 0]]`:
/// 
/// ```
/// [[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]], [[10, 11, 12]]]
/// ```
/// 
/// The output tensor has shape `[1, 2, 2, 3]` and value:
/// 
/// ```
/// x = [[[[1, 2, 3], [4, 5, 6]],
///       [[7, 8, 9], [10, 11, 12]]]]
/// ```
/// 
/// (3) For the following input of shape `[4, 2, 2, 1]`, `block_shape = [2, 2]`, and
///     `crops = [[0, 0], [0, 0]]`:
/// 
/// ```
/// x = [[[[1], [3]], [[9], [11]]],
///      [[[2], [4]], [[10], [12]]],
///      [[[5], [7]], [[13], [15]]],
///      [[[6], [8]], [[14], [16]]]]
/// ```
/// 
/// The output tensor has shape `[1, 4, 4, 1]` and value:
/// 
/// ```
/// x = [[[1],   [2],  [3],  [4]],
///      [[5],   [6],  [7],  [8]],
///      [[9],  [10], [11],  [12]],
///      [[13], [14], [15],  [16]]]
/// ```
/// 
/// (4) For the following input of shape `[8, 1, 3, 1]`, `block_shape = [2, 2]`, and
///     `crops = [[0, 0], [2, 0]]`:
/// 
/// ```
/// x = [[[[0], [1], [3]]], [[[0], [9], [11]]],
///      [[[0], [2], [4]]], [[[0], [10], [12]]],
///      [[[0], [5], [7]]], [[[0], [13], [15]]],
///      [[[0], [6], [8]]], [[[0], [14], [16]]]]
/// ```
/// 
/// The output tensor has shape `[2, 2, 4, 1]` and value:
/// 
/// ```
/// x = [[[[1],   [2],  [3],  [4]],
///       [[5],   [6],  [7],  [8]]],
///      [[[9],  [10], [11],  [12]],
///       [[13], [14], [15],  [16]]]]
/// ```
/// - Parameter tblockShape: 
/// - Parameter tcrops: 
/// - Returns: 
///	output: 
public func batchToSpaceND(operationName: String? = nil, input: Output, blockShape: Output, crops: Output, tblockShape: Any.Type, tcrops: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Tblock_shape"] = tblockShape
	attrs["Tcrops"] = tcrops
	let opspec = OpSpec(
		type: "BatchToSpaceND",
		name: (operationName ?? "Type"),
		input: [input, blockShape, crops],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes the gradient of the crop_and_resize op wrt the input boxes tensor.
/// - Parameter grads: A 4-D tensor of shape `[num_boxes, crop_height, crop_width, depth]`.
/// - Parameter image: A 4-D tensor of shape `[batch, image_height, image_width, depth]`.
/// Both `image_height` and `image_width` need to be positive.
/// - Parameter boxes: A 2-D tensor of shape `[num_boxes, 4]`. The `i`-th row of the tensor
/// specifies the coordinates of a box in the `box_ind[i]` image and is specified
/// in normalized coordinates `[y1, x1, y2, x2]`. A normalized coordinate value of
/// `y` is mapped to the image coordinate at `y  *  (image_height - 1)`, so as the
/// `[0, 1]` interval of normalized image height is mapped to
/// `[0, image_height - 1] in image height coordinates. We do allow y1 > y2, in
/// which case the sampled crop is an up-down flipped version of the original
/// image. The width dimension is treated similarly. Normalized coordinates
/// outside the `[0, 1]` range are allowed, in which case we use
/// `extrapolation_value` to extrapolate the input image values.
/// - Parameter boxInd: A 1-D tensor of shape `[num_boxes]` with int32 values in `[0, batch)`.
/// The value of `box_ind[i]` specifies the image that the `i`-th box refers to.
/// - Parameter method: A string specifying the interpolation method. Only 'bilinear' is
/// supported for now.
/// - Returns: 
///	output: A 2-D tensor of shape `[num_boxes, 4]`.
public func cropAndResizeGradBoxes(operationName: String? = nil, grads: Output, image: Output, boxes: Output, boxInd: Output, method: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["method"] = method
	let opspec = OpSpec(
		type: "CropAndResizeGradBoxes",
		name: (operationName ?? "Type"),
		input: [grads, image, boxes, boxInd],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Update relevant entries in ' * var' and ' * accum' according to the momentum scheme.
///Set use_nesterov = True if you want to use Nesterov momentum.
/// 
/// That is for rows we have grad for, we update var and accum as follows:
/// 
/// accum = accum  *  momentum + grad
/// var -= lr  *  accum
/// - Parameter `var`: Should be from a Variable().
/// - Parameter accum: Should be from a Variable().
/// - Parameter lr: Learning rate. Must be a scalar.
/// - Parameter grad: The gradient.
/// - Parameter indices: A vector of indices into the first dimension of var and accum.
/// - Parameter momentum: Momentum. Must be a scalar.
/// - Parameter tindices: 
/// - Parameter useLocking: If `True`, updating of the var and accum tensors will be protected
/// by a lock; otherwise the behavior is undefined, but may exhibit less
/// contention.
/// - Parameter useNesterov: If `True`, the tensor passed to compute grad will be
/// var - lr  *  momentum  *  accum, so in the end, the var you get is actually
/// var - lr  *  momentum  *  accum.
/// - Returns: 
///	out: Same as "var".
public func sparseApplyMomentum(operationName: String? = nil, `var`: Output, accum: Output, lr: Output, grad: Output, indices: Output, momentum: Output, tindices: Any.Type, useLocking: Bool, useNesterov: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Tindices"] = tindices
	attrs["use_locking"] = useLocking
	attrs["use_nesterov"] = useNesterov
	let opspec = OpSpec(
		type: "SparseApplyMomentum",
		name: (operationName ?? "Type"),
		input: [`var`, accum, lr, grad, indices, momentum],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Extracts a glimpse from the input tensor.
///Returns a set of windows called glimpses extracted at location
/// `offsets` from the input tensor. If the windows only partially
/// overlaps the inputs, the non overlapping areas will be filled with
/// random noise.
/// 
/// The result is a 4-D tensor of shape `[batch_size, glimpse_height,
/// glimpse_width, channels]`. The channels and batch dimensions are the
/// same as that of the input tensor. The height and width of the output
/// windows are specified in the `size` parameter.
/// 
/// The argument `normalized` and `centered` controls how the windows are built:
/// 
///  *  If the coordinates are normalized but not centered, 0.0 and 1.0
///   correspond to the minimum and maximum of each height and width
///   dimension.
///  *  If the coordinates are both normalized and centered, they range from
///   -1.0 to 1.0. The coordinates (-1.0, -1.0) correspond to the upper
///   left corner, the lower right corner is located at (1.0, 1.0) and the
///   center is at (0, 0).
///  *  If the coordinates are not normalized they are interpreted as
///   numbers of pixels.
/// - Parameter input: A 4-D float tensor of shape `[batch_size, height, width, channels]`.
/// - Parameter size: A 1-D tensor of 2 elements containing the size of the glimpses
/// to extract.  The glimpse height must be specified first, following
/// by the glimpse width.
/// - Parameter offsets: A 2-D integer tensor of shape `[batch_size, 2]` containing
/// the y, x locations of the center of each window.
/// - Parameter centered: indicates if the offset coordinates are centered relative to
/// the image, in which case the (0, 0) offset is relative to the center
/// of the input images. If false, the (0,0) offset corresponds to the
/// upper left corner of the input images.
/// - Parameter normalized: indicates if the offset coordinates are normalized.
/// - Parameter uniformNoise: indicates if the noise should be generated using a
/// uniform distribution or a Gaussian distribution.
/// - Returns: 
///	glimpse: A tensor representing the glimpses `[batch_size,
/// glimpse_height, glimpse_width, channels]`.
public func extractGlimpse(operationName: String? = nil, input: Output, size: Output, offsets: Output, centered: Bool, normalized: Bool, uniformNoise: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["centered"] = centered
	attrs["normalized"] = normalized
	attrs["uniform_noise"] = uniformNoise
	let opspec = OpSpec(
		type: "ExtractGlimpse",
		name: (operationName ?? "Type"),
		input: [input, size, offsets],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Update relevant entries in ' * var' according to the Ftrl-proximal scheme.
///That is for rows we have grad for, we update var, accum and linear as follows:
/// grad_with_shrinkage = grad + 2  *  l2_shrinkage  *  var
/// accum_new = accum + grad_with_shrinkage  *  grad_with_shrinkage
/// linear += grad_with_shrinkage +
///     (accum_new// ^(-lr_power) - accum// ^(-lr_power)) / lr  *  var
/// quadratic = 1.0 / (accum_new// ^(lr_power)  *  lr) + 2  *  l2
/// var = (sign(linear)  *  l1 - linear) / quadratic if |linear| > l1 else 0.0
/// accum = accum_new
/// - Parameter `var`: Should be from a Variable().
/// - Parameter accum: Should be from a Variable().
/// - Parameter linear: Should be from a Variable().
/// - Parameter grad: The gradient.
/// - Parameter indices: A vector of indices into the first dimension of var and accum.
/// - Parameter lr: Scaling factor. Must be a scalar.
/// - Parameter l1: L1 regularization. Must be a scalar.
/// - Parameter l2: L2 shrinkage regulariation. Must be a scalar.
/// - Parameter l2Shrinkage: 
/// - Parameter lrPower: Scaling factor. Must be a scalar.
/// - Parameter tindices: 
/// - Parameter useLocking: If `True`, updating of the var and accum tensors will be protected
/// by a lock; otherwise the behavior is undefined, but may exhibit less
/// contention.
public func resourceSparseApplyFtrlV2(operationName: String? = nil, `var`: Output, accum: Output, linear: Output, grad: Output, indices: Output, lr: Output, l1: Output, l2: Output, l2Shrinkage: Output, lrPower: Output, tindices: Any.Type, useLocking: Bool) throws -> Operation { 
	var attrs = [String : Any]()
	attrs["Tindices"] = tindices
	attrs["use_locking"] = useLocking
	let opspec = OpSpec(
		type: "ResourceSparseApplyFtrlV2",
		name: (operationName ?? "Type"),
		input: [`var`, accum, linear, grad, indices, lr, l1, l2, l2Shrinkage, lrPower],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Encode audio data using the WAV file format.
///This operation will generate a string suitable to be saved out to create a .wav
/// audio file. It will be encoded in the 16-bit PCM format. It takes in float
/// values in the range -1.0f to 1.0f, and any outside that value will be clamped to
/// that range.
/// 
/// `audio` is a 2-D float Tensor of shape `[length, channels]`.
/// `sample_rate` is a scalar Tensor holding the rate to use (e.g. 44100).
/// - Parameter audio: 2-D with shape `[length, channels]`.
/// - Parameter sampleRate: Scalar containing the sample frequency.
/// - Returns: 
///	contents: 0-D. WAV-encoded file contents.
public func encodeWav(operationName: String? = nil, audio: Output, sampleRate: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "EncodeWav",
		name: (operationName ?? "Type"),
		input: [audio, sampleRate],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Generate a single randomly distorted bounding box for an image.
///Bounding box annotations are often supplied in addition to ground-truth labels
/// in image recognition or object localization tasks. A common technique for
/// training such a system is to randomly distort an image while preserving
/// its content, i.e.  * data augmentation * . This Op outputs a randomly distorted
/// localization of an object, i.e. bounding box, given an `image_size`,
/// `bounding_boxes` and a series of constraints.
/// 
/// The output of this Op is a single bounding box that may be used to crop the
/// original image. The output is returned as 3 tensors: `begin`, `size` and
/// `bboxes`. The first 2 tensors can be fed directly into `tf.slice` to crop the
/// image. The latter may be supplied to `tf.image.draw_bounding_boxes` to visualize
/// what the bounding box looks like.
/// 
/// Bounding boxes are supplied and returned as `[y_min, x_min, y_max, x_max]`. The
/// bounding box coordinates are floats in `[0.0, 1.0]` relative to the width and
/// height of the underlying image.
/// 
/// For example,
/// 
/// ```python
///     # Generate a single distorted bounding box.
///     begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
///         tf.shape(image),
///         bounding_boxes=bounding_boxes)
/// 
///     # Draw the bounding box in an image summary.
///     image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0),
///                                                   bbox_for_draw)
///     tf.image_summary('images_with_box', image_with_box)
/// 
///     # Employ the bounding box to distort the image.
///     distorted_image = tf.slice(image, begin, size)
/// ```
/// 
/// Note that if no bounding box information is available, setting
/// `use_image_if_no_bounding_boxes = true` will assume there is a single implicit
/// bounding box covering the whole image. If `use_image_if_no_bounding_boxes` is
/// false and no bounding boxes are supplied, an error is raised.
/// - Parameter imageSize: 1-D, containing `[height, width, channels]`.
/// - Parameter boundingBoxes: 3-D with shape `[batch, N, 4]` describing the N bounding boxes
/// associated with the image.
/// - Parameter minObjectCovered: The cropped area of the image must contain at least this
/// fraction of any bounding box supplied. The value of this parameter should be
/// non-negative. In the case of 0, the cropped area does not need to overlap
/// any of the bounding boxes supplied.
/// - Parameter seed: If either `seed` or `seed2` are set to non-zero, the random number
/// generator is seeded by the given `seed`.  Otherwise, it is seeded by a random
/// seed.
/// - Parameter seed2: A second seed to avoid seed collision.
/// - Parameter aspectRatioRange: The cropped area of the image must have an aspect ratio =
/// width / height within this range.
/// - Parameter areaRange: The cropped area of the image must contain a fraction of the
/// supplied image within in this range.
/// - Parameter maxAttempts: Number of attempts at generating a cropped region of the image
/// of the specified constraints. After `max_attempts` failures, return the entire
/// image.
/// - Parameter useImageIfNoBoundingBoxes: Controls behavior if no bounding boxes supplied.
/// If true, assume an implicit bounding box covering the whole input. If false,
/// raise an error.
/// - Returns: 
///	begin: 1-D, containing `[offset_height, offset_width, 0]`. Provide as input to
/// `tf.slice`.
///	size: 1-D, containing `[target_height, target_width, -1]`. Provide as input to
/// `tf.slice`.
///	bboxes: 3-D with shape `[1, 1, 4]` containing the distorted bounding box.
/// Provide as input to `tf.image.draw_bounding_boxes`.
public func sampleDistortedBoundingBoxV2(operationName: String? = nil, imageSize: Output, boundingBoxes: Output, minObjectCovered: Output, seed: UInt8, seed2: UInt8, aspectRatioRange: [Float], areaRange: [Float], maxAttempts: UInt8, useImageIfNoBoundingBoxes: Bool) throws -> (begin: Output, size: Output, bboxes: Output) { 
	var attrs = [String : Any]()
	attrs["seed"] = seed
	attrs["seed2"] = seed2
	attrs["aspect_ratio_range"] = aspectRatioRange
	attrs["area_range"] = areaRange
	attrs["max_attempts"] = maxAttempts
	attrs["use_image_if_no_bounding_boxes"] = useImageIfNoBoundingBoxes
	let opspec = OpSpec(
		type: "SampleDistortedBoundingBoxV2",
		name: (operationName ?? "Type"),
		input: [imageSize, boundingBoxes, minObjectCovered],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (begin: op.output(at: 0), size: op.output(at: 1), bboxes: op.output(at: 2))
} 

///Adjust the saturation of one or more images.
///`images` is a tensor of at least 3 dimensions.  The last dimension is
/// interpretted as channels, and must be three.
/// 
/// The input image is considered in the RGB colorspace. Conceptually, the RGB
/// colors are first mapped into HSV. A scale is then applied all the saturation
/// values, and then remapped back to RGB colorspace.
/// - Parameter images: Images to adjust.  At least 3-D.
/// - Parameter scale: A float scale to add to the saturation.
/// - Returns: 
///	output: The hue-adjusted image or images.
public func adjustSaturation(operationName: String? = nil, images: Output, scale: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "AdjustSaturation",
		name: (operationName ?? "Type"),
		input: [images, scale],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes the sign and the log of the absolute value of the determinant of
///one or more square matrices.
/// 
/// The input is a tensor of shape `[N, M, M]` whose inner-most 2 dimensions
/// form square matrices. The outputs are two tensors containing the signs and
/// absolute values of the log determinants for all N input submatrices
/// `[..., :, :]` such that the determinant = sign * exp(log_abs_determinant).
/// The log_abs_determinant is computed as det(P) * sum(log(diag(LU))) where LU
/// is the LU decomposition of the input and P is the corresponding
/// permutation matrix.
/// - Parameter input: Shape is `[N, M, M]`.
/// - Returns: 
///	sign: The signs of the log determinants of the inputs. Shape is `[N]`.
///	log_abs_determinant: The logs of the absolute values of the determinants
/// of the N input matrices.  Shape is `[N]`.
public func logMatrixDeterminant(operationName: String? = nil, input: Output) throws -> (sign: Output, logAbsDeterminant: Output) { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "LogMatrixDeterminant",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (sign: op.output(at: 0), logAbsDeterminant: op.output(at: 1))
} 

///Resize `images` to `size` using bilinear interpolation.
///Input images can be of different types but output images are always float.
/// - Parameter images: 4-D with shape `[batch, height, width, channels]`.
/// - Parameter size: = A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
/// new size for the images.
/// - Parameter alignCorners: If true, rescale input by (new_height - 1) / (height - 1), which
/// exactly aligns the 4 corners of images and resized images. If false, rescale
/// by new_height / height. Treat similarly the width dimension.
/// - Returns: 
///	resized_images: 4-D with shape
/// `[batch, new_height, new_width, channels]`.
public func resizeBilinear(operationName: String? = nil, images: Output, size: Output, alignCorners: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["align_corners"] = alignCorners
	let opspec = OpSpec(
		type: "ResizeBilinear",
		name: (operationName ?? "Type"),
		input: [images, size],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Returns a tensor that may be mutated, but only persists within a single step.
///This is an experimental op for internal use only and it is possible to use this
/// op in unsafe ways.  DO NOT USE unless you fully understand the risks.
/// 
/// It is the caller's responsibility to ensure that 'ref' is eventually passed to a
/// matching 'DestroyTemporaryVariable' op after all other uses have completed.
/// 
/// Outputs a ref to the tensor state so it may be read or modified.
/// 
///   E.g.
///       var = state_ops._temporary_variable([1, 2], types.float_)
///       var_name = var.op.name
///       var = state_ops.assign(var, [[4.0, 5.0]])
///       var = state_ops.assign_add(var, [[6.0, 7.0]])
///       final = state_ops._destroy_temporary_variable(var, var_name=var_name)
/// - Parameter shape: The shape of the variable tensor.
/// - Parameter dtype: The type of elements in the variable tensor.
/// - Parameter varName: Overrides the name used for the temporary variable resource. Default
/// value is the name of the 'TemporaryVariable' op (which is guaranteed unique).
/// - Returns: 
///	ref: A reference to the variable tensor.
public func temporaryVariable(operationName: String? = nil, shape: Shape, dtype: Any.Type, varName: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["shape"] = shape
	attrs["dtype"] = dtype
	attrs["var_name"] = varName
	let opspec = OpSpec(
		type: "TemporaryVariable",
		name: (operationName ?? "Type"),
		input: [],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///JPEG-encode an image.
///`image` is a 3-D uint8 Tensor of shape `[height, width, channels]`.
/// 
/// The attr `format` can be used to override the color format of the encoded
/// output.  Values can be:
/// 
///  *    `''`: Use a default format based on the number of channels in the image.
///  *    `grayscale`: Output a grayscale JPEG image.  The `channels` dimension
///     of `image` must be 1.
///  *    `rgb`: Output an RGB JPEG image. The `channels` dimension
///     of `image` must be 3.
/// 
/// If `format` is not specified or is the empty string, a default format is picked
/// in function of the number of channels in `image`:
/// 
///  *    1: Output a grayscale image.
///  *    3: Output an RGB image.
/// - Parameter image: 3-D with shape `[height, width, channels]`.
/// - Parameter format: Per pixel image format.
/// - Parameter quality: Quality of the compression from 0 to 100 (higher is better and slower).
/// - Parameter progressive: If True, create a JPEG that loads progressively (coarse to fine).
/// - Parameter optimizeSize: If True, spend CPU/RAM to reduce size with no quality change.
/// - Parameter chromaDownsampling: See http://en.wikipedia.org/wiki/Chroma_subsampling.
/// - Parameter densityUnit: Unit used to specify `x_density` and `y_density`:
/// pixels per inch (`'in'`) or centimeter (`'cm'`).
/// - Parameter xDensity: Horizontal pixels per density unit.
/// - Parameter yDensity: Vertical pixels per density unit.
/// - Parameter xmpMetadata: If not empty, embed this XMP metadata in the image header.
/// - Returns: 
///	contents: 0-D. JPEG-encoded image.
public func encodeJpeg(operationName: String? = nil, image: Output, format: String, quality: UInt8, progressive: Bool, optimizeSize: Bool, chromaDownsampling: Bool, densityUnit: String, xDensity: UInt8, yDensity: UInt8, xmpMetadata: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["format"] = format
	attrs["quality"] = quality
	attrs["progressive"] = progressive
	attrs["optimize_size"] = optimizeSize
	attrs["chroma_downsampling"] = chromaDownsampling
	attrs["density_unit"] = densityUnit
	attrs["x_density"] = xDensity
	attrs["y_density"] = yDensity
	attrs["xmp_metadata"] = xmpMetadata
	let opspec = OpSpec(
		type: "EncodeJpeg",
		name: (operationName ?? "Type"),
		input: [image],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Op returns the number of incomplete elements in the underlying container.
/// - Parameter capacity: 
/// - Parameter memoryLimit: 
/// - Parameter dtypes: 
/// - Parameter container: 
/// - Parameter sharedName: 
/// - Returns: 
///	size: 
public func orderedMapIncompleteSize(operationName: String? = nil, capacity: UInt8, memoryLimit: UInt8, dtypes: [Any.Type], container: String, sharedName: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["capacity"] = capacity
	attrs["memory_limit"] = memoryLimit
	attrs["dtypes"] = dtypes
	attrs["container"] = container
	attrs["shared_name"] = sharedName
	let opspec = OpSpec(
		type: "OrderedMapIncompleteSize",
		name: (operationName ?? "Type"),
		input: [],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Resize quantized `images` to `size` using quantized bilinear interpolation.
///Input images and output images must be quantized types.
/// - Parameter images: 4-D with shape `[batch, height, width, channels]`.
/// - Parameter size: = A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
/// new size for the images.
/// - Parameter min: 
/// - Parameter max: 
/// - Parameter alignCorners: If true, rescale input by (new_height - 1) / (height - 1), which
/// exactly aligns the 4 corners of images and resized images. If false, rescale
/// by new_height / height. Treat similarly the width dimension.
/// - Returns: 
///	resized_images: 4-D with shape
/// `[batch, new_height, new_width, channels]`.
///	out_min: 
///	out_max: 
public func quantizedResizeBilinear(operationName: String? = nil, images: Output, size: Output, min: Output, max: Output, alignCorners: Bool) throws -> (resizedImages: Output, outMin: Output, outMax: Output) { 
	var attrs = [String : Any]()
	attrs["align_corners"] = alignCorners
	let opspec = OpSpec(
		type: "QuantizedResizeBilinear",
		name: (operationName ?? "Type"),
		input: [images, size, min, max],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (resizedImages: op.output(at: 0), outMin: op.output(at: 1), outMax: op.output(at: 2))
} 

///Batch normalization.
///This op is deprecated. Prefer `tf.nn.batch_normalization`.
/// - Parameter t: A 4D input Tensor.
/// - Parameter m: A 1D mean Tensor with size matching the last dimension of t.
/// This is the first output from tf.nn.moments,
/// or a saved moving average thereof.
/// - Parameter v: A 1D variance Tensor with size matching the last dimension of t.
/// This is the second output from tf.nn.moments,
/// or a saved moving average thereof.
/// - Parameter beta: A 1D beta Tensor with size matching the last dimension of t.
/// An offset to be added to the normalized tensor.
/// - Parameter gamma: A 1D gamma Tensor with size matching the last dimension of t.
/// If "scale_after_normalization" is true, this tensor will be multiplied
/// with the normalized tensor.
/// - Parameter varianceEpsilon: A small float number to avoid dividing by 0.
/// - Parameter scaleAfterNormalization: A bool indicating whether the resulted tensor
/// needs to be multiplied with gamma.
/// - Returns: 
///	result: 
public func batchNormWithGlobalNormalization(operationName: String? = nil, t: Output, m: Output, v: Output, beta: Output, gamma: Output, varianceEpsilon: Float, scaleAfterNormalization: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["variance_epsilon"] = varianceEpsilon
	attrs["scale_after_normalization"] = scaleAfterNormalization
	let opspec = OpSpec(
		type: "BatchNormWithGlobalNormalization",
		name: (operationName ?? "Type"),
		input: [t, m, v, beta, gamma],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Encode strings into web-safe base64 format.
///Refer to the following article for more information on base64 format:
/// en.wikipedia.org/wiki/Base64. Base64 strings may have padding with '=' at the
/// end so that the encoded has length multiple of 4. See Padding section of the
/// link above.
/// 
/// Web-safe means that the encoder uses - and _ instead of + and /.
/// - Parameter input: Strings to be encoded.
/// - Parameter pad: Bool whether padding is applied at the ends.
/// - Returns: 
///	output: Input strings encoded in base64.
public func encodeBase64(operationName: String? = nil, input: Output, pad: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["pad"] = pad
	let opspec = OpSpec(
		type: "EncodeBase64",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes gradients for SparseSegmentSqrtN.
///Returns tensor "output" with same shape as grad, except for dimension 0 whose
/// value is output_dim0.
/// - Parameter grad: gradient propagated to the SparseSegmentSqrtN op.
/// - Parameter indices: indices passed to the corresponding SparseSegmentSqrtN op.
/// - Parameter segmentIds: segment_ids passed to the corresponding SparseSegmentSqrtN op.
/// - Parameter outputDim0: dimension 0 of "data" passed to SparseSegmentSqrtN op.
/// - Parameter tidx: 
/// - Returns: 
///	output: 
public func sparseSegmentSqrtNGrad(operationName: String? = nil, grad: Output, indices: Output, segmentIds: Output, outputDim0: Output, tidx: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Tidx"] = tidx
	let opspec = OpSpec(
		type: "SparseSegmentSqrtNGrad",
		name: (operationName ?? "Type"),
		input: [grad, indices, segmentIds, outputDim0],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Fake-quantize the 'inputs' tensor of type float and one of the shapes: `[d]`,
///`[b, d]` `[b, h, w, d]` via per-channel floats `min` and `max` of shape `[d]`
/// to 'outputs' tensor of same shape as `inputs`.
/// 
/// `[min; max]` define the clamping range for the `inputs` data.
/// `inputs` values are quantized into the quantization range (`[0; 2// ^num_bits - 1]`
/// when `narrow_range` is false and `[1; 2// ^num_bits - 1]` when it is true) and
/// then de-quantized and output as floats in `[min; max]` interval.
/// `num_bits` is the bitwidth of the quantization; between 2 and 8, inclusive.
/// 
/// This operation has a gradient and thus allows for training `min` and `max`
/// values.
/// - Parameter inputs: 
/// - Parameter min: 
/// - Parameter max: 
/// - Parameter numBits: 
/// - Parameter narrowRange: 
/// - Returns: 
///	outputs: 
public func fakeQuantWithMinMaxVarsPerChannel(operationName: String? = nil, inputs: Output, min: Output, max: Output, numBits: UInt8, narrowRange: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["num_bits"] = numBits
	attrs["narrow_range"] = narrowRange
	let opspec = OpSpec(
		type: "FakeQuantWithMinMaxVarsPerChannel",
		name: (operationName ?? "Type"),
		input: [inputs, min, max],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Converts the given string representing a handle to an iterator to a resource.
/// - Parameter stringHandle: A string representation of the given handle.
/// - Parameter outputTypes: If specified, defines the type of each tuple component in an
/// element produced by the resulting iterator.
/// - Parameter outputShapes: If specified, defines the shape of each tuple component in an
/// element produced by the resulting iterator.
/// - Returns: 
///	resource_handle: A handle to an iterator resource.
public func iteratorFromStringHandle(operationName: String? = nil, stringHandle: Output, outputTypes: [Any.Type], outputShapes: [Shape]) throws -> Output { 
	var attrs = [String : Any]()
	attrs["output_types"] = outputTypes
	attrs["output_shapes"] = outputShapes
	let opspec = OpSpec(
		type: "IteratorFromStringHandle",
		name: (operationName ?? "Type"),
		input: [stringHandle],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Creates or finds a child frame, and makes `data` available to the child frame.
///This op is used together with `Exit` to create loops in the graph.
/// The unique `frame_name` is used by the `Executor` to identify frames. If
/// `is_constant` is true, `output` is a constant in the child frame; otherwise
/// it may be changed in the child frame. At most `parallel_iterations` iterations
/// are run in parallel in the child frame.
/// - Parameter data: The tensor to be made available to the child frame.
/// - Parameter frameName: The name of the child frame.
/// - Parameter isConstant: If true, the output is constant within the child frame.
/// - Parameter parallelIterations: The number of iterations allowed to run in parallel.
/// - Returns: 
///	output: The same tensor as `data`.
public func enter(operationName: String? = nil, data: Output, frameName: String, isConstant: Bool, parallelIterations: UInt8) throws -> Output { 
	var attrs = [String : Any]()
	attrs["frame_name"] = frameName
	attrs["is_constant"] = isConstant
	attrs["parallel_iterations"] = parallelIterations
	let opspec = OpSpec(
		type: "Enter",
		name: (operationName ?? "Type"),
		input: [data],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///PNG-encode an image.
///`image` is a 3-D uint8 or uint16 Tensor of shape `[height, width, channels]`
/// where `channels` is:
/// 
///  *    1: for grayscale.
///  *    2: for grayscale + alpha.
///  *    3: for RGB.
///  *    4: for RGBA.
/// 
/// The ZLIB compression level, `compression`, can be -1 for the PNG-encoder
/// default or a value from 0 to 9.  9 is the highest compression level, generating
/// the smallest output, but is slower.
/// - Parameter image: 3-D with shape `[height, width, channels]`.
/// - Parameter compression: Compression level.
/// - Returns: 
///	contents: 0-D. PNG-encoded image.
public func encodePng(operationName: String? = nil, image: Output, compression: UInt8) throws -> Output { 
	var attrs = [String : Any]()
	attrs["compression"] = compression
	let opspec = OpSpec(
		type: "EncodePng",
		name: (operationName ?? "Type"),
		input: [image],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Gets the next output from the given iterator.
/// - Parameter iterator: 
/// - Parameter outputTypes: 
/// - Parameter outputShapes: 
/// - Returns: 
///	components: 
public func iteratorGetNext(operationName: String? = nil, iterator: Output, outputTypes: [Any.Type], outputShapes: [Shape]) throws -> Output { 
	var attrs = [String : Any]()
	attrs["output_types"] = outputTypes
	attrs["output_shapes"] = outputShapes
	let opspec = OpSpec(
		type: "IteratorGetNext",
		name: (operationName ?? "Type"),
		input: [iterator],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Outputs random values from a normal distribution.
///The generated values will have mean 0 and standard deviation 1.
/// - Parameter shape: The shape of the output tensor.
/// - Parameter seed: If either `seed` or `seed2` are set to be non-zero, the random number
/// generator is seeded by the given seed.  Otherwise, it is seeded by a
/// random seed.
/// - Parameter seed2: A second seed to avoid seed collision.
/// - Parameter dtype: The type of the output.
/// - Returns: 
///	output: A tensor of the specified shape filled with random normal values.
public func randomStandardNormal(operationName: String? = nil, shape: Output, seed: UInt8, seed2: UInt8, dtype: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["seed"] = seed
	attrs["seed2"] = seed2
	attrs["dtype"] = dtype
	let opspec = OpSpec(
		type: "RandomStandardNormal",
		name: (operationName ?? "Type"),
		input: [shape],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Produces the average pool of the input tensor for quantized types.
/// - Parameter input: 4-D with shape `[batch, height, width, channels]`.
/// - Parameter minInput: The float value that the lowest quantized input value represents.
/// - Parameter maxInput: The float value that the highest quantized input value represents.
/// - Parameter ksize: The size of the window for each dimension of the input tensor.
/// The length must be 4 to match the number of dimensions of the input.
/// - Parameter strides: The stride of the sliding window for each dimension of the input
/// tensor.  The length must be 4 to match the number of dimensions of the input.
/// - Parameter padding: The type of padding algorithm to use.
/// - Returns: 
///	output: 
///	min_output: The float value that the lowest quantized output value represents.
///	max_output: The float value that the highest quantized output value represents.
public func quantizedAvgPool(operationName: String? = nil, input: Output, minInput: Output, maxInput: Output, ksize: [Int64], strides: [Int64], padding: String) throws -> (output: Output, minOutput: Output, maxOutput: Output) { 
	var attrs = [String : Any]()
	attrs["ksize"] = ksize
	attrs["strides"] = strides
	attrs["padding"] = padding
	let opspec = OpSpec(
		type: "QuantizedAvgPool",
		name: (operationName ?? "Type"),
		input: [input, minInput, maxInput],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (output: op.output(at: 0), minOutput: op.output(at: 1), maxOutput: op.output(at: 2))
} 

///Gather slices from the variable pointed to by `resource` according to `indices`.
///`indices` must be an integer tensor of any dimension (usually 0-D or 1-D).
/// Produces an output tensor with shape `indices.shape + params.shape[1:]` where:
/// 
/// ```python
///     # Scalar indices
///     output[:, ..., :] = params[indices, :, ... :]
/// 
///     # Vector indices
///     output[i, :, ..., :] = params[indices[i], :, ... :]
/// 
///     # Higher rank indices
///     output[i, ..., j, :, ... :] = params[indices[i, ..., j], :, ..., :]
/// ```
/// - Parameter resource: 
/// - Parameter indices: 
/// - Parameter validateIndices: 
/// - Parameter dtype: 
/// - Parameter tindices: 
/// - Returns: 
///	output: 
public func resourceGather(operationName: String? = nil, resource: Output, indices: Output, validateIndices: Bool, dtype: Any.Type, tindices: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["validate_indices"] = validateIndices
	attrs["dtype"] = dtype
	attrs["Tindices"] = tindices
	let opspec = OpSpec(
		type: "ResourceGather",
		name: (operationName ?? "Type"),
		input: [resource, indices],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Adjust the contrast of one or more images.
///`images` is a tensor of at least 3 dimensions.  The last 3 dimensions are
/// interpreted as `[height, width, channels]`.  The other dimensions only
/// represent a collection of images, such as `[batch, height, width, channels].`
/// 
/// Contrast is adjusted independently for each channel of each image.
/// 
/// For each channel, the Op first computes the mean of the image pixels in the
/// channel and then adjusts each component of each pixel to
/// `(x - mean)  *  contrast_factor + mean`.
/// - Parameter images: Images to adjust.  At least 3-D.
/// - Parameter contrastFactor: A float multiplier for adjusting contrast.
/// - Returns: 
///	output: The contrast-adjusted image or images.
public func adjustContrastv2(operationName: String? = nil, images: Output, contrastFactor: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "AdjustContrastv2",
		name: (operationName ?? "Type"),
		input: [images, contrastFactor],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Makes a "one-shot" iterator that can be iterated only once.
///A one-shot iterator bundles the logic for defining the dataset and
/// the state of the iterator in a single op, which allows simple input
/// pipelines to be defined without an additional initialization
/// ("MakeIterator") step.
/// 
/// One-shot iterators have the following limitations:
/// 
///  *  They do not support parameterization: all logic for creating the underlying
///   dataset must be bundled in the `dataset_factory` function.
///  *  They are not resettable. Once a one-shot iterator reaches the end of its
///   underlying dataset, subsequent "IteratorGetNext" operations on that
///   iterator will always produce an `OutOfRange` error.
/// 
/// For greater flexibility, use "Iterator" and "MakeIterator" to define
/// an iterator using an arbitrary subgraph, which may capture tensors
/// (including fed values) as parameters, and which may be reset multiple
/// times by rerunning "MakeIterator".
/// - Parameter datasetFactory: A function of type `() -> DT_VARIANT`, where the returned
/// DT_VARIANT is a dataset.
/// - Parameter outputTypes: 
/// - Parameter outputShapes: 
/// - Parameter container: 
/// - Parameter sharedName: 
/// - Returns: 
///	handle: A handle to the iterator that can be passed to an "IteratorGetNext"
/// op.
public func oneShotIterator(operationName: String? = nil, datasetFactory: Tensorflow_NameAttrList, outputTypes: [Any.Type], outputShapes: [Shape], container: String, sharedName: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["dataset_factory"] = datasetFactory
	attrs["output_types"] = outputTypes
	attrs["output_shapes"] = outputShapes
	attrs["container"] = container
	attrs["shared_name"] = sharedName
	let opspec = OpSpec(
		type: "OneShotIterator",
		name: (operationName ?? "Type"),
		input: [],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Outputs random values from a normal distribution. The parameters may each be a
///scalar which applies to the entire output, or a vector of length shape[0] which
/// stores the parameters for each batch.
/// - Parameter shape: The shape of the output tensor. Batches are indexed by the 0th dimension.
/// - Parameter means: The mean parameter of each batch.
/// - Parameter stdevs: The standard deviation parameter of each batch. Must be greater than 0.
/// - Parameter minvals: The minimum cutoff. May be -infinity.
/// - Parameter maxvals: The maximum cutoff. May be +infinity, and must be more than the minval
/// for each batch.
/// - Parameter seed: If either `seed` or `seed2` are set to be non-zero, the random number
/// generator is seeded by the given seed.  Otherwise, it is seeded by a
/// random seed.
/// - Parameter seed2: A second seed to avoid seed collision.
/// - Parameter dtype: The type of the output.
/// - Returns: 
///	output: A matrix of shape num_batches x samples_per_batch, filled with random
/// truncated normal values using the parameters for each row.
public func parameterizedTruncatedNormal(operationName: String? = nil, shape: Output, means: Output, stdevs: Output, minvals: Output, maxvals: Output, seed: UInt8, seed2: UInt8, dtype: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["seed"] = seed
	attrs["seed2"] = seed2
	attrs["dtype"] = dtype
	let opspec = OpSpec(
		type: "ParameterizedTruncatedNormal",
		name: (operationName ?? "Type"),
		input: [shape, means, stdevs, minvals, maxvals],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Dequeues `n` tuples of one or more tensors from the given queue.
///This operation is not supported by all queues.  If a queue does not support
/// DequeueUpTo, then an Unimplemented error is returned.
/// 
/// If the queue is closed and there are more than 0 but less than `n`
/// elements remaining, then instead of returning an OutOfRange error like
/// QueueDequeueMany, less than `n` elements are returned immediately.  If
/// the queue is closed and there are 0 elements left in the queue, then
/// an OutOfRange error is returned just like in QueueDequeueMany.
/// Otherwise the behavior is identical to QueueDequeueMany:
/// 
/// This operation concatenates queue-element component tensors along the
/// 0th dimension to make a single component tensor.  All of the components
/// in the dequeued tuple will have size `n` in the 0th dimension.
/// 
/// This operation has k outputs, where `k` is the number of components in
/// the tuples stored in the given queue, and output `i` is the ith
/// component of the dequeued tuple.
/// - Parameter handle: The handle to a queue.
/// - Parameter n: The number of tuples to dequeue.
/// - Parameter componentTypes: The type of each component in a tuple.
/// - Parameter timeoutMs: If the queue has fewer than n elements, this operation
/// will block for up to timeout_ms milliseconds.
/// Note: This option is not supported yet.
/// - Returns: 
///	components: One or more tensors that were dequeued as a tuple.
public func queueDequeueUpTo(operationName: String? = nil, handle: Output, n: Output, componentTypes: [Any.Type], timeoutMs: UInt8) throws -> Output { 
	var attrs = [String : Any]()
	attrs["component_types"] = componentTypes
	attrs["timeout_ms"] = timeoutMs
	let opspec = OpSpec(
		type: "QueueDequeueUpTo",
		name: (operationName ?? "Type"),
		input: [handle, n],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Restores the state of the `iterator` from the checkpoint saved at `path` using "SaveIterator".
/// - Parameter iterator: 
/// - Parameter path: 
public func restoreIterator(operationName: String? = nil, iterator: Output, path: Output) throws -> Operation { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "RestoreIterator",
		name: (operationName ?? "Type"),
		input: [iterator, path],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Update entries in ' * var' and ' * accum' according to the proximal adagrad scheme.
/// - Parameter `var`: Should be from a Variable().
/// - Parameter gradientAccumulator: Should be from a Variable().
/// - Parameter gradientSquaredAccumulator: Should be from a Variable().
/// - Parameter grad: The gradient.
/// - Parameter indices: A vector of indices into the first dimension of var and accum.
/// - Parameter lr: Learning rate. Must be a scalar.
/// - Parameter l1: L1 regularization. Must be a scalar.
/// - Parameter l2: L2 regularization. Must be a scalar.
/// - Parameter globalStep: Training step number. Must be a scalar.
/// - Parameter tindices: 
/// - Parameter useLocking: If True, updating of the var and accum tensors will be protected by
/// a lock; otherwise the behavior is undefined, but may exhibit less contention.
public func resourceSparseApplyAdagradDA(operationName: String? = nil, `var`: Output, gradientAccumulator: Output, gradientSquaredAccumulator: Output, grad: Output, indices: Output, lr: Output, l1: Output, l2: Output, globalStep: Output, tindices: Any.Type, useLocking: Bool) throws -> Operation { 
	var attrs = [String : Any]()
	attrs["Tindices"] = tindices
	attrs["use_locking"] = useLocking
	let opspec = OpSpec(
		type: "ResourceSparseApplyAdagradDA",
		name: (operationName ?? "Type"),
		input: [`var`, gradientAccumulator, gradientSquaredAccumulator, grad, indices, lr, l1, l2, globalStep],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Creates a dataset that emits the records from one or more TFRecord files.
/// - Parameter filenames: A scalar or vector containing the name(s) of the file(s) to be
/// read.
/// - Parameter compressionType: A scalar containing either (i) the empty string (no
/// compression), (ii) "ZLIB", or (iii) "GZIP".
/// - Parameter bufferSize: A scalar representing the number of bytes to buffer. A value of
/// 0 means no buffering will be performed.
/// - Returns: 
///	handle: 
public func tFRecordDataset(operationName: String? = nil, filenames: Output, compressionType: Output, bufferSize: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "TFRecordDataset",
		name: (operationName ?? "Type"),
		input: [filenames, compressionType, bufferSize],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Forwards `data` to the output port determined by `pred`.
///If `pred` is true, the `data` input is forwarded to `output_true`. Otherwise,
/// the data goes to `output_false`.
/// 
/// See also `RefSwitch` and `Merge`.
/// - Parameter data: The tensor to be forwarded to the appropriate output.
/// - Parameter pred: A scalar that specifies which output port will receive data.
/// - Returns: 
///	output_false: If `pred` is false, data will be forwarded to this output.
///	output_true: If `pred` is true, data will be forwarded to this output.
public func `switch`(operationName: String? = nil, data: Output, pred: Output) throws -> (outputFalse: Output, outputTrue: Output) { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "`switch`",
		name: (operationName ?? "Type"),
		input: [data, pred],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (outputFalse: op.output(at: 0), outputTrue: op.output(at: 1))
} 

///Generates values in an interval.
///A sequence of `num` evenly-spaced values are generated beginning at `start`.
/// If `num > 1`, the values in the sequence increase by `stop - start / num - 1`,
/// so that the last one is exactly `stop`.
/// 
/// For example:
/// 
/// ```
/// tf.linspace(10.0, 12.0, 3, name="linspace") => [ 10.0  11.0  12.0]
/// ```
/// - Parameter start: First entry in the range.
/// - Parameter stop: Last entry in the range.
/// - Parameter num: Number of values to generate.
/// - Parameter tidx: 
/// - Returns: 
///	output: 1-D. The generated values.
public func linSpace(operationName: String? = nil, start: Output, stop: Output, num: Output, tidx: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Tidx"] = tidx
	let opspec = OpSpec(
		type: "LinSpace",
		name: (operationName ?? "Type"),
		input: [start, stop, num],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Calculates the CTC Loss (log probability) for each batch entry.  Also calculates
///the gradient.  This class performs the softmax operation for you, so inputs
/// should be e.g. linear projections of outputs by an LSTM.
/// - Parameter inputs: 3-D, shape: `(max_time x batch_size x num_classes)`, the logits.
/// - Parameter labelsIndices: The indices of a `SparseTensor<int32, 2>`.
/// `labels_indices(i, :) == [b, t]` means `labels_values(i)` stores the id for
/// `(batch b, time t)`.
/// - Parameter labelsValues: The values (labels) associated with the given batch and time.
/// - Parameter sequenceLength: A vector containing sequence lengths (batch).
/// - Parameter preprocessCollapseRepeated: Scalar, if true then repeated labels are
/// collapsed prior to the CTC calculation.
/// - Parameter ctcMergeRepeated: Scalar.  If set to false,  * during *  CTC calculation
/// repeated non-blank labels will not be merged and are interpreted as
/// individual labels.  This is a simplified version of CTC.
/// - Parameter ignoreLongerOutputsThanInputs: Scalar. If set to true, during CTC
/// calculation, items that have longer output sequences than input sequences
/// are skipped: they don't contribute to the loss term and have zero-gradient.
/// - Returns: 
///	loss: A vector (batch) containing log-probabilities.
///	gradient: The gradient of `loss`.  3-D, shape:
/// `(max_time x batch_size x num_classes)`.
public func cTCLoss(operationName: String? = nil, inputs: Output, labelsIndices: Output, labelsValues: Output, sequenceLength: Output, preprocessCollapseRepeated: Bool, ctcMergeRepeated: Bool, ignoreLongerOutputsThanInputs: Bool) throws -> (loss: Output, gradient: Output) { 
	var attrs = [String : Any]()
	attrs["preprocess_collapse_repeated"] = preprocessCollapseRepeated
	attrs["ctc_merge_repeated"] = ctcMergeRepeated
	attrs["ignore_longer_outputs_than_inputs"] = ignoreLongerOutputsThanInputs
	let opspec = OpSpec(
		type: "CTCLoss",
		name: (operationName ?? "Type"),
		input: [inputs, labelsIndices, labelsValues, sequenceLength],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (loss: op.output(at: 0), gradient: op.output(at: 1))
} 

///Creates a dataset that emits the records from one or more binary files.
/// - Parameter filenames: A scalar or a vector containing the name(s) of the file(s) to be
/// read.
/// - Parameter headerBytes: A scalar representing the number of bytes to skip at the
/// beginning of a file.
/// - Parameter recordBytes: A scalar representing the number of bytes in each record.
/// - Parameter footerBytes: A scalar representing the number of bytes to skip at the end
/// of a file.
/// - Parameter bufferSize: A scalar representing the number of bytes to buffer. Must be > 0.
/// - Returns: 
///	handle: 
public func fixedLengthRecordDataset(operationName: String? = nil, filenames: Output, headerBytes: Output, recordBytes: Output, footerBytes: Output, bufferSize: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "FixedLengthRecordDataset",
		name: (operationName ?? "Type"),
		input: [filenames, headerBytes, recordBytes, footerBytes, bufferSize],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Sparse update entries in ' * var' and ' * accum' according to FOBOS algorithm.
///That is for rows we have grad for, we update var and accum as follows:
/// accum += grad  *  grad
/// prox_v = var
/// prox_v -= lr  *  grad  *  (1 / sqrt(accum))
/// var = sign(prox_v)/(1+lr * l2)  *  max{|prox_v|-lr * l1,0}
/// - Parameter `var`: Should be from a Variable().
/// - Parameter accum: Should be from a Variable().
/// - Parameter lr: Learning rate. Must be a scalar.
/// - Parameter l1: L1 regularization. Must be a scalar.
/// - Parameter l2: L2 regularization. Must be a scalar.
/// - Parameter grad: The gradient.
/// - Parameter indices: A vector of indices into the first dimension of var and accum.
/// - Parameter tindices: 
/// - Parameter useLocking: If True, updating of the var and accum tensors will be protected by
/// a lock; otherwise the behavior is undefined, but may exhibit less contention.
/// - Returns: 
///	out: Same as "var".
public func sparseApplyProximalAdagrad(operationName: String? = nil, `var`: Output, accum: Output, lr: Output, l1: Output, l2: Output, grad: Output, indices: Output, tindices: Any.Type, useLocking: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Tindices"] = tindices
	attrs["use_locking"] = useLocking
	let opspec = OpSpec(
		type: "SparseApplyProximalAdagrad",
		name: (operationName ?? "Type"),
		input: [`var`, accum, lr, l1, l2, grad, indices],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Op returns the number of elements in the underlying container.
/// - Parameter capacity: 
/// - Parameter memoryLimit: 
/// - Parameter dtypes: 
/// - Parameter container: 
/// - Parameter sharedName: 
/// - Returns: 
///	size: 
public func mapSize(operationName: String? = nil, capacity: UInt8, memoryLimit: UInt8, dtypes: [Any.Type], container: String, sharedName: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["capacity"] = capacity
	attrs["memory_limit"] = memoryLimit
	attrs["dtypes"] = dtypes
	attrs["container"] = container
	attrs["shared_name"] = sharedName
	let opspec = OpSpec(
		type: "MapSize",
		name: (operationName ?? "Type"),
		input: [],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Solves systems of linear equations.
///`Matrix` is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
/// form square matrices. `Rhs` is a tensor of shape `[..., M, K]`. The `output` is
/// a tensor shape `[..., M, K]`.  If `adjoint` is `False` then each output matrix
/// satisfies `matrix[..., :, :]  *  output[..., :, :] = rhs[..., :, :]`.
/// If `adjoint` is `True` then each output matrix satisfies
/// `adjoint(matrix[..., :, :])  *  output[..., :, :] = rhs[..., :, :]`.
/// - Parameter matrix: Shape is `[..., M, M]`.
/// - Parameter rhs: Shape is `[..., M, K]`.
/// - Parameter adjoint: Boolean indicating whether to solve with `matrix` or its (block-wise)
/// adjoint.
/// - Returns: 
///	output: Shape is `[..., M, K]`.
public func matrixSolve(operationName: String? = nil, matrix: Output, rhs: Output, adjoint: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["adjoint"] = adjoint
	let opspec = OpSpec(
		type: "MatrixSolve",
		name: (operationName ?? "Type"),
		input: [matrix, rhs],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes hyperbolic sine of x element-wise.
/// - Parameter x: 
/// - Returns: 
///	y: 
public func sinh(operationName: String? = nil, x: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Sinh",
		name: (operationName ?? "Type"),
		input: [x],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 


/// - Parameter diagonal: 
/// - Returns: 
///	output: 
public func batchMatrixDiag(operationName: String? = nil, diagonal: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "BatchMatrixDiag",
		name: (operationName ?? "Type"),
		input: [diagonal],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Creates a dataset that executes a SQL query and emits rows of the result set.
/// - Parameter driverName: The database type. Currently, the only supported type is 'sqlite'.
/// - Parameter dataSourceName: A connection string to connect to the database.
/// - Parameter query: A SQL query to execute.
/// - Parameter outputTypes: 
/// - Parameter outputShapes: 
/// - Returns: 
///	handle: 
public func sqlDataset(operationName: String? = nil, driverName: Output, dataSourceName: Output, query: Output, outputTypes: [Any.Type], outputShapes: [Shape]) throws -> Output { 
	var attrs = [String : Any]()
	attrs["output_types"] = outputTypes
	attrs["output_shapes"] = outputShapes
	let opspec = OpSpec(
		type: "SqlDataset",
		name: (operationName ?? "Type"),
		input: [driverName, dataSourceName, query],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes the sum along segments of a tensor.
///Read @{$math_ops#segmentation$the section on segmentation} for an explanation of
/// segments.
/// 
/// Computes a tensor such that
/// \\(output_i = \sum_j data_j\\) where sum is over `j` such
/// that `segment_ids[j] == i`.
/// 
/// If the sum is empty for a given segment ID `i`, `output[i] = 0`.
/// 
/// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
/// <img style="width:100%" src="https://www.tensorflow.org/images/SegmentSum.png" alt>
/// </div>
/// - Parameter data: 
/// - Parameter segmentIds: A 1-D tensor whose rank is equal to the rank of `data`'s
/// first dimension.  Values should be sorted and can be repeated.
/// - Parameter tindices: 
/// - Returns: 
///	output: Has same shape as data, except for dimension 0 which
/// has size `k`, the number of segments.
public func segmentSum(operationName: String? = nil, data: Output, segmentIds: Output, tindices: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Tindices"] = tindices
	let opspec = OpSpec(
		type: "SegmentSum",
		name: (operationName ?? "Type"),
		input: [data, segmentIds],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Creates a dataset that emits the lines of one or more text files.
/// - Parameter filenames: A scalar or a vector containing the name(s) of the file(s) to be
/// read.
/// - Parameter compressionType: A scalar containing either (i) the empty string (no
/// compression), (ii) "ZLIB", or (iii) "GZIP".
/// - Parameter bufferSize: A scalar containing the number of bytes to buffer.
/// - Returns: 
///	handle: 
public func textLineDataset(operationName: String? = nil, filenames: Output, compressionType: Output, bufferSize: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "TextLineDataset",
		name: (operationName ?? "Type"),
		input: [filenames, compressionType, bufferSize],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Performs 3D average pooling on the input.
/// - Parameter input: Shape `[batch, depth, rows, cols, channels]` tensor to pool over.
/// - Parameter ksize: 1-D tensor of length 5. The size of the window for each dimension of
/// the input tensor. Must have `ksize[0] = ksize[4] = 1`.
/// - Parameter strides: 1-D tensor of length 5. The stride of the sliding window for each
/// dimension of `input`. Must have `strides[0] = strides[4] = 1`.
/// - Parameter padding: The type of padding algorithm to use.
/// - Parameter dataFormat: The data format of the input and output data. With the
/// default format "NDHWC", the data is stored in the order of:
///     [batch, in_depth, in_height, in_width, in_channels].
/// Alternatively, the format could be "NCDHW", the data storage order is:
///     [batch, in_channels, in_depth, in_height, in_width].
/// - Returns: 
///	output: The average pooled output tensor.
public func avgPool3D(operationName: String? = nil, input: Output, ksize: [Int64], strides: [Int64], padding: String, dataFormat: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["ksize"] = ksize
	attrs["strides"] = strides
	attrs["padding"] = padding
	attrs["data_format"] = dataFormat
	let opspec = OpSpec(
		type: "AvgPool3D",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Deprecated, use StackCloseV2.
/// - Parameter handle: 
public func stackClose(operationName: String? = nil, handle: Output) throws -> Operation { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "StackClose",
		name: (operationName ?? "Type"),
		input: [handle],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Assigns a new value to a variable.
///Any ReadVariableOp with a control dependency on this op is guaranteed to return
/// this value or a subsequent newer value of the variable.
/// - Parameter resource: handle to the resource in which to store the variable.
/// - Parameter value: the value to set the new tensor to use.
/// - Parameter dtype: the dtype of the value.
public func assignVariableOp(operationName: String? = nil, resource: Output, value: Output, dtype: Any.Type) throws -> Operation { 
	var attrs = [String : Any]()
	attrs["dtype"] = dtype
	let opspec = OpSpec(
		type: "AssignVariableOp",
		name: (operationName ?? "Type"),
		input: [resource, value],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Resize `images` to `size` using bicubic interpolation.
///Input images can be of different types but output images are always float.
/// - Parameter images: 4-D with shape `[batch, height, width, channels]`.
/// - Parameter size: = A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
/// new size for the images.
/// - Parameter alignCorners: If true, rescale input by (new_height - 1) / (height - 1), which
/// exactly aligns the 4 corners of images and resized images. If false, rescale
/// by new_height / height. Treat similarly the width dimension.
/// - Returns: 
///	resized_images: 4-D with shape
/// `[batch, new_height, new_width, channels]`.
public func resizeBicubic(operationName: String? = nil, images: Output, size: Output, alignCorners: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["align_corners"] = alignCorners
	let opspec = OpSpec(
		type: "ResizeBicubic",
		name: (operationName ?? "Type"),
		input: [images, size],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Convert one or more images from HSV to RGB.
///Outputs a tensor of the same shape as the `images` tensor, containing the RGB
/// value of the pixels. The output is only well defined if the value in `images`
/// are in `[0,1]`.
/// 
/// See `rgb_to_hsv` for a description of the HSV encoding.
/// - Parameter images: 1-D or higher rank. HSV data to convert. Last dimension must be size 3.
/// - Returns: 
///	output: `images` converted to RGB.
public func hSVToRGB(operationName: String? = nil, images: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "HSVToRGB",
		name: (operationName ?? "Type"),
		input: [images],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Creates a dataset that caches elements from `input_dataset`.
///A CacheDataset will iterate over the input_dataset, and store tensors. If the
/// cache already exists, the cache will be used. If the cache is inappropriate
/// (e.g. cannot be opened, contains tensors of the wrong shape / size), an error
/// will the returned when used.
/// - Parameter inputDataset: 
/// - Parameter filename: A path on the filesystem where we should cache the dataset. Note: this
/// will be a directory.
/// - Parameter outputTypes: 
/// - Parameter outputShapes: 
/// - Returns: 
///	handle: 
public func cacheDataset(operationName: String? = nil, inputDataset: Output, filename: Output, outputTypes: [Any.Type], outputShapes: [Shape]) throws -> Output { 
	var attrs = [String : Any]()
	attrs["output_types"] = outputTypes
	attrs["output_shapes"] = outputShapes
	let opspec = OpSpec(
		type: "CacheDataset",
		name: (operationName ?? "Type"),
		input: [inputDataset, filename],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Outputs random values from the Poisson distribution(s) described by rate.
///This op uses two algorithms, depending on rate. If rate >= 10, then
/// the algorithm by Hormann is used to acquire samples via
/// transformation-rejection.
/// See http://www.sciencedirect.com/science/article/pii/0167668793909974.
/// 
/// Otherwise, Knuth's algorithm is used to acquire samples via multiplying uniform
/// random variables.
/// See Donald E. Knuth (1969). Seminumerical Algorithms. The Art of Computer
/// Programming, Volume 2. Addison Wesley
/// - Parameter shape: 1-D integer tensor. Shape of independent samples to draw from each
/// distribution described by the shape parameters given in rate.
/// - Parameter rate: A tensor in which each scalar is a "rate" parameter describing the
/// associated poisson distribution.
/// - Parameter seed: If either `seed` or `seed2` are set to be non-zero, the random number
/// generator is seeded by the given seed.  Otherwise, it is seeded by a
/// random seed.
/// - Parameter seed2: A second seed to avoid seed collision.
/// - Parameter s: 
/// - Parameter r: 
/// - Parameter dtype: 
/// - Returns: 
///	output: A tensor with shape `shape + shape(rate)`. Each slice
/// `[:, ..., :, i0, i1, ...iN]` contains the samples drawn for
/// `rate[i0, i1, ...iN]`.
public func randomPoissonV2(operationName: String? = nil, shape: Output, rate: Output, seed: UInt8, seed2: UInt8, s: Any.Type, r: Any.Type, dtype: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["seed"] = seed
	attrs["seed2"] = seed2
	attrs["S"] = s
	attrs["R"] = r
	attrs["dtype"] = dtype
	let opspec = OpSpec(
		type: "RandomPoissonV2",
		name: (operationName ?? "Type"),
		input: [shape, rate],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Creates a dataset that shuffles elements from `input_dataset` pseudorandomly.
/// - Parameter inputDataset: 
/// - Parameter bufferSize: The number of output elements to buffer in an iterator over
/// this dataset. Compare with the `min_after_dequeue` attr when creating a
/// `RandomShuffleQueue`.
/// - Parameter seed: A scalar seed for the random number generator. If either seed or
/// seed2 is set to be non-zero, the random number generator is seeded
/// by the given seed.  Otherwise, a random seed is used.
/// - Parameter seed2: A second scalar seed to avoid seed collision.
/// - Parameter reshuffleEachIteration: If true, each iterator over this dataset will be given
/// a different pseudorandomly generated seed, based on a sequence seeded by the
/// `seed` and `seed2` inputs. If false, each iterator will be given the same
/// seed, and repeated iteration over this dataset will yield the exact same
/// sequence of results.
/// - Parameter outputTypes: 
/// - Parameter outputShapes: 
/// - Returns: 
///	handle: 
public func shuffleDataset(operationName: String? = nil, inputDataset: Output, bufferSize: Output, seed: Output, seed2: Output, reshuffleEachIteration: Bool, outputTypes: [Any.Type], outputShapes: [Shape]) throws -> Output { 
	var attrs = [String : Any]()
	attrs["reshuffle_each_iteration"] = reshuffleEachIteration
	attrs["output_types"] = outputTypes
	attrs["output_shapes"] = outputShapes
	let opspec = OpSpec(
		type: "ShuffleDataset",
		name: (operationName ?? "Type"),
		input: [inputDataset, bufferSize, seed, seed2],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Concatenates a list of `N` tensors along the first dimension.
///The input tensors are all required to have size 1 in the first dimension.
/// 
/// For example:
/// 
/// ```
/// # 'x' is [[1, 4]]
/// # 'y' is [[2, 5]]
/// # 'z' is [[3, 6]]
/// parallel_concat([x, y, z]) => [[1, 4], [2, 5], [3, 6]]  # Pack along first dim.
/// ```
/// 
/// The difference between concat and parallel_concat is that concat requires all
/// of the inputs be computed before the operation will begin but doesn't require
/// that the input shapes be known during graph construction.  Parallel concat
/// will copy pieces of the input into the output as they become available, in
/// some situations this can provide a performance benefit.
/// - Parameter values: Tensors to be concatenated. All must have size 1 in the first dimension
/// and same shape.
/// - Parameter n: 
/// - Parameter shape: the final shape of the result; should be equal to the shapes of any input
/// but with the number of input values in the first dimension.
/// - Returns: 
///	output: The concatenated tensor.
public func parallelConcat(operationName: String? = nil, values: [Output], n: UInt8, shape: Shape) throws -> Output { 
	var attrs = [String : Any]()
	attrs["N"] = n
	attrs["shape"] = shape
	let opspec = OpSpec(
		type: "ParallelConcat",
		name: (operationName ?? "Type"),
		input: [values],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Delete the TensorArray from its resource container.
///This enables the user to close and release the resource in the middle
/// of a step/run.
/// - Parameter handle: The handle to a TensorArray (output of TensorArray or TensorArrayGrad).
public func tensorArrayCloseV3(operationName: String? = nil, handle: Output) throws -> Operation { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "TensorArrayCloseV3",
		name: (operationName ?? "Type"),
		input: [handle],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Creates a dataset with a range of values. Corresponds to python's xrange.
/// - Parameter start: corresponds to start in python's xrange().
/// - Parameter stop: corresponds to stop in python's xrange().
/// - Parameter step: corresponds to step in python's xrange().
/// - Parameter outputTypes: 
/// - Parameter outputShapes: 
/// - Returns: 
///	handle: 
public func rangeDataset(operationName: String? = nil, start: Output, stop: Output, step: Output, outputTypes: [Any.Type], outputShapes: [Shape]) throws -> Output { 
	var attrs = [String : Any]()
	attrs["output_types"] = outputTypes
	attrs["output_shapes"] = outputShapes
	let opspec = OpSpec(
		type: "RangeDataset",
		name: (operationName ?? "Type"),
		input: [start, stop, step],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///V2 format specific: merges the metadata files of sharded checkpoints.  The
///result is one logical checkpoint, with one physical metadata file and renamed
/// data files.
/// 
/// Intended for "grouping" multiple checkpoints in a sharded checkpoint setup.
/// 
/// If delete_old_dirs is true, attempts to delete recursively the dirname of each
/// path in the input checkpoint_prefixes.  This is useful when those paths are non
/// user-facing temporary locations.
/// - Parameter checkpointPrefixes: prefixes of V2 checkpoints to merge.
/// - Parameter destinationPrefix: scalar.  The desired final prefix.  Allowed to be the same
/// as one of the checkpoint_prefixes.
/// - Parameter deleteOldDirs: see above.
public func mergeV2Checkpoints(operationName: String? = nil, checkpointPrefixes: Output, destinationPrefix: Output, deleteOldDirs: Bool) throws -> Operation { 
	var attrs = [String : Any]()
	attrs["delete_old_dirs"] = deleteOldDirs
	let opspec = OpSpec(
		type: "MergeV2Checkpoints",
		name: (operationName ?? "Type"),
		input: [checkpointPrefixes, destinationPrefix],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Creates a dataset that zips together `input_datasets`.
/// - Parameter inputDatasets: 
/// - Parameter outputTypes: 
/// - Parameter outputShapes: 
/// - Parameter n: 
/// - Returns: 
///	handle: 
public func zipDataset(operationName: String? = nil, inputDatasets: [Output], outputTypes: [Any.Type], outputShapes: [Shape], n: UInt8) throws -> Output { 
	var attrs = [String : Any]()
	attrs["output_types"] = outputTypes
	attrs["output_shapes"] = outputShapes
	attrs["N"] = n
	let opspec = OpSpec(
		type: "ZipDataset",
		name: (operationName ?? "Type"),
		input: [inputDatasets],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Closes the given queue.
///This operation signals that no more elements will be enqueued in the
/// given queue. Subsequent Enqueue(Many) operations will fail.
/// Subsequent Dequeue(Many) operations will continue to succeed if
/// sufficient elements remain in the queue. Subsequent Dequeue(Many)
/// operations that would block will fail immediately.
/// - Parameter handle: The handle to a queue.
/// - Parameter cancelPendingEnqueues: If true, all pending enqueue requests that are
/// blocked on the given queue will be canceled.
public func queueClose(operationName: String? = nil, handle: Output, cancelPendingEnqueues: Bool) throws -> Operation { 
	var attrs = [String : Any]()
	attrs["cancel_pending_enqueues"] = cancelPendingEnqueues
	let opspec = OpSpec(
		type: "QueueClose",
		name: (operationName ?? "Type"),
		input: [handle],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///A queue that randomizes the order of elements.
/// - Parameter componentTypes: The type of each component in a value.
/// - Parameter shapes: The shape of each component in a value. The length of this attr must
/// be either 0 or the same as the length of component_types. If the length of
/// this attr is 0, the shapes of queue elements are not constrained, and
/// only one element may be dequeued at a time.
/// - Parameter capacity: The upper bound on the number of elements in this queue.
/// Negative numbers mean no limit.
/// - Parameter minAfterDequeue: Dequeue will block unless there would be this
/// many elements after the dequeue or the queue is closed. This
/// ensures a minimum level of mixing of elements.
/// - Parameter seed: If either seed or seed2 is set to be non-zero, the random number
/// generator is seeded by the given seed.  Otherwise, a random seed is used.
/// - Parameter seed2: A second seed to avoid seed collision.
/// - Parameter container: If non-empty, this queue is placed in the given container.
/// Otherwise, a default container is used.
/// - Parameter sharedName: If non-empty, this queue will be shared under the given name
/// across multiple sessions.
/// - Returns: 
///	handle: The handle to the queue.
public func randomShuffleQueue(operationName: String? = nil, componentTypes: [Any.Type], shapes: [Shape], capacity: UInt8, minAfterDequeue: UInt8, seed: UInt8, seed2: UInt8, container: String, sharedName: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["component_types"] = componentTypes
	attrs["shapes"] = shapes
	attrs["capacity"] = capacity
	attrs["min_after_dequeue"] = minAfterDequeue
	attrs["seed"] = seed
	attrs["seed2"] = seed2
	attrs["container"] = container
	attrs["shared_name"] = sharedName
	let opspec = OpSpec(
		type: "RandomShuffleQueue",
		name: (operationName ?? "Type"),
		input: [],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Restores tensors from a V2 checkpoint.
///For backward compatibility with the V1 format, this Op currently allows
/// restoring from a V1 checkpoint as well:
///   - This Op first attempts to find the V2 index file pointed to by "prefix", and
///     if found proceed to read it as a V2 checkpoint;
///   - Otherwise the V1 read path is invoked.
/// Relying on this behavior is not recommended, as the ability to fall back to read
/// V1 might be deprecated and eventually removed.
/// 
/// By default, restores the named tensors in full.  If the caller wishes to restore
/// specific slices of stored tensors, "shape_and_slices" should be non-empty
/// strings and correspondingly well-formed.
/// 
/// Callers must ensure all the named tensors are indeed stored in the checkpoint.
/// - Parameter prefix: Must have a single element.  The prefix of a V2 checkpoint.
/// - Parameter tensorNames: shape {N}.  The names of the tensors to be restored.
/// - Parameter shapeAndSlices: shape {N}.  The slice specs of the tensors to be restored.
/// Empty strings indicate that they are non-partitioned tensors.
/// - Parameter dtypes: shape {N}.  The list of expected dtype for the tensors.  Must match
/// those stored in the checkpoint.
/// - Returns: 
///	tensors: shape {N}.  The restored tensors, whose shapes are read from the
/// checkpoint directly.
public func restoreV2(operationName: String? = nil, prefix: Output, tensorNames: Output, shapeAndSlices: Output, dtypes: [Any.Type]) throws -> Output { 
	var attrs = [String : Any]()
	attrs["dtypes"] = dtypes
	let opspec = OpSpec(
		type: "RestoreV2",
		name: (operationName ?? "Type"),
		input: [prefix, tensorNames, shapeAndSlices],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Creates a dataset that yields a SparseTensor for each element of the input.
/// - Parameter inputDataset: A handle to an input dataset. Must have a single component.
/// - Parameter batchSize: A scalar representing the number of elements to accumulate in a
/// batch.
/// - Parameter rowShape: A vector representing the dense shape of each row in the produced
/// SparseTensor. The shape may be partially specified, using `-1` to indicate
/// that a particular dimension should use the maximum size of all batch elements.
/// - Parameter outputTypes: 
/// - Parameter outputShapes: 
/// - Returns: 
///	handle: 
public func denseToSparseBatchDataset(operationName: String? = nil, inputDataset: Output, batchSize: Output, rowShape: Output, outputTypes: [Any.Type], outputShapes: [Shape]) throws -> Output { 
	var attrs = [String : Any]()
	attrs["output_types"] = outputTypes
	attrs["output_shapes"] = outputShapes
	let opspec = OpSpec(
		type: "DenseToSparseBatchDataset",
		name: (operationName ?? "Type"),
		input: [inputDataset, batchSize, rowShape],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Add all input tensors element wise.
/// - Parameter inputs: Must all be the same size and shape.
/// - Parameter n: 
/// - Returns: 
///	sum: 
public func addN(operationName: String? = nil, inputs: [Output], n: UInt8) throws -> Output { 
	var attrs = [String : Any]()
	attrs["N"] = n
	let opspec = OpSpec(
		type: "AddN",
		name: (operationName ?? "Type"),
		input: [inputs],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Takes the given number of completed elements from a barrier.
///This operation concatenates completed-element component tensors along
/// the 0th dimension to make a single component tensor.
/// 
/// Elements come out of the barrier when they are complete, and in the order
/// in which they were placed into the barrier.  The indices output provides
/// information about the batch in which each element was originally inserted
/// into the barrier.
/// - Parameter handle: The handle to a barrier.
/// - Parameter numElements: A single-element tensor containing the number of elements to
/// take.
/// - Parameter componentTypes: The type of each component in a value.
/// - Parameter allowSmallBatch: Allow to return less than num_elements items if barrier is
/// already closed.
/// - Parameter waitForIncomplete: 
/// - Parameter timeoutMs: If the queue is empty, this operation will block for up to
/// timeout_ms milliseconds.
/// Note: This option is not supported yet.
/// - Returns: 
///	indices: A one-dimensional tensor of indices, with length num_elems.
/// These indices refer to the batch in which the values were placed into the
/// barrier (starting with MIN_LONG and increasing with each BarrierInsertMany).
///	keys: A one-dimensional tensor of keys, with length num_elements.
///	values: One any-dimensional tensor per component in a barrier element. All
/// values have length num_elements in the 0th dimension.
public func barrierTakeMany(operationName: String? = nil, handle: Output, numElements: Output, componentTypes: [Any.Type], allowSmallBatch: Bool, waitForIncomplete: Bool, timeoutMs: UInt8) throws -> (indices: Output, keys: Output, values: Output) { 
	var attrs = [String : Any]()
	attrs["component_types"] = componentTypes
	attrs["allow_small_batch"] = allowSmallBatch
	attrs["wait_for_incomplete"] = waitForIncomplete
	attrs["timeout_ms"] = timeoutMs
	let opspec = OpSpec(
		type: "BarrierTakeMany",
		name: (operationName ?? "Type"),
		input: [handle, numElements],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (indices: op.output(at: 0), keys: op.output(at: 1), values: op.output(at: 2))
} 

///Deprecated. Use TensorArrayV3
/// - Parameter size: 
/// - Parameter dtype: 
/// - Parameter elementShape: 
/// - Parameter dynamicSize: 
/// - Parameter clearAfterRead: 
/// - Parameter tensorArrayName: 
/// - Returns: 
///	handle: 
public func tensorArrayV2(operationName: String? = nil, size: Output, dtype: Any.Type, elementShape: Shape, dynamicSize: Bool, clearAfterRead: Bool, tensorArrayName: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["dtype"] = dtype
	attrs["element_shape"] = elementShape
	attrs["dynamic_size"] = dynamicSize
	attrs["clear_after_read"] = clearAfterRead
	attrs["tensor_array_name"] = tensorArrayName
	let opspec = OpSpec(
		type: "TensorArrayV2",
		name: (operationName ?? "Type"),
		input: [size],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Creates a dataset containing elements of `input_dataset` matching `predicate`.
///The `predicate` function must return a scalar boolean and accept the
/// following arguments:
/// 
///  *  One tensor for each component of an element of `input_dataset`.
///  *  One tensor for each value in `other_arguments`.
/// - Parameter inputDataset: 
/// - Parameter otherArguments: A list of tensors, typically values that were captured when
/// building a closure for `predicate`.
/// - Parameter predicate: A function returning a scalar boolean.
/// - Parameter targuments: 
/// - Parameter outputTypes: 
/// - Parameter outputShapes: 
/// - Returns: 
///	handle: 
public func filterDataset(operationName: String? = nil, inputDataset: Output, otherArguments: Output, predicate: Tensorflow_NameAttrList, targuments: [Any.Type], outputTypes: [Any.Type], outputShapes: [Shape]) throws -> Output { 
	var attrs = [String : Any]()
	attrs["predicate"] = predicate
	attrs["Targuments"] = targuments
	attrs["output_types"] = outputTypes
	attrs["output_shapes"] = outputShapes
	let opspec = OpSpec(
		type: "FilterDataset",
		name: (operationName ?? "Type"),
		input: [inputDataset, otherArguments],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes gradients of max pooling function.
/// - Parameter origInput: The original input tensor.
/// - Parameter origOutput: The original output tensor.
/// - Parameter grad: Output backprop of shape `[batch, depth, rows, cols, channels]`.
/// - Parameter ksize: 1-D tensor of length 5. The size of the window for each dimension of
/// the input tensor. Must have `ksize[0] = ksize[4] = 1`.
/// - Parameter strides: 1-D tensor of length 5. The stride of the sliding window for each
/// dimension of `input`. Must have `strides[0] = strides[4] = 1`.
/// - Parameter padding: The type of padding algorithm to use.
/// - Parameter dataFormat: The data format of the input and output data. With the
/// default format "NDHWC", the data is stored in the order of:
///     [batch, in_depth, in_height, in_width, in_channels].
/// Alternatively, the format could be "NCDHW", the data storage order is:
///     [batch, in_channels, in_depth, in_height, in_width].
/// - Parameter tInput: 
/// - Returns: 
///	output: 
public func maxPool3DGrad(operationName: String? = nil, origInput: Output, origOutput: Output, grad: Output, ksize: [Int64], strides: [Int64], padding: String, dataFormat: String, tInput: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["ksize"] = ksize
	attrs["strides"] = strides
	attrs["padding"] = padding
	attrs["data_format"] = dataFormat
	attrs["TInput"] = tInput
	let opspec = OpSpec(
		type: "MaxPool3DGrad",
		name: (operationName ?? "Type"),
		input: [origInput, origOutput, grad],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Creates a dataset that applies `f` to the outputs of `input_dataset`.
///Unlike MapDataset, the `f` in InterleaveDataset is expected to return
/// a Dataset variant, and InterleaveDataset will flatten successive
/// results into a single Dataset. Unlike FlatMapDataset,
/// InterleaveDataset will interleave sequences of up to `block_length`
/// consecutive elements from `cycle_length` input elements.
/// - Parameter inputDataset: 
/// - Parameter otherArguments: 
/// - Parameter cycleLength: 
/// - Parameter blockLength: 
/// - Parameter f: A function mapping elements of `input_dataset`, concatenated with
/// `other_arguments`, to a Dataset variant that contains elements matching
/// `output_types` and `output_shapes`.
/// - Parameter targuments: 
/// - Parameter outputTypes: 
/// - Parameter outputShapes: 
/// - Returns: 
///	handle: 
public func interleaveDataset(operationName: String? = nil, inputDataset: Output, otherArguments: Output, cycleLength: Output, blockLength: Output, f: Tensorflow_NameAttrList, targuments: [Any.Type], outputTypes: [Any.Type], outputShapes: [Shape]) throws -> Output { 
	var attrs = [String : Any]()
	attrs["f"] = f
	attrs["Targuments"] = targuments
	attrs["output_types"] = outputTypes
	attrs["output_shapes"] = outputShapes
	let opspec = OpSpec(
		type: "InterleaveDataset",
		name: (operationName ?? "Type"),
		input: [inputDataset, otherArguments, cycleLength, blockLength],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Returns the number of records this Reader has produced.
///This is the same as the number of ReaderRead executions that have
/// succeeded.
/// - Parameter readerHandle: Handle to a Reader.
/// - Returns: 
///	records_produced: 
public func readerNumRecordsProducedV2(operationName: String? = nil, readerHandle: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "ReaderNumRecordsProducedV2",
		name: (operationName ?? "Type"),
		input: [readerHandle],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Creates a dataset that asynchronously prefetches elements from `input_dataset`.
/// - Parameter inputDataset: 
/// - Parameter bufferSize: The maximum number of elements to buffer in an iterator over
/// this dataset.
/// - Parameter outputTypes: 
/// - Parameter outputShapes: 
/// - Returns: 
///	handle: 
public func prefetchDataset(operationName: String? = nil, inputDataset: Output, bufferSize: Output, outputTypes: [Any.Type], outputShapes: [Shape]) throws -> Output { 
	var attrs = [String : Any]()
	attrs["output_types"] = outputTypes
	attrs["output_shapes"] = outputShapes
	let opspec = OpSpec(
		type: "PrefetchDataset",
		name: (operationName ?? "Type"),
		input: [inputDataset, bufferSize],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Creates a sequence of numbers.
///This operation creates a sequence of numbers that begins at `start` and
/// extends by increments of `delta` up to but not including `limit`.
/// 
/// For example:
/// 
/// ```
/// # 'start' is 3
/// # 'limit' is 18
/// # 'delta' is 3
/// tf.range(start, limit, delta) ==> [3, 6, 9, 12, 15]
/// ```
/// - Parameter start: 0-D (scalar). First entry in the sequence.
/// - Parameter limit: 0-D (scalar). Upper limit of sequence, exclusive.
/// - Parameter delta: 0-D (scalar). Optional. Default is 1. Number that increments `start`.
/// - Parameter tidx: 
/// - Returns: 
///	output: 1-D.
public func range(operationName: String? = nil, start: Output, limit: Output, delta: Output, tidx: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Tidx"] = tidx
	let opspec = OpSpec(
		type: "Range",
		name: (operationName ?? "Type"),
		input: [start, limit, delta],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Creates a dataset that applies `f` to the outputs of `input_dataset`.
///Unlike MapDataset, the `f` in FlatMapDataset is expected to return a
/// Dataset variant, and FlatMapDataset will flatten successive results
/// into a single Dataset.
/// - Parameter inputDataset: 
/// - Parameter otherArguments: 
/// - Parameter f: A function mapping elements of `input_dataset`, concatenated with
/// `other_arguments`, to a Dataset variant that contains elements matching
/// `output_types` and `output_shapes`.
/// - Parameter targuments: 
/// - Parameter outputTypes: 
/// - Parameter outputShapes: 
/// - Returns: 
///	handle: 
public func flatMapDataset(operationName: String? = nil, inputDataset: Output, otherArguments: Output, f: Tensorflow_NameAttrList, targuments: [Any.Type], outputTypes: [Any.Type], outputShapes: [Shape]) throws -> Output { 
	var attrs = [String : Any]()
	attrs["f"] = f
	attrs["Targuments"] = targuments
	attrs["output_types"] = outputTypes
	attrs["output_shapes"] = outputShapes
	let opspec = OpSpec(
		type: "FlatMapDataset",
		name: (operationName ?? "Type"),
		input: [inputDataset, otherArguments],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Outputs a `Summary` protocol buffer with a histogram.
///The generated
/// [`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
/// has one summary value containing a histogram for `values`.
/// 
/// This op reports an `InvalidArgument` error if any value is not finite.
/// - Parameter tag: Scalar.  Tag to use for the `Summary.Value`.
/// - Parameter values: Any shape. Values to use to build the histogram.
/// - Returns: 
///	summary: Scalar. Serialized `Summary` protocol buffer.
public func histogramSummary(operationName: String? = nil, tag: Output, values: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "HistogramSummary",
		name: (operationName ?? "Type"),
		input: [tag, values],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Pop the element at the top of the stack.
/// - Parameter handle: The handle to a stack.
/// - Parameter elemType: The type of the elem that is popped.
/// - Returns: 
///	elem: The tensor that is popped from the top of the stack.
public func stackPopV2(operationName: String? = nil, handle: Output, elemType: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["elem_type"] = elemType
	let opspec = OpSpec(
		type: "StackPopV2",
		name: (operationName ?? "Type"),
		input: [handle],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes the gradients of 3-D convolution with respect to the input.
/// - Parameter inputSizes: An integer vector representing the tensor shape of `input`,
/// where `input` is a 5-D
/// `[batch, depth, rows, cols, in_channels]` tensor.
/// - Parameter filter: Shape `[depth, rows, cols, in_channels, out_channels]`.
/// `in_channels` must match between `input` and `filter`.
/// - Parameter outBackprop: Backprop signal of shape `[batch, out_depth, out_rows, out_cols,
/// out_channels]`.
/// - Parameter strides: 1-D tensor of length 5. The stride of the sliding window for each
/// dimension of `input`. Must have `strides[0] = strides[4] = 1`.
/// - Parameter padding: The type of padding algorithm to use.
/// - Parameter dataFormat: The data format of the input and output data. With the
/// default format "NDHWC", the data is stored in the order of:
///     [batch, in_depth, in_height, in_width, in_channels].
/// Alternatively, the format could be "NCDHW", the data storage order is:
///     [batch, in_channels, in_depth, in_height, in_width].
/// - Returns: 
///	output: 
public func conv3DBackpropInputV2(operationName: String? = nil, inputSizes: Output, filter: Output, outBackprop: Output, strides: [Int64], padding: String, dataFormat: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["strides"] = strides
	attrs["padding"] = padding
	attrs["data_format"] = dataFormat
	let opspec = OpSpec(
		type: "Conv3DBackpropInputV2",
		name: (operationName ?? "Type"),
		input: [inputSizes, filter, outBackprop],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes the gradient of bilinear interpolation.
/// - Parameter grads: 4-D with shape `[batch, height, width, channels]`.
/// - Parameter originalImage: 4-D with shape `[batch, orig_height, orig_width, channels]`,
/// The image tensor that was resized.
/// - Parameter alignCorners: If true, rescale grads by (orig_height - 1) / (height - 1), which
/// exactly aligns the 4 corners of grads and original_image. If false, rescale by
/// orig_height / height. Treat similarly the width dimension.
/// - Returns: 
///	output: 4-D with shape `[batch, orig_height, orig_width, channels]`.
/// Gradients with respect to the input image. Input image must have been
/// float or double.
public func resizeBilinearGrad(operationName: String? = nil, grads: Output, originalImage: Output, alignCorners: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["align_corners"] = alignCorners
	let opspec = OpSpec(
		type: "ResizeBilinearGrad",
		name: (operationName ?? "Type"),
		input: [grads, originalImage],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Creates a dataset that skips `count` elements from the `input_dataset`.
/// - Parameter inputDataset: 
/// - Parameter count: A scalar representing the number of elements from the `input_dataset`
/// that should be skipped.  If count is -1, skips everything.
/// - Parameter outputTypes: 
/// - Parameter outputShapes: 
/// - Returns: 
///	handle: 
public func skipDataset(operationName: String? = nil, inputDataset: Output, count: Output, outputTypes: [Any.Type], outputShapes: [Shape]) throws -> Output { 
	var attrs = [String : Any]()
	attrs["output_types"] = outputTypes
	attrs["output_shapes"] = outputShapes
	let opspec = OpSpec(
		type: "SkipDataset",
		name: (operationName ?? "Type"),
		input: [inputDataset, count],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes the "logical and" of elements across dimensions of a tensor.
///Reduces `input` along the dimensions given in `reduction_indices`. Unless
/// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
/// `reduction_indices`. If `keep_dims` is true, the reduced dimensions are
/// retained with length 1.
/// - Parameter input: The tensor to reduce.
/// - Parameter reductionIndices: The dimensions to reduce. Must be in the range
/// `[-rank(input), rank(input))`.
/// - Parameter keepDims: If true, retain reduced dimensions with length 1.
/// - Parameter tidx: 
/// - Returns: 
///	output: The reduced tensor.
public func all(operationName: String? = nil, input: Output, reductionIndices: Output, keepDims: Bool, tidx: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["keep_dims"] = keepDims
	attrs["Tidx"] = tidx
	let opspec = OpSpec(
		type: "All",
		name: (operationName ?? "Type"),
		input: [input, reductionIndices],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Returns the number of records this Reader has produced.
///This is the same as the number of ReaderRead executions that have
/// succeeded.
/// - Parameter readerHandle: Handle to a Reader.
/// - Returns: 
///	records_produced: 
public func readerNumRecordsProduced(operationName: String? = nil, readerHandle: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "ReaderNumRecordsProduced",
		name: (operationName ?? "Type"),
		input: [readerHandle],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Creates a dataset that contains `count` elements from the `input_dataset`.
/// - Parameter inputDataset: 
/// - Parameter count: A scalar representing the number of elements from the `input_dataset`
/// that should be taken. A value of `-1` indicates that all of `input_dataset`
/// is taken.
/// - Parameter outputTypes: 
/// - Parameter outputShapes: 
/// - Returns: 
///	handle: 
public func takeDataset(operationName: String? = nil, inputDataset: Output, count: Output, outputTypes: [Any.Type], outputShapes: [Shape]) throws -> Output { 
	var attrs = [String : Any]()
	attrs["output_types"] = outputTypes
	attrs["output_shapes"] = outputShapes
	let opspec = OpSpec(
		type: "TakeDataset",
		name: (operationName ?? "Type"),
		input: [inputDataset, count],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Returns the truth value of (x == y) element-wise.
/// * NOTE * : `Equal` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
/// - Parameter x: 
/// - Parameter y: 
/// - Returns: 
///	z: 
public func equal(operationName: String? = nil, x: Output, y: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Equal",
		name: (operationName ?? "Type"),
		input: [x, y],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Applies set operation along last dimension of 2 `SparseTensor` inputs.
///See SetOperationOp::SetOperationFromContext for values of `set_operation`.
/// 
/// If `validate_indices` is `True`, `SparseToSparseSetOperation` validates the
/// order and range of `set1` and `set2` indices.
/// 
/// Input `set1` is a `SparseTensor` represented by `set1_indices`, `set1_values`,
/// and `set1_shape`. For `set1` ranked `n`, 1st `n-1` dimensions must be the same
/// as `set2`. Dimension `n` contains values in a set, duplicates are allowed but
/// ignored.
/// 
/// Input `set2` is a `SparseTensor` represented by `set2_indices`, `set2_values`,
/// and `set2_shape`. For `set2` ranked `n`, 1st `n-1` dimensions must be the same
/// as `set1`. Dimension `n` contains values in a set, duplicates are allowed but
/// ignored.
/// 
/// If `validate_indices` is `True`, this op validates the order and range of `set1`
/// and `set2` indices.
/// 
/// Output `result` is a `SparseTensor` represented by `result_indices`,
/// `result_values`, and `result_shape`. For `set1` and `set2` ranked `n`, this
/// has rank `n` and the same 1st `n-1` dimensions as `set1` and `set2`. The `nth`
/// dimension contains the result of `set_operation` applied to the corresponding
/// `[0...n-1]` dimension of `set`.
/// - Parameter set1Indices: 2D `Tensor`, indices of a `SparseTensor`. Must be in row-major
/// order.
/// - Parameter set1Values: 1D `Tensor`, values of a `SparseTensor`. Must be in row-major
/// order.
/// - Parameter set1Shape: 1D `Tensor`, shape of a `SparseTensor`. `set1_shape[0...n-1]` must
/// be the same as `set2_shape[0...n-1]`, `set1_shape[n]` is the
/// max set size across `0...n-1` dimensions.
/// - Parameter set2Indices: 2D `Tensor`, indices of a `SparseTensor`. Must be in row-major
/// order.
/// - Parameter set2Values: 1D `Tensor`, values of a `SparseTensor`. Must be in row-major
/// order.
/// - Parameter set2Shape: 1D `Tensor`, shape of a `SparseTensor`. `set2_shape[0...n-1]` must
/// be the same as `set1_shape[0...n-1]`, `set2_shape[n]` is the
/// max set size across `0...n-1` dimensions.
/// - Parameter setOperation: 
/// - Parameter validateIndices: 
/// - Returns: 
///	result_indices: 2D indices of a `SparseTensor`.
///	result_values: 1D values of a `SparseTensor`.
///	result_shape: 1D `Tensor` shape of a `SparseTensor`. `result_shape[0...n-1]` is
/// the same as the 1st `n-1` dimensions of `set1` and `set2`, `result_shape[n]`
/// is the max result set size across all `0...n-1` dimensions.
public func sparseToSparseSetOperation(operationName: String? = nil, set1Indices: Output, set1Values: Output, set1Shape: Output, set2Indices: Output, set2Values: Output, set2Shape: Output, setOperation: String, validateIndices: Bool) throws -> (resultIndices: Output, resultValues: Output, resultShape: Output) { 
	var attrs = [String : Any]()
	attrs["set_operation"] = setOperation
	attrs["validate_indices"] = validateIndices
	let opspec = OpSpec(
		type: "SparseToSparseSetOperation",
		name: (operationName ?? "Type"),
		input: [set1Indices, set1Values, set1Shape, set2Indices, set2Values, set2Shape],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (resultIndices: op.output(at: 0), resultValues: op.output(at: 1), resultShape: op.output(at: 2))
} 

///Performs a padding as a preprocess during a convolution.
///Similar to FusedResizeAndPadConv2d, this op allows for an optimized
/// implementation where the spatial padding transformation stage is fused with the
/// im2col lookup, but in this case without the bilinear filtering required for
/// resizing. Fusing the padding prevents the need to write out the intermediate
/// results as whole tensors, reducing memory pressure, and we can get some latency
/// gains by merging the transformation calculations.
/// The data_format attribute for Conv2D isn't supported by this op, and 'NHWC'
/// order is used instead.
/// Internally this op uses a single per-graph scratch buffer, which means that it
/// will block if multiple versions are being run in parallel. This is because this
/// operator is primarily an optimization to minimize memory usage.
/// - Parameter input: 4-D with shape `[batch, in_height, in_width, in_channels]`.
/// - Parameter paddings: A two-column matrix specifying the padding sizes. The number of
/// rows must be the same as the rank of `input`.
/// - Parameter filter: 4-D with shape
/// `[filter_height, filter_width, in_channels, out_channels]`.
/// - Parameter mode: 
/// - Parameter strides: 1-D of length 4.  The stride of the sliding window for each dimension
/// of `input`. Must be in the same order as the dimension specified with format.
/// - Parameter padding: The type of padding algorithm to use.
/// - Returns: 
///	output: 
public func fusedPadConv2D(operationName: String? = nil, input: Output, paddings: Output, filter: Output, mode: String, strides: [Int64], padding: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["mode"] = mode
	attrs["strides"] = strides
	attrs["padding"] = padding
	let opspec = OpSpec(
		type: "FusedPadConv2D",
		name: (operationName ?? "Type"),
		input: [input, paddings, filter],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Updates the table to associates keys with values.
///The tensor `keys` must be of the same type as the keys of the table.
/// The tensor `values` must be of the type of the table values.
/// - Parameter tableHandle: Handle to the table.
/// - Parameter keys: Any shape.  Keys to look up.
/// - Parameter values: Values to associate with keys.
/// - Parameter tin: 
/// - Parameter tout: 
public func lookupTableInsertV2(operationName: String? = nil, tableHandle: Output, keys: Output, values: Output, tin: Any.Type, tout: Any.Type) throws -> Operation { 
	var attrs = [String : Any]()
	attrs["Tin"] = tin
	attrs["Tout"] = tout
	let opspec = OpSpec(
		type: "LookupTableInsertV2",
		name: (operationName ?? "Type"),
		input: [tableHandle, keys, values],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///For each key, assigns the respective value to the specified component.
///If a key is not found in the barrier, this operation will create a new
/// incomplete element. If a key is found in the barrier, and the element
/// already has a value at component_index, this operation will fail with
/// INVALID_ARGUMENT, and leave the barrier in an undefined state.
/// - Parameter handle: The handle to a barrier.
/// - Parameter keys: A one-dimensional tensor of keys, with length n.
/// - Parameter values: An any-dimensional tensor of values, which are associated with the
/// respective keys. The 0th dimension must have length n.
/// - Parameter componentIndex: The component of the barrier elements that is being assigned.
public func barrierInsertMany(operationName: String? = nil, handle: Output, keys: Output, values: Output, componentIndex: UInt8) throws -> Operation { 
	var attrs = [String : Any]()
	attrs["component_index"] = componentIndex
	let opspec = OpSpec(
		type: "BarrierInsertMany",
		name: (operationName ?? "Type"),
		input: [handle, keys, values],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Elementwise computes the bitwise AND of `x` and `y`.
///The result will have those bits set, that are set in both `x` and `y`. The
/// computation is performed on the underlying representations of `x` and `y`.
/// - Parameter x: 
/// - Parameter y: 
/// - Returns: 
///	z: 
public func bitwiseAnd(operationName: String? = nil, x: Output, y: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "BitwiseAnd",
		name: (operationName ?? "Type"),
		input: [x, y],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Deprecated. Use TensorArraySplitV3
/// - Parameter handle: 
/// - Parameter value: 
/// - Parameter lengths: 
/// - Parameter flowIn: 
/// - Returns: 
///	flow_out: 
public func tensorArraySplitV2(operationName: String? = nil, handle: Output, value: Output, lengths: Output, flowIn: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "TensorArraySplitV2",
		name: (operationName ?? "Type"),
		input: [handle, value, lengths, flowIn],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Op removes and returns the values associated with the key
///from the underlying container.   If the underlying container
/// does not contain this key, the op will block until it does.
/// - Parameter key: 
/// - Parameter indices: 
/// - Parameter capacity: 
/// - Parameter memoryLimit: 
/// - Parameter dtypes: 
/// - Parameter container: 
/// - Parameter sharedName: 
/// - Returns: 
///	values: 
public func orderedMapUnstage(operationName: String? = nil, key: Output, indices: Output, capacity: UInt8, memoryLimit: UInt8, dtypes: [Any.Type], container: String, sharedName: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["capacity"] = capacity
	attrs["memory_limit"] = memoryLimit
	attrs["dtypes"] = dtypes
	attrs["container"] = container
	attrs["shared_name"] = sharedName
	let opspec = OpSpec(
		type: "OrderedMapUnstage",
		name: (operationName ?? "Type"),
		input: [key, indices],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Performs average pooling on the input.
///Each entry in `output` is the mean of the corresponding size `ksize`
/// window in `value`.
/// - Parameter value: 4-D with shape `[batch, height, width, channels]`.
/// - Parameter ksize: The size of the sliding window for each dimension of `value`.
/// - Parameter strides: The stride of the sliding window for each dimension of `value`.
/// - Parameter padding: The type of padding algorithm to use.
/// - Parameter dataFormat: Specify the data format of the input and output data. With the
/// default format "NHWC", the data is stored in the order of:
///     [batch, in_height, in_width, in_channels].
/// Alternatively, the format could be "NCHW", the data storage order of:
///     [batch, in_channels, in_height, in_width].
/// - Returns: 
///	output: The average pooled output tensor.
public func avgPool(operationName: String? = nil, value: Output, ksize: [Int64], strides: [Int64], padding: String, dataFormat: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["ksize"] = ksize
	attrs["strides"] = strides
	attrs["padding"] = padding
	attrs["data_format"] = dataFormat
	let opspec = OpSpec(
		type: "AvgPool",
		name: (operationName ?? "Type"),
		input: [value],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 


/// - Parameter input: 
/// - Returns: 
///	output: 
public func batchFFT2D(operationName: String? = nil, input: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "BatchFFT2D",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Op returns the number of incomplete elements in the underlying container.
/// - Parameter capacity: 
/// - Parameter memoryLimit: 
/// - Parameter dtypes: 
/// - Parameter container: 
/// - Parameter sharedName: 
/// - Returns: 
///	size: 
public func mapIncompleteSize(operationName: String? = nil, capacity: UInt8, memoryLimit: UInt8, dtypes: [Any.Type], container: String, sharedName: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["capacity"] = capacity
	attrs["memory_limit"] = memoryLimit
	attrs["dtypes"] = dtypes
	attrs["container"] = container
	attrs["shared_name"] = sharedName
	let opspec = OpSpec(
		type: "MapIncompleteSize",
		name: (operationName ?? "Type"),
		input: [],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes the Eigen Decomposition of a batch of square self-adjoint matrices.
///The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
/// form square matrices, with the same constraints as the single matrix
/// SelfAdjointEig.
/// 
/// The result is a [..., M+1, M] matrix with [..., 0,:] containing the
/// eigenvalues, and subsequent [...,1:, :] containing the eigenvectors.
/// - Parameter input: Shape is `[..., M, M]`.
/// - Returns: 
///	output: Shape is `[..., M+1, M]`.
public func selfAdjointEig(operationName: String? = nil, input: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "SelfAdjointEig",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Sends the named tensor from send_device to recv_device.
///_HostSend requires its input on host memory whereas _Send requires its
/// input on device memory.
/// - Parameter tensor: The tensor to send.
/// - Parameter tensorName: The name of the tensor to send.
/// - Parameter sendDevice: The name of the device sending the tensor.
/// - Parameter sendDeviceIncarnation: The current incarnation of send_device.
/// - Parameter recvDevice: The name of the device receiving the tensor.
/// - Parameter clientTerminated: If set to true, this indicates that the node was added
/// to the graph as a result of a client-side feed or fetch of Tensor data,
/// in which case the corresponding send or recv is expected to be managed
/// locally by the caller.
public func hostSend(operationName: String? = nil, tensor: Output, tensorName: String, sendDevice: String, sendDeviceIncarnation: UInt8, recvDevice: String, clientTerminated: Bool) throws -> Operation { 
	var attrs = [String : Any]()
	attrs["tensor_name"] = tensorName
	attrs["send_device"] = sendDevice
	attrs["send_device_incarnation"] = sendDeviceIncarnation
	attrs["recv_device"] = recvDevice
	attrs["client_terminated"] = clientTerminated
	let opspec = OpSpec(
		type: "_HostSend",
		name: (operationName ?? "Type"),
		input: [tensor],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Restore a Reader to its initial clean state.
/// - Parameter readerHandle: Handle to a Reader.
public func readerResetV2(operationName: String? = nil, readerHandle: Output) throws -> Operation { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "ReaderResetV2",
		name: (operationName ?? "Type"),
		input: [readerHandle],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Op returns the number of elements in the underlying container.
/// - Parameter capacity: 
/// - Parameter memoryLimit: 
/// - Parameter dtypes: 
/// - Parameter container: 
/// - Parameter sharedName: 
/// - Returns: 
///	size: 
public func orderedMapSize(operationName: String? = nil, capacity: UInt8, memoryLimit: UInt8, dtypes: [Any.Type], container: String, sharedName: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["capacity"] = capacity
	attrs["memory_limit"] = memoryLimit
	attrs["dtypes"] = dtypes
	attrs["container"] = container
	attrs["shared_name"] = sharedName
	let opspec = OpSpec(
		type: "OrderedMapSize",
		name: (operationName ?? "Type"),
		input: [],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Makes its input available to the next iteration.
/// - Parameter data: The tensor to be made available to the next iteration.
/// - Returns: 
///	output: The same tensor as `data`.
public func refNextIteration(operationName: String? = nil, data: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "RefNextIteration",
		name: (operationName ?? "Type"),
		input: [data],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Op peeks at the values at the specified key.  If the
///underlying container does not contain this key
/// this op will block until it does.   This Op is optimized for
/// performance.
/// - Parameter key: 
/// - Parameter indices: 
/// - Parameter capacity: 
/// - Parameter memoryLimit: 
/// - Parameter dtypes: 
/// - Parameter container: 
/// - Parameter sharedName: 
/// - Returns: 
///	values: 
public func orderedMapPeek(operationName: String? = nil, key: Output, indices: Output, capacity: UInt8, memoryLimit: UInt8, dtypes: [Any.Type], container: String, sharedName: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["capacity"] = capacity
	attrs["memory_limit"] = memoryLimit
	attrs["dtypes"] = dtypes
	attrs["container"] = container
	attrs["shared_name"] = sharedName
	let opspec = OpSpec(
		type: "OrderedMapPeek",
		name: (operationName ?? "Type"),
		input: [key, indices],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Decode a JPEG-encoded image to a uint8 tensor.
///The attr `channels` indicates the desired number of color channels for the
/// decoded image.
/// 
/// Accepted values are:
/// 
///  *    0: Use the number of channels in the JPEG-encoded image.
///  *    1: output a grayscale image.
///  *    3: output an RGB image.
/// 
/// If needed, the JPEG-encoded image is transformed to match the requested number
/// of color channels.
/// 
/// The attr `ratio` allows downscaling the image by an integer factor during
/// decoding.  Allowed values are: 1, 2, 4, and 8.  This is much faster than
/// downscaling the image later.
/// 
/// 
/// This op also supports decoding PNGs and non-animated GIFs since the interface is
/// the same, though it is cleaner to use `tf.image.decode_image`.
/// - Parameter contents: 0-D.  The JPEG-encoded image.
/// - Parameter channels: Number of color channels for the decoded image.
/// - Parameter ratio: Downscaling ratio.
/// - Parameter fancyUpscaling: If true use a slower but nicer upscaling of the
/// chroma planes (yuv420/422 only).
/// - Parameter tryRecoverTruncated: If true try to recover an image from truncated input.
/// - Parameter acceptableFraction: The minimum required fraction of lines before a truncated
/// input is accepted.
/// - Parameter dctMethod: string specifying a hint about the algorithm used for
/// decompression.  Defaults to "" which maps to a system-specific
/// default.  Currently valid values are ["INTEGER_FAST",
/// "INTEGER_ACCURATE"].  The hint may be ignored (e.g., the internal
/// jpeg library changes to a version that does not have that specific
/// option.)
/// - Returns: 
///	image: 3-D with shape `[height, width, channels]`..
public func decodeJpeg(operationName: String? = nil, contents: Output, channels: UInt8, ratio: UInt8, fancyUpscaling: Bool, tryRecoverTruncated: Bool, acceptableFraction: Float, dctMethod: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["channels"] = channels
	attrs["ratio"] = ratio
	attrs["fancy_upscaling"] = fancyUpscaling
	attrs["try_recover_truncated"] = tryRecoverTruncated
	attrs["acceptable_fraction"] = acceptableFraction
	attrs["dct_method"] = dctMethod
	let opspec = OpSpec(
		type: "DecodeJpeg",
		name: (operationName ?? "Type"),
		input: [contents],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Op removes all elements in the underlying container.
/// - Parameter capacity: 
/// - Parameter memoryLimit: 
/// - Parameter dtypes: 
/// - Parameter container: 
/// - Parameter sharedName: 
public func mapClear(operationName: String? = nil, capacity: UInt8, memoryLimit: UInt8, dtypes: [Any.Type], container: String, sharedName: String) throws -> Operation { 
	var attrs = [String : Any]()
	attrs["capacity"] = capacity
	attrs["memory_limit"] = memoryLimit
	attrs["dtypes"] = dtypes
	attrs["container"] = container
	attrs["shared_name"] = sharedName
	let opspec = OpSpec(
		type: "MapClear",
		name: (operationName ?? "Type"),
		input: [],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Dequeues a tuple of one or more tensors from the given queue.
///This operation has k outputs, where k is the number of components
/// in the tuples stored in the given queue, and output i is the ith
/// component of the dequeued tuple.
/// 
/// N.B. If the queue is empty, this operation will block until an element
/// has been dequeued (or 'timeout_ms' elapses, if specified).
/// - Parameter handle: The handle to a queue.
/// - Parameter componentTypes: The type of each component in a tuple.
/// - Parameter timeoutMs: If the queue is empty, this operation will block for up to
/// timeout_ms milliseconds.
/// Note: This option is not supported yet.
/// - Returns: 
///	components: One or more tensors that were dequeued as a tuple.
public func queueDequeue(operationName: String? = nil, handle: Output, componentTypes: [Any.Type], timeoutMs: UInt8) throws -> Output { 
	var attrs = [String : Any]()
	attrs["component_types"] = componentTypes
	attrs["timeout_ms"] = timeoutMs
	let opspec = OpSpec(
		type: "QueueDequeue",
		name: (operationName ?? "Type"),
		input: [handle],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///2D real-valued fast Fourier transform.
///Computes the 2-dimensional discrete Fourier transform of a real-valued signal
/// over the inner-most 2 dimensions of `input`.
/// 
/// Since the DFT of a real signal is Hermitian-symmetric, `RFFT2D` only returns the
/// `fft_length / 2 + 1` unique components of the FFT for the inner-most dimension
/// of `output`: the zero-frequency term, followed by the `fft_length / 2`
/// positive-frequency terms.
/// 
/// Along each axis `RFFT2D` is computed on, if `fft_length` is smaller than the
/// corresponding dimension of `input`, the dimension is cropped. If it is larger,
/// the dimension is padded with zeros.
/// - Parameter input: A float32 tensor.
/// - Parameter fftLength: An int32 tensor of shape [2]. The FFT length for each dimension.
/// - Returns: 
///	output: A complex64 tensor of the same rank as `input`. The inner-most 2
///   dimensions of `input` are replaced with their 2D Fourier transform. The
///   inner-most dimension contains `fft_length / 2 + 1` unique frequency
///   components.
/// 
/// @compatibility(numpy)
/// Equivalent to np.fft.rfft2
/// @end_compatibility
public func rfft2D(operationName: String? = nil, input: Output, fftLength: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "RFFT2D",
		name: (operationName ?? "Type"),
		input: [input, fftLength],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes the Gauss error function of `x` element-wise.
/// - Parameter x: 
/// - Returns: 
///	y: 
public func erf(operationName: String? = nil, x: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Erf",
		name: (operationName ?? "Type"),
		input: [x],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Cast x of type SrcT to y of DstT.
/// - Parameter x: 
/// - Parameter srcT: 
/// - Parameter dstT: 
/// - Returns: 
///	y: 
public func cast(operationName: String? = nil, x: Output, srcT: Any.Type, dstT: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["SrcT"] = srcT
	attrs["DstT"] = dstT
	let opspec = OpSpec(
		type: "Cast",
		name: (operationName ?? "Type"),
		input: [x],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 


/// - Parameter matrix: 
/// - Parameter rhs: 
/// - Parameter lower: 
/// - Parameter adjoint: 
/// - Returns: 
///	output: 
public func batchMatrixTriangularSolve(operationName: String? = nil, matrix: Output, rhs: Output, lower: Bool, adjoint: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["lower"] = lower
	attrs["adjoint"] = adjoint
	let opspec = OpSpec(
		type: "BatchMatrixTriangularSolve",
		name: (operationName ?? "Type"),
		input: [matrix, rhs],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes second-order gradients of the maxpooling function.
/// - Parameter input: The original input.
/// - Parameter grad: 4-D with shape `[batch, height, width, channels]`.  Gradients w.r.t. the
/// input of `max_pool`.
/// - Parameter argmax: The indices of the maximum values chosen for each output of `max_pool`.
/// - Parameter ksize: The size of the window for each dimension of the input tensor.
/// - Parameter strides: The stride of the sliding window for each dimension of the
/// input tensor.
/// - Parameter padding: The type of padding algorithm to use.
/// - Parameter targmax: 
/// - Returns: 
///	output: Gradients of gradients w.r.t. the input of `max_pool`.
public func maxPoolGradGradWithArgmax(operationName: String? = nil, input: Output, grad: Output, argmax: Output, ksize: [Int64], strides: [Int64], padding: String, targmax: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["ksize"] = ksize
	attrs["strides"] = strides
	attrs["padding"] = padding
	attrs["Targmax"] = targmax
	let opspec = OpSpec(
		type: "MaxPoolGradGradWithArgmax",
		name: (operationName ?? "Type"),
		input: [input, grad, argmax],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Returns the truth value of (x < y) element-wise.
/// * NOTE * : `Less` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
/// - Parameter x: 
/// - Parameter y: 
/// - Returns: 
///	z: 
public func less(operationName: String? = nil, x: Output, y: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Less",
		name: (operationName ?? "Type"),
		input: [x, y],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Applies set operation along last dimension of 2 `Tensor` inputs.
///See SetOperationOp::SetOperationFromContext for values of `set_operation`.
/// 
/// Output `result` is a `SparseTensor` represented by `result_indices`,
/// `result_values`, and `result_shape`. For `set1` and `set2` ranked `n`, this
/// has rank `n` and the same 1st `n-1` dimensions as `set1` and `set2`. The `nth`
/// dimension contains the result of `set_operation` applied to the corresponding
/// `[0...n-1]` dimension of `set`.
/// - Parameter set1: `Tensor` with rank `n`. 1st `n-1` dimensions must be the same as `set2`.
/// Dimension `n` contains values in a set, duplicates are allowed but ignored.
/// - Parameter set2: `Tensor` with rank `n`. 1st `n-1` dimensions must be the same as `set1`.
/// Dimension `n` contains values in a set, duplicates are allowed but ignored.
/// - Parameter setOperation: 
/// - Parameter validateIndices: 
/// - Returns: 
///	result_indices: 2D indices of a `SparseTensor`.
///	result_values: 1D values of a `SparseTensor`.
///	result_shape: 1D `Tensor` shape of a `SparseTensor`. `result_shape[0...n-1]` is
/// the same as the 1st `n-1` dimensions of `set1` and `set2`, `result_shape[n]`
/// is the max result set size across all `0...n-1` dimensions.
public func denseToDenseSetOperation(operationName: String? = nil, set1: Output, set2: Output, setOperation: String, validateIndices: Bool) throws -> (resultIndices: Output, resultValues: Output, resultShape: Output) { 
	var attrs = [String : Any]()
	attrs["set_operation"] = setOperation
	attrs["validate_indices"] = validateIndices
	let opspec = OpSpec(
		type: "DenseToDenseSetOperation",
		name: (operationName ?? "Type"),
		input: [set1, set2],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (resultIndices: op.output(at: 0), resultValues: op.output(at: 1), resultShape: op.output(at: 2))
} 

///Returns true if queue is closed.
///This operation returns true if the queue is closed and false if the queue
/// is open.
/// - Parameter handle: The handle to a queue.
/// - Returns: 
///	is_closed: 
public func queueIsClosedV2(operationName: String? = nil, handle: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "QueueIsClosedV2",
		name: (operationName ?? "Type"),
		input: [handle],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Local Response Normalization.
///The 4-D `input` tensor is treated as a 3-D array of 1-D vectors (along the last
/// dimension), and each vector is normalized independently.  Within a given vector,
/// each component is divided by the weighted, squared sum of inputs within
/// `depth_radius`.  In detail,
/// 
///     sqr_sum[a, b, c, d] =
///         sum(input[a, b, c, d - depth_radius : d + depth_radius + 1]  *  *  2)
///     output = input / (bias + alpha  *  sqr_sum)  *  *  beta
/// 
/// For details, see [Krizhevsky et al., ImageNet classification with deep
/// convolutional neural networks (NIPS 2012)](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks).
/// - Parameter input: 4-D.
/// - Parameter depthRadius: 0-D.  Half-width of the 1-D normalization window.
/// - Parameter bias: An offset (usually positive to avoid dividing by 0).
/// - Parameter alpha: A scale factor, usually positive.
/// - Parameter beta: An exponent.
/// - Returns: 
///	output: 
public func lrn(operationName: String? = nil, input: Output, depthRadius: UInt8, bias: Float, alpha: Float, beta: Float) throws -> Output { 
	var attrs = [String : Any]()
	attrs["depth_radius"] = depthRadius
	attrs["bias"] = bias
	attrs["alpha"] = alpha
	attrs["beta"] = beta
	let opspec = OpSpec(
		type: "LRN",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Compute the Hurwitz zeta function \\(\zeta(x, q)\\).
///The Hurwitz zeta function is defined as:
/// 
/// 
/// \\(\zeta(x, q) = \sum_{n=0}// ^{\infty} (q + n)// ^{-x}\\)
/// - Parameter x: 
/// - Parameter q: 
/// - Returns: 
///	z: 
public func zeta(operationName: String? = nil, x: Output, q: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Zeta",
		name: (operationName ?? "Type"),
		input: [x, q],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Deprecated. Use TensorArrayGradV3
/// - Parameter handle: 
/// - Parameter flowIn: 
/// - Parameter source: 
/// - Returns: 
///	grad_handle: 
public func tensorArrayGradV2(operationName: String? = nil, handle: Output, flowIn: Output, source: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["source"] = source
	let opspec = OpSpec(
		type: "TensorArrayGradV2",
		name: (operationName ?? "Type"),
		input: [handle, flowIn],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Outputs a `Summary` protocol buffer with images.
///The summary has up to `max_images` summary values containing images. The
/// images are built from `tensor` which must be 4-D with shape `[batch_size,
/// height, width, channels]` and where `channels` can be:
/// 
///  *   1: `tensor` is interpreted as Grayscale.
///  *   3: `tensor` is interpreted as RGB.
///  *   4: `tensor` is interpreted as RGBA.
/// 
/// The images have the same number of channels as the input tensor. For float
/// input, the values are normalized one image at a time to fit in the range
/// `[0, 255]`.  `uint8` values are unchanged.  The op uses two different
/// normalization algorithms:
/// 
///  *   If the input values are all positive, they are rescaled so the largest one
///    is 255.
/// 
///  *   If any input value is negative, the values are shifted so input value 0.0
///    is at 127.  They are then rescaled so that either the smallest value is 0,
///    or the largest one is 255.
/// 
/// The `tag` argument is a scalar `Tensor` of type `string`.  It is used to
/// build the `tag` of the summary values:
/// 
///  *   If `max_images` is 1, the summary value tag is ' * tag * /image'.
///  *   If `max_images` is greater than 1, the summary value tags are
///    generated sequentially as ' * tag * /image/0', ' * tag * /image/1', etc.
/// 
/// The `bad_color` argument is the color to use in the generated images for
/// non-finite input values.  It is a `unit8` 1-D tensor of length `channels`.
/// Each element must be in the range `[0, 255]` (It represents the value of a
/// pixel in the output image).  Non-finite values in the input tensor are
/// replaced by this tensor in the output image.  The default value is the color
/// red.
/// - Parameter tag: Scalar. Used to build the `tag` attribute of the summary values.
/// - Parameter tensor: 4-D of shape `[batch_size, height, width, channels]` where
/// `channels` is 1, 3, or 4.
/// - Parameter maxImages: Max number of batch elements to generate images for.
/// - Parameter badColor: Color to use for pixels with non-finite values.
/// - Returns: 
///	summary: Scalar. Serialized `Summary` protocol buffer.
public func imageSummary(operationName: String? = nil, tag: Output, tensor: Output, maxImages: UInt8, badColor: Tensor) throws -> Output { 
	var attrs = [String : Any]()
	attrs["max_images"] = maxImages
	attrs["bad_color"] = badColor
	let opspec = OpSpec(
		type: "ImageSummary",
		name: (operationName ?? "Type"),
		input: [tag, tensor],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///A Reader that outputs the entire contents of a file as a value.
///To use, enqueue filenames in a Queue.  The output of ReaderRead will
/// be a filename (key) and the contents of that file (value).
/// - Parameter container: If non-empty, this reader is placed in the given container.
/// Otherwise, a default container is used.
/// - Parameter sharedName: If non-empty, this reader is named in the given bucket
/// with this shared_name. Otherwise, the node name is used instead.
/// - Returns: 
///	reader_handle: The handle to reference the Reader.
public func wholeFileReaderV2(operationName: String? = nil, container: String, sharedName: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["container"] = container
	attrs["shared_name"] = sharedName
	let opspec = OpSpec(
		type: "WholeFileReaderV2",
		name: (operationName ?? "Type"),
		input: [],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Op removes and returns a random (key, value)
///from the underlying container.   If the underlying container
/// does not contain elements, the op will block until it does.
/// - Parameter indices: 
/// - Parameter capacity: 
/// - Parameter memoryLimit: 
/// - Parameter dtypes: 
/// - Parameter container: 
/// - Parameter sharedName: 
/// - Returns: 
///	key: 
///	values: 
public func mapUnstageNoKey(operationName: String? = nil, indices: Output, capacity: UInt8, memoryLimit: UInt8, dtypes: [Any.Type], container: String, sharedName: String) throws -> (key: Output, values: Output) { 
	var attrs = [String : Any]()
	attrs["capacity"] = capacity
	attrs["memory_limit"] = memoryLimit
	attrs["dtypes"] = dtypes
	attrs["container"] = container
	attrs["shared_name"] = sharedName
	let opspec = OpSpec(
		type: "MapUnstageNoKey",
		name: (operationName ?? "Type"),
		input: [indices],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (key: op.output(at: 0), values: op.output(at: 1))
} 

///Returns locations of true values in a boolean tensor.
///This operation returns the coordinates of true elements in `input`. The
/// coordinates are returned in a 2-D tensor where the first dimension (rows)
/// represents the number of true elements, and the second dimension (columns)
/// represents the coordinates of the true elements. Keep in mind, the shape of
/// the output tensor can vary depending on how many true values there are in
/// `input`. Indices are output in row-major order.
/// 
/// For example:
/// 
/// ```
/// # 'input' tensor is [[True, False]
/// #                    [True, False]]
/// # 'input' has two true values, so output has two coordinates.
/// # 'input' has rank of 2, so coordinates have two indices.
/// where(input) ==> [[0, 0],
///                   [1, 0]]
/// 
/// # `input` tensor is [[[True, False]
/// #                     [True, False]]
/// #                    [[False, True]
/// #                     [False, True]]
/// #                    [[False, False]
/// #                     [False, True]]]
/// # 'input' has 5 true values, so output has 5 coordinates.
/// # 'input' has rank of 3, so coordinates have three indices.
/// where(input) ==> [[0, 0, 0],
///                   [0, 1, 0],
///                   [1, 0, 1],
///                   [1, 1, 1],
///                   [2, 1, 1]]
/// ```
/// - Parameter input: 
/// - Returns: 
///	index: 
public func `where`(operationName: String? = nil, input: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "`where`",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Transforms a spectrogram into a form that's useful for speech recognition.
///Mel Frequency Cepstral Coefficients are a way of representing audio data that's
/// been effective as an input feature for machine learning. They are created by
/// taking the spectrum of a spectrogram (a 'cepstrum'), and discarding some of the
/// higher frequencies that are less significant to the human ear. They have a long
/// history in the speech recognition world, and https://en.wikipedia.org/wiki/Mel-frequency_cepstrum
/// is a good resource to learn more.
/// - Parameter spectrogram: Typically produced by the Spectrogram op, with magnitude_squared
/// set to true.
/// - Parameter sampleRate: How many samples per second the source audio used.
/// - Parameter upperFrequencyLimit: The highest frequency to use when calculating the
/// ceptstrum.
/// - Parameter lowerFrequencyLimit: The lowest frequency to use when calculating the
/// ceptstrum.
/// - Parameter filterbankChannelCount: Resolution of the Mel bank used internally.
/// - Parameter dctCoefficientCount: How many output channels to produce per time slice.
/// - Returns: 
///	output: 
public func mfcc(operationName: String? = nil, spectrogram: Output, sampleRate: Output, upperFrequencyLimit: Float, lowerFrequencyLimit: Float, filterbankChannelCount: UInt8, dctCoefficientCount: UInt8) throws -> Output { 
	var attrs = [String : Any]()
	attrs["upper_frequency_limit"] = upperFrequencyLimit
	attrs["lower_frequency_limit"] = lowerFrequencyLimit
	attrs["filterbank_channel_count"] = filterbankChannelCount
	attrs["dct_coefficient_count"] = dctCoefficientCount
	let opspec = OpSpec(
		type: "Mfcc",
		name: (operationName ?? "Type"),
		input: [spectrogram, sampleRate],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 


/// - Parameter input: 
/// - Returns: 
///	diagonal: 
public func batchMatrixDiagPart(operationName: String? = nil, input: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "BatchMatrixDiagPart",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Deprecated. Disallowed in GraphDef version >= 2.
/// - Parameter images: 
/// - Parameter contrastFactor: 
/// - Parameter minValue: 
/// - Parameter maxValue: 
/// - Returns: 
///	output: 
public func adjustContrast(operationName: String? = nil, images: Output, contrastFactor: Output, minValue: Output, maxValue: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "AdjustContrast",
		name: (operationName ?? "Type"),
		input: [images, contrastFactor, minValue, maxValue],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Resize `images` to `size` using nearest neighbor interpolation.
/// - Parameter images: 4-D with shape `[batch, height, width, channels]`.
/// - Parameter size: = A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
/// new size for the images.
/// - Parameter alignCorners: If true, rescale input by (new_height - 1) / (height - 1), which
/// exactly aligns the 4 corners of images and resized images. If false, rescale
/// by new_height / height. Treat similarly the width dimension.
/// - Returns: 
///	resized_images: 4-D with shape
/// `[batch, new_height, new_width, channels]`.
public func resizeNearestNeighbor(operationName: String? = nil, images: Output, size: Output, alignCorners: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["align_corners"] = alignCorners
	let opspec = OpSpec(
		type: "ResizeNearestNeighbor",
		name: (operationName ?? "Type"),
		input: [images, size],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Serialize an `N`-minibatch `SparseTensor` into an `[N, 3]` string `Tensor`.
///The `SparseTensor` must have rank `R` greater than 1, and the first dimension
/// is treated as the minibatch dimension.  Elements of the `SparseTensor`
/// must be sorted in increasing order of this first dimension.  The serialized
/// `SparseTensor` objects going into each row of `serialized_sparse` will have
/// rank `R-1`.
/// 
/// The minibatch size `N` is extracted from `sparse_shape[0]`.
/// - Parameter sparseIndices: 2-D.  The `indices` of the minibatch `SparseTensor`.
/// - Parameter sparseValues: 1-D.  The `values` of the minibatch `SparseTensor`.
/// - Parameter sparseShape: 1-D.  The `shape` of the minibatch `SparseTensor`.
/// - Returns: 
///	serialized_sparse: 
public func serializeManySparse(operationName: String? = nil, sparseIndices: Output, sparseValues: Output, sparseShape: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "SerializeManySparse",
		name: (operationName ?? "Type"),
		input: [sparseIndices, sparseValues, sparseShape],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Stage (key, values) in the underlying container which behaves like a hashtable.
/// - Parameter key: int64
/// - Parameter indices: 
/// - Parameter values: a list of tensors
/// dtypes A list of data types that inserted values should adhere to.
/// - Parameter capacity: Maximum number of elements in the Staging Area. If > 0, inserts
/// on the container will block when the capacity is reached.
/// - Parameter memoryLimit: 
/// - Parameter dtypes: 
/// - Parameter fakeDtypes: 
/// - Parameter container: If non-empty, this queue is placed in the given container. Otherwise,
/// a default container is used.
/// - Parameter sharedName: It is necessary to match this name to the matching Unstage Op.
public func mapStage(operationName: String? = nil, key: Output, indices: Output, values: Output, capacity: UInt8, memoryLimit: UInt8, dtypes: [Any.Type], fakeDtypes: [Any.Type], container: String, sharedName: String) throws -> Operation { 
	var attrs = [String : Any]()
	attrs["capacity"] = capacity
	attrs["memory_limit"] = memoryLimit
	attrs["dtypes"] = dtypes
	attrs["fake_dtypes"] = fakeDtypes
	attrs["container"] = container
	attrs["shared_name"] = sharedName
	let opspec = OpSpec(
		type: "MapStage",
		name: (operationName ?? "Type"),
		input: [key, indices, values],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Performs greedy decoding on the logits given in inputs.
///A note about the attribute merge_repeated: if enabled, when
/// consecutive logits' maximum indices are the same, only the first of
/// these is emitted.  Labeling the blank ' * ', the sequence "A B B  *  B B"
/// becomes "A B B" if merge_repeated = True and "A B B B B" if
/// merge_repeated = False.
/// 
/// Regardless of the value of merge_repeated, if the maximum index of a given
/// time and batch corresponds to the blank, index `(num_classes - 1)`, no new
/// element is emitted.
/// - Parameter inputs: 3-D, shape: `(max_time x batch_size x num_classes)`, the logits.
/// - Parameter sequenceLength: A vector containing sequence lengths, size `(batch_size)`.
/// - Parameter mergeRepeated: If True, merge repeated classes in output.
/// - Returns: 
///	decoded_indices: Indices matrix, size `(total_decoded_outputs x 2)`,
/// of a `SparseTensor<int64, 2>`.  The rows store: [batch, time].
///	decoded_values: Values vector, size: `(total_decoded_outputs)`,
/// of a `SparseTensor<int64, 2>`.  The vector stores the decoded classes.
///	decoded_shape: Shape vector, size `(2)`, of the decoded SparseTensor.
/// Values are: `[batch_size, max_decoded_length]`.
///	log_probability: Matrix, size `(batch_size x 1)`, containing sequence
/// log-probabilities.
public func cTCGreedyDecoder(operationName: String? = nil, inputs: Output, sequenceLength: Output, mergeRepeated: Bool) throws -> (decodedIndices: Output, decodedValues: Output, decodedShape: Output, logProbability: Output) { 
	var attrs = [String : Any]()
	attrs["merge_repeated"] = mergeRepeated
	let opspec = OpSpec(
		type: "CTCGreedyDecoder",
		name: (operationName ?? "Type"),
		input: [inputs, sequenceLength],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (decodedIndices: op.output(at: 0), decodedValues: op.output(at: 1), decodedShape: op.output(at: 2), logProbability: op.output(at: 3))
} 

///Scatter `updates` into a new (initially zero) tensor according to `indices`.
///Creates a new tensor by applying sparse `updates` to individual
/// values or slices within a zero tensor of the given `shape` according to
/// indices.  This operator is the inverse of the @{tf.gather_nd} operator which
/// extracts values or slices from a given tensor.
/// 
///  *  * WARNING *  * : The order in which updates are applied is nondeterministic, so the
/// output will be nondeterministic if `indices` contains duplicates.
/// 
/// `indices` is an integer tensor containing indices into a new tensor of shape
/// `shape`.  The last dimension of `indices` can be at most the rank of `shape`:
/// 
///     indices.shape[-1] <= shape.rank
/// 
/// The last dimension of `indices` corresponds to indices into elements
/// (if `indices.shape[-1] = shape.rank`) or slices
/// (if `indices.shape[-1] < shape.rank`) along dimension `indices.shape[-1]` of
/// `shape`.  `updates` is a tensor with shape
/// 
///     indices.shape[:-1] + shape[indices.shape[-1]:]
/// 
/// The simplest form of scatter is to insert individual elements in a tensor by
/// index. For example, say we want to insert 4 scattered elements in a rank-1
/// tensor with 8 elements.
/// 
/// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
/// <img style="width:100%" src="https://www.tensorflow.org/images/ScatterNd1.png" alt>
/// </div>
/// 
/// In Python, this scatter operation would look like this:
/// 
/// ```python
///     indices = tf.constant([[4], [3], [1], [7]])
///     updates = tf.constant([9, 10, 11, 12])
///     shape = tf.constant([8])
///     scatter = tf.scatter_nd(indices, updates, shape)
///     with tf.Session() as sess:
///       print(sess.run(scatter))
/// ```
/// 
/// The resulting tensor would look like this:
/// 
///     [0, 11, 0, 10, 9, 0, 0, 12]
/// 
/// We can also, insert entire slices of a higher rank tensor all at once. For
/// example, if we wanted to insert two slices in the first dimension of a
/// rank-3 tensor with two matrices of new values.
/// 
/// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
/// <img style="width:100%" src="https://www.tensorflow.org/images/ScatterNd2.png" alt>
/// </div>
/// 
/// In Python, this scatter operation would look like this:
/// 
/// ```python
///     indices = tf.constant([[0], [2]])
///     updates = tf.constant([[[5, 5, 5, 5], [6, 6, 6, 6],
///                             [7, 7, 7, 7], [8, 8, 8, 8]],
///                            [[5, 5, 5, 5], [6, 6, 6, 6],
///                             [7, 7, 7, 7], [8, 8, 8, 8]]])
///     shape = tf.constant([4, 4, 4])
///     scatter = tf.scatter_nd(indices, updates, shape)
///     with tf.Session() as sess:
///       print(sess.run(scatter))
/// ```
/// 
/// The resulting tensor would look like this:
/// 
///     [[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
///      [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
///      [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
///      [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]
/// - Parameter indices: Index tensor.
/// - Parameter updates: Updates to scatter into output.
/// - Parameter shape: 1-D. The shape of the resulting tensor.
/// - Parameter tindices: 
/// - Returns: 
///	output: A new tensor with the given shape and updates applied according
/// to the indices.
public func scatterNd(operationName: String? = nil, indices: Output, updates: Output, shape: Output, tindices: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Tindices"] = tindices
	let opspec = OpSpec(
		type: "ScatterNd",
		name: (operationName ?? "Type"),
		input: [indices, updates, shape],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Op returns the number of elements in the underlying container.
/// - Parameter capacity: 
/// - Parameter memoryLimit: 
/// - Parameter dtypes: 
/// - Parameter container: 
/// - Parameter sharedName: 
/// - Returns: 
///	size: 
public func stageSize(operationName: String? = nil, capacity: UInt8, memoryLimit: UInt8, dtypes: [Any.Type], container: String, sharedName: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["capacity"] = capacity
	attrs["memory_limit"] = memoryLimit
	attrs["dtypes"] = dtypes
	attrs["container"] = container
	attrs["shared_name"] = sharedName
	let opspec = OpSpec(
		type: "StageSize",
		name: (operationName ?? "Type"),
		input: [],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes the gradient for the inverse of `x` wrt its input.
///Specifically, `grad = -dy  *  y * y`, where `y = 1/x`, and `dy`
/// is the corresponding input gradient.
/// - Parameter y: 
/// - Parameter dy: 
/// - Returns: 
///	z: 
public func reciprocalGrad(operationName: String? = nil, y: Output, dy: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "ReciprocalGrad",
		name: (operationName ?? "Type"),
		input: [y, dy],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Reshapes a quantized tensor as per the Reshape op.
///```
/// - Parameter tensor: 
/// - Parameter shape: Defines the shape of the output tensor.
/// - Parameter inputMin: The minimum value of the input.
/// - Parameter inputMax: The maximum value of the input.
/// - Parameter tshape: 
/// - Returns: 
///	output: 
///	output_min: This value is copied from input_min.
///	output_max: This value is copied from input_max.
public func quantizedReshape(operationName: String? = nil, tensor: Output, shape: Output, inputMin: Output, inputMax: Output, tshape: Any.Type) throws -> (output: Output, outputMin: Output, outputMax: Output) { 
	var attrs = [String : Any]()
	attrs["Tshape"] = tshape
	let opspec = OpSpec(
		type: "QuantizedReshape",
		name: (operationName ?? "Type"),
		input: [tensor, shape, inputMin, inputMax],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (output: op.output(at: 0), outputMin: op.output(at: 1), outputMax: op.output(at: 2))
} 

///Creates a dataset that applies `f` to the outputs of `input_dataset`.
/// - Parameter inputDataset: 
/// - Parameter otherArguments: 
/// - Parameter f: 
/// - Parameter targuments: 
/// - Parameter outputTypes: 
/// - Parameter outputShapes: 
/// - Returns: 
///	handle: 
public func mapDataset(operationName: String? = nil, inputDataset: Output, otherArguments: Output, f: Tensorflow_NameAttrList, targuments: [Any.Type], outputTypes: [Any.Type], outputShapes: [Shape]) throws -> Output { 
	var attrs = [String : Any]()
	attrs["f"] = f
	attrs["Targuments"] = targuments
	attrs["output_types"] = outputTypes
	attrs["output_shapes"] = outputShapes
	let opspec = OpSpec(
		type: "MapDataset",
		name: (operationName ?? "Type"),
		input: [inputDataset, otherArguments],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Quantize the 'input' tensor of type float to 'output' tensor of type 'T'.
///[min_range, max_range] are scalar floats that specify the range for
/// the 'input' data. The 'mode' attribute controls exactly which calculations are
/// used to convert the float values to their quantized equivalents.
/// 
/// In 'MIN_COMBINED' mode, each value of the tensor will undergo the following:
/// 
/// ```
/// out[i] = (in[i] - min_range)  *  range(T) / (max_range - min_range)
/// if T == qint8, out[i] -= (range(T) + 1) / 2.0
/// ```
/// here `range(T) = numeric_limits<T>::max() - numeric_limits<T>::min()`
/// 
///  * MIN_COMBINED Mode Example * 
/// 
/// Assume the input is type float and has a possible range of [0.0, 6.0] and the
/// output type is quint8 ([0, 255]). The min_range and max_range values should be
/// specified as 0.0 and 6.0. Quantizing from float to quint8 will multiply each
/// value of the input by 255/6 and cast to quint8.
/// 
/// If the output type was qint8 ([-128, 127]), the operation will additionally
/// subtract each value by 128 prior to casting, so that the range of values aligns
/// with the range of qint8.
/// 
/// If the mode is 'MIN_FIRST', then this approach is used:
/// 
/// ```
/// number_of_steps = 1 << (# of bits in T)
/// range_adjust = number_of_steps / (number_of_steps - 1)
/// range = (range_max - range_min)  *  range_adjust
/// range_scale = number_of_steps / range
/// quantized = round(input  *  range_scale) - round(range_min  *  range_scale) +
///   numeric_limits<T>::min()
/// quantized = max(quantized, numeric_limits<T>::min())
/// quantized = min(quantized, numeric_limits<T>::max())
/// ```
/// 
/// The biggest difference between this and MIN_COMBINED is that the minimum range
/// is rounded first, before it's subtracted from the rounded value. With
/// MIN_COMBINED, a small bias is introduced where repeated iterations of quantizing
/// and dequantizing will introduce a larger and larger error.
/// 
///  * SCALED mode Example * 
/// 
/// `SCALED` mode matches the quantization approach used in
/// `QuantizeAndDequantize{V2|V3}`.
/// 
/// If the mode is `SCALED`, we do not use the full range of the output type,
/// choosing to elide the lowest possible value for symmetry (e.g., output range is
/// -127 to 127, not -128 to 127 for signed 8 bit quantization), so that 0.0 maps to
/// 0.
/// 
/// We first find the range of values in our tensor. The
/// range we use is always centered on 0, so we find m such that
/// ```c++
///   m = max(abs(input_min), abs(input_max))
/// ```
/// 
/// Our input tensor range is then `[-m, m]`.
/// 
/// Next, we choose our fixed-point quantization buckets, `[min_fixed, max_fixed]`.
/// If T is signed, this is
/// ```
///   num_bits = sizeof(T)  *  8
///   [min_fixed, max_fixed] =
///       [-(1 << (num_bits - 1) - 1), (1 << (num_bits - 1)) - 1]
/// ```
/// 
/// Otherwise, if T is unsigned, the fixed-point range is
/// ```
///   [min_fixed, max_fixed] = [0, (1 << num_bits) - 1]
/// ```
/// 
/// From this we compute our scaling factor, s:
/// ```c++
///   s = (max_fixed - min_fixed) / (2  *  m)
/// ```
/// 
/// Now we can quantize the elements of our tensor:
/// ```c++
/// result = (input  *  s).round_to_nearest()
/// ```
/// 
/// One thing to watch out for is that the operator may choose to adjust the
/// requested minimum and maximum values slightly during the quantization process,
/// so you should always use the output ports as the range for further calculations.
/// For example, if the requested minimum and maximum values are close to equal,
/// they will be separated by a small epsilon value to prevent ill-formed quantized
/// buffers from being created. Otherwise, you can end up with buffers where all the
/// quantized values map to the same float value, which causes problems for
/// operations that have to perform further calculations on them.
/// - Parameter input: 
/// - Parameter minRange: The minimum scalar value possibly produced for the input.
/// - Parameter maxRange: The maximum scalar value possibly produced for the input.
/// - Parameter mode: 
/// - Returns: 
///	output: The quantized data produced from the float input.
///	output_min: The actual minimum scalar value used for the output.
///	output_max: The actual maximum scalar value used for the output.
public func quantizeV2(operationName: String? = nil, input: Output, minRange: Output, maxRange: Output, mode: String) throws -> (output: Output, outputMin: Output, outputMax: Output) { 
	var attrs = [String : Any]()
	attrs["mode"] = mode
	let opspec = OpSpec(
		type: "QuantizeV2",
		name: (operationName ?? "Type"),
		input: [input, minRange, maxRange],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (output: op.output(at: 0), outputMin: op.output(at: 1), outputMax: op.output(at: 2))
} 

///Op is similar to a lightweight Dequeue.
///The basic functionality is similar to dequeue with many fewer
/// capabilities and options.  This Op is optimized for performance.
/// - Parameter capacity: 
/// - Parameter memoryLimit: 
/// - Parameter dtypes: 
/// - Parameter container: 
/// - Parameter sharedName: 
/// - Returns: 
///	values: 
public func unstage(operationName: String? = nil, capacity: UInt8, memoryLimit: UInt8, dtypes: [Any.Type], container: String, sharedName: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["capacity"] = capacity
	attrs["memory_limit"] = memoryLimit
	attrs["dtypes"] = dtypes
	attrs["container"] = container
	attrs["shared_name"] = sharedName
	let opspec = OpSpec(
		type: "Unstage",
		name: (operationName ?? "Type"),
		input: [],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Stage values similar to a lightweight Enqueue.
///The basic functionality of this Op is similar to a queue with many
/// fewer capabilities and options.  This Op is optimized for performance.
/// - Parameter values: a list of tensors
/// dtypes A list of data types that inserted values should adhere to.
/// - Parameter capacity: Maximum number of elements in the Staging Area. If > 0, inserts
/// on the container will block when the capacity is reached.
/// - Parameter memoryLimit: The maximum number of bytes allowed for Tensors in the Staging Area.
/// If > 0, inserts will block until sufficient space is available.
/// - Parameter dtypes: 
/// - Parameter container: If non-empty, this queue is placed in the given container. Otherwise,
/// a default container is used.
/// - Parameter sharedName: It is necessary to match this name to the matching Unstage Op.
public func stage(operationName: String? = nil, values: Output, capacity: UInt8, memoryLimit: UInt8, dtypes: [Any.Type], container: String, sharedName: String) throws -> Operation { 
	var attrs = [String : Any]()
	attrs["capacity"] = capacity
	attrs["memory_limit"] = memoryLimit
	attrs["dtypes"] = dtypes
	attrs["container"] = container
	attrs["shared_name"] = sharedName
	let opspec = OpSpec(
		type: "Stage",
		name: (operationName ?? "Type"),
		input: [values],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Creates a dataset that emits each dim-0 slice of `components` once.
/// - Parameter components: 
/// - Parameter toutputTypes: 
/// - Parameter outputShapes: 
/// - Returns: 
///	handle: 
public func tensorSliceDataset(operationName: String? = nil, components: Output, toutputTypes: [Any.Type], outputShapes: [Shape]) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Toutput_types"] = toutputTypes
	attrs["output_shapes"] = outputShapes
	let opspec = OpSpec(
		type: "TensorSliceDataset",
		name: (operationName ?? "Type"),
		input: [components],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 


/// - Parameter input: 
/// - Returns: 
///	output: 
public func batchIFFT(operationName: String? = nil, input: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "BatchIFFT",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 


/// - Parameter input: 
/// - Returns: 
///	output: 
public func batchMatrixDeterminant(operationName: String? = nil, input: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "BatchMatrixDeterminant",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Delete the tensor specified by its handle in the session.
/// - Parameter handle: The handle for a tensor stored in the session state.
public func deleteSessionTensor(operationName: String? = nil, handle: Output) throws -> Operation { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "DeleteSessionTensor",
		name: (operationName ?? "Type"),
		input: [handle],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Computes sigmoid of `x` element-wise.
///Specifically, `y = 1 / (1 + exp(-x))`.
/// - Parameter x: 
/// - Returns: 
///	y: 
public func sigmoid(operationName: String? = nil, x: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Sigmoid",
		name: (operationName ?? "Type"),
		input: [x],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Bitcasts a tensor from one type to another without copying data.
///Given a tensor `input`, this operation returns a tensor that has the same buffer
/// data as `input` with datatype `type`.
/// 
/// If the input datatype `T` is larger than the output datatype `type` then the
/// shape changes from [...] to [..., sizeof(`T`)/sizeof(`type`)].
/// 
/// If `T` is smaller than `type`, the operator requires that the rightmost
/// dimension be equal to sizeof(`type`)/sizeof(`T`). The shape then goes from
/// [..., sizeof(`type`)/sizeof(`T`)] to [...].
/// 
///  * NOTE * : Bitcast is implemented as a low-level cast, so machines with different
/// endian orderings will give different results.
/// - Parameter input: 
/// - Parameter type: 
/// - Returns: 
///	output: 
public func bitcast(operationName: String? = nil, input: Output, type: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["type"] = type
	let opspec = OpSpec(
		type: "Bitcast",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Store the input tensor in the state of the current session.
/// - Parameter value: The tensor to be stored.
/// - Returns: 
///	handle: The handle for the tensor stored in the session state, represented
/// as a ResourceHandle object.
public func getSessionHandleV2(operationName: String? = nil, value: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "GetSessionHandleV2",
		name: (operationName ?? "Type"),
		input: [value],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes the number of complete elements in the given barrier.
/// - Parameter handle: The handle to a barrier.
/// - Returns: 
///	size: The number of complete elements (i.e. those with all of their value
/// components set) in the barrier.
public func barrierReadySize(operationName: String? = nil, handle: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "BarrierReadySize",
		name: (operationName ?? "Type"),
		input: [handle],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Defines a barrier that persists across different graph executions.
///A barrier represents a key-value map, where each key is a string, and
/// each value is a tuple of tensors.
/// 
/// At runtime, the barrier contains 'complete' and 'incomplete'
/// elements. A complete element has defined tensors for all components of
/// its value tuple, and may be accessed using BarrierTakeMany. An
/// incomplete element has some undefined components in its value tuple,
/// and may be updated using BarrierInsertMany.
/// - Parameter componentTypes: The type of each component in a value.
/// - Parameter shapes: The shape of each component in a value. Each shape must be 1 in the
/// first dimension. The length of this attr must be the same as the length of
/// component_types.
/// - Parameter capacity: The capacity of the barrier.  The default capacity is MAX_INT32,
/// which is the largest capacity of the underlying queue.
/// - Parameter container: If non-empty, this barrier is placed in the given container.
/// Otherwise, a default container is used.
/// - Parameter sharedName: If non-empty, this barrier will be shared under the given name
/// across multiple sessions.
/// - Returns: 
///	handle: The handle to the barrier.
public func barrier(operationName: String? = nil, componentTypes: [Any.Type], shapes: [Shape], capacity: UInt8, container: String, sharedName: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["component_types"] = componentTypes
	attrs["shapes"] = shapes
	attrs["capacity"] = capacity
	attrs["container"] = container
	attrs["shared_name"] = sharedName
	let opspec = OpSpec(
		type: "Barrier",
		name: (operationName ?? "Type"),
		input: [],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Creates a dataset that emits `components` as a tuple of tensors once.
/// - Parameter components: 
/// - Parameter toutputTypes: 
/// - Parameter outputShapes: 
/// - Returns: 
///	handle: 
public func tensorDataset(operationName: String? = nil, components: Output, toutputTypes: [Any.Type], outputShapes: [Shape]) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Toutput_types"] = toutputTypes
	attrs["output_shapes"] = outputShapes
	let opspec = OpSpec(
		type: "TensorDataset",
		name: (operationName ?? "Type"),
		input: [components],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Returns x + y element-wise, working on quantized buffers.
/// - Parameter x: 
/// - Parameter y: 
/// - Parameter minX: The float value that the lowest quantized `x` value represents.
/// - Parameter maxX: The float value that the highest quantized `x` value represents.
/// - Parameter minY: The float value that the lowest quantized `y` value represents.
/// - Parameter maxY: The float value that the highest quantized `y` value represents.
/// - Parameter t1: 
/// - Parameter t2: 
/// - Parameter toutput: 
/// - Returns: 
///	z: 
///	min_z: The float value that the lowest quantized output value represents.
///	max_z: The float value that the highest quantized output value represents.
/// 
///  * NOTE * : `QuantizedAdd` supports limited forms of broadcasting. More about
/// broadcasting [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
public func quantizedAdd(operationName: String? = nil, x: Output, y: Output, minX: Output, maxX: Output, minY: Output, maxY: Output, t1: Any.Type, t2: Any.Type, toutput: Any.Type) throws -> (z: Output, minZ: Output, maxZ: Output) { 
	var attrs = [String : Any]()
	attrs["T1"] = t1
	attrs["T2"] = t2
	attrs["Toutput"] = toutput
	let opspec = OpSpec(
		type: "QuantizedAdd",
		name: (operationName ?? "Type"),
		input: [x, y, minX, maxX, minY, maxY],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (z: op.output(at: 0), minZ: op.output(at: 1), maxZ: op.output(at: 2))
} 

///Creates a dataset that splits a SparseTensor into elements row-wise.
/// - Parameter indices: 
/// - Parameter values: 
/// - Parameter denseShape: 
/// - Parameter tvalues: 
/// - Returns: 
///	handle: 
public func sparseTensorSliceDataset(operationName: String? = nil, indices: Output, values: Output, denseShape: Output, tvalues: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Tvalues"] = tvalues
	let opspec = OpSpec(
		type: "SparseTensorSliceDataset",
		name: (operationName ?? "Type"),
		input: [indices, values, denseShape],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Training via negative sampling.
/// - Parameter wIn: input word embedding.
/// - Parameter wOut: output word embedding.
/// - Parameter examples: A vector of word ids.
/// - Parameter labels: A vector of word ids.
/// - Parameter lr: 
/// - Parameter vocabCount: Count of words in the vocabulary.
/// - Parameter numNegativeSamples: Number of negative samples per example.
public func negTrain(operationName: String? = nil, wIn: Output, wOut: Output, examples: Output, labels: Output, lr: Output, vocabCount: [Int64], numNegativeSamples: UInt8) throws -> Operation { 
	var attrs = [String : Any]()
	attrs["vocab_count"] = vocabCount
	attrs["num_negative_samples"] = numNegativeSamples
	let opspec = OpSpec(
		type: "NegTrain",
		name: (operationName ?? "Type"),
		input: [wIn, wOut, examples, labels, lr],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Generates labels for candidate sampling with a learned unigram distribution.
///See explanations of candidate sampling and the data formats at
/// go/candidate-sampling.
/// 
/// For each batch, this op picks a single set of sampled candidate labels.
/// 
/// The advantages of sampling candidates per-batch are simplicity and the
/// possibility of efficient dense matrix multiplication. The disadvantage is that
/// the sampled candidates must be chosen independently of the context and of the
/// true labels.
/// - Parameter trueClasses: A batch_size  *  num_true matrix, in which each row contains the
/// IDs of the num_true target_classes in the corresponding original label.
/// - Parameter numTrue: Number of true labels per context.
/// - Parameter numSampled: Number of candidates to randomly sample.
/// - Parameter unique: If unique is true, we sample with rejection, so that all sampled
/// candidates in a batch are unique. This requires some approximation to
/// estimate the post-rejection sampling probabilities.
/// - Parameter rangeMax: The sampler will sample integers from the interval [0, range_max).
/// - Parameter seed: If either seed or seed2 are set to be non-zero, the random number
/// generator is seeded by the given seed.  Otherwise, it is seeded by a
/// random seed.
/// - Parameter seed2: An second seed to avoid seed collision.
/// - Returns: 
///	sampled_candidates: A vector of length num_sampled, in which each element is
/// the ID of a sampled candidate.
///	true_expected_count: A batch_size  *  num_true matrix, representing
/// the number of times each candidate is expected to occur in a batch
/// of sampled candidates. If unique=true, then this is a probability.
///	sampled_expected_count: A vector of length num_sampled, for each sampled
/// candidate representing the number of times the candidate is expected
/// to occur in a batch of sampled candidates.  If unique=true, then this is a
/// probability.
public func threadUnsafeUnigramCandidateSampler(operationName: String? = nil, trueClasses: Output, numTrue: UInt8, numSampled: UInt8, unique: Bool, rangeMax: UInt8, seed: UInt8, seed2: UInt8) throws -> (sampledCandidates: Output, trueExpectedCount: Output, sampledExpectedCount: Output) { 
	var attrs = [String : Any]()
	attrs["num_true"] = numTrue
	attrs["num_sampled"] = numSampled
	attrs["unique"] = unique
	attrs["range_max"] = rangeMax
	attrs["seed"] = seed
	attrs["seed2"] = seed2
	let opspec = OpSpec(
		type: "ThreadUnsafeUnigramCandidateSampler",
		name: (operationName ?? "Type"),
		input: [trueClasses],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (sampledCandidates: op.output(at: 0), trueExpectedCount: op.output(at: 1), sampledExpectedCount: op.output(at: 2))
} 

///Delete the stack from its resource container.
/// - Parameter handle: The handle to a stack.
public func stackCloseV2(operationName: String? = nil, handle: Output) throws -> Operation { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "StackCloseV2",
		name: (operationName ?? "Type"),
		input: [handle],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Deprecated. Use TensorArrayCloseV3
/// - Parameter handle: 
public func tensorArrayCloseV2(operationName: String? = nil, handle: Output) throws -> Operation { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "TensorArrayCloseV2",
		name: (operationName ?? "Type"),
		input: [handle],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 


/// - Parameter input: 
/// - Parameter numLower: 
/// - Parameter numUpper: 
/// - Returns: 
///	band: 
public func batchMatrixBandPart(operationName: String? = nil, input: Output, numLower: Output, numUpper: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "BatchMatrixBandPart",
		name: (operationName ?? "Type"),
		input: [input, numLower, numUpper],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 


/// - Parameter handle: 
public func tensorArrayClose(operationName: String? = nil, handle: Output) throws -> Operation { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "TensorArrayClose",
		name: (operationName ?? "Type"),
		input: [handle],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Returns x / y element-wise.
/// * NOTE * : `Div` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
/// - Parameter x: 
/// - Parameter y: 
/// - Returns: 
///	z: 
public func div(operationName: String? = nil, x: Output, y: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Div",
		name: (operationName ?? "Type"),
		input: [x, y],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Flushes and closes the summary writer.
///Also removes it from the resource manager. To reopen, use another
/// CreateSummaryFileWriter op.
/// - Parameter writer: A handle to the summary writer resource.
public func closeSummaryWriter(operationName: String? = nil, writer: Output) throws -> Operation { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "CloseSummaryWriter",
		name: (operationName ?? "Type"),
		input: [writer],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Deprecated. Use TensorArraySizeV3
/// - Parameter handle: 
/// - Parameter flowIn: 
/// - Returns: 
///	size: 
public func tensorArraySizeV2(operationName: String? = nil, handle: Output, flowIn: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "TensorArraySizeV2",
		name: (operationName ?? "Type"),
		input: [handle, flowIn],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Returns element-wise remainder of division. When `x < 0` xor `y < 0` is
///true, this follows Python semantics in that the result here is consistent
/// with a flooring divide. E.g. `floor(x / y)  *  y + mod(x, y) = x`.
/// 
///  * NOTE * : `FloorMod` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
/// - Parameter x: 
/// - Parameter y: 
/// - Returns: 
///	z: 
public func floorMod(operationName: String? = nil, x: Output, y: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "FloorMod",
		name: (operationName ?? "Type"),
		input: [x, y],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Returns the set of files matching one or more glob patterns.
///Note that this routine only supports wildcard characters in the
/// basename portion of the pattern, not in the directory portion.
/// - Parameter pattern: Shell wildcard pattern(s). Scalar or vector of type string.
/// - Returns: 
///	filenames: A vector of matching filenames.
public func matchingFiles(operationName: String? = nil, pattern: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "MatchingFiles",
		name: (operationName ?? "Type"),
		input: [pattern],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Restores a tensor from checkpoint files.
///Reads a tensor stored in one or several files. If there are several files (for
/// instance because a tensor was saved as slices), `file_pattern` may contain
/// wildcard symbols (` * ` and `?`) in the filename portion only, not in the
/// directory portion.
/// 
/// If a `file_pattern` matches several files, `preferred_shard` can be used to hint
/// in which file the requested tensor is likely to be found. This op will first
/// open the file at index `preferred_shard` in the list of matching files and try
/// to restore tensors from that file.  Only if some tensors or tensor slices are
/// not found in that first file, then the Op opens all the files. Setting
/// `preferred_shard` to match the value passed as the `shard` input
/// of a matching `Save` Op may speed up Restore.  This attribute only affects
/// performance, not correctness.  The default value -1 means files are processed in
/// order.
/// 
/// See also `RestoreSlice`.
/// - Parameter filePattern: Must have a single element. The pattern of the files from
/// which we read the tensor.
/// - Parameter tensorName: Must have a single element. The name of the tensor to be
/// restored.
/// - Parameter dt: The type of the tensor to be restored.
/// - Parameter preferredShard: Index of file to open first if multiple files match
/// `file_pattern`.
/// - Returns: 
///	tensor: The restored tensor.
public func restore(operationName: String? = nil, filePattern: Output, tensorName: Output, dt: Any.Type, preferredShard: UInt8) throws -> Output { 
	var attrs = [String : Any]()
	attrs["dt"] = dt
	attrs["preferred_shard"] = preferredShard
	let opspec = OpSpec(
		type: "Restore",
		name: (operationName ?? "Type"),
		input: [filePattern, tensorName],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes hyperbolic tangent of `x` element-wise.
/// - Parameter x: 
/// - Returns: 
///	y: 
public func tanh(operationName: String? = nil, x: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Tanh",
		name: (operationName ?? "Type"),
		input: [x],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes the gradient of the crop_and_resize op wrt the input image tensor.
/// - Parameter grads: A 4-D tensor of shape `[num_boxes, crop_height, crop_width, depth]`.
/// - Parameter boxes: A 2-D tensor of shape `[num_boxes, 4]`. The `i`-th row of the tensor
/// specifies the coordinates of a box in the `box_ind[i]` image and is specified
/// in normalized coordinates `[y1, x1, y2, x2]`. A normalized coordinate value of
/// `y` is mapped to the image coordinate at `y  *  (image_height - 1)`, so as the
/// `[0, 1]` interval of normalized image height is mapped to
/// `[0, image_height - 1] in image height coordinates. We do allow y1 > y2, in
/// which case the sampled crop is an up-down flipped version of the original
/// image. The width dimension is treated similarly. Normalized coordinates
/// outside the `[0, 1]` range are allowed, in which case we use
/// `extrapolation_value` to extrapolate the input image values.
/// - Parameter boxInd: A 1-D tensor of shape `[num_boxes]` with int32 values in `[0, batch)`.
/// The value of `box_ind[i]` specifies the image that the `i`-th box refers to.
/// - Parameter imageSize: A 1-D tensor with value `[batch, image_height, image_width, depth]`
/// containing the original image size. Both `image_height` and `image_width` need
/// to be positive.
/// - Parameter method: A string specifying the interpolation method. Only 'bilinear' is
/// supported for now.
/// - Returns: 
///	output: A 4-D tensor of shape `[batch, image_height, image_width, depth]`.
public func cropAndResizeGradImage(operationName: String? = nil, grads: Output, boxes: Output, boxInd: Output, imageSize: Output, method: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["method"] = method
	let opspec = OpSpec(
		type: "CropAndResizeGradImage",
		name: (operationName ?? "Type"),
		input: [grads, boxes, boxInd, imageSize],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes Quantized Rectified Linear X: `min(max(features, 0), max_value)`
/// - Parameter features: 
/// - Parameter maxValue: 
/// - Parameter minFeatures: The float value that the lowest quantized value represents.
/// - Parameter maxFeatures: The float value that the highest quantized value represents.
/// - Parameter tinput: 
/// - Parameter outType: 
/// - Returns: 
///	activations: Has the same output shape as "features".
///	min_activations: The float value that the lowest quantized value represents.
///	max_activations: The float value that the highest quantized value represents.
public func quantizedReluX(operationName: String? = nil, features: Output, maxValue: Output, minFeatures: Output, maxFeatures: Output, tinput: Any.Type, outType: Any.Type) throws -> (activations: Output, minActivations: Output, maxActivations: Output) { 
	var attrs = [String : Any]()
	attrs["Tinput"] = tinput
	attrs["out_type"] = outType
	let opspec = OpSpec(
		type: "QuantizedReluX",
		name: (operationName ?? "Type"),
		input: [features, maxValue, minFeatures, maxFeatures],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (activations: op.output(at: 0), minActivations: op.output(at: 1), maxActivations: op.output(at: 2))
} 

///Extracts the average gradient in the given ConditionalAccumulator.
///The op blocks until sufficient (i.e., more than num_required)
/// gradients have been accumulated.  If the accumulator has already
/// aggregated more than num_required gradients, it returns the average of
/// the accumulated gradients.  Also automatically increments the recorded
/// global_step in the accumulator by 1, and resets the aggregate to 0.
/// - Parameter handle: The handle to an accumulator.
/// - Parameter numRequired: Number of gradients required before we return an aggregate.
/// - Parameter dtype: The data type of accumulated gradients. Needs to correspond to the type
/// of the accumulator.
/// - Returns: 
///	average: The average of the accumulated gradients.
public func accumulatorTakeGradient(operationName: String? = nil, handle: Output, numRequired: Output, dtype: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["dtype"] = dtype
	let opspec = OpSpec(
		type: "AccumulatorTakeGradient",
		name: (operationName ?? "Type"),
		input: [handle, numRequired],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Update ' * var' according to the Ftrl-proximal scheme.
///accum_new = accum + grad  *  grad
/// linear += grad + (accum_new// ^(-lr_power) - accum// ^(-lr_power)) / lr  *  var
/// quadratic = 1.0 / (accum_new// ^(lr_power)  *  lr) + 2  *  l2
/// var = (sign(linear)  *  l1 - linear) / quadratic if |linear| > l1 else 0.0
/// accum = accum_new
/// - Parameter `var`: Should be from a Variable().
/// - Parameter accum: Should be from a Variable().
/// - Parameter linear: Should be from a Variable().
/// - Parameter grad: The gradient.
/// - Parameter lr: Scaling factor. Must be a scalar.
/// - Parameter l1: L1 regulariation. Must be a scalar.
/// - Parameter l2: L2 regulariation. Must be a scalar.
/// - Parameter lrPower: Scaling factor. Must be a scalar.
/// - Parameter useLocking: If `True`, updating of the var and accum tensors will be protected
/// by a lock; otherwise the behavior is undefined, but may exhibit less
/// contention.
/// - Returns: 
///	out: Same as "var".
public func applyFtrl(operationName: String? = nil, `var`: Output, accum: Output, linear: Output, grad: Output, lr: Output, l1: Output, l2: Output, lrPower: Output, useLocking: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["use_locking"] = useLocking
	let opspec = OpSpec(
		type: "ApplyFtrl",
		name: (operationName ?? "Type"),
		input: [`var`, accum, linear, grad, lr, l1, l2, lrPower],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Inverse real-valued fast Fourier transform.
///Computes the inverse 1-dimensional discrete Fourier transform of a real-valued
/// signal over the inner-most dimension of `input`.
/// 
/// The inner-most dimension of `input` is assumed to be the result of `RFFT`: the
/// `fft_length / 2 + 1` unique components of the DFT of a real-valued signal. If
/// `fft_length` is not provided, it is computed from the size of the inner-most
/// dimension of `input` (`fft_length = 2  *  (inner - 1)`). If the FFT length used to
/// compute `input` is odd, it should be provided since it cannot be inferred
/// properly.
/// 
/// Along the axis `IRFFT` is computed on, if `fft_length / 2 + 1` is smaller
/// than the corresponding dimension of `input`, the dimension is cropped. If it is
/// larger, the dimension is padded with zeros.
/// - Parameter input: A complex64 tensor.
/// - Parameter fftLength: An int32 tensor of shape [1]. The FFT length.
/// - Returns: 
///	output: A float32 tensor of the same rank as `input`. The inner-most
///   dimension of `input` is replaced with the `fft_length` samples of its inverse
///   1D Fourier transform.
/// 
/// @compatibility(numpy)
/// Equivalent to np.fft.irfft
/// @end_compatibility
public func irfft(operationName: String? = nil, input: Output, fftLength: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "IRFFT",
		name: (operationName ?? "Type"),
		input: [input, fftLength],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Compare values of `input` to `threshold` and pack resulting bits into a `uint8`.
///Each comparison returns a boolean `true` (if `input_value > threshold`)
/// or and `false` otherwise.
/// 
/// This operation is useful for Locality-Sensitive-Hashing (LSH) and other
/// algorithms that use hashing approximations of cosine and `L2` distances;
/// codes can be generated from an input via:
/// 
/// ```python
/// codebook_size = 50
/// codebook_bits = codebook_size  *  32
/// codebook = tf.get_variable('codebook', [x.shape[-1].value, codebook_bits],
///                            dtype=x.dtype,
///                            initializer=tf.orthogonal_initializer())
/// codes = compare_and_threshold(tf.matmul(x, codebook), threshold=0.)
/// codes = tf.bitcast(codes, tf.int32)  # go from uint8 to int32
/// # now codes has shape x.shape[:-1] + [codebook_size]
/// ```
/// 
///  *  * NOTE *  * : Currently, the innermost dimension of the tensor must be divisible
/// by 8.
/// 
/// Given an `input` shaped `[s0, s1, ..., s_n]`, the output is
/// a `uint8` tensor shaped `[s0, s1, ..., s_n / 8]`.
/// - Parameter input: Values to compare against `threshold` and bitpack.
/// - Parameter threshold: Threshold to compare against.
/// - Returns: 
///	output: The bitpacked comparisons.
public func compareAndBitpack(operationName: String? = nil, input: Output, threshold: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "CompareAndBitpack",
		name: (operationName ?? "Type"),
		input: [input, threshold],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Saves the state of the `iterator` at `path`.
///This state can be restored using "RestoreIterator".
/// - Parameter iterator: 
/// - Parameter path: 
public func saveIterator(operationName: String? = nil, iterator: Output, path: Output) throws -> Operation { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "SaveIterator",
		name: (operationName ?? "Type"),
		input: [iterator, path],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Converts one or more images from RGB to HSV.
///Outputs a tensor of the same shape as the `images` tensor, containing the HSV
/// value of the pixels. The output is only well defined if the value in `images`
/// are in `[0,1]`.
/// 
/// `output[..., 0]` contains hue, `output[..., 1]` contains saturation, and
/// `output[..., 2]` contains value. All HSV values are in `[0,1]`. A hue of 0
/// corresponds to pure red, hue 1/3 is pure green, and 2/3 is pure blue.
/// - Parameter images: 1-D or higher rank. RGB data to convert. Last dimension must be size 3.
/// - Returns: 
///	output: `images` converted to HSV.
public func rGBToHSV(operationName: String? = nil, images: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "RGBToHSV",
		name: (operationName ?? "Type"),
		input: [images],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Deprecated. Use TensorArrayScatterV3
/// - Parameter handle: 
/// - Parameter indices: 
/// - Parameter value: 
/// - Parameter flowIn: 
/// - Returns: 
///	flow_out: 
public func tensorArrayScatterV2(operationName: String? = nil, handle: Output, indices: Output, value: Output, flowIn: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "TensorArrayScatterV2",
		name: (operationName ?? "Type"),
		input: [handle, indices, value, flowIn],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Converts each string in the input Tensor to its hash mod by a number of buckets.
///The hash function is deterministic on the content of the string within the
/// process and will never change. However, it is not suitable for cryptography.
/// This function may be used when CPU time is scarce and inputs are trusted or
/// unimportant. There is a risk of adversaries constructing inputs that all hash
/// to the same bucket. To prevent this problem, use a strong hash function with
/// `tf.string_to_hash_bucket_strong`.
/// - Parameter input: The strings to assign a hash bucket.
/// - Parameter numBuckets: The number of buckets.
/// - Returns: 
///	output: A Tensor of the same shape as the input `string_tensor`.
public func stringToHashBucketFast(operationName: String? = nil, input: Output, numBuckets: UInt8) throws -> Output { 
	var attrs = [String : Any]()
	attrs["num_buckets"] = numBuckets
	let opspec = OpSpec(
		type: "StringToHashBucketFast",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Assign `value` to the sliced l-value reference of `ref`.
///The values of `value` are assigned to the positions in the variable
/// `ref` that are selected by the slice parameters. The slice parameters
/// `begin, `end`, `strides`, etc. work exactly as in `StridedSlice`.
/// 
/// NOTE this op currently does not support broadcasting and so `value`'s
/// shape must be exactly the shape produced by the slice of `ref`.
/// - Parameter ref: 
/// - Parameter begin: 
/// - Parameter end: 
/// - Parameter strides: 
/// - Parameter value: 
/// - Parameter index: 
/// - Parameter beginMask: 
/// - Parameter endMask: 
/// - Parameter ellipsisMask: 
/// - Parameter newAxisMask: 
/// - Parameter shrinkAxisMask: 
/// - Returns: 
///	output_ref: 
public func stridedSliceAssign(operationName: String? = nil, ref: Output, begin: Output, end: Output, strides: Output, value: Output, index: Any.Type, beginMask: UInt8, endMask: UInt8, ellipsisMask: UInt8, newAxisMask: UInt8, shrinkAxisMask: UInt8) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Index"] = index
	attrs["begin_mask"] = beginMask
	attrs["end_mask"] = endMask
	attrs["ellipsis_mask"] = ellipsisMask
	attrs["new_axis_mask"] = newAxisMask
	attrs["shrink_axis_mask"] = shrinkAxisMask
	let opspec = OpSpec(
		type: "StridedSliceAssign",
		name: (operationName ?? "Type"),
		input: [ref, begin, end, strides, value],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Creates a handle to a Variable resource.
/// - Parameter container: the container this variable is placed in.
/// - Parameter sharedName: the name by which this variable is referred to.
/// - Parameter dtype: the type of this variable. Must agree with the dtypes
/// of all ops using this variable.
/// - Parameter shape: The (possibly partially specified) shape of this variable.
/// - Returns: 
///	resource: 
public func varHandleOp(operationName: String? = nil, container: String, sharedName: String, dtype: Any.Type, shape: Shape) throws -> Output { 
	var attrs = [String : Any]()
	attrs["container"] = container
	attrs["shared_name"] = sharedName
	attrs["dtype"] = dtype
	attrs["shape"] = shape
	let opspec = OpSpec(
		type: "VarHandleOp",
		name: (operationName ?? "Type"),
		input: [],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 


/// - Parameter handle: 
/// - Parameter indices: 
/// - Parameter value: 
/// - Parameter flowIn: 
/// - Returns: 
///	flow_out: 
public func tensorArrayScatter(operationName: String? = nil, handle: Output, indices: Output, value: Output, flowIn: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "TensorArrayScatter",
		name: (operationName ?? "Type"),
		input: [handle, indices, value, flowIn],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Partitions `data` into `num_partitions` tensors using indices from `partitions`.
///For each index tuple `js` of size `partitions.ndim`, the slice `data[js, ...]`
/// becomes part of `outputs[partitions[js]]`.  The slices with `partitions[js] = i`
/// are placed in `outputs[i]` in lexicographic order of `js`, and the first
/// dimension of `outputs[i]` is the number of entries in `partitions` equal to `i`.
/// In detail,
/// 
/// ```python
///     outputs[i].shape = [sum(partitions == i)] + data.shape[partitions.ndim:]
/// 
///     outputs[i] = pack([data[js, ...] for js if partitions[js] == i])
/// ```
/// 
/// `data.shape` must start with `partitions.shape`.
/// 
/// For example:
/// 
/// ```python
///     # Scalar partitions.
///     partitions = 1
///     num_partitions = 2
///     data = [10, 20]
///     outputs[0] = []  # Empty with shape [0, 2]
///     outputs[1] = [[10, 20]]
/// 
///     # Vector partitions.
///     partitions = [0, 0, 1, 1, 0]
///     num_partitions = 2
///     data = [10, 20, 30, 40, 50]
///     outputs[0] = [10, 20, 50]
///     outputs[1] = [30, 40]
/// ```
/// 
/// See `dynamic_stitch` for an example on how to merge partitions back.
/// 
/// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
/// <img style="width:100%" src="https://www.tensorflow.org/images/DynamicPartition.png" alt>
/// </div>
/// - Parameter data: 
/// - Parameter partitions: Any shape.  Indices in the range `[0, num_partitions)`.
/// - Parameter numPartitions: The number of partitions to output.
/// - Returns: 
///	outputs: 
public func dynamicPartition(operationName: String? = nil, data: Output, partitions: Output, numPartitions: UInt8) throws -> Output { 
	var attrs = [String : Any]()
	attrs["num_partitions"] = numPartitions
	let opspec = OpSpec(
		type: "DynamicPartition",
		name: (operationName ?? "Type"),
		input: [data, partitions],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Deprecated. Do not use.
/// - Parameter resource: 
/// - Returns: 
///	handle: 
public func fakeQueue(operationName: String? = nil, resource: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "FakeQueue",
		name: (operationName ?? "Type"),
		input: [resource],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 


/// - Parameter handle: 
/// - Parameter flowIn: 
/// - Parameter dtype: 
/// - Parameter elementShape: 
/// - Returns: 
///	value: 
public func tensorArrayPack(operationName: String? = nil, handle: Output, flowIn: Output, dtype: Any.Type, elementShape: Shape) throws -> Output { 
	var attrs = [String : Any]()
	attrs["dtype"] = dtype
	attrs["element_shape"] = elementShape
	let opspec = OpSpec(
		type: "TensorArrayPack",
		name: (operationName ?? "Type"),
		input: [handle, flowIn],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes the gradient of morphological 2-D dilation with respect to the filter.
/// - Parameter input: 4-D with shape `[batch, in_height, in_width, depth]`.
/// - Parameter filter: 3-D with shape `[filter_height, filter_width, depth]`.
/// - Parameter outBackprop: 4-D with shape `[batch, out_height, out_width, depth]`.
/// - Parameter strides: 1-D of length 4. The stride of the sliding window for each dimension of
/// the input tensor. Must be: `[1, stride_height, stride_width, 1]`.
/// - Parameter rates: 1-D of length 4. The input stride for atrous morphological dilation.
/// Must be: `[1, rate_height, rate_width, 1]`.
/// - Parameter padding: The type of padding algorithm to use.
/// - Returns: 
///	filter_backprop: 3-D with shape `[filter_height, filter_width, depth]`.
public func dilation2DBackpropFilter(operationName: String? = nil, input: Output, filter: Output, outBackprop: Output, strides: [Int64], rates: [Int64], padding: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["strides"] = strides
	attrs["rates"] = rates
	attrs["padding"] = padding
	let opspec = OpSpec(
		type: "Dilation2DBackpropFilter",
		name: (operationName ?? "Type"),
		input: [input, filter, outBackprop],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Pads a tensor.
///This operation pads `input` according to the `paddings` and `constant_values`
/// you specify. `paddings` is an integer tensor with shape `[Dn, 2]`, where n is
/// the rank of `input`. For each dimension D of `input`, `paddings[D, 0]` indicates
/// how many padding values to add before the contents of `input` in that dimension,
/// and `paddings[D, 1]` indicates how many padding values to add after the contents
/// of `input` in that dimension. `constant_values` is a scalar tensor of the same
/// type as `input` that indicates the value to use for padding `input`.
/// 
/// The padded size of each dimension D of the output is:
/// 
/// `paddings(D, 0) + input.dim_size(D) + paddings(D, 1)`
/// 
/// For example:
/// 
/// ```
/// # 't' is [[1, 1], [2, 2]]
/// # 'paddings' is [[1, 1], [2, 2]]
/// # 'constant_values' is 0
/// # rank of 't' is 2
/// pad(t, paddings) ==> [[0, 0, 0, 0, 0, 0]
///                       [0, 0, 1, 1, 0, 0]
///                       [0, 0, 2, 2, 0, 0]
///                       [0, 0, 0, 0, 0, 0]]
/// ```
/// - Parameter input: 
/// - Parameter paddings: 
/// - Parameter constantValues: 
/// - Parameter tpaddings: 
/// - Returns: 
///	output: 
public func padV2(operationName: String? = nil, input: Output, paddings: Output, constantValues: Output, tpaddings: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Tpaddings"] = tpaddings
	let opspec = OpSpec(
		type: "PadV2",
		name: (operationName ?? "Type"),
		input: [input, paddings, constantValues],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 


/// - Parameter input: 
/// - Parameter computeV: 
/// - Returns: 
///	e: 
///	v: 
public func batchSelfAdjointEigV2(operationName: String? = nil, input: Output, computeV: Bool) throws -> (e: Output, v: Output) { 
	var attrs = [String : Any]()
	attrs["compute_v"] = computeV
	let opspec = OpSpec(
		type: "BatchSelfAdjointEigV2",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (e: op.output(at: 0), v: op.output(at: 1))
} 

///Computes the gradient for the tanh of `x` wrt its input.
///Specifically, `grad = dy  *  (1 - y * y)`, where `y = tanh(x)`, and `dy`
/// is the corresponding input gradient.
/// - Parameter y: 
/// - Parameter dy: 
/// - Returns: 
///	z: 
public func tanhGrad(operationName: String? = nil, y: Output, dy: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "TanhGrad",
		name: (operationName ?? "Type"),
		input: [y, dy],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Creates a dataset that applies `f` to the outputs of `input_dataset`.
///Unlike a "MapDataset", which applies `f` sequentially, this dataset invokes up
/// to `num_parallel_calls` copies of `f` in parallel.
/// - Parameter inputDataset: 
/// - Parameter otherArguments: 
/// - Parameter numParallelCalls: The number of concurrent invocations of `f` that process
/// elements from `input_dataset` in parallel.
/// - Parameter f: 
/// - Parameter targuments: 
/// - Parameter outputTypes: 
/// - Parameter outputShapes: 
/// - Returns: 
///	handle: 
public func parallelMapDataset(operationName: String? = nil, inputDataset: Output, otherArguments: Output, numParallelCalls: Output, f: Tensorflow_NameAttrList, targuments: [Any.Type], outputTypes: [Any.Type], outputShapes: [Shape]) throws -> Output { 
	var attrs = [String : Any]()
	attrs["f"] = f
	attrs["Targuments"] = targuments
	attrs["output_types"] = outputTypes
	attrs["output_shapes"] = outputShapes
	let opspec = OpSpec(
		type: "ParallelMapDataset",
		name: (operationName ?? "Type"),
		input: [inputDataset, otherArguments, numParallelCalls],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Unpacks a given dimension of a rank-`R` tensor into `num` rank-`(R-1)` tensors.
///Unpacks `num` tensors from `value` by chipping it along the `axis` dimension.
/// For example, given a tensor of shape `(A, B, C, D)`;
/// 
/// If `axis == 0` then the i'th tensor in `output` is the slice `value[i, :, :, :]`
///   and each tensor in `output` will have shape `(B, C, D)`. (Note that the
///   dimension unpacked along is gone, unlike `split`).
/// 
/// If `axis == 1` then the i'th tensor in `output` is the slice `value[:, i, :, :]`
///   and each tensor in `output` will have shape `(A, C, D)`.
/// Etc.
/// 
/// This is the opposite of `pack`.
/// - Parameter value: 1-D or higher, with `axis` dimension size equal to `num`.
/// - Parameter num: 
/// - Parameter axis: Dimension along which to unpack.  Negative values wrap around, so the
/// valid range is `[-R, R)`.
/// - Returns: 
///	output: The list of tensors unpacked from `value`.
public func unpack(operationName: String? = nil, value: Output, num: UInt8, axis: UInt8) throws -> Output { 
	var attrs = [String : Any]()
	attrs["num"] = num
	attrs["axis"] = axis
	let opspec = OpSpec(
		type: "Unpack",
		name: (operationName ?? "Type"),
		input: [value],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes the max of elements across dimensions of a SparseTensor.
///This Op takes a SparseTensor and is the sparse counterpart to
/// `tf.reduce_max()`.  In particular, this Op also returns a dense `Tensor`
/// instead of a sparse one.
/// 
/// Reduces `sp_input` along the dimensions given in `reduction_axes`.  Unless
/// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
/// `reduction_axes`. If `keep_dims` is true, the reduced dimensions are retained
/// with length 1.
/// 
/// If `reduction_axes` has no entries, all dimensions are reduced, and a tensor
/// with a single element is returned.  Additionally, the axes can be negative,
/// which are interpreted according to the indexing rules in Python.
/// - Parameter inputIndices: 2-D.  `N x R` matrix with the indices of non-empty values in a
/// SparseTensor, possibly not in canonical ordering.
/// - Parameter inputValues: 1-D.  `N` non-empty values corresponding to `input_indices`.
/// - Parameter inputShape: 1-D.  Shape of the input SparseTensor.
/// - Parameter reductionAxes: 1-D.  Length-`K` vector containing the reduction axes.
/// - Parameter keepDims: If true, retain reduced dimensions with length 1.
/// - Returns: 
///	output: `R-K`-D.  The reduced Tensor.
public func sparseReduceMax(operationName: String? = nil, inputIndices: Output, inputValues: Output, inputShape: Output, reductionAxes: Output, keepDims: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["keep_dims"] = keepDims
	let opspec = OpSpec(
		type: "SparseReduceMax",
		name: (operationName ?? "Type"),
		input: [inputIndices, inputValues, inputShape, reductionAxes],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes the Max along segments of a tensor.
///Read @{$math_ops#segmentation$the section on segmentation} for an explanation of
/// segments.
/// 
/// This operator is similar to the [unsorted segment sum operator](../../../api_docs/python/math_ops.md#UnsortedSegmentSum).
/// Instead of computing the sum over segments, it computes the maximum
/// such that:
/// 
/// \\(output_i = \max_j data_j\\) where max is over `j` such
/// that `segment_ids[j] == i`.
/// 
/// If the maximum is empty for a given segment ID `i`, it outputs the smallest possible value for specific numeric type,
///  `output[i] = numeric_limits<T>::min()`.
/// 
/// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
/// <img style="width:100%" src="https://www.tensorflow.org/images/UnsortedSegmentMax.png" alt>
/// </div>
/// - Parameter data: 
/// - Parameter segmentIds: A 1-D tensor whose rank is equal to the rank of `data`'s
/// first dimension.
/// - Parameter numSegments: 
/// - Parameter tindices: 
/// - Returns: 
///	output: Has same shape as data, except for dimension 0 which
/// has size `num_segments`.
public func unsortedSegmentMax(operationName: String? = nil, data: Output, segmentIds: Output, numSegments: Output, tindices: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Tindices"] = tindices
	let opspec = OpSpec(
		type: "UnsortedSegmentMax",
		name: (operationName ?? "Type"),
		input: [data, segmentIds, numSegments],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Dequeues a tuple of one or more tensors from the given queue.
///This operation has k outputs, where k is the number of components
/// in the tuples stored in the given queue, and output i is the ith
/// component of the dequeued tuple.
/// 
/// N.B. If the queue is empty, this operation will block until an element
/// has been dequeued (or 'timeout_ms' elapses, if specified).
/// - Parameter handle: The handle to a queue.
/// - Parameter componentTypes: The type of each component in a tuple.
/// - Parameter timeoutMs: If the queue is empty, this operation will block for up to
/// timeout_ms milliseconds.
/// Note: This option is not supported yet.
/// - Returns: 
///	components: One or more tensors that were dequeued as a tuple.
public func queueDequeueV2(operationName: String? = nil, handle: Output, componentTypes: [Any.Type], timeoutMs: UInt8) throws -> Output { 
	var attrs = [String : Any]()
	attrs["component_types"] = componentTypes
	attrs["timeout_ms"] = timeoutMs
	let opspec = OpSpec(
		type: "QueueDequeueV2",
		name: (operationName ?? "Type"),
		input: [handle],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Subtracts a value from the current value of a variable.
///Any ReadVariableOp which depends directly or indirectly on this assign is
/// guaranteed to see the incremented value or a subsequent newer one.
/// 
/// Outputs the incremented value, which can be used to totally order the
/// increments to this variable.
/// - Parameter resource: handle to the resource in which to store the variable.
/// - Parameter value: the value by which the variable will be incremented.
/// - Parameter dtype: the dtype of the value.
public func assignSubVariableOp(operationName: String? = nil, resource: Output, value: Output, dtype: Any.Type) throws -> Operation { 
	var attrs = [String : Any]()
	attrs["dtype"] = dtype
	let opspec = OpSpec(
		type: "AssignSubVariableOp",
		name: (operationName ?? "Type"),
		input: [resource, value],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Gradient for batch normalization.
///Note that the size of 4D Tensors are defined by either "NHWC" or "NCHW".
/// The size of 1D Tensors matches the dimension C of the 4D Tensors.
/// - Parameter yBackprop: A 4D Tensor for the gradient with respect to y.
/// - Parameter x: A 4D Tensor for input data.
/// - Parameter scale: A 1D Tensor for scaling factor, to scale the normalized x.
/// - Parameter reserveSpace1: When is_training is True, a 1D Tensor for the computed batch
/// mean to be reused in gradient computation. When is_training is
/// False, a 1D Tensor for the population mean to be reused in both
/// 1st and 2nd order gradient computation.
/// - Parameter reserveSpace2: When is_training is True, a 1D Tensor for the computed batch
/// variance (inverted variance in the cuDNN case) to be reused in
/// gradient computation. When is_training is False, a 1D Tensor
/// for the population variance to be reused in both 1st and 2nd
/// order gradient computation.
/// - Parameter epsilon: A small float number added to the variance of x.
/// - Parameter dataFormat: The data format for y_backprop, x, x_backprop.
/// Either "NHWC" (default) or "NCHW".
/// - Parameter isTraining: A bool value to indicate the operation is for training (default)
/// or inference.
/// - Returns: 
///	x_backprop: A 4D Tensor for the gradient with respect to x.
///	scale_backprop: A 1D Tensor for the gradient with respect to scale.
///	offset_backprop: A 1D Tensor for the gradient with respect to offset.
///	reserve_space_3: Unused placeholder to match the mean input in FusedBatchNorm.
///	reserve_space_4: Unused placeholder to match the variance input
/// in FusedBatchNorm.
public func fusedBatchNormGrad(operationName: String? = nil, yBackprop: Output, x: Output, scale: Output, reserveSpace1: Output, reserveSpace2: Output, epsilon: Float, dataFormat: String, isTraining: Bool) throws -> (xBackprop: Output, scaleBackprop: Output, offsetBackprop: Output, reserveSpace3: Output, reserveSpace4: Output) { 
	var attrs = [String : Any]()
	attrs["epsilon"] = epsilon
	attrs["data_format"] = dataFormat
	attrs["is_training"] = isTraining
	let opspec = OpSpec(
		type: "FusedBatchNormGrad",
		name: (operationName ?? "Type"),
		input: [yBackprop, x, scale, reserveSpace1, reserveSpace2],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (xBackprop: op.output(at: 0), scaleBackprop: op.output(at: 1), offsetBackprop: op.output(at: 2), reserveSpace3: op.output(at: 3), reserveSpace4: op.output(at: 4))
} 

///Convert CSV records to tensors. Each column maps to one tensor.
///RFC 4180 format is expected for the CSV records.
/// (https://tools.ietf.org/html/rfc4180)
/// Note that we allow leading and trailing spaces with int or float field.
/// - Parameter records: Each string is a record/row in the csv and all records should have
/// the same format.
/// - Parameter recordDefaults: One tensor per column of the input record, with either a
/// scalar default value for that column or empty if the column is required.
/// - Parameter outType: 
/// - Parameter fieldDelim: char delimiter to separate fields in a record.
/// - Parameter useQuoteDelim: If false, treats double quotation marks as regular
/// characters inside of the string fields (ignoring RFC 4180, Section 2,
/// Bullet 5).
/// - Parameter naValue: Additional string to recognize as NA/NaN.
/// - Returns: 
///	output: Each tensor will have the same shape as records.
public func decodeCSV(operationName: String? = nil, records: Output, recordDefaults: Output, outType: [Any.Type], fieldDelim: String, useQuoteDelim: Bool, naValue: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["OUT_TYPE"] = outType
	attrs["field_delim"] = fieldDelim
	attrs["use_quote_delim"] = useQuoteDelim
	attrs["na_value"] = naValue
	let opspec = OpSpec(
		type: "DecodeCSV",
		name: (operationName ?? "Type"),
		input: [records, recordDefaults],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Constructs a tensor by tiling a given tensor.
///This operation creates a new tensor by replicating `input` `multiples` times.
/// The output tensor's i'th dimension has `input.dims(i)  *  multiples[i]` elements,
/// and the values of `input` are replicated `multiples[i]` times along the 'i'th
/// dimension. For example, tiling `[a b c d]` by `[2]` produces
/// `[a b c d a b c d]`.
/// - Parameter input: 1-D or higher.
/// - Parameter multiples: 1-D. Length must be the same as the number of dimensions in `input`
/// - Parameter tmultiples: 
/// - Returns: 
///	output: 
public func tile(operationName: String? = nil, input: Output, multiples: Output, tmultiples: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Tmultiples"] = tmultiples
	let opspec = OpSpec(
		type: "Tile",
		name: (operationName ?? "Type"),
		input: [input, multiples],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Outputs a `Summary` protocol buffer with a tensor.
///This op is being phased out in favor of TensorSummaryV2, which lets callers pass
/// a tag as well as a serialized SummaryMetadata proto string that contains
/// plugin-specific data. We will keep this op to maintain backwards compatibility.
/// - Parameter tensor: A tensor to serialize.
/// - Parameter description: A json-encoded SummaryDescription proto.
/// - Parameter labels: An unused list of strings.
/// - Parameter displayName: An unused string.
/// - Returns: 
///	summary: 
public func tensorSummary(operationName: String? = nil, tensor: Output, description: String, labels: [String], displayName: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["description"] = description
	attrs["labels"] = labels
	attrs["display_name"] = displayName
	let opspec = OpSpec(
		type: "TensorSummary",
		name: (operationName ?? "Type"),
		input: [tensor],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Generate a single randomly distorted bounding box for an image.
///Bounding box annotations are often supplied in addition to ground-truth labels
/// in image recognition or object localization tasks. A common technique for
/// training such a system is to randomly distort an image while preserving
/// its content, i.e.  * data augmentation * . This Op outputs a randomly distorted
/// localization of an object, i.e. bounding box, given an `image_size`,
/// `bounding_boxes` and a series of constraints.
/// 
/// The output of this Op is a single bounding box that may be used to crop the
/// original image. The output is returned as 3 tensors: `begin`, `size` and
/// `bboxes`. The first 2 tensors can be fed directly into `tf.slice` to crop the
/// image. The latter may be supplied to `tf.image.draw_bounding_boxes` to visualize
/// what the bounding box looks like.
/// 
/// Bounding boxes are supplied and returned as `[y_min, x_min, y_max, x_max]`. The
/// bounding box coordinates are floats in `[0.0, 1.0]` relative to the width and
/// height of the underlying image.
/// 
/// For example,
/// 
/// ```python
///     # Generate a single distorted bounding box.
///     begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
///         tf.shape(image),
///         bounding_boxes=bounding_boxes)
/// 
///     # Draw the bounding box in an image summary.
///     image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0),
///                                                   bbox_for_draw)
///     tf.image_summary('images_with_box', image_with_box)
/// 
///     # Employ the bounding box to distort the image.
///     distorted_image = tf.slice(image, begin, size)
/// ```
/// 
/// Note that if no bounding box information is available, setting
/// `use_image_if_no_bounding_boxes = true` will assume there is a single implicit
/// bounding box covering the whole image. If `use_image_if_no_bounding_boxes` is
/// false and no bounding boxes are supplied, an error is raised.
/// - Parameter imageSize: 1-D, containing `[height, width, channels]`.
/// - Parameter boundingBoxes: 3-D with shape `[batch, N, 4]` describing the N bounding boxes
/// associated with the image.
/// - Parameter seed: If either `seed` or `seed2` are set to non-zero, the random number
/// generator is seeded by the given `seed`.  Otherwise, it is seeded by a random
/// seed.
/// - Parameter seed2: A second seed to avoid seed collision.
/// - Parameter minObjectCovered: The cropped area of the image must contain at least this
/// fraction of any bounding box supplied. The value of this parameter should be
/// non-negative. In the case of 0, the cropped area does not need to overlap
/// any of the bounding boxes supplied.
/// - Parameter aspectRatioRange: The cropped area of the image must have an aspect ratio =
/// width / height within this range.
/// - Parameter areaRange: The cropped area of the image must contain a fraction of the
/// supplied image within in this range.
/// - Parameter maxAttempts: Number of attempts at generating a cropped region of the image
/// of the specified constraints. After `max_attempts` failures, return the entire
/// image.
/// - Parameter useImageIfNoBoundingBoxes: Controls behavior if no bounding boxes supplied.
/// If true, assume an implicit bounding box covering the whole input. If false,
/// raise an error.
/// - Returns: 
///	begin: 1-D, containing `[offset_height, offset_width, 0]`. Provide as input to
/// `tf.slice`.
///	size: 1-D, containing `[target_height, target_width, -1]`. Provide as input to
/// `tf.slice`.
///	bboxes: 3-D with shape `[1, 1, 4]` containing the distorted bounding box.
/// Provide as input to `tf.image.draw_bounding_boxes`.
public func sampleDistortedBoundingBox(operationName: String? = nil, imageSize: Output, boundingBoxes: Output, seed: UInt8, seed2: UInt8, minObjectCovered: Float, aspectRatioRange: [Float], areaRange: [Float], maxAttempts: UInt8, useImageIfNoBoundingBoxes: Bool) throws -> (begin: Output, size: Output, bboxes: Output) { 
	var attrs = [String : Any]()
	attrs["seed"] = seed
	attrs["seed2"] = seed2
	attrs["min_object_covered"] = minObjectCovered
	attrs["aspect_ratio_range"] = aspectRatioRange
	attrs["area_range"] = areaRange
	attrs["max_attempts"] = maxAttempts
	attrs["use_image_if_no_bounding_boxes"] = useImageIfNoBoundingBoxes
	let opspec = OpSpec(
		type: "SampleDistortedBoundingBox",
		name: (operationName ?? "Type"),
		input: [imageSize, boundingBoxes],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (begin: op.output(at: 0), size: op.output(at: 1), bboxes: op.output(at: 2))
} 

///Decode the first frame of a BMP-encoded image to a uint8 tensor.
///The attr `channels` indicates the desired number of color channels for the
/// decoded image.
/// 
/// Accepted values are:
/// 
///  *    0: Use the number of channels in the BMP-encoded image.
///  *    3: output an RGB image.
///  *    4: output an RGBA image.
/// - Parameter contents: 0-D.  The BMP-encoded image.
/// - Parameter channels: 
/// - Returns: 
///	image: 3-D with shape `[height, width, channels]`. RGB order
public func decodeBmp(operationName: String? = nil, contents: Output, channels: UInt8) throws -> Output { 
	var attrs = [String : Any]()
	attrs["channels"] = channels
	let opspec = OpSpec(
		type: "DecodeBmp",
		name: (operationName ?? "Type"),
		input: [contents],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes softsign: `features / (abs(features) + 1)`.
/// - Parameter features: 
/// - Returns: 
///	activations: 
public func softsign(operationName: String? = nil, features: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Softsign",
		name: (operationName ?? "Type"),
		input: [features],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Produces a visualization of audio data over time.
///Spectrograms are a standard way of representing audio information as a series of
/// slices of frequency information, one slice for each window of time. By joining
/// these together into a sequence, they form a distinctive fingerprint of the sound
/// over time.
/// 
/// This op expects to receive audio data as an input, stored as floats in the range
/// -1 to 1, together with a window width in samples, and a stride specifying how
/// far to move the window between slices. From this it generates a three
/// dimensional output. The lowest dimension has an amplitude value for each
/// frequency during that time slice. The next dimension is time, with successive
/// frequency slices. The final dimension is for the channels in the input, so a
/// stereo audio input would have two here for example.
/// 
/// This means the layout when converted and saved as an image is rotated 90 degrees
/// clockwise from a typical spectrogram. Time is descending down the Y axis, and
/// the frequency decreases from left to right.
/// 
/// Each value in the result represents the square root of the sum of the real and
/// imaginary parts of an FFT on the current window of samples. In this way, the
/// lowest dimension represents the power of each frequency in the current window,
/// and adjacent windows are concatenated in the next dimension.
/// 
/// To get a more intuitive and visual look at what this operation does, you can run
/// tensorflow/examples/wav_to_spectrogram to read in an audio file and save out the
/// resulting spectrogram as a PNG image.
/// - Parameter input: Float representation of audio data.
/// - Parameter windowSize: How wide the input window is in samples. For the highest efficiency
/// this should be a power of two, but other values are accepted.
/// - Parameter stride: How widely apart the center of adjacent sample windows should be.
/// - Parameter magnitudeSquared: Whether to return the squared magnitude or just the
/// magnitude. Using squared magnitude can avoid extra calculations.
/// - Returns: 
///	spectrogram: 3D representation of the audio frequencies as an image.
public func audioSpectrogram(operationName: String? = nil, input: Output, windowSize: UInt8, stride: UInt8, magnitudeSquared: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["window_size"] = windowSize
	attrs["stride"] = stride
	attrs["magnitude_squared"] = magnitudeSquared
	let opspec = OpSpec(
		type: "AudioSpectrogram",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 


/// - Parameter handle: 
/// - Parameter index: 
/// - Parameter flowIn: 
/// - Parameter dtype: 
/// - Returns: 
///	value: 
public func tensorArrayRead(operationName: String? = nil, handle: Output, index: Output, flowIn: Output, dtype: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["dtype"] = dtype
	let opspec = OpSpec(
		type: "TensorArrayRead",
		name: (operationName ?? "Type"),
		input: [handle, index, flowIn],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Flips all bits elementwise.
///The result will have exactly those bits set, that are not set in `x`. The
/// computation is performed on the underlying representation of x.
/// - Parameter x: 
/// - Returns: 
///	y: 
public func invert(operationName: String? = nil, x: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Invert",
		name: (operationName ?? "Type"),
		input: [x],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes the gradients of 3-D convolution with respect to the filter.
/// - Parameter input: Shape `[batch, depth, rows, cols, in_channels]`.
/// - Parameter filterSizes: An integer vector representing the tensor shape of `filter`,
/// where `filter` is a 5-D
/// `[filter_depth, filter_height, filter_width, in_channels, out_channels]`
/// tensor.
/// - Parameter outBackprop: Backprop signal of shape `[batch, out_depth, out_rows, out_cols,
/// out_channels]`.
/// - Parameter strides: 1-D tensor of length 5. The stride of the sliding window for each
/// dimension of `input`. Must have `strides[0] = strides[4] = 1`.
/// - Parameter padding: The type of padding algorithm to use.
/// - Parameter dataFormat: The data format of the input and output data. With the
/// default format "NDHWC", the data is stored in the order of:
///     [batch, in_depth, in_height, in_width, in_channels].
/// Alternatively, the format could be "NCDHW", the data storage order is:
///     [batch, in_channels, in_depth, in_height, in_width].
/// - Returns: 
///	output: 
public func conv3DBackpropFilterV2(operationName: String? = nil, input: Output, filterSizes: Output, outBackprop: Output, strides: [Int64], padding: String, dataFormat: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["strides"] = strides
	attrs["padding"] = padding
	attrs["data_format"] = dataFormat
	let opspec = OpSpec(
		type: "Conv3DBackpropFilterV2",
		name: (operationName ?? "Type"),
		input: [input, filterSizes, outBackprop],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Op removes all elements in the underlying container.
/// - Parameter capacity: 
/// - Parameter memoryLimit: 
/// - Parameter dtypes: 
/// - Parameter container: 
/// - Parameter sharedName: 
public func stageClear(operationName: String? = nil, capacity: UInt8, memoryLimit: UInt8, dtypes: [Any.Type], container: String, sharedName: String) throws -> Operation { 
	var attrs = [String : Any]()
	attrs["capacity"] = capacity
	attrs["memory_limit"] = memoryLimit
	attrs["dtypes"] = dtypes
	attrs["container"] = container
	attrs["shared_name"] = sharedName
	let opspec = OpSpec(
		type: "StageClear",
		name: (operationName ?? "Type"),
		input: [],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Creates an empty hash table.
///This op creates a mutable hash table, specifying the type of its keys and
/// values. Each value must be a scalar. Data can be inserted into the table using
/// the insert operations. It does not support the initialization operation.
/// - Parameter container: If non-empty, this table is placed in the given container.
/// Otherwise, a default container is used.
/// - Parameter sharedName: If non-empty, this table is shared under the given name across
/// multiple sessions.
/// - Parameter useNodeNameSharing: If true and shared_name is empty, the table is shared
/// using the node name.
/// - Parameter keyDtype: Type of the table keys.
/// - Parameter valueDtype: Type of the table values.
/// - Returns: 
///	table_handle: Handle to a table.
public func mutableHashTable(operationName: String? = nil, container: String, sharedName: String, useNodeNameSharing: Bool, keyDtype: Any.Type, valueDtype: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["container"] = container
	attrs["shared_name"] = sharedName
	attrs["use_node_name_sharing"] = useNodeNameSharing
	attrs["key_dtype"] = keyDtype
	attrs["value_dtype"] = valueDtype
	let opspec = OpSpec(
		type: "MutableHashTable",
		name: (operationName ?? "Type"),
		input: [],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Applies a sparse gradient to a given accumulator.
///Does not add if local_step is smaller than the accumulator's
/// global_step.
/// - Parameter handle: The handle to a accumulator.
/// - Parameter localStep: The local_step value at which the sparse gradient was computed.
/// - Parameter gradientIndices: Indices of the sparse gradient to be accumulated. Must be a
/// vector.
/// - Parameter gradientValues: Values are the non-zero slices of the gradient, and must have
/// the same first dimension as indices, i.e., the nnz represented by indices and
/// values must be consistent.
/// - Parameter gradientShape: Shape of the sparse gradient to be accumulated.
/// - Parameter dtype: The data type of accumulated gradients. Needs to correspond to the type
/// of the accumulator.
/// - Parameter hasKnownShape: Boolean indicating whether gradient_shape is unknown, in which
/// case the input is ignored during validation.
public func sparseAccumulatorApplyGradient(operationName: String? = nil, handle: Output, localStep: Output, gradientIndices: Output, gradientValues: Output, gradientShape: Output, dtype: Any.Type, hasKnownShape: Bool) throws -> Operation { 
	var attrs = [String : Any]()
	attrs["dtype"] = dtype
	attrs["has_known_shape"] = hasKnownShape
	let opspec = OpSpec(
		type: "SparseAccumulatorApplyGradient",
		name: (operationName ?? "Type"),
		input: [handle, localStep, gradientIndices, gradientValues, gradientShape],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Elementwise computes the bitwise OR of `x` and `y`.
///The result will have those bits set, that are set in `x`, `y` or both. The
/// computation is performed on the underlying representations of `x` and `y`.
/// - Parameter x: 
/// - Parameter y: 
/// - Returns: 
///	z: 
public func bitwiseOr(operationName: String? = nil, x: Output, y: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "BitwiseOr",
		name: (operationName ?? "Type"),
		input: [x, y],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///The backward operation for "BiasAdd" on the "bias" tensor.
///It accumulates all the values from out_backprop into the feature dimension.
/// For NHWC data format, the feature dimension is the last. For NCHW data format,
/// the feature dimension is the third-to-last.
/// - Parameter outBackprop: Any number of dimensions.
/// - Parameter dataFormat: Specify the data format of the input and output data. With the
/// default format "NHWC", the bias tensor will be added to the last dimension
/// of the value tensor.
/// Alternatively, the format could be "NCHW", the data storage order of:
///     [batch, in_channels, in_height, in_width].
/// The tensor will be added to "in_channels", the third-to-the-last
///     dimension.
/// - Returns: 
///	output: 1-D with size the feature dimension of `out_backprop`.
public func biasAddGrad(operationName: String? = nil, outBackprop: Output, dataFormat: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["data_format"] = dataFormat
	let opspec = OpSpec(
		type: "BiasAddGrad",
		name: (operationName ?? "Type"),
		input: [outBackprop],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes tan of x element-wise.
/// - Parameter x: 
/// - Returns: 
///	y: 
public func tan(operationName: String? = nil, x: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Tan",
		name: (operationName ?? "Type"),
		input: [x],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Add a `SparseTensor` to a `SparseTensorsMap` return its handle.
///A `SparseTensor` is represented by three tensors: `sparse_indices`,
/// `sparse_values`, and `sparse_shape`.
/// 
/// This operator takes the given `SparseTensor` and adds it to a container
/// object (a `SparseTensorsMap`).  A unique key within this container is generated
/// in the form of an `int64`, and this is the value that is returned.
/// 
/// The `SparseTensor` can then be read out as part of a minibatch by passing
/// the key as a vector element to `TakeManySparseFromTensorsMap`.  To ensure
/// the correct `SparseTensorsMap` is accessed, ensure that the same
/// `container` and `shared_name` are passed to that Op.  If no `shared_name`
/// is provided here, instead use the  * name *  of the Operation created by calling
/// `AddSparseToTensorsMap` as the `shared_name` passed to
/// `TakeManySparseFromTensorsMap`.  Ensure the Operations are colocated.
/// - Parameter sparseIndices: 2-D.  The `indices` of the `SparseTensor`.
/// - Parameter sparseValues: 1-D.  The `values` of the `SparseTensor`.
/// - Parameter sparseShape: 1-D.  The `shape` of the `SparseTensor`.
/// - Parameter container: The container name for the `SparseTensorsMap` created by this op.
/// - Parameter sharedName: The shared name for the `SparseTensorsMap` created by this op.
/// If blank, the new Operation's unique name is used.
/// - Returns: 
///	sparse_handle: 0-D.  The handle of the `SparseTensor` now stored in the
/// `SparseTensorsMap`.
public func addSparseToTensorsMap(operationName: String? = nil, sparseIndices: Output, sparseValues: Output, sparseShape: Output, container: String, sharedName: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["container"] = container
	attrs["shared_name"] = sharedName
	let opspec = OpSpec(
		type: "AddSparseToTensorsMap",
		name: (operationName ?? "Type"),
		input: [sparseIndices, sparseValues, sparseShape],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes a 2-D convolution given 4-D `input` and `filter` tensors.
///Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
/// and a filter / kernel tensor of shape
/// `[filter_height, filter_width, in_channels, out_channels]`, this op
/// performs the following:
/// 
/// 1. Flattens the filter to a 2-D matrix with shape
///    `[filter_height  *  filter_width  *  in_channels, output_channels]`.
/// 2. Extracts image patches from the input tensor to form a  * virtual * 
///    tensor of shape `[batch, out_height, out_width,
///    filter_height  *  filter_width  *  in_channels]`.
/// 3. For each patch, right-multiplies the filter matrix and the image patch
///    vector.
/// 
/// In detail, with the default NHWC format,
/// 
///     output[b, i, j, k] =
///         sum_{di, dj, q} input[b, strides[1]  *  i + di, strides[2]  *  j + dj, q]  * 
///                         filter[di, dj, q, k]
/// 
/// Must have `strides[0] = strides[3] = 1`.  For the most common case of the same
/// horizontal and vertices strides, `strides = [1, stride, stride, 1]`.
/// - Parameter input: A 4-D tensor. The dimension order is interpreted according to the value
/// of `data_format`, see below for details.
/// - Parameter filter: A 4-D tensor of shape
/// `[filter_height, filter_width, in_channels, out_channels]`
/// - Parameter strides: 1-D tensor of length 4.  The stride of the sliding window for each
/// dimension of `input`. The dimension order is determined by the value of
///   `data_format`, see below for details.
/// - Parameter useCudnnOnGpu: 
/// - Parameter padding: The type of padding algorithm to use.
/// - Parameter dataFormat: Specify the data format of the input and output data. With the
/// default format "NHWC", the data is stored in the order of:
///     [batch, height, width, channels].
/// Alternatively, the format could be "NCHW", the data storage order of:
///     [batch, channels, height, width].
/// - Returns: 
///	output: A 4-D tensor. The dimension order is determined by the value of
/// `data_format`, see below for details.
public func conv2D(operationName: String? = nil, input: Output, filter: Output, strides: [Int64], useCudnnOnGpu: Bool, padding: String, dataFormat: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["strides"] = strides
	attrs["use_cudnn_on_gpu"] = useCudnnOnGpu
	attrs["padding"] = padding
	attrs["data_format"] = dataFormat
	let opspec = OpSpec(
		type: "Conv2D",
		name: (operationName ?? "Type"),
		input: [input, filter],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Creates a dataset that applies `f` to the outputs of `input_dataset`.
///The resulting dataset is similar to the `InterleaveDataset`, with the exception
/// that if retrieving the next value from a dataset would cause the requester to
/// block, it will skip that input dataset. This dataset is especially useful
/// when loading data from a variable-latency datastores (e.g. HDFS, GCS), as it
/// allows the training step to proceed so long as some data is available.
/// 
/// !! WARNING !! This dataset is not deterministic!
/// - Parameter inputDataset: 
/// - Parameter otherArguments: 
/// - Parameter cycleLength: 
/// - Parameter blockLength: 
/// - Parameter f: A function mapping elements of `input_dataset`, concatenated with
/// `other_arguments`, to a Dataset variant that contains elements matching
/// `output_types` and `output_shapes`.
/// - Parameter targuments: 
/// - Parameter outputTypes: 
/// - Parameter outputShapes: 
/// - Returns: 
///	handle: 
public func sloppyInterleaveDataset(operationName: String? = nil, inputDataset: Output, otherArguments: Output, cycleLength: Output, blockLength: Output, f: Tensorflow_NameAttrList, targuments: [Any.Type], outputTypes: [Any.Type], outputShapes: [Shape]) throws -> Output { 
	var attrs = [String : Any]()
	attrs["f"] = f
	attrs["Targuments"] = targuments
	attrs["output_types"] = outputTypes
	attrs["output_shapes"] = outputShapes
	let opspec = OpSpec(
		type: "SloppyInterleaveDataset",
		name: (operationName ?? "Type"),
		input: [inputDataset, otherArguments, cycleLength, blockLength],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Replaces the contents of the table with the specified keys and values.
///The tensor `keys` must be of the same type as the keys of the table.
/// The tensor `values` must be of the type of the table values.
/// - Parameter tableHandle: Handle to the table.
/// - Parameter keys: Any shape.  Keys to look up.
/// - Parameter values: Values to associate with keys.
/// - Parameter tin: 
/// - Parameter tout: 
public func lookupTableImportV2(operationName: String? = nil, tableHandle: Output, keys: Output, values: Output, tin: Any.Type, tout: Any.Type) throws -> Operation { 
	var attrs = [String : Any]()
	attrs["Tin"] = tin
	attrs["Tout"] = tout
	let opspec = OpSpec(
		type: "LookupTableImportV2",
		name: (operationName ?? "Type"),
		input: [tableHandle, keys, values],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Returns the shape of the variable pointed to by `resource`.
///This operation returns a 1-D integer tensor representing the shape of `input`.
/// 
/// For example:
/// 
/// ```
/// # 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
/// shape(t) ==> [2, 2, 3]
/// ```
/// - Parameter input: 
/// - Parameter outType: 
/// - Returns: 
///	output: 
public func variableShape(operationName: String? = nil, input: Output, outType: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["out_type"] = outType
	let opspec = OpSpec(
		type: "VariableShape",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Returns element-wise largest integer not greater than x.
/// - Parameter x: 
/// - Returns: 
///	y: 
public func floor(operationName: String? = nil, x: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Floor",
		name: (operationName ?? "Type"),
		input: [x],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Enqueues a tuple of one or more tensors in the given queue.
///The components input has k elements, which correspond to the components of
/// tuples stored in the given queue.
/// 
/// N.B. If the queue is full, this operation will block until the given
/// element has been enqueued (or 'timeout_ms' elapses, if specified).
/// - Parameter handle: The handle to a queue.
/// - Parameter components: One or more tensors from which the enqueued tensors should be taken.
/// - Parameter tcomponents: 
/// - Parameter timeoutMs: If the queue is full, this operation will block for up to
/// timeout_ms milliseconds.
/// Note: This option is not supported yet.
public func queueEnqueue(operationName: String? = nil, handle: Output, components: Output, tcomponents: [Any.Type], timeoutMs: UInt8) throws -> Operation { 
	var attrs = [String : Any]()
	attrs["Tcomponents"] = tcomponents
	attrs["timeout_ms"] = timeoutMs
	let opspec = OpSpec(
		type: "QueueEnqueue",
		name: (operationName ?? "Type"),
		input: [handle, components],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Get the current size of the TensorArray.
/// - Parameter handle: The handle to a TensorArray (output of TensorArray or TensorArrayGrad).
/// - Parameter flowIn: A float scalar that enforces proper chaining of operations.
/// - Returns: 
///	size: The current size of the TensorArray.
public func tensorArraySizeV3(operationName: String? = nil, handle: Output, flowIn: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "TensorArraySizeV3",
		name: (operationName ?? "Type"),
		input: [handle, flowIn],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Returns shape of tensors.
///This operation returns N 1-D integer tensors representing shape of `input[i]s`.
/// - Parameter input: 
/// - Parameter n: 
/// - Parameter outType: 
/// - Returns: 
///	output: 
public func shapeN(operationName: String? = nil, input: [Output], n: UInt8, outType: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["N"] = n
	attrs["out_type"] = outType
	let opspec = OpSpec(
		type: "ShapeN",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes the sum of elements across dimensions of a SparseTensor.
///This Op takes a SparseTensor and is the sparse counterpart to
/// `tf.reduce_sum()`.  In contrast to SparseReduceSum, this Op returns a
/// SparseTensor.
/// 
/// Reduces `sp_input` along the dimensions given in `reduction_axes`.  Unless
/// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
/// `reduction_axes`. If `keep_dims` is true, the reduced dimensions are retained
/// with length 1.
/// 
/// If `reduction_axes` has no entries, all dimensions are reduced, and a tensor
/// with a single element is returned.  Additionally, the axes can be negative,
/// which are interpreted according to the indexing rules in Python.
/// - Parameter inputIndices: 2-D.  `N x R` matrix with the indices of non-empty values in a
/// SparseTensor, possibly not in canonical ordering.
/// - Parameter inputValues: 1-D.  `N` non-empty values corresponding to `input_indices`.
/// - Parameter inputShape: 1-D.  Shape of the input SparseTensor.
/// - Parameter reductionAxes: 1-D.  Length-`K` vector containing the reduction axes.
/// - Parameter keepDims: If true, retain reduced dimensions with length 1.
/// - Returns: 
///	output_indices: 
///	output_values: 
///	output_shape: 
public func sparseReduceSumSparse(operationName: String? = nil, inputIndices: Output, inputValues: Output, inputShape: Output, reductionAxes: Output, keepDims: Bool) throws -> (outputIndices: Output, outputValues: Output, outputShape: Output) { 
	var attrs = [String : Any]()
	attrs["keep_dims"] = keepDims
	let opspec = OpSpec(
		type: "SparseReduceSumSparse",
		name: (operationName ?? "Type"),
		input: [inputIndices, inputValues, inputShape, reductionAxes],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (outputIndices: op.output(at: 0), outputValues: op.output(at: 1), outputShape: op.output(at: 2))
} 

///Enqueues zero or more tuples of one or more tensors in the given queue.
///This operation slices each component tensor along the 0th dimension to
/// make multiple queue elements. All of the tuple components must have the
/// same size in the 0th dimension.
/// 
/// The components input has k elements, which correspond to the components of
/// tuples stored in the given queue.
/// 
/// N.B. If the queue is full, this operation will block until the given
/// elements have been enqueued (or 'timeout_ms' elapses, if specified).
/// - Parameter handle: The handle to a queue.
/// - Parameter components: One or more tensors from which the enqueued tensors should
/// be taken.
/// - Parameter tcomponents: 
/// - Parameter timeoutMs: If the queue is too full, this operation will block for up
/// to timeout_ms milliseconds.
/// Note: This option is not supported yet.
public func queueEnqueueManyV2(operationName: String? = nil, handle: Output, components: Output, tcomponents: [Any.Type], timeoutMs: UInt8) throws -> Operation { 
	var attrs = [String : Any]()
	attrs["Tcomponents"] = tcomponents
	attrs["timeout_ms"] = timeoutMs
	let opspec = OpSpec(
		type: "QueueEnqueueManyV2",
		name: (operationName ?? "Type"),
		input: [handle, components],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Fast Fourier transform.
///Computes the 1-dimensional discrete Fourier transform over the inner-most
/// dimension of `input`.
/// - Parameter input: A complex64 tensor.
/// - Returns: 
///	output: A complex64 tensor of the same shape as `input`. The inner-most
///   dimension of `input` is replaced with its 1D Fourier transform.
/// 
/// @compatibility(numpy)
/// Equivalent to np.fft.fft
/// @end_compatibility
public func fft(operationName: String? = nil, input: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "FFT",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Concat the elements from the TensorArray into value `value`.
///Takes `T` elements of shapes
/// 
///   ```
///   (n0 x d0 x d1 x ...), (n1 x d0 x d1 x ...), ..., (n(T-1) x d0 x d1 x ...)
///   ```
/// 
/// and concatenates them into a Tensor of shape:
/// 
///   ```(n0 + n1 + ... + n(T-1) x d0 x d1 x ...)```
/// 
/// All elements must have the same shape (excepting the first dimension).
/// - Parameter handle: The handle to a TensorArray.
/// - Parameter flowIn: A float scalar that enforces proper chaining of operations.
/// - Parameter dtype: The type of the elem that is returned.
/// - Parameter elementShapeExcept0: The expected shape of an element, if known,
/// excluding the first dimension. Used to validate the shapes of
/// TensorArray elements. If this shape is not fully specified, concatenating
/// zero-size TensorArrays is an error.
/// - Returns: 
///	value: All of the elements in the TensorArray, concatenated along the first
/// axis.
///	lengths: A vector of the row sizes of the original T elements in the
/// value output.  In the example above, this would be the values:
/// `(n1, n2, ..., n(T-1))`.
public func tensorArrayConcatV3(operationName: String? = nil, handle: Output, flowIn: Output, dtype: Any.Type, elementShapeExcept0: Shape) throws -> (value: Output, lengths: Output) { 
	var attrs = [String : Any]()
	attrs["dtype"] = dtype
	attrs["element_shape_except0"] = elementShapeExcept0
	let opspec = OpSpec(
		type: "TensorArrayConcatV3",
		name: (operationName ?? "Type"),
		input: [handle, flowIn],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (value: op.output(at: 0), lengths: op.output(at: 1))
} 

///Update ' * var' according to the adadelta scheme.
///accum = rho()  *  accum + (1 - rho())  *  grad.square();
/// update = (update_accum + epsilon).sqrt()  *  (accum + epsilon()).rsqrt()  *  grad;
/// update_accum = rho()  *  update_accum + (1 - rho())  *  update.square();
/// var -= update;
/// - Parameter `var`: Should be from a Variable().
/// - Parameter accum: Should be from a Variable().
/// - Parameter accumUpdate: Should be from a Variable().
/// - Parameter lr: Scaling factor. Must be a scalar.
/// - Parameter rho: Decay factor. Must be a scalar.
/// - Parameter epsilon: Constant factor. Must be a scalar.
/// - Parameter grad: The gradient.
/// - Parameter useLocking: If True, updating of the var, accum and update_accum tensors will be protected by
/// a lock; otherwise the behavior is undefined, but may exhibit less contention.
public func resourceApplyAdadelta(operationName: String? = nil, `var`: Output, accum: Output, accumUpdate: Output, lr: Output, rho: Output, epsilon: Output, grad: Output, useLocking: Bool) throws -> Operation { 
	var attrs = [String : Any]()
	attrs["use_locking"] = useLocking
	let opspec = OpSpec(
		type: "ResourceApplyAdadelta",
		name: (operationName ?? "Type"),
		input: [`var`, accum, accumUpdate, lr, rho, epsilon, grad],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Push an element onto the tensor_array.
/// - Parameter handle: The handle to a TensorArray.
/// - Parameter index: The position to write to inside the TensorArray.
/// - Parameter value: The tensor to write to the TensorArray.
/// - Parameter flowIn: A float scalar that enforces proper chaining of operations.
/// - Returns: 
///	flow_out: A float scalar that enforces proper chaining of operations.
public func tensorArrayWriteV3(operationName: String? = nil, handle: Output, index: Output, value: Output, flowIn: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "TensorArrayWriteV3",
		name: (operationName ?? "Type"),
		input: [handle, index, value, flowIn],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Creates an empty hash table.
///This op creates a mutable hash table, specifying the type of its keys and
/// values. Each value must be a vector. Data can be inserted into the table using
/// the insert operations. It does not support the initialization operation.
/// - Parameter container: If non-empty, this table is placed in the given container.
/// Otherwise, a default container is used.
/// - Parameter sharedName: If non-empty, this table is shared under the given name across
/// multiple sessions.
/// - Parameter useNodeNameSharing: 
/// - Parameter keyDtype: Type of the table keys.
/// - Parameter valueDtype: Type of the table values.
/// - Parameter valueShape: 
/// - Returns: 
///	table_handle: Handle to a table.
public func mutableHashTableOfTensors(operationName: String? = nil, container: String, sharedName: String, useNodeNameSharing: Bool, keyDtype: Any.Type, valueDtype: Any.Type, valueShape: Shape) throws -> Output { 
	var attrs = [String : Any]()
	attrs["container"] = container
	attrs["shared_name"] = sharedName
	attrs["use_node_name_sharing"] = useNodeNameSharing
	attrs["key_dtype"] = keyDtype
	attrs["value_dtype"] = valueDtype
	attrs["value_shape"] = valueShape
	let opspec = OpSpec(
		type: "MutableHashTableOfTensors",
		name: (operationName ?? "Type"),
		input: [],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Creates a TensorArray for storing the gradients of values in the given handle.
///If the given TensorArray gradient already exists, returns a reference to it.
/// 
/// Locks the size of the original TensorArray by disabling its dynamic size flag.
/// 
///  *  * A note about the input flow_in: *  * 
/// 
/// The handle flow_in forces the execution of the gradient lookup to occur
/// only after certain other operations have occurred.  For example, when
/// the forward TensorArray is dynamically sized, writes to this TensorArray
/// may resize the object.  The gradient TensorArray is statically sized based
/// on the size of the forward TensorArray when this operation executes.
/// Furthermore, the size of the forward TensorArray is frozen by this call.
/// As a result, the flow is used to ensure that the call to generate the gradient
/// TensorArray only happens after all writes are executed.
/// 
/// In the case of dynamically sized TensorArrays, gradient computation should
/// only be performed on read operations that have themselves been chained via
/// flow to occur only after all writes have executed. That way the final size
/// of the forward TensorArray is known when this operation is called.
/// 
///  *  * A note about the source attribute: *  * 
/// 
/// TensorArray gradient calls use an accumulator TensorArray object.  If
/// multiple gradients are calculated and run in the same session, the multiple
/// gradient nodes may accidentally flow through the same accumulator TensorArray.
/// This double counts and generally breaks the TensorArray gradient flow.
/// 
/// The solution is to identify which gradient call this particular
/// TensorArray gradient is being called in.  This is performed by identifying
/// a unique string (e.g. "gradients", "gradients_1", ...) from the input
/// gradient Tensor's name.  This string is used as a suffix when creating
/// the TensorArray gradient object here (the attribute `source`).
/// 
/// The attribute `source` is added as a suffix to the forward TensorArray's
/// name when performing the creation / lookup, so that each separate gradient
/// calculation gets its own TensorArray accumulator.
/// - Parameter handle: The handle to the forward TensorArray.
/// - Parameter flowIn: A float scalar that enforces proper chaining of operations.
/// - Parameter source: The gradient source string, used to decide which gradient TensorArray
/// to return.
/// - Returns: 
///	grad_handle: 
///	flow_out: 
public func tensorArrayGradV3(operationName: String? = nil, handle: Output, flowIn: Output, source: String) throws -> (gradHandle: Output, flowOut: Output) { 
	var attrs = [String : Any]()
	attrs["source"] = source
	let opspec = OpSpec(
		type: "TensorArrayGradV3",
		name: (operationName ?? "Type"),
		input: [handle, flowIn],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (gradHandle: op.output(at: 0), flowOut: op.output(at: 1))
} 

///Computes the max of elements across dimensions of a SparseTensor.
///This Op takes a SparseTensor and is the sparse counterpart to
/// `tf.reduce_max()`.  In contrast to SparseReduceMax, this Op returns a
/// SparseTensor.
/// 
/// Reduces `sp_input` along the dimensions given in `reduction_axes`.  Unless
/// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
/// `reduction_axes`. If `keep_dims` is true, the reduced dimensions are retained
/// with length 1.
/// 
/// If `reduction_axes` has no entries, all dimensions are reduced, and a tensor
/// with a single element is returned.  Additionally, the axes can be negative,
/// which are interpreted according to the indexing rules in Python.
/// - Parameter inputIndices: 2-D.  `N x R` matrix with the indices of non-empty values in a
/// SparseTensor, possibly not in canonical ordering.
/// - Parameter inputValues: 1-D.  `N` non-empty values corresponding to `input_indices`.
/// - Parameter inputShape: 1-D.  Shape of the input SparseTensor.
/// - Parameter reductionAxes: 1-D.  Length-`K` vector containing the reduction axes.
/// - Parameter keepDims: If true, retain reduced dimensions with length 1.
/// - Returns: 
///	output_indices: 
///	output_values: 
///	output_shape: 
public func sparseReduceMaxSparse(operationName: String? = nil, inputIndices: Output, inputValues: Output, inputShape: Output, reductionAxes: Output, keepDims: Bool) throws -> (outputIndices: Output, outputValues: Output, outputShape: Output) { 
	var attrs = [String : Any]()
	attrs["keep_dims"] = keepDims
	let opspec = OpSpec(
		type: "SparseReduceMaxSparse",
		name: (operationName ?? "Type"),
		input: [inputIndices, inputValues, inputShape, reductionAxes],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (outputIndices: op.output(at: 0), outputValues: op.output(at: 1), outputShape: op.output(at: 2))
} 

///Forwards the ref tensor `data` to the output port determined by `pred`.
///If `pred` is true, the `data` input is forwarded to `output_true`. Otherwise,
/// the data goes to `output_false`.
/// 
/// See also `Switch` and `Merge`.
/// - Parameter data: The ref tensor to be forwarded to the appropriate output.
/// - Parameter pred: A scalar that specifies which output port will receive data.
/// - Returns: 
///	output_false: If `pred` is false, data will be forwarded to this output.
///	output_true: If `pred` is true, data will be forwarded to this output.
public func refSwitch(operationName: String? = nil, data: Output, pred: Output) throws -> (outputFalse: Output, outputTrue: Output) { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "RefSwitch",
		name: (operationName ?? "Type"),
		input: [data, pred],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (outputFalse: op.output(at: 0), outputTrue: op.output(at: 1))
} 

///Returns x // y element-wise.
/// * NOTE * : `FloorDiv` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
/// - Parameter x: 
/// - Parameter y: 
/// - Returns: 
///	z: 
public func floorDiv(operationName: String? = nil, x: Output, y: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "FloorDiv",
		name: (operationName ?? "Type"),
		input: [x, y],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Update ' * var' according to the proximal adagrad scheme.
/// - Parameter `var`: Should be from a Variable().
/// - Parameter gradientAccumulator: Should be from a Variable().
/// - Parameter gradientSquaredAccumulator: Should be from a Variable().
/// - Parameter grad: The gradient.
/// - Parameter lr: Scaling factor. Must be a scalar.
/// - Parameter l1: L1 regularization. Must be a scalar.
/// - Parameter l2: L2 regularization. Must be a scalar.
/// - Parameter globalStep: Training step number. Must be a scalar.
/// - Parameter useLocking: If True, updating of the var and accum tensors will be protected by
/// a lock; otherwise the behavior is undefined, but may exhibit less contention.
/// - Returns: 
///	out: Same as "var".
public func applyAdagradDA(operationName: String? = nil, `var`: Output, gradientAccumulator: Output, gradientSquaredAccumulator: Output, grad: Output, lr: Output, l1: Output, l2: Output, globalStep: Output, useLocking: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["use_locking"] = useLocking
	let opspec = OpSpec(
		type: "ApplyAdagradDA",
		name: (operationName ?? "Type"),
		input: [`var`, gradientAccumulator, gradientSquaredAccumulator, grad, lr, l1, l2, globalStep],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///An array of Tensors of given size.
///Write data via Write and read via Read or Pack.
/// - Parameter size: The size of the array.
/// - Parameter dtype: The type of the elements on the tensor_array.
/// - Parameter elementShape: The expected shape of an element, if known. Used to
/// validate the shapes of TensorArray elements. If this shape is not
/// fully specified, gathering zero-size TensorArrays is an error.
/// - Parameter dynamicSize: A boolean that determines whether writes to the TensorArray
/// are allowed to grow the size.  By default, this is not allowed.
/// - Parameter clearAfterRead: If true (default), Tensors in the TensorArray are cleared
/// after being read.  This disables multiple read semantics but allows early
/// release of memory.
/// - Parameter tensorArrayName: Overrides the name used for the temporary tensor_array
/// resource. Default value is the name of the 'TensorArray' op (which
/// is guaranteed unique).
/// - Returns: 
///	handle: The handle to the TensorArray.
///	flow: A scalar used to control gradient flow.
public func tensorArrayV3(operationName: String? = nil, size: Output, dtype: Any.Type, elementShape: Shape, dynamicSize: Bool, clearAfterRead: Bool, tensorArrayName: String) throws -> (handle: Output, flow: Output) { 
	var attrs = [String : Any]()
	attrs["dtype"] = dtype
	attrs["element_shape"] = elementShape
	attrs["dynamic_size"] = dynamicSize
	attrs["clear_after_read"] = clearAfterRead
	attrs["tensor_array_name"] = tensorArrayName
	let opspec = OpSpec(
		type: "TensorArrayV3",
		name: (operationName ?? "Type"),
		input: [size],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (handle: op.output(at: 0), flow: op.output(at: 1))
} 

///A queue that produces elements in first-in first-out order.
///Variable-size shapes are allowed by setting the corresponding shape dimensions
/// to 0 in the shape attr.  In this case DequeueMany will pad up to the maximum
/// size of any given element in the minibatch.  See below for details.
/// - Parameter componentTypes: The type of each component in a value.
/// - Parameter shapes: The shape of each component in a value. The length of this attr must
/// be either 0 or the same as the length of component_types.
/// Shapes of fixed rank but variable size are allowed by setting
/// any shape dimension to -1.  In this case, the inputs' shape may vary along
/// the given dimension, and DequeueMany will pad the given dimension with
/// zeros up to the maximum shape of all elements in the given batch.
/// If the length of this attr is 0, different queue elements may have
/// different ranks and shapes, but only one element may be dequeued at a time.
/// - Parameter capacity: The upper bound on the number of elements in this queue.
/// Negative numbers mean no limit.
/// - Parameter container: If non-empty, this queue is placed in the given container.
/// Otherwise, a default container is used.
/// - Parameter sharedName: If non-empty, this queue will be shared under the given name
/// across multiple sessions.
/// - Returns: 
///	handle: The handle to the queue.
public func paddingFIFOQueue(operationName: String? = nil, componentTypes: [Any.Type], shapes: [Shape], capacity: UInt8, container: String, sharedName: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["component_types"] = componentTypes
	attrs["shapes"] = shapes
	attrs["capacity"] = capacity
	attrs["container"] = container
	attrs["shared_name"] = sharedName
	let opspec = OpSpec(
		type: "PaddingFIFOQueue",
		name: (operationName ?? "Type"),
		input: [],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Outputs random values from the Poisson distribution(s) described by rate.
///This op uses two algorithms, depending on rate. If rate >= 10, then
/// the algorithm by Hormann is used to acquire samples via
/// transformation-rejection.
/// See http://www.sciencedirect.com/science/article/pii/0167668793909974.
/// 
/// Otherwise, Knuth's algorithm is used to acquire samples via multiplying uniform
/// random variables.
/// See Donald E. Knuth (1969). Seminumerical Algorithms. The Art of Computer
/// Programming, Volume 2. Addison Wesley
/// - Parameter shape: 1-D integer tensor. Shape of independent samples to draw from each
/// distribution described by the shape parameters given in rate.
/// - Parameter rate: A tensor in which each scalar is a "rate" parameter describing the
/// associated poisson distribution.
/// - Parameter seed: If either `seed` or `seed2` are set to be non-zero, the random number
/// generator is seeded by the given seed.  Otherwise, it is seeded by a
/// random seed.
/// - Parameter seed2: A second seed to avoid seed collision.
/// - Parameter s: 
/// - Parameter dtype: 
/// - Returns: 
///	output: A tensor with shape `shape + shape(rate)`. Each slice
/// `[:, ..., :, i0, i1, ...iN]` contains the samples drawn for
/// `rate[i0, i1, ...iN]`. The dtype of the output matches the dtype of
/// rate.
public func randomPoisson(operationName: String? = nil, shape: Output, rate: Output, seed: UInt8, seed2: UInt8, s: Any.Type, dtype: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["seed"] = seed
	attrs["seed2"] = seed2
	attrs["S"] = s
	attrs["dtype"] = dtype
	let opspec = OpSpec(
		type: "RandomPoisson",
		name: (operationName ?? "Type"),
		input: [shape, rate],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Add an `N`-minibatch `SparseTensor` to a `SparseTensorsMap`, return `N` handles.
///A `SparseTensor` of rank `R` is represented by three tensors: `sparse_indices`,
/// `sparse_values`, and `sparse_shape`, where
/// 
/// ```sparse_indices.shape[1] == sparse_shape.shape[0] == R```
/// 
/// An `N`-minibatch of `SparseTensor` objects is represented as a `SparseTensor`
/// having a first `sparse_indices` column taking values between `[0, N)`, where
/// the minibatch size `N == sparse_shape[0]`.
/// 
/// The input `SparseTensor` must have rank `R` greater than 1, and the first
/// dimension is treated as the minibatch dimension.  Elements of the `SparseTensor`
/// must be sorted in increasing order of this first dimension.  The stored
/// `SparseTensor` objects pointed to by each row of the output `sparse_handles`
/// will have rank `R-1`.
/// 
/// The `SparseTensor` values can then be read out as part of a minibatch by passing
/// the given keys as vector elements to `TakeManySparseFromTensorsMap`.  To ensure
/// the correct `SparseTensorsMap` is accessed, ensure that the same
/// `container` and `shared_name` are passed to that Op.  If no `shared_name`
/// is provided here, instead use the  * name *  of the Operation created by calling
/// `AddManySparseToTensorsMap` as the `shared_name` passed to
/// `TakeManySparseFromTensorsMap`.  Ensure the Operations are colocated.
/// - Parameter sparseIndices: 2-D.  The `indices` of the minibatch `SparseTensor`.
/// `sparse_indices[:, 0]` must be ordered values in `[0, N)`.
/// - Parameter sparseValues: 1-D.  The `values` of the minibatch `SparseTensor`.
/// - Parameter sparseShape: 1-D.  The `shape` of the minibatch `SparseTensor`.
/// The minibatch size `N == sparse_shape[0]`.
/// - Parameter container: The container name for the `SparseTensorsMap` created by this op.
/// - Parameter sharedName: The shared name for the `SparseTensorsMap` created by this op.
/// If blank, the new Operation's unique name is used.
/// - Returns: 
///	sparse_handles: 1-D.  The handles of the `SparseTensor` now stored in the
/// `SparseTensorsMap`.  Shape: `[N]`.
public func addManySparseToTensorsMap(operationName: String? = nil, sparseIndices: Output, sparseValues: Output, sparseShape: Output, container: String, sharedName: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["container"] = container
	attrs["shared_name"] = sharedName
	let opspec = OpSpec(
		type: "AddManySparseToTensorsMap",
		name: (operationName ?? "Type"),
		input: [sparseIndices, sparseValues, sparseShape],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes square of x element-wise.
///I.e., \\(y = x  *  x = x// ^2\\).
/// - Parameter x: 
/// - Returns: 
///	y: 
public func square(operationName: String? = nil, x: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Square",
		name: (operationName ?? "Type"),
		input: [x],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///A Reader that outputs the queued work as both the key and value.
///To use, enqueue strings in a Queue.  ReaderRead will take the front
/// work string and output (work, work).
/// - Parameter container: If non-empty, this reader is placed in the given container.
/// Otherwise, a default container is used.
/// - Parameter sharedName: If non-empty, this reader is named in the given bucket
/// with this shared_name. Otherwise, the node name is used instead.
/// - Returns: 
///	reader_handle: The handle to reference the Reader.
public func identityReader(operationName: String? = nil, container: String, sharedName: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["container"] = container
	attrs["shared_name"] = sharedName
	let opspec = OpSpec(
		type: "IdentityReader",
		name: (operationName ?? "Type"),
		input: [],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Quantizes then dequantizes a tensor.
///This is almost identical to QuantizeAndDequantizeV2, except that num_bits is a
/// tensor, so its value can change during training.
/// - Parameter input: 
/// - Parameter inputMin: 
/// - Parameter inputMax: 
/// - Parameter numBits: 
/// - Parameter signedInput: 
/// - Parameter rangeGiven: 
/// - Returns: 
///	output: 
public func quantizeAndDequantizeV3(operationName: String? = nil, input: Output, inputMin: Output, inputMax: Output, numBits: Output, signedInput: Bool, rangeGiven: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["signed_input"] = signedInput
	attrs["range_given"] = rangeGiven
	let opspec = OpSpec(
		type: "QuantizeAndDequantizeV3",
		name: (operationName ?? "Type"),
		input: [input, inputMin, inputMax, numBits],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Deprecated, use StackPopV2.
/// - Parameter handle: 
/// - Parameter elemType: 
/// - Returns: 
///	elem: 
public func stackPop(operationName: String? = nil, handle: Output, elemType: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["elem_type"] = elemType
	let opspec = OpSpec(
		type: "StackPop",
		name: (operationName ?? "Type"),
		input: [handle],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Scatter the data from the input value into specific TensorArray elements.
///`indices` must be a vector, its length must match the first dim of `value`.
/// - Parameter handle: The handle to a TensorArray.
/// - Parameter indices: The locations at which to write the tensor elements.
/// - Parameter value: The concatenated tensor to write to the TensorArray.
/// - Parameter flowIn: A float scalar that enforces proper chaining of operations.
/// - Returns: 
///	flow_out: A float scalar that enforces proper chaining of operations.
public func tensorArrayScatterV3(operationName: String? = nil, handle: Output, indices: Output, value: Output, flowIn: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "TensorArrayScatterV3",
		name: (operationName ?? "Type"),
		input: [handle, indices, value, flowIn],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes the absolute value of a tensor.
///Given a tensor `x`, this operation returns a tensor containing the absolute
/// value of each element in `x`. For example, if x is an input element and y is
/// an output element, this operation computes \\(y = |x|\\).
/// - Parameter x: 
/// - Returns: 
///	y: 
public func abs(operationName: String? = nil, x: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Abs",
		name: (operationName ?? "Type"),
		input: [x],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Read an element from the TensorArray into output `value`.
/// - Parameter handle: The handle to a TensorArray.
/// - Parameter index: 
/// - Parameter flowIn: A float scalar that enforces proper chaining of operations.
/// - Parameter dtype: The type of the elem that is returned.
/// - Returns: 
///	value: The tensor that is read from the TensorArray.
public func tensorArrayReadV3(operationName: String? = nil, handle: Output, index: Output, flowIn: Output, dtype: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["dtype"] = dtype
	let opspec = OpSpec(
		type: "TensorArrayReadV3",
		name: (operationName ?? "Type"),
		input: [handle, index, flowIn],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Adds `bias` to `value`.
///This is a deprecated version of BiasAdd and will be soon removed.
/// 
/// This is a special case of `tf.add` where `bias` is restricted to be 1-D.
/// Broadcasting is supported, so `value` may have any number of dimensions.
/// - Parameter value: Any number of dimensions.
/// - Parameter bias: 1-D with size the last dimension of `value`.
/// - Returns: 
///	output: Broadcasted sum of `value` and `bias`.
public func biasAddV1(operationName: String? = nil, value: Output, bias: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "BiasAddV1",
		name: (operationName ?? "Type"),
		input: [value, bias],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Returns the truth value of x OR y element-wise.
/// * NOTE * : `LogicalOr` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
/// - Parameter x: 
/// - Parameter y: 
/// - Returns: 
///	z: 
public func logicalOr(operationName: String? = nil, x: Output, y: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "LogicalOr",
		name: (operationName ?? "Type"),
		input: [x, y],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Deprecated, use StackPushV2.
/// - Parameter handle: 
/// - Parameter elem: 
/// - Parameter swapMemory: 
/// - Returns: 
///	output: 
public func stackPush(operationName: String? = nil, handle: Output, elem: Output, swapMemory: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["swap_memory"] = swapMemory
	let opspec = OpSpec(
		type: "StackPush",
		name: (operationName ?? "Type"),
		input: [handle, elem],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///A Reader that outputs the records from a TensorFlow Records file.
/// - Parameter container: If non-empty, this reader is placed in the given container.
/// Otherwise, a default container is used.
/// - Parameter sharedName: If non-empty, this reader is named in the given bucket
/// with this shared_name. Otherwise, the node name is used instead.
/// - Parameter compressionType: 
/// - Returns: 
///	reader_handle: The handle to reference the Reader.
public func tFRecordReaderV2(operationName: String? = nil, container: String, sharedName: String, compressionType: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["container"] = container
	attrs["shared_name"] = sharedName
	attrs["compression_type"] = compressionType
	let opspec = OpSpec(
		type: "TFRecordReaderV2",
		name: (operationName ?? "Type"),
		input: [],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 


/// - Parameter handle: 
/// - Parameter flowIn: 
/// - Parameter dtype: 
/// - Parameter elementShapeExcept0: 
/// - Returns: 
///	value: 
///	lengths: 
public func tensorArrayConcat(operationName: String? = nil, handle: Output, flowIn: Output, dtype: Any.Type, elementShapeExcept0: Shape) throws -> (value: Output, lengths: Output) { 
	var attrs = [String : Any]()
	attrs["dtype"] = dtype
	attrs["element_shape_except0"] = elementShapeExcept0
	let opspec = OpSpec(
		type: "TensorArrayConcat",
		name: (operationName ?? "Type"),
		input: [handle, flowIn],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (value: op.output(at: 0), lengths: op.output(at: 1))
} 

///Generates labels for candidate sampling with a log-uniform distribution.
///See explanations of candidate sampling and the data formats at
/// go/candidate-sampling.
/// 
/// For each batch, this op picks a single set of sampled candidate labels.
/// 
/// The advantages of sampling candidates per-batch are simplicity and the
/// possibility of efficient dense matrix multiplication. The disadvantage is that
/// the sampled candidates must be chosen independently of the context and of the
/// true labels.
/// - Parameter trueClasses: A batch_size  *  num_true matrix, in which each row contains the
/// IDs of the num_true target_classes in the corresponding original label.
/// - Parameter numTrue: Number of true labels per context.
/// - Parameter numSampled: Number of candidates to randomly sample.
/// - Parameter unique: If unique is true, we sample with rejection, so that all sampled
/// candidates in a batch are unique. This requires some approximation to
/// estimate the post-rejection sampling probabilities.
/// - Parameter rangeMax: The sampler will sample integers from the interval [0, range_max).
/// - Parameter seed: If either seed or seed2 are set to be non-zero, the random number
/// generator is seeded by the given seed.  Otherwise, it is seeded by a
/// random seed.
/// - Parameter seed2: An second seed to avoid seed collision.
/// - Returns: 
///	sampled_candidates: A vector of length num_sampled, in which each element is
/// the ID of a sampled candidate.
///	true_expected_count: A batch_size  *  num_true matrix, representing
/// the number of times each candidate is expected to occur in a batch
/// of sampled candidates. If unique=true, then this is a probability.
///	sampled_expected_count: A vector of length num_sampled, for each sampled
/// candidate representing the number of times the candidate is expected
/// to occur in a batch of sampled candidates.  If unique=true, then this is a
/// probability.
public func logUniformCandidateSampler(operationName: String? = nil, trueClasses: Output, numTrue: UInt8, numSampled: UInt8, unique: Bool, rangeMax: UInt8, seed: UInt8, seed2: UInt8) throws -> (sampledCandidates: Output, trueExpectedCount: Output, sampledExpectedCount: Output) { 
	var attrs = [String : Any]()
	attrs["num_true"] = numTrue
	attrs["num_sampled"] = numSampled
	attrs["unique"] = unique
	attrs["range_max"] = rangeMax
	attrs["seed"] = seed
	attrs["seed2"] = seed2
	let opspec = OpSpec(
		type: "LogUniformCandidateSampler",
		name: (operationName ?? "Type"),
		input: [trueClasses],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (sampledCandidates: op.output(at: 0), trueExpectedCount: op.output(at: 1), sampledExpectedCount: op.output(at: 2))
} 

///Computes gradients for SparseSegmentMean.
///Returns tensor "output" with same shape as grad, except for dimension 0 whose
/// value is output_dim0.
/// - Parameter grad: gradient propagated to the SparseSegmentMean op.
/// - Parameter indices: indices passed to the corresponding SparseSegmentMean op.
/// - Parameter segmentIds: segment_ids passed to the corresponding SparseSegmentMean op.
/// - Parameter outputDim0: dimension 0 of "data" passed to SparseSegmentMean op.
/// - Parameter tidx: 
/// - Returns: 
///	output: 
public func sparseSegmentMeanGrad(operationName: String? = nil, grad: Output, indices: Output, segmentIds: Output, outputDim0: Output, tidx: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Tidx"] = tidx
	let opspec = OpSpec(
		type: "SparseSegmentMeanGrad",
		name: (operationName ?? "Type"),
		input: [grad, indices, segmentIds, outputDim0],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Gather slices from `params` into a Tensor with shape specified by `indices`.
///`indices` is an K-dimensional integer tensor, best thought of as a
/// (K-1)-dimensional tensor of indices into `params`, where each element defines a
/// slice of `params`:
/// 
///     output[i_0, ..., i_{K-2}] = params[indices[i0, ..., i_{K-2}]]
/// 
/// Whereas in @{tf.gather} `indices` defines slices into the first
/// dimension of `params`, in `tf.gather_nd`, `indices` defines slices into the
/// first `N` dimensions of `params`, where `N = indices.shape[-1]`.
/// 
/// The last dimension of `indices` can be at most the rank of
/// `params`:
/// 
///     indices.shape[-1] <= params.rank
/// 
/// The last dimension of `indices` corresponds to elements
/// (if `indices.shape[-1] == params.rank`) or slices
/// (if `indices.shape[-1] < params.rank`) along dimension `indices.shape[-1]`
/// of `params`.  The output tensor has shape
/// 
///     indices.shape[:-1] + params.shape[indices.shape[-1]:]
/// 
/// Some examples below.
/// 
/// Simple indexing into a matrix:
/// 
/// ```python
///     indices = [[0, 0], [1, 1]]
///     params = [['a', 'b'], ['c', 'd']]
///     output = ['a', 'd']
/// ```
/// 
/// Slice indexing into a matrix:
/// 
/// ```python
///     indices = [[1], [0]]
///     params = [['a', 'b'], ['c', 'd']]
///     output = [['c', 'd'], ['a', 'b']]
/// ```
/// 
/// Indexing into a 3-tensor:
/// 
/// ```python
///     indices = [[1]]
///     params = [[['a0', 'b0'], ['c0', 'd0']],
///               [['a1', 'b1'], ['c1', 'd1']]]
///     output = [[['a1', 'b1'], ['c1', 'd1']]]
/// 
/// 
///     indices = [[0, 1], [1, 0]]
///     params = [[['a0', 'b0'], ['c0', 'd0']],
///               [['a1', 'b1'], ['c1', 'd1']]]
///     output = [['c0', 'd0'], ['a1', 'b1']]
/// 
/// 
///     indices = [[0, 0, 1], [1, 0, 1]]
///     params = [[['a0', 'b0'], ['c0', 'd0']],
///               [['a1', 'b1'], ['c1', 'd1']]]
///     output = ['b0', 'b1']
/// ```
/// 
/// Batched indexing into a matrix:
/// 
/// ```python
///     indices = [[[0, 0]], [[0, 1]]]
///     params = [['a', 'b'], ['c', 'd']]
///     output = [['a'], ['b']]
/// ```
/// 
/// Batched slice indexing into a matrix:
/// 
/// ```python
///     indices = [[[1]], [[0]]]
///     params = [['a', 'b'], ['c', 'd']]
///     output = [[['c', 'd']], [['a', 'b']]]
/// ```
/// 
/// Batched indexing into a 3-tensor:
/// 
/// ```python
///     indices = [[[1]], [[0]]]
///     params = [[['a0', 'b0'], ['c0', 'd0']],
///               [['a1', 'b1'], ['c1', 'd1']]]
///     output = [[[['a1', 'b1'], ['c1', 'd1']]],
///               [[['a0', 'b0'], ['c0', 'd0']]]]
/// 
///     indices = [[[0, 1], [1, 0]], [[0, 0], [1, 1]]]
///     params = [[['a0', 'b0'], ['c0', 'd0']],
///               [['a1', 'b1'], ['c1', 'd1']]]
///     output = [[['c0', 'd0'], ['a1', 'b1']],
///               [['a0', 'b0'], ['c1', 'd1']]]
/// 
/// 
///     indices = [[[0, 0, 1], [1, 0, 1]], [[0, 1, 1], [1, 1, 0]]]
///     params = [[['a0', 'b0'], ['c0', 'd0']],
///               [['a1', 'b1'], ['c1', 'd1']]]
///     output = [['b0', 'b1'], ['d0', 'c1']]
/// ```
/// - Parameter params: The tensor from which to gather values.
/// - Parameter indices: Index tensor.
/// - Parameter tparams: 
/// - Parameter tindices: 
/// - Returns: 
///	output: Values from `params` gathered from indices given by `indices`, with
/// shape `indices.shape[:-1] + params.shape[indices.shape[-1]:]`.
public func gatherNd(operationName: String? = nil, params: Output, indices: Output, tparams: Any.Type, tindices: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Tparams"] = tparams
	attrs["Tindices"] = tindices
	let opspec = OpSpec(
		type: "GatherNd",
		name: (operationName ?? "Type"),
		input: [params, indices],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Op removes all elements in the underlying container.
/// - Parameter capacity: 
/// - Parameter memoryLimit: 
/// - Parameter dtypes: 
/// - Parameter container: 
/// - Parameter sharedName: 
public func orderedMapClear(operationName: String? = nil, capacity: UInt8, memoryLimit: UInt8, dtypes: [Any.Type], container: String, sharedName: String) throws -> Operation { 
	var attrs = [String : Any]()
	attrs["capacity"] = capacity
	attrs["memory_limit"] = memoryLimit
	attrs["dtypes"] = dtypes
	attrs["container"] = container
	attrs["shared_name"] = sharedName
	let opspec = OpSpec(
		type: "OrderedMapClear",
		name: (operationName ?? "Type"),
		input: [],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Closes the given queue.
///This operation signals that no more elements will be enqueued in the
/// given queue. Subsequent Enqueue(Many) operations will fail.
/// Subsequent Dequeue(Many) operations will continue to succeed if
/// sufficient elements remain in the queue. Subsequent Dequeue(Many)
/// operations that would block will fail immediately.
/// - Parameter handle: The handle to a queue.
/// - Parameter cancelPendingEnqueues: If true, all pending enqueue requests that are
/// blocked on the given queue will be canceled.
public func queueCloseV2(operationName: String? = nil, handle: Output, cancelPendingEnqueues: Bool) throws -> Operation { 
	var attrs = [String : Any]()
	attrs["cancel_pending_enqueues"] = cancelPendingEnqueues
	let opspec = OpSpec(
		type: "QueueCloseV2",
		name: (operationName ?? "Type"),
		input: [handle],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Looks up keys in a table, outputs the corresponding values.
///The tensor `keys` must of the same type as the keys of the table.
/// The output `values` is of the type of the table values.
/// 
/// The scalar `default_value` is the value output for keys not present in the
/// table. It must also be of the same type as the table values.
/// - Parameter tableHandle: Handle to the table.
/// - Parameter keys: Any shape.  Keys to look up.
/// - Parameter defaultValue: 
/// - Parameter tin: 
/// - Parameter tout: 
/// - Returns: 
///	values: Same shape as `keys`.  Values found in the table, or `default_values`
/// for missing keys.
public func lookupTableFind(operationName: String? = nil, tableHandle: Output, keys: Output, defaultValue: Output, tin: Any.Type, tout: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Tin"] = tin
	attrs["Tout"] = tout
	let opspec = OpSpec(
		type: "LookupTableFind",
		name: (operationName ?? "Type"),
		input: [tableHandle, keys, defaultValue],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes rectified linear: `max(features, 0)`.
/// - Parameter features: 
/// - Returns: 
///	activations: 
public func relu(operationName: String? = nil, features: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Relu",
		name: (operationName ?? "Type"),
		input: [features],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Interleave the values from the `data` tensors into a single tensor.
///Builds a merged tensor such that
/// 
/// ```python
///     merged[indices[m][i, ..., j], ...] = data[m][i, ..., j, ...]
/// ```
/// 
/// For example, if each `indices[m]` is scalar or vector, we have
/// 
/// ```python
///     # Scalar indices:
///     merged[indices[m], ...] = data[m][...]
/// 
///     # Vector indices:
///     merged[indices[m][i], ...] = data[m][i, ...]
/// ```
/// 
/// Each `data[i].shape` must start with the corresponding `indices[i].shape`,
/// and the rest of `data[i].shape` must be constant w.r.t. `i`.  That is, we
/// must have `data[i].shape = indices[i].shape + constant`.  In terms of this
/// `constant`, the output shape is
/// 
///     merged.shape = [max(indices)] + constant
/// 
/// Values are merged in order, so if an index appears in both `indices[m][i]` and
/// `indices[n][j]` for `(m,i) < (n,j)` the slice `data[n][j]` will appear in the
/// merged result. If you do not need this guarantee, ParallelDynamicStitch might
/// perform better on some devices.
/// 
/// For example:
/// 
/// ```python
///     indices[0] = 6
///     indices[1] = [4, 1]
///     indices[2] = [[5, 2], [0, 3]]
///     data[0] = [61, 62]
///     data[1] = [[41, 42], [11, 12]]
///     data[2] = [[[51, 52], [21, 22]], [[1, 2], [31, 32]]]
///     merged = [[1, 2], [11, 12], [21, 22], [31, 32], [41, 42],
///               [51, 52], [61, 62]]
/// ```
/// 
/// This method can be used to merge partitions created by `dynamic_partition`
/// as illustrated on the following example:
/// 
/// ```python
///     # Apply function (increments x_i) on elements for which a certain condition
///     # apply (x_i != -1 in this example).
///     x=tf.constant([0.1, -1., 5.2, 4.3, -1., 7.4])
///     condition_mask=tf.not_equal(x,tf.constant(-1.))
///     partitioned_data = tf.dynamic_partition(
///         x, tf.cast(condition_mask, tf.int32) , 2)
///     partitioned_data[1] = partitioned_data[1] + 1.0
///     condition_indices = tf.dynamic_partition(
///         tf.range(tf.shape(x)[0]), tf.cast(condition_mask, tf.int32) , 2)
///     x = tf.dynamic_stitch(condition_indices, partitioned_data)
///     # Here x=[1.1, -1., 6.2, 5.3, -1, 8.4], the -1. values remain
///     # unchanged.
/// ```
/// 
/// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
/// <img style="width:100%" src="https://www.tensorflow.org/images/DynamicStitch.png" alt>
/// </div>
/// - Parameter indices: 
/// - Parameter data: 
/// - Parameter n: 
/// - Returns: 
///	merged: 
public func dynamicStitch(operationName: String? = nil, indices: [Output], data: [Output], n: UInt8) throws -> Output { 
	var attrs = [String : Any]()
	attrs["N"] = n
	let opspec = OpSpec(
		type: "DynamicStitch",
		name: (operationName ?? "Type"),
		input: [indices, data],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///var: Should be from a Variable().
/// - Parameter `var`: 
/// - Parameter accum: Should be from a Variable().
/// - Parameter accumUpdate: : Should be from a Variable().
/// - Parameter lr: Learning rate. Must be a scalar.
/// - Parameter rho: Decay factor. Must be a scalar.
/// - Parameter epsilon: Constant factor. Must be a scalar.
/// - Parameter grad: The gradient.
/// - Parameter indices: A vector of indices into the first dimension of var and accum.
/// - Parameter tindices: 
/// - Parameter useLocking: If True, updating of the var and accum tensors will be protected by
/// a lock; otherwise the behavior is undefined, but may exhibit less contention.
/// - Returns: 
///	out: Same as "var".
public func sparseApplyAdadelta(operationName: String? = nil, `var`: Output, accum: Output, accumUpdate: Output, lr: Output, rho: Output, epsilon: Output, grad: Output, indices: Output, tindices: Any.Type, useLocking: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Tindices"] = tindices
	attrs["use_locking"] = useLocking
	let opspec = OpSpec(
		type: "SparseApplyAdadelta",
		name: (operationName ?? "Type"),
		input: [`var`, accum, accumUpdate, lr, rho, epsilon, grad, indices],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Reshapes a SparseTensor to represent values in a new dense shape.
///This operation has the same semantics as reshape on the represented dense
/// tensor.  The `input_indices` are recomputed based on the requested `new_shape`.
/// 
/// If one component of `new_shape` is the special value -1, the size of that
/// dimension is computed so that the total dense size remains constant.  At
/// most one component of `new_shape` can be -1.  The number of dense elements
/// implied by `new_shape` must be the same as the number of dense elements
/// originally implied by `input_shape`.
/// 
/// Reshaping does not affect the order of values in the SparseTensor.
/// 
/// If the input tensor has rank `R_in` and `N` non-empty values, and `new_shape`
/// has length `R_out`, then `input_indices` has shape `[N, R_in]`,
/// `input_shape` has length `R_in`, `output_indices` has shape `[N, R_out]`, and
/// `output_shape` has length `R_out`.
/// - Parameter inputIndices: 2-D.  `N x R_in` matrix with the indices of non-empty values in a
/// SparseTensor.
/// - Parameter inputShape: 1-D.  `R_in` vector with the input SparseTensor's dense shape.
/// - Parameter newShape: 1-D.  `R_out` vector with the requested new dense shape.
/// - Returns: 
///	output_indices: 2-D.  `N x R_out` matrix with the updated indices of non-empty
/// values in the output SparseTensor.
///	output_shape: 1-D.  `R_out` vector with the full dense shape of the output
/// SparseTensor.  This is the same as `new_shape` but with any -1 dimensions
/// filled in.
public func sparseReshape(operationName: String? = nil, inputIndices: Output, inputShape: Output, newShape: Output) throws -> (outputIndices: Output, outputShape: Output) { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "SparseReshape",
		name: (operationName ?? "Type"),
		input: [inputIndices, inputShape, newShape],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (outputIndices: op.output(at: 0), outputShape: op.output(at: 1))
} 

///Computes the complex absolute value of a tensor.
///Given a tensor `x` of complex numbers, this operation returns a tensor of type
/// `float` or `double` that is the absolute value of each element in `x`. All
/// elements in `x` must be complex numbers of the form \\(a + bj\\). The absolute
/// value is computed as \\( \sqrt{a// ^2 + b// ^2}\\).
/// - Parameter x: 
/// - Parameter tout: 
/// - Returns: 
///	y: 
public func complexAbs(operationName: String? = nil, x: Output, tout: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Tout"] = tout
	let opspec = OpSpec(
		type: "ComplexAbs",
		name: (operationName ?? "Type"),
		input: [x],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Deprecated. Use TensorArrayConcatV3
/// - Parameter handle: 
/// - Parameter flowIn: 
/// - Parameter dtype: 
/// - Parameter elementShapeExcept0: 
/// - Returns: 
///	value: 
///	lengths: 
public func tensorArrayConcatV2(operationName: String? = nil, handle: Output, flowIn: Output, dtype: Any.Type, elementShapeExcept0: Shape) throws -> (value: Output, lengths: Output) { 
	var attrs = [String : Any]()
	attrs["dtype"] = dtype
	attrs["element_shape_except0"] = elementShapeExcept0
	let opspec = OpSpec(
		type: "TensorArrayConcatV2",
		name: (operationName ?? "Type"),
		input: [handle, flowIn],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (value: op.output(at: 0), lengths: op.output(at: 1))
} 

///Applies sparse addition to `input` using individual values or slices
///from `updates` according to indices `indices`.  The updates are non-aliasing:
/// `input` is only modified in-place if no other operations will use it.
/// Otherwise, a copy of `input` is made.  This operation has a gradient with
/// respect to both `input` and `updates`.
/// 
/// `input` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.
/// 
/// `indices` must be integer tensor, containing indices into `input`.
/// It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.
/// 
/// The innermost dimension of `indices` (with length `K`) corresponds to
/// indices into elements (if `K = P`) or `(P-K)`-dimensional slices
/// (if `K < P`) along the `K`th dimension of `input`.
/// 
/// `updates` is `Tensor` of rank `Q-1+P-K` with shape:
/// 
/// ```
/// [d_0, ..., d_{Q-2}, input.shape[K], ..., input.shape[P-1]].
/// ```
/// 
/// For example, say we want to add 4 scattered elements to a rank-1 tensor to 8
/// elements. In Python, that addition would look like this:
/// 
///     input = tf.constant([1, 2, 3, 4, 5, 6, 7, 8])
///     indices = tf.constant([[4], [3], [1], [7]])
///     updates = tf.constant([9, 10, 11, 12])
///     output = tf.scatter_nd_non_aliasing_add(input, indices, updates)
///     with tf.Session() as sess:
///       print(sess.run(output))
/// 
/// The resulting value `output` would look like this:
/// 
///     [1, 13, 3, 14, 14, 6, 7, 20]
/// 
/// See @{tf.scatter_nd} for more details about how to make updates to slices.
/// - Parameter input: A Tensor.
/// - Parameter indices: A Tensor. Must be one of the following types: `int32`, `int64`.
/// A tensor of indices into `input`.
/// - Parameter updates: A Tensor. Must have the same type as ref. A tensor of updated values
/// to add to `input`.
/// - Parameter tindices: 
/// - Returns: 
///	output: A `Tensor` with the same shape as `input`, containing values of `input`
/// updated with `updates`.
public func scatterNdNonAliasingAdd(operationName: String? = nil, input: Output, indices: Output, updates: Output, tindices: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Tindices"] = tindices
	let opspec = OpSpec(
		type: "ScatterNdNonAliasingAdd",
		name: (operationName ?? "Type"),
		input: [input, indices, updates],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Converts each string in the input Tensor to its hash mod by a number of buckets.
///The hash function is deterministic on the content of the string within the
/// process. The hash function is a keyed hash function, where attribute `key`
/// defines the key of the hash function. `key` is an array of 2 elements.
/// 
/// A strong hash is important when inputs may be malicious, e.g. URLs with
/// additional components. Adversaries could try to make their inputs hash to the
/// same bucket for a denial-of-service attack or to skew the results. A strong
/// hash prevents this by making it difficult, if not infeasible, to compute inputs
/// that hash to the same bucket. This comes at a cost of roughly 4x higher compute
/// time than `tf.string_to_hash_bucket_fast`.
/// - Parameter input: The strings to assign a hash bucket.
/// - Parameter numBuckets: The number of buckets.
/// - Parameter key: The key for the keyed hash function passed as a list of two uint64
/// elements.
/// - Returns: 
///	output: A Tensor of the same shape as the input `string_tensor`.
public func stringToHashBucketStrong(operationName: String? = nil, input: Output, numBuckets: UInt8, key: [Int64]) throws -> Output { 
	var attrs = [String : Any]()
	attrs["num_buckets"] = numBuckets
	attrs["key"] = key
	let opspec = OpSpec(
		type: "StringToHashBucketStrong",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Draws samples from a multinomial distribution.
/// - Parameter logits: 2-D Tensor with shape `[batch_size, num_classes]`.  Each slice `[i, :]`
/// represents the unnormalized log probabilities for all classes.
/// - Parameter numSamples: 0-D.  Number of independent samples to draw for each row slice.
/// - Parameter seed: If either seed or seed2 is set to be non-zero, the internal random number
/// generator is seeded by the given seed.  Otherwise, a random seed is used.
/// - Parameter seed2: A second seed to avoid seed collision.
/// - Returns: 
///	output: 2-D Tensor with shape `[batch_size, num_samples]`.  Each slice `[i, :]`
/// contains the drawn class labels with range `[0, num_classes)`.
public func multinomial(operationName: String? = nil, logits: Output, numSamples: Output, seed: UInt8, seed2: UInt8) throws -> Output { 
	var attrs = [String : Any]()
	attrs["seed"] = seed
	attrs["seed2"] = seed2
	let opspec = OpSpec(
		type: "Multinomial",
		name: (operationName ?? "Type"),
		input: [logits, numSamples],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Serialize a `SparseTensor` into a string 3-vector (1-D `Tensor`) object.
/// - Parameter sparseIndices: 2-D.  The `indices` of the `SparseTensor`.
/// - Parameter sparseValues: 1-D.  The `values` of the `SparseTensor`.
/// - Parameter sparseShape: 1-D.  The `shape` of the `SparseTensor`.
/// - Returns: 
///	serialized_sparse: 
public func serializeSparse(operationName: String? = nil, sparseIndices: Output, sparseValues: Output, sparseShape: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "SerializeSparse",
		name: (operationName ?? "Type"),
		input: [sparseIndices, sparseValues, sparseShape],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Deprecated, use StackV2.
/// - Parameter elemType: 
/// - Parameter stackName: 
/// - Returns: 
///	handle: 
public func stack(operationName: String? = nil, elemType: Any.Type, stackName: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["elem_type"] = elemType
	attrs["stack_name"] = stackName
	let opspec = OpSpec(
		type: "Stack",
		name: (operationName ?? "Type"),
		input: [],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///A Reader that outputs the records from a TensorFlow Records file.
/// - Parameter container: If non-empty, this reader is placed in the given container.
/// Otherwise, a default container is used.
/// - Parameter sharedName: If non-empty, this reader is named in the given bucket
/// with this shared_name. Otherwise, the node name is used instead.
/// - Parameter compressionType: 
/// - Returns: 
///	reader_handle: The handle to reference the Reader.
public func tFRecordReader(operationName: String? = nil, container: String, sharedName: String, compressionType: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["container"] = container
	attrs["shared_name"] = sharedName
	attrs["compression_type"] = compressionType
	let opspec = OpSpec(
		type: "TFRecordReader",
		name: (operationName ?? "Type"),
		input: [],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Performs max pooling on the input.
/// - Parameter input: 4-D input to pool over.
/// - Parameter ksize: The size of the window for each dimension of the input tensor.
/// - Parameter strides: The stride of the sliding window for each dimension of the
/// input tensor.
/// - Parameter padding: The type of padding algorithm to use.
/// - Parameter dataFormat: Specify the data format of the input and output data. With the
/// default format "NHWC", the data is stored in the order of:
///     [batch, in_height, in_width, in_channels].
/// Alternatively, the format could be "NCHW", the data storage order of:
///     [batch, in_channels, in_height, in_width].
/// - Returns: 
///	output: The max pooled output tensor.
public func maxPool(operationName: String? = nil, input: Output, ksize: [Int64], strides: [Int64], padding: String, dataFormat: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["ksize"] = ksize
	attrs["strides"] = strides
	attrs["padding"] = padding
	attrs["data_format"] = dataFormat
	let opspec = OpSpec(
		type: "MaxPool",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes the ids of the positions in sampled_candidates that match true_labels.
///When doing log-odds NCE, the result of this op should be passed through a
/// SparseToDense op, then added to the logits of the sampled candidates. This has
/// the effect of 'removing' the sampled labels that match the true labels by
/// making the classifier sure that they are sampled labels.
/// - Parameter trueClasses: The true_classes output of UnpackSparseLabels.
/// - Parameter sampledCandidates: The sampled_candidates output of CandidateSampler.
/// - Parameter numTrue: Number of true labels per context.
/// - Parameter seed: If either seed or seed2 are set to be non-zero, the random number
/// generator is seeded by the given seed.  Otherwise, it is seeded by a
/// random seed.
/// - Parameter seed2: An second seed to avoid seed collision.
/// - Returns: 
///	indices: A vector of indices corresponding to rows of true_candidates.
///	ids: A vector of IDs of positions in sampled_candidates that match a true_label
/// for the row with the corresponding index in indices.
///	weights: A vector of the same length as indices and ids, in which each element
/// is -FLOAT_MAX.
public func computeAccidentalHits(operationName: String? = nil, trueClasses: Output, sampledCandidates: Output, numTrue: UInt8, seed: UInt8, seed2: UInt8) throws -> (indices: Output, ids: Output, weights: Output) { 
	var attrs = [String : Any]()
	attrs["num_true"] = numTrue
	attrs["seed"] = seed
	attrs["seed2"] = seed2
	let opspec = OpSpec(
		type: "ComputeAccidentalHits",
		name: (operationName ?? "Type"),
		input: [trueClasses, sampledCandidates],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (indices: op.output(at: 0), ids: op.output(at: 1), weights: op.output(at: 2))
} 

///Dequeues `n` tuples of one or more tensors from the given queue.
///If the queue is closed and there are fewer than `n` elements, then an
/// OutOfRange error is returned.
/// 
/// This operation concatenates queue-element component tensors along the
/// 0th dimension to make a single component tensor.  All of the components
/// in the dequeued tuple will have size `n` in the 0th dimension.
/// 
/// This operation has `k` outputs, where `k` is the number of components in
/// the tuples stored in the given queue, and output `i` is the ith
/// component of the dequeued tuple.
/// 
/// N.B. If the queue is empty, this operation will block until `n` elements
/// have been dequeued (or 'timeout_ms' elapses, if specified).
/// - Parameter handle: The handle to a queue.
/// - Parameter n: The number of tuples to dequeue.
/// - Parameter componentTypes: The type of each component in a tuple.
/// - Parameter timeoutMs: If the queue has fewer than n elements, this operation
/// will block for up to timeout_ms milliseconds.
/// Note: This option is not supported yet.
/// - Returns: 
///	components: One or more tensors that were dequeued as a tuple.
public func queueDequeueMany(operationName: String? = nil, handle: Output, n: Output, componentTypes: [Any.Type], timeoutMs: UInt8) throws -> Output { 
	var attrs = [String : Any]()
	attrs["component_types"] = componentTypes
	attrs["timeout_ms"] = timeoutMs
	let opspec = OpSpec(
		type: "QueueDequeueMany",
		name: (operationName ?? "Type"),
		input: [handle, n],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Deserialize and concatenate `SparseTensors` from a serialized minibatch.
///The input `serialized_sparse` must be a string matrix of shape `[N x 3]` where
/// `N` is the minibatch size and the rows correspond to packed outputs of
/// `SerializeSparse`.  The ranks of the original `SparseTensor` objects
/// must all match.  When the final `SparseTensor` is created, it has rank one
/// higher than the ranks of the incoming `SparseTensor` objects
/// (they have been concatenated along a new row dimension).
/// 
/// The output `SparseTensor` object's shape values for all dimensions but the
/// first are the max across the input `SparseTensor` objects' shape values
/// for the corresponding dimensions.  Its first shape value is `N`, the minibatch
/// size.
/// 
/// The input `SparseTensor` objects' indices are assumed ordered in
/// standard lexicographic order.  If this is not the case, after this
/// step run `SparseReorder` to restore index ordering.
/// 
/// For example, if the serialized input is a `[2 x 3]` matrix representing two
/// original `SparseTensor` objects:
/// 
///     index = [ 0]
///             [10]
///             [20]
///     values = [1, 2, 3]
///     shape = [50]
/// 
/// and
/// 
///     index = [ 2]
///             [10]
///     values = [4, 5]
///     shape = [30]
/// 
/// then the final deserialized `SparseTensor` will be:
/// 
///     index = [0  0]
///             [0 10]
///             [0 20]
///             [1  2]
///             [1 10]
///     values = [1, 2, 3, 4, 5]
///     shape = [2 50]
/// - Parameter serializedSparse: 2-D, The `N` serialized `SparseTensor` objects.
/// Must have 3 columns.
/// - Parameter dtype: The `dtype` of the serialized `SparseTensor` objects.
/// - Returns: 
///	sparse_indices: 
///	sparse_values: 
///	sparse_shape: 
public func deserializeManySparse(operationName: String? = nil, serializedSparse: Output, dtype: Any.Type) throws -> (sparseIndices: Output, sparseValues: Output, sparseShape: Output) { 
	var attrs = [String : Any]()
	attrs["dtype"] = dtype
	let opspec = OpSpec(
		type: "DeserializeManySparse",
		name: (operationName ?? "Type"),
		input: [serializedSparse],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (sparseIndices: op.output(at: 0), sparseValues: op.output(at: 1), sparseShape: op.output(at: 2))
} 

///A conditional accumulator for aggregating sparse gradients.
///The accumulator accepts gradients marked with local_step greater or
/// equal to the most recent global_step known to the accumulator. The
/// average can be extracted from the accumulator, provided sufficient
/// gradients have been accumulated. Extracting the average automatically
/// resets the aggregate to 0, and increments the global_step recorded by
/// the accumulator.
/// - Parameter dtype: The type of the value being accumulated.
/// - Parameter shape: The shape of the values.
/// - Parameter container: If non-empty, this accumulator is placed in the given container.
/// Otherwise, a default container is used.
/// - Parameter sharedName: If non-empty, this accumulator will be shared under the given name
/// across multiple sessions.
/// - Returns: 
///	handle: The handle to the accumulator.
public func sparseConditionalAccumulator(operationName: String? = nil, dtype: Any.Type, shape: Shape, container: String, sharedName: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["dtype"] = dtype
	attrs["shape"] = shape
	attrs["container"] = container
	attrs["shared_name"] = sharedName
	let opspec = OpSpec(
		type: "SparseConditionalAccumulator",
		name: (operationName ?? "Type"),
		input: [],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///A conditional accumulator for aggregating gradients.
///The accumulator accepts gradients marked with local_step greater or
/// equal to the most recent global_step known to the accumulator. The
/// average can be extracted from the accumulator, provided sufficient
/// gradients have been accumulated. Extracting the average automatically
/// resets the aggregate to 0, and increments the global_step recorded by
/// the accumulator.
/// - Parameter dtype: The type of the value being accumulated.
/// - Parameter shape: The shape of the values, can be [], in which case shape is unknown.
/// - Parameter container: If non-empty, this accumulator is placed in the given container.
/// Otherwise, a default container is used.
/// - Parameter sharedName: If non-empty, this accumulator will be shared under the
/// given name across multiple sessions.
/// - Returns: 
///	handle: The handle to the accumulator.
public func conditionalAccumulator(operationName: String? = nil, dtype: Any.Type, shape: Shape, container: String, sharedName: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["dtype"] = dtype
	attrs["shape"] = shape
	attrs["container"] = container
	attrs["shared_name"] = sharedName
	let opspec = OpSpec(
		type: "ConditionalAccumulator",
		name: (operationName ?? "Type"),
		input: [],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Extract the shape information of a JPEG-encoded image.
///This op only parses the image header, so it is much faster than DecodeJpeg.
/// - Parameter contents: 0-D. The JPEG-encoded image.
/// - Parameter outputType: (Optional) The output type of the operation (int32 or int64).
/// Defaults to int32.
/// - Returns: 
///	image_shape: 1-D. The image shape with format [height, width, channels].
public func extractJpegShape(operationName: String? = nil, contents: Output, outputType: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["output_type"] = outputType
	let opspec = OpSpec(
		type: "ExtractJpegShape",
		name: (operationName ?? "Type"),
		input: [contents],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 


/// - Parameter input: 
/// - Returns: 
///	output: 
public func batchFFT(operationName: String? = nil, input: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "BatchFFT",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Returns the number of gradients aggregated in the given accumulators.
/// - Parameter handle: The handle to an accumulator.
/// - Returns: 
///	num_accumulated: The number of gradients aggregated in the given accumulator.
public func accumulatorNumAccumulated(operationName: String? = nil, handle: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "AccumulatorNumAccumulated",
		name: (operationName ?? "Type"),
		input: [handle],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 


/// - Parameter input: 
/// - Parameter adjoint: 
/// - Returns: 
///	output: 
public func batchMatrixInverse(operationName: String? = nil, input: Output, adjoint: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["adjoint"] = adjoint
	let opspec = OpSpec(
		type: "BatchMatrixInverse",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Update ' * var' according to the centered RMSProp algorithm.
///The centered RMSProp algorithm uses an estimate of the centered second moment
/// (i.e., the variance) for normalization, as opposed to regular RMSProp, which
/// uses the (uncentered) second moment. This often helps with training, but is
/// slightly more expensive in terms of computation and memory.
/// 
/// Note that in dense implementation of this algorithm, mg, ms, and mom will
/// update even if the grad is zero, but in this sparse implementation, mg, ms,
/// and mom will not update in iterations during which the grad is zero.
/// 
/// mean_square = decay  *  mean_square + (1-decay)  *  gradient  *  *  2
/// mean_grad = decay  *  mean_grad + (1-decay)  *  gradient
/// Delta = learning_rate  *  gradient / sqrt(mean_square + epsilon - mean_grad  *  *  2)
/// 
/// ms <- rho  *  ms_{t-1} + (1-rho)  *  grad  *  grad
/// mom <- momentum  *  mom_{t-1} + lr  *  grad / sqrt(ms + epsilon)
/// var <- var - mom
/// - Parameter `var`: Should be from a Variable().
/// - Parameter mg: Should be from a Variable().
/// - Parameter ms: Should be from a Variable().
/// - Parameter mom: Should be from a Variable().
/// - Parameter lr: Scaling factor. Must be a scalar.
/// - Parameter rho: Decay rate. Must be a scalar.
/// - Parameter momentum: 
/// - Parameter epsilon: Ridge term. Must be a scalar.
/// - Parameter grad: The gradient.
/// - Parameter indices: A vector of indices into the first dimension of var, ms and mom.
/// - Parameter tindices: 
/// - Parameter useLocking: If `True`, updating of the var, mg, ms, and mom tensors is
/// protected by a lock; otherwise the behavior is undefined, but may exhibit less
/// contention.
public func resourceSparseApplyCenteredRMSProp(operationName: String? = nil, `var`: Output, mg: Output, ms: Output, mom: Output, lr: Output, rho: Output, momentum: Output, epsilon: Output, grad: Output, indices: Output, tindices: Any.Type, useLocking: Bool) throws -> Operation { 
	var attrs = [String : Any]()
	attrs["Tindices"] = tindices
	attrs["use_locking"] = useLocking
	let opspec = OpSpec(
		type: "ResourceSparseApplyCenteredRMSProp",
		name: (operationName ?? "Type"),
		input: [`var`, mg, ms, mom, lr, rho, momentum, epsilon, grad, indices],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Computes the number of elements in the given queue.
/// - Parameter handle: The handle to a queue.
/// - Returns: 
///	size: The number of elements in the given queue.
public func queueSizeV2(operationName: String? = nil, handle: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "QueueSizeV2",
		name: (operationName ?? "Type"),
		input: [handle],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 


/// - Parameter input: 
/// - Returns: 
///	output: 
public func batchSelfAdjointEig(operationName: String? = nil, input: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "BatchSelfAdjointEig",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Returns the min of x and y (i.e. x < y ? x : y) element-wise.
/// * NOTE * : `Minimum` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
/// - Parameter x: 
/// - Parameter y: 
/// - Returns: 
///	z: 
public func minimum(operationName: String? = nil, x: Output, y: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Minimum",
		name: (operationName ?? "Type"),
		input: [x, y],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Returns true if queue is closed.
///This operation returns true if the queue is closed and false if the queue
/// is open.
/// - Parameter handle: The handle to a queue.
/// - Returns: 
///	is_closed: 
public func queueIsClosed(operationName: String? = nil, handle: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "QueueIsClosed",
		name: (operationName ?? "Type"),
		input: [handle],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Split the data from the input value into TensorArray elements.
///Assuming that `lengths` takes on values
/// 
///   ```(n0, n1, ..., n(T-1))```
/// 
/// and that `value` has shape
/// 
///   ```(n0 + n1 + ... + n(T-1) x d0 x d1 x ...)```,
/// 
/// this splits values into a TensorArray with T tensors.
/// 
/// TensorArray index t will be the subtensor of values with starting position
/// 
///   ```(n0 + n1 + ... + n(t-1), 0, 0, ...)```
/// 
/// and having size
/// 
///   ```nt x d0 x d1 x ...```
/// - Parameter handle: The handle to a TensorArray.
/// - Parameter value: The concatenated tensor to write to the TensorArray.
/// - Parameter lengths: The vector of lengths, how to split the rows of value into the
/// TensorArray.
/// - Parameter flowIn: A float scalar that enforces proper chaining of operations.
/// - Returns: 
///	flow_out: A float scalar that enforces proper chaining of operations.
public func tensorArraySplitV3(operationName: String? = nil, handle: Output, value: Output, lengths: Output, flowIn: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "TensorArraySplitV3",
		name: (operationName ?? "Type"),
		input: [handle, value, lengths, flowIn],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Update relevant entries in ' * var' according to the Ftrl-proximal scheme.
///That is for rows we have grad for, we update var, accum and linear as follows:
/// accum_new = accum + grad  *  grad
/// linear += grad + (accum_new// ^(-lr_power) - accum// ^(-lr_power)) / lr  *  var
/// quadratic = 1.0 / (accum_new// ^(lr_power)  *  lr) + 2  *  l2
/// var = (sign(linear)  *  l1 - linear) / quadratic if |linear| > l1 else 0.0
/// accum = accum_new
/// - Parameter `var`: Should be from a Variable().
/// - Parameter accum: Should be from a Variable().
/// - Parameter linear: Should be from a Variable().
/// - Parameter grad: The gradient.
/// - Parameter indices: A vector of indices into the first dimension of var and accum.
/// - Parameter lr: Scaling factor. Must be a scalar.
/// - Parameter l1: L1 regularization. Must be a scalar.
/// - Parameter l2: L2 regularization. Must be a scalar.
/// - Parameter lrPower: Scaling factor. Must be a scalar.
/// - Parameter tindices: 
/// - Parameter useLocking: If `True`, updating of the var and accum tensors will be protected
/// by a lock; otherwise the behavior is undefined, but may exhibit less
/// contention.
/// - Returns: 
///	out: Same as "var".
public func sparseApplyFtrl(operationName: String? = nil, `var`: Output, accum: Output, linear: Output, grad: Output, indices: Output, lr: Output, l1: Output, l2: Output, lrPower: Output, tindices: Any.Type, useLocking: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Tindices"] = tindices
	attrs["use_locking"] = useLocking
	let opspec = OpSpec(
		type: "SparseApplyFtrl",
		name: (operationName ?? "Type"),
		input: [`var`, accum, linear, grad, indices, lr, l1, l2, lrPower],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Sparse update ' * var' as FOBOS algorithm with fixed learning rate.
///That is for rows we have grad for, we update var as follows:
/// prox_v = var - alpha  *  grad
/// var = sign(prox_v)/(1+alpha * l2)  *  max{|prox_v|-alpha * l1,0}
/// - Parameter `var`: Should be from a Variable().
/// - Parameter alpha: Scaling factor. Must be a scalar.
/// - Parameter l1: L1 regularization. Must be a scalar.
/// - Parameter l2: L2 regularization. Must be a scalar.
/// - Parameter grad: The gradient.
/// - Parameter indices: A vector of indices into the first dimension of var and accum.
/// - Parameter tindices: 
/// - Parameter useLocking: If True, the subtraction will be protected by a lock;
/// otherwise the behavior is undefined, but may exhibit less contention.
public func resourceSparseApplyProximalGradientDescent(operationName: String? = nil, `var`: Output, alpha: Output, l1: Output, l2: Output, grad: Output, indices: Output, tindices: Any.Type, useLocking: Bool) throws -> Operation { 
	var attrs = [String : Any]()
	attrs["Tindices"] = tindices
	attrs["use_locking"] = useLocking
	let opspec = OpSpec(
		type: "ResourceSparseApplyProximalGradientDescent",
		name: (operationName ?? "Type"),
		input: [`var`, alpha, l1, l2, grad, indices],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///A queue that produces elements in first-in first-out order.
/// - Parameter componentTypes: The type of each component in a value.
/// - Parameter shapes: The shape of each component in a value. The length of this attr must
/// be either 0 or the same as the length of component_types. If the length of
/// this attr is 0, the shapes of queue elements are not constrained, and
/// only one element may be dequeued at a time.
/// - Parameter capacity: The upper bound on the number of elements in this queue.
/// Negative numbers mean no limit.
/// - Parameter container: If non-empty, this queue is placed in the given container.
/// Otherwise, a default container is used.
/// - Parameter sharedName: If non-empty, this queue will be shared under the given name
/// across multiple sessions.
/// - Returns: 
///	handle: The handle to the queue.
public func fIFOQueue(operationName: String? = nil, componentTypes: [Any.Type], shapes: [Shape], capacity: UInt8, container: String, sharedName: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["component_types"] = componentTypes
	attrs["shapes"] = shapes
	attrs["capacity"] = capacity
	attrs["container"] = container
	attrs["shared_name"] = sharedName
	let opspec = OpSpec(
		type: "FIFOQueue",
		name: (operationName ?? "Type"),
		input: [],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Op removes and returns the (key, value) element with the smallest
///key from the underlying container.   If the underlying container
/// does not contain elements, the op will block until it does.
/// - Parameter indices: 
/// - Parameter capacity: 
/// - Parameter memoryLimit: 
/// - Parameter dtypes: 
/// - Parameter container: 
/// - Parameter sharedName: 
/// - Returns: 
///	key: 
///	values: 
public func orderedMapUnstageNoKey(operationName: String? = nil, indices: Output, capacity: UInt8, memoryLimit: UInt8, dtypes: [Any.Type], container: String, sharedName: String) throws -> (key: Output, values: Output) { 
	var attrs = [String : Any]()
	attrs["capacity"] = capacity
	attrs["memory_limit"] = memoryLimit
	attrs["dtypes"] = dtypes
	attrs["container"] = container
	attrs["shared_name"] = sharedName
	let opspec = OpSpec(
		type: "OrderedMapUnstageNoKey",
		name: (operationName ?? "Type"),
		input: [indices],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (key: op.output(at: 0), values: op.output(at: 1))
} 

///Returns element-wise integer closest to x.
///If the result is midway between two representable values,
/// the even representable is chosen.
/// For example:
/// 
/// ```
/// rint(-1.5) ==> -2.0
/// rint(0.5000001) ==> 1.0
/// rint([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0]) ==> [-2., -2., -0., 0., 2., 2., 2.]
/// ```
/// - Parameter x: 
/// - Returns: 
///	y: 
public func rint(operationName: String? = nil, x: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Rint",
		name: (operationName ?? "Type"),
		input: [x],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///A queue that produces elements in first-in first-out order.
///Variable-size shapes are allowed by setting the corresponding shape dimensions
/// to 0 in the shape attr.  In this case DequeueMany will pad up to the maximum
/// size of any given element in the minibatch.  See below for details.
/// - Parameter componentTypes: The type of each component in a value.
/// - Parameter shapes: The shape of each component in a value. The length of this attr must
/// be either 0 or the same as the length of component_types.
/// Shapes of fixed rank but variable size are allowed by setting
/// any shape dimension to -1.  In this case, the inputs' shape may vary along
/// the given dimension, and DequeueMany will pad the given dimension with
/// zeros up to the maximum shape of all elements in the given batch.
/// If the length of this attr is 0, different queue elements may have
/// different ranks and shapes, but only one element may be dequeued at a time.
/// - Parameter capacity: The upper bound on the number of elements in this queue.
/// Negative numbers mean no limit.
/// - Parameter container: If non-empty, this queue is placed in the given container.
/// Otherwise, a default container is used.
/// - Parameter sharedName: If non-empty, this queue will be shared under the given name
/// across multiple sessions.
/// - Returns: 
///	handle: The handle to the queue.
public func paddingFIFOQueueV2(operationName: String? = nil, componentTypes: [Any.Type], shapes: [Shape], capacity: UInt8, container: String, sharedName: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["component_types"] = componentTypes
	attrs["shapes"] = shapes
	attrs["capacity"] = capacity
	attrs["container"] = container
	attrs["shared_name"] = sharedName
	let opspec = OpSpec(
		type: "PaddingFIFOQueueV2",
		name: (operationName ?? "Type"),
		input: [],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 


/// - Parameter size: 
/// - Parameter dtype: 
/// - Parameter dynamicSize: 
/// - Parameter clearAfterRead: 
/// - Parameter tensorArrayName: 
/// - Parameter elementShape: 
/// - Returns: 
///	handle: 
public func tensorArray(operationName: String? = nil, size: Output, dtype: Any.Type, dynamicSize: Bool, clearAfterRead: Bool, tensorArrayName: String, elementShape: Shape) throws -> Output { 
	var attrs = [String : Any]()
	attrs["dtype"] = dtype
	attrs["dynamic_size"] = dynamicSize
	attrs["clear_after_read"] = clearAfterRead
	attrs["tensor_array_name"] = tensorArrayName
	attrs["element_shape"] = elementShape
	let opspec = OpSpec(
		type: "TensorArray",
		name: (operationName ?? "Type"),
		input: [size],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Raise a exception to abort the process when called.
///If exit_without_error is true, the process will exit normally,
/// otherwise it will exit with a SIGABORT signal.
/// 
/// Returns nothing but an exception.
/// - Parameter errorMsg: A string which is the message associated with the exception.
/// - Parameter exitWithoutError: 
public func abort(operationName: String? = nil, errorMsg: String, exitWithoutError: Bool) throws -> Operation { 
	var attrs = [String : Any]()
	attrs["error_msg"] = errorMsg
	attrs["exit_without_error"] = exitWithoutError
	let opspec = OpSpec(
		type: "Abort",
		name: (operationName ?? "Type"),
		input: [],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Resize `images` to `size` using area interpolation.
///Input images can be of different types but output images are always float.
/// 
/// Each output pixel is computed by first transforming the pixel's footprint into
/// the input tensor and then averaging the pixels that intersect the footprint. An
/// input pixel's contribution to the average is weighted by the fraction of its
/// area that intersects the footprint.  This is the same as OpenCV's INTER_AREA.
/// - Parameter images: 4-D with shape `[batch, height, width, channels]`.
/// - Parameter size: = A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
/// new size for the images.
/// - Parameter alignCorners: If true, rescale input by (new_height - 1) / (height - 1), which
/// exactly aligns the 4 corners of images and resized images. If false, rescale
/// by new_height / height. Treat similarly the width dimension.
/// - Returns: 
///	resized_images: 4-D with shape
/// `[batch, new_height, new_width, channels]`.
public func resizeArea(operationName: String? = nil, images: Output, size: Output, alignCorners: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["align_corners"] = alignCorners
	let opspec = OpSpec(
		type: "ResizeArea",
		name: (operationName ?? "Type"),
		input: [images, size],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Extracts crops from the input image tensor and bilinearly resizes them (possibly
///with aspect ratio change) to a common output size specified by `crop_size`. This
/// is more general than the `crop_to_bounding_box` op which extracts a fixed size
/// slice from the input image and does not allow resizing or aspect ratio change.
/// 
/// Returns a tensor with `crops` from the input `image` at positions defined at the
/// bounding box locations in `boxes`. The cropped boxes are all resized (with
/// bilinear interpolation) to a fixed `size = [crop_height, crop_width]`. The
/// result is a 4-D tensor `[num_boxes, crop_height, crop_width, depth]`.
/// - Parameter image: A 4-D tensor of shape `[batch, image_height, image_width, depth]`.
/// Both `image_height` and `image_width` need to be positive.
/// - Parameter boxes: A 2-D tensor of shape `[num_boxes, 4]`. The `i`-th row of the tensor
/// specifies the coordinates of a box in the `box_ind[i]` image and is specified
/// in normalized coordinates `[y1, x1, y2, x2]`. A normalized coordinate value of
/// `y` is mapped to the image coordinate at `y  *  (image_height - 1)`, so as the
/// `[0, 1]` interval of normalized image height is mapped to
/// `[0, image_height - 1]` in image height coordinates. We do allow `y1` > `y2`, in
/// which case the sampled crop is an up-down flipped version of the original
/// image. The width dimension is treated similarly. Normalized coordinates
/// outside the `[0, 1]` range are allowed, in which case we use
/// `extrapolation_value` to extrapolate the input image values.
/// - Parameter boxInd: A 1-D tensor of shape `[num_boxes]` with int32 values in `[0, batch)`.
/// The value of `box_ind[i]` specifies the image that the `i`-th box refers to.
/// - Parameter cropSize: A 1-D tensor of 2 elements, `size = [crop_height, crop_width]`. All
/// cropped image patches are resized to this size. The aspect ratio of the image
/// content is not preserved. Both `crop_height` and `crop_width` need to be
/// positive.
/// - Parameter method: A string specifying the interpolation method. Only 'bilinear' is
/// supported for now.
/// - Parameter extrapolationValue: Value used for extrapolation, when applicable.
/// - Returns: 
///	crops: A 4-D tensor of shape `[num_boxes, crop_height, crop_width, depth]`.
public func cropAndResize(operationName: String? = nil, image: Output, boxes: Output, boxInd: Output, cropSize: Output, method: String, extrapolationValue: Float) throws -> Output { 
	var attrs = [String : Any]()
	attrs["method"] = method
	attrs["extrapolation_value"] = extrapolationValue
	let opspec = OpSpec(
		type: "CropAndResize",
		name: (operationName ?? "Type"),
		input: [image, boxes, boxInd, cropSize],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Deprecated. Use TensorArrayGatherV3
/// - Parameter handle: 
/// - Parameter indices: 
/// - Parameter flowIn: 
/// - Parameter dtype: 
/// - Parameter elementShape: 
/// - Returns: 
///	value: 
public func tensorArrayGatherV2(operationName: String? = nil, handle: Output, indices: Output, flowIn: Output, dtype: Any.Type, elementShape: Shape) throws -> Output { 
	var attrs = [String : Any]()
	attrs["dtype"] = dtype
	attrs["element_shape"] = elementShape
	let opspec = OpSpec(
		type: "TensorArrayGatherV2",
		name: (operationName ?? "Type"),
		input: [handle, indices, flowIn],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Returns the element-wise max of two SparseTensors.
///Assumes the two SparseTensors have the same shape, i.e., no broadcasting.
/// - Parameter aIndices: 2-D.  `N x R` matrix with the indices of non-empty values in a
/// SparseTensor, in the canonical lexicographic ordering.
/// - Parameter aValues: 1-D.  `N` non-empty values corresponding to `a_indices`.
/// - Parameter aShape: 1-D.  Shape of the input SparseTensor.
/// - Parameter bIndices: counterpart to `a_indices` for the other operand.
/// - Parameter bValues: counterpart to `a_values` for the other operand; must be of the same dtype.
/// - Parameter bShape: counterpart to `a_shape` for the other operand; the two shapes must be equal.
/// - Returns: 
///	output_indices: 2-D.  The indices of the output SparseTensor.
///	output_values: 1-D.  The values of the output SparseTensor.
public func sparseSparseMaximum(operationName: String? = nil, aIndices: Output, aValues: Output, aShape: Output, bIndices: Output, bValues: Output, bShape: Output) throws -> (outputIndices: Output, outputValues: Output) { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "SparseSparseMaximum",
		name: (operationName ?? "Type"),
		input: [aIndices, aValues, aShape, bIndices, bValues, bShape],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (outputIndices: op.output(at: 0), outputValues: op.output(at: 1))
} 

///Decode the first frame of a GIF-encoded image to a uint8 tensor.
///GIF with frame or transparency compression are not supported
/// convert animated GIF from compressed to uncompressed by:
/// 
///     convert $src.gif -coalesce $dst.gif
/// 
/// This op also supports decoding JPEGs and PNGs, though it is cleaner to use
/// `tf.image.decode_image`.
/// - Parameter contents: 0-D.  The GIF-encoded image.
/// - Returns: 
///	image: 4-D with shape `[num_frames, height, width, 3]`. RGB order
public func decodeGif(operationName: String? = nil, contents: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "DecodeGif",
		name: (operationName ?? "Type"),
		input: [contents],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Transforms a vector of brain.Example protos (as strings) into typed tensors.
/// - Parameter serialized: A vector containing a batch of binary serialized Example protos.
/// - Parameter names: A vector containing the names of the serialized protos.
/// May contain, for example, table key (descriptive) names for the
/// corresponding serialized protos.  These are purely useful for debugging
/// purposes, and the presence of values here has no effect on the output.
/// May also be an empty vector if no names are available.
/// If non-empty, this vector must be the same length as "serialized".
/// - Parameter sparseKeys: A list of Nsparse string Tensors (scalars).
/// The keys expected in the Examples' features associated with sparse values.
/// - Parameter denseKeys: A list of Ndense string Tensors (scalars).
/// The keys expected in the Examples' features associated with dense values.
/// - Parameter denseDefaults: A list of Ndense Tensors (some may be empty).
/// dense_defaults[j] provides default values
/// when the example's feature_map lacks dense_key[j].  If an empty Tensor is
/// provided for dense_defaults[j], then the Feature dense_keys[j] is required.
/// The input type is inferred from dense_defaults[j], even when it's empty.
/// If dense_defaults[j] is not empty, and dense_shapes[j] is fully defined,
/// then the shape of dense_defaults[j] must match that of dense_shapes[j].
/// If dense_shapes[j] has an undefined major dimension (variable strides dense
/// feature), dense_defaults[j] must contain a single element:
/// the padding element.
/// - Parameter nsparse: 
/// - Parameter ndense: 
/// - Parameter sparseTypes: A list of Nsparse types; the data types of data in each Feature
/// given in sparse_keys.
/// Currently the ParseExample supports DT_FLOAT (FloatList),
/// DT_INT64 (Int64List), and DT_STRING (BytesList).
/// - Parameter tdense: 
/// - Parameter denseShapes: A list of Ndense shapes; the shapes of data in each Feature
/// given in dense_keys.
/// The number of elements in the Feature corresponding to dense_key[j]
/// must always equal dense_shapes[j].NumEntries().
/// If dense_shapes[j] == (D0, D1, ..., DN) then the shape of output
/// Tensor dense_values[j] will be (|serialized|, D0, D1, ..., DN):
/// The dense outputs are just the inputs row-stacked by batch.
/// This works for dense_shapes[j] = (-1, D1, ..., DN).  In this case
/// the shape of the output Tensor dense_values[j] will be
/// (|serialized|, M, D1, .., DN), where M is the maximum number of blocks
/// of elements of length D1  *  ....  *  DN, across all minibatch entries
/// in the input.  Any minibatch entry with less than M blocks of elements of
/// length D1  *  ...  *  DN will be padded with the corresponding default_value
/// scalar element along the second dimension.
/// - Returns: 
///	sparse_indices: 
///	sparse_values: 
///	sparse_shapes: 
///	dense_values: 
public func parseExample(operationName: String? = nil, serialized: Output, names: Output, sparseKeys: Output, denseKeys: Output, denseDefaults: Output, nsparse: UInt8, ndense: UInt8, sparseTypes: [Any.Type], tdense: [Any.Type], denseShapes: [Shape]) throws -> (sparseIndices: Output, sparseValues: Output, sparseShapes: Output, denseValues: Output) { 
	var attrs = [String : Any]()
	attrs["Nsparse"] = nsparse
	attrs["Ndense"] = ndense
	attrs["sparse_types"] = sparseTypes
	attrs["Tdense"] = tdense
	attrs["dense_shapes"] = denseShapes
	let opspec = OpSpec(
		type: "ParseExample",
		name: (operationName ?? "Type"),
		input: [serialized, names, sparseKeys, denseKeys, denseDefaults],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (sparseIndices: op.output(at: 0), sparseValues: op.output(at: 1), sparseShapes: op.output(at: 2), denseValues: op.output(at: 3))
} 

///Computes inverse hyperbolic tangent of x element-wise.
/// - Parameter x: 
/// - Returns: 
///	y: 
public func atanh(operationName: String? = nil, x: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Atanh",
		name: (operationName ?? "Type"),
		input: [x],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Makes a new iterator from the given `dataset` and stores it in `iterator`.
///This operation may be executed multiple times. Each execution will reset the
/// iterator in `iterator` to the first element of `dataset`.
/// - Parameter dataset: 
/// - Parameter iterator: 
public func makeIterator(operationName: String? = nil, dataset: Output, iterator: Output) throws -> Operation { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "MakeIterator",
		name: (operationName ?? "Type"),
		input: [dataset, iterator],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Return substrings from `Tensor` of strings.
///For each string in the input `Tensor`, creates a substring starting at index
/// `pos` with a total length of `len`.
/// 
/// If `len` defines a substring that would extend beyond the length of the input
/// string, then as many characters as possible are used.
/// 
/// If `pos` is negative or specifies a character index larger than any of the input
/// strings, then an `InvalidArgumentError` is thrown.
/// 
/// `pos` and `len` must have the same shape, otherwise a `ValueError` is thrown on
/// Op creation.
/// 
///  * NOTE * : `Substr` supports broadcasting up to two dimensions. More about
/// broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
/// 
/// ---
/// 
/// Examples
/// 
/// Using scalar `pos` and `len`:
/// 
/// ```python
/// input = [b'Hello', b'World']
/// position = 1
/// length = 3
/// 
/// output = [b'ell', b'orl']
/// ```
/// 
/// Using `pos` and `len` with same shape as `input`:
/// 
/// ```python
/// input = [[b'ten', b'eleven', b'twelve'],
///          [b'thirteen', b'fourteen', b'fifteen'],
///          [b'sixteen', b'seventeen', b'eighteen']]
/// position = [[1, 2, 3],
///             [1, 2, 3],
///             [1, 2, 3]]
/// length =   [[2, 3, 4],
///             [4, 3, 2],
///             [5, 5, 5]]
/// 
/// output = [[b'en', b'eve', b'lve'],
///           [b'hirt', b'urt', b'te'],
///           [b'ixtee', b'vente', b'hteen']]
/// ```
/// 
/// Broadcasting `pos` and `len` onto `input`:
/// 
/// ```
/// input = [[b'ten', b'eleven', b'twelve'],
///          [b'thirteen', b'fourteen', b'fifteen'],
///          [b'sixteen', b'seventeen', b'eighteen'],
///          [b'nineteen', b'twenty', b'twentyone']]
/// position = [1, 2, 3]
/// length =   [1, 2, 3]
/// 
/// output = [[b'e', b'ev', b'lve'],
///           [b'h', b'ur', b'tee'],
///           [b'i', b've', b'hte'],
///           [b'i', b'en', b'nty']]
/// ```
/// 
/// Broadcasting `input` onto `pos` and `len`:
/// 
/// ```
/// input = b'thirteen'
/// position = [1, 5, 7]
/// length =   [3, 2, 1]
/// 
/// output = [b'hir', b'ee', b'n']
/// ```
/// - Parameter input: Tensor of strings
/// - Parameter pos: Scalar defining the position of first character in each substring
/// - Parameter len: Scalar defining the number of characters to include in each substring
/// - Returns: 
///	output: Tensor of substrings
public func substr(operationName: String? = nil, input: Output, pos: Output, len: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Substr",
		name: (operationName ?? "Type"),
		input: [input, pos, len],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Extract `patches` from `images` and put them in the "depth" output dimension.
/// - Parameter images: 4-D Tensor with shape `[batch, in_rows, in_cols, depth]`.
/// - Parameter ksizes: The size of the sliding window for each dimension of `images`.
/// - Parameter strides: 1-D of length 4. How far the centers of two consecutive patches are in
/// the images. Must be: `[1, stride_rows, stride_cols, 1]`.
/// - Parameter rates: 1-D of length 4. Must be: `[1, rate_rows, rate_cols, 1]`. This is the
/// input stride, specifying how far two consecutive patch samples are in the
/// input. Equivalent to extracting patches with
/// `patch_sizes_eff = patch_sizes + (patch_sizes - 1)  *  (rates - 1)`, followed by
/// subsampling them spatially by a factor of `rates`. This is equivalent to
/// `rate` in dilated (a.k.a. Atrous) convolutions.
/// - Parameter padding: The type of padding algorithm to use.
/// 
/// We specify the size-related attributes as:
/// 
/// ```python
///       ksizes = [1, ksize_rows, ksize_cols, 1]
///       strides = [1, strides_rows, strides_cols, 1]
///       rates = [1, rates_rows, rates_cols, 1]
/// ```
/// - Returns: 
///	patches: 4-D Tensor with shape `[batch, out_rows, out_cols, ksize_rows  * 
/// ksize_cols  *  depth]` containing image patches with size
/// `ksize_rows x ksize_cols x depth` vectorized in the "depth" dimension. Note
/// `out_rows` and `out_cols` are the dimensions of the output patches.
public func extractImagePatches(operationName: String? = nil, images: Output, ksizes: [Int64], strides: [Int64], rates: [Int64], padding: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["ksizes"] = ksizes
	attrs["strides"] = strides
	attrs["rates"] = rates
	attrs["padding"] = padding
	let opspec = OpSpec(
		type: "ExtractImagePatches",
		name: (operationName ?? "Type"),
		input: [images],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes the difference between two lists of numbers or strings.
///Given a list `x` and a list `y`, this operation returns a list `out` that
/// represents all values that are in `x` but not in `y`. The returned list `out`
/// is sorted in the same order that the numbers appear in `x` (duplicates are
/// preserved). This operation also returns a list `idx` that represents the
/// position of each `out` element in `x`. In other words:
/// 
/// `out[i] = x[idx[i]] for i in [0, 1, ..., len(out) - 1]`
/// 
/// For example, given this input:
/// 
/// ```
/// x = [1, 2, 3, 4, 5, 6]
/// y = [1, 3, 5]
/// ```
/// 
/// This operation would return:
/// 
/// ```
/// out ==> [2, 4, 6]
/// idx ==> [1, 3, 5]
/// ```
/// - Parameter x: 1-D. Values to keep.
/// - Parameter y: 1-D. Values to remove.
/// - Parameter outIdx: 
/// - Returns: 
///	out: 1-D. Values present in `x` but not in `y`.
///	idx: 1-D. Positions of `x` values preserved in `out`.
public func listDiff(operationName: String? = nil, x: Output, y: Output, outIdx: Any.Type) throws -> (out: Output, idx: Output) { 
	var attrs = [String : Any]()
	attrs["out_idx"] = outIdx
	let opspec = OpSpec(
		type: "ListDiff",
		name: (operationName ?? "Type"),
		input: [x, y],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (out: op.output(at: 0), idx: op.output(at: 1))
} 

///Generates labels for candidate sampling with a learned unigram distribution.
///A unigram sampler could use a fixed unigram distribution read from a
/// file or passed in as an in-memory array instead of building up the distribution
/// from data on the fly. There is also an option to skew the distribution by
/// applying a distortion power to the weights.
/// 
/// The vocabulary file should be in CSV-like format, with the last field
/// being the weight associated with the word.
/// 
/// For each batch, this op picks a single set of sampled candidate labels.
/// 
/// The advantages of sampling candidates per-batch are simplicity and the
/// possibility of efficient dense matrix multiplication. The disadvantage is that
/// the sampled candidates must be chosen independently of the context and of the
/// true labels.
/// - Parameter trueClasses: A batch_size  *  num_true matrix, in which each row contains the
/// IDs of the num_true target_classes in the corresponding original label.
/// - Parameter numTrue: Number of true labels per context.
/// - Parameter numSampled: Number of candidates to randomly sample.
/// - Parameter unique: If unique is true, we sample with rejection, so that all sampled
/// candidates in a batch are unique. This requires some approximation to
/// estimate the post-rejection sampling probabilities.
/// - Parameter rangeMax: The sampler will sample integers from the interval [0, range_max).
/// - Parameter vocabFile: Each valid line in this file (which should have a CSV-like format)
/// corresponds to a valid word ID. IDs are in sequential order, starting from
/// num_reserved_ids. The last entry in each line is expected to be a value
/// corresponding to the count or relative probability. Exactly one of vocab_file
/// and unigrams needs to be passed to this op.
/// - Parameter distortion: The distortion is used to skew the unigram probability distribution.
/// Each weight is first raised to the distortion's power before adding to the
/// internal unigram distribution. As a result, distortion = 1.0 gives regular
/// unigram sampling (as defined by the vocab file), and distortion = 0.0 gives
/// a uniform distribution.
/// - Parameter numReservedIds: Optionally some reserved IDs can be added in the range [0,
/// ..., num_reserved_ids) by the users. One use case is that a special unknown
/// word token is used as ID 0. These IDs will have a sampling probability of 0.
/// - Parameter numShards: A sampler can be used to sample from a subset of the original range
/// in order to speed up the whole computation through parallelism. This parameter
/// (together with 'shard') indicates the number of partitions that are being
/// used in the overall computation.
/// - Parameter shard: A sampler can be used to sample from a subset of the original range
/// in order to speed up the whole computation through parallelism. This parameter
/// (together with 'num_shards') indicates the particular partition number of a
/// sampler op, when partitioning is being used.
/// - Parameter unigrams: A list of unigram counts or probabilities, one per ID in sequential
/// order. Exactly one of vocab_file and unigrams should be passed to this op.
/// - Parameter seed: If either seed or seed2 are set to be non-zero, the random number
/// generator is seeded by the given seed.  Otherwise, it is seeded by a
/// random seed.
/// - Parameter seed2: An second seed to avoid seed collision.
/// - Returns: 
///	sampled_candidates: A vector of length num_sampled, in which each element is
/// the ID of a sampled candidate.
///	true_expected_count: A batch_size  *  num_true matrix, representing
/// the number of times each candidate is expected to occur in a batch
/// of sampled candidates. If unique=true, then this is a probability.
///	sampled_expected_count: A vector of length num_sampled, for each sampled
/// candidate representing the number of times the candidate is expected
/// to occur in a batch of sampled candidates.  If unique=true, then this is a
/// probability.
public func fixedUnigramCandidateSampler(operationName: String? = nil, trueClasses: Output, numTrue: UInt8, numSampled: UInt8, unique: Bool, rangeMax: UInt8, vocabFile: String, distortion: Float, numReservedIds: UInt8, numShards: UInt8, shard: UInt8, unigrams: [Float], seed: UInt8, seed2: UInt8) throws -> (sampledCandidates: Output, trueExpectedCount: Output, sampledExpectedCount: Output) { 
	var attrs = [String : Any]()
	attrs["num_true"] = numTrue
	attrs["num_sampled"] = numSampled
	attrs["unique"] = unique
	attrs["range_max"] = rangeMax
	attrs["vocab_file"] = vocabFile
	attrs["distortion"] = distortion
	attrs["num_reserved_ids"] = numReservedIds
	attrs["num_shards"] = numShards
	attrs["shard"] = shard
	attrs["unigrams"] = unigrams
	attrs["seed"] = seed
	attrs["seed2"] = seed2
	let opspec = OpSpec(
		type: "FixedUnigramCandidateSampler",
		name: (operationName ?? "Type"),
		input: [trueClasses],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (sampledCandidates: op.output(at: 0), trueExpectedCount: op.output(at: 1), sampledExpectedCount: op.output(at: 2))
} 

///Generate a sharded filename. The filename is printf formatted as
///   %s-%05d-of-%05d, basename, shard, num_shards.
/// - Parameter basename: 
/// - Parameter shard: 
/// - Parameter numShards: 
/// - Returns: 
///	filename: 
public func shardedFilename(operationName: String? = nil, basename: Output, shard: Output, numShards: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "ShardedFilename",
		name: (operationName ?? "Type"),
		input: [basename, shard, numShards],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Decode web-safe base64-encoded strings.
///Input may or may not have padding at the end. See EncodeBase64 for padding.
/// Web-safe means that input must use - and _ instead of + and /.
/// - Parameter input: Base64 strings to decode.
/// - Returns: 
///	output: Decoded strings.
public func decodeBase64(operationName: String? = nil, input: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "DecodeBase64",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes the inverse of one or more square invertible matrices or their
///adjoints (conjugate transposes).
/// 
/// The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
/// form square matrices. The output is a tensor of the same shape as the input
/// containing the inverse for all input submatrices `[..., :, :]`.
/// 
/// The op uses LU decomposition with partial pivoting to compute the inverses.
/// 
/// If a matrix is not invertible there is no guarantee what the op does. It
/// may detect the condition and raise an exception or it may simply return a
/// garbage result.
/// - Parameter input: Shape is `[..., M, M]`.
/// - Parameter adjoint: 
/// - Returns: 
///	output: Shape is `[..., M, M]`.
/// 
/// @compatibility(numpy)
/// Equivalent to np.linalg.inv
/// @end_compatibility
public func matrixInverse(operationName: String? = nil, input: Output, adjoint: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["adjoint"] = adjoint
	let opspec = OpSpec(
		type: "MatrixInverse",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes the gradients of 3-D convolution with respect to the input.
/// - Parameter input: Shape `[batch, depth, rows, cols, in_channels]`.
/// - Parameter filter: Shape `[depth, rows, cols, in_channels, out_channels]`.
/// `in_channels` must match between `input` and `filter`.
/// - Parameter outBackprop: Backprop signal of shape `[batch, out_depth, out_rows, out_cols,
/// out_channels]`.
/// - Parameter strides: 1-D tensor of length 5. The stride of the sliding window for each
/// dimension of `input`. Must have `strides[0] = strides[4] = 1`.
/// - Parameter padding: The type of padding algorithm to use.
/// - Returns: 
///	output: 
public func conv3DBackpropInput(operationName: String? = nil, input: Output, filter: Output, outBackprop: Output, strides: [Int64], padding: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["strides"] = strides
	attrs["padding"] = padding
	let opspec = OpSpec(
		type: "Conv3DBackpropInput",
		name: (operationName ?? "Type"),
		input: [input, filter, outBackprop],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes a 2-D depthwise convolution given 4-D `input` and `filter` tensors.
///Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
/// and a filter / kernel tensor of shape
/// `[filter_height, filter_width, in_channels, channel_multiplier]`, containing
/// `in_channels` convolutional filters of depth 1, `depthwise_conv2d` applies
/// a different filter to each input channel (expanding from 1 channel to
/// `channel_multiplier` channels for each), then concatenates the results
/// together. Thus, the output has `in_channels  *  channel_multiplier` channels.
/// 
/// ```
/// for k in 0..in_channels-1
///   for q in 0..channel_multiplier-1
///     output[b, i, j, k  *  channel_multiplier + q] =
///       sum_{di, dj} input[b, strides[1]  *  i + di, strides[2]  *  j + dj, k]  * 
///                         filter[di, dj, k, q]
/// ```
/// 
/// Must have `strides[0] = strides[3] = 1`.  For the most common case of the same
/// horizontal and vertices strides, `strides = [1, stride, stride, 1]`.
/// - Parameter input: 
/// - Parameter filter: 
/// - Parameter strides: 1-D of length 4.  The stride of the sliding window for each dimension
/// of `input`.
/// - Parameter padding: The type of padding algorithm to use.
/// - Parameter dataFormat: Specify the data format of the input and output data. With the
/// default format "NHWC", the data is stored in the order of:
///     [batch, height, width, channels].
/// Alternatively, the format could be "NCHW", the data storage order of:
///     [batch, channels, height, width].
/// - Returns: 
///	output: 
public func depthwiseConv2dNative(operationName: String? = nil, input: Output, filter: Output, strides: [Int64], padding: String, dataFormat: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["strides"] = strides
	attrs["padding"] = padding
	attrs["data_format"] = dataFormat
	let opspec = OpSpec(
		type: "DepthwiseConv2dNative",
		name: (operationName ?? "Type"),
		input: [input, filter],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Generates labels for candidate sampling with a learned unigram distribution.
///See explanations of candidate sampling and the data formats at
/// go/candidate-sampling.
/// 
/// For each batch, this op picks a single set of sampled candidate labels.
/// 
/// The advantages of sampling candidates per-batch are simplicity and the
/// possibility of efficient dense matrix multiplication. The disadvantage is that
/// the sampled candidates must be chosen independently of the context and of the
/// true labels.
/// - Parameter trueClasses: A batch_size  *  num_true matrix, in which each row contains the
/// IDs of the num_true target_classes in the corresponding original label.
/// - Parameter numTrue: Number of true labels per context.
/// - Parameter numSampled: Number of candidates to randomly sample.
/// - Parameter unique: If unique is true, we sample with rejection, so that all sampled
/// candidates in a batch are unique. This requires some approximation to
/// estimate the post-rejection sampling probabilities.
/// - Parameter rangeMax: The sampler will sample integers from the interval [0, range_max).
/// - Parameter seed: If either seed or seed2 are set to be non-zero, the random number
/// generator is seeded by the given seed.  Otherwise, it is seeded by a
/// random seed.
/// - Parameter seed2: An second seed to avoid seed collision.
/// - Returns: 
///	sampled_candidates: A vector of length num_sampled, in which each element is
/// the ID of a sampled candidate.
///	true_expected_count: A batch_size  *  num_true matrix, representing
/// the number of times each candidate is expected to occur in a batch
/// of sampled candidates. If unique=true, then this is a probability.
///	sampled_expected_count: A vector of length num_sampled, for each sampled
/// candidate representing the number of times the candidate is expected
/// to occur in a batch of sampled candidates.  If unique=true, then this is a
/// probability.
public func learnedUnigramCandidateSampler(operationName: String? = nil, trueClasses: Output, numTrue: UInt8, numSampled: UInt8, unique: Bool, rangeMax: UInt8, seed: UInt8, seed2: UInt8) throws -> (sampledCandidates: Output, trueExpectedCount: Output, sampledExpectedCount: Output) { 
	var attrs = [String : Any]()
	attrs["num_true"] = numTrue
	attrs["num_sampled"] = numSampled
	attrs["unique"] = unique
	attrs["range_max"] = rangeMax
	attrs["seed"] = seed
	attrs["seed2"] = seed2
	let opspec = OpSpec(
		type: "LearnedUnigramCandidateSampler",
		name: (operationName ?? "Type"),
		input: [trueClasses],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (sampledCandidates: op.output(at: 0), trueExpectedCount: op.output(at: 1), sampledExpectedCount: op.output(at: 2))
} 

///Destroys the temporary variable and returns its final value.
///Sets output to the value of the Tensor pointed to by 'ref', then destroys
/// the temporary variable called 'var_name'.
/// All other uses of 'ref'  * must *  have executed before this op.
/// This is typically achieved by chaining the ref through each assign op, or by
/// using control dependencies.
/// 
/// Outputs the final value of the tensor pointed to by 'ref'.
/// - Parameter ref: A reference to the temporary variable tensor.
/// - Parameter varName: Name of the temporary variable, usually the name of the matching
/// 'TemporaryVariable' op.
/// - Returns: 
///	value: 
public func destroyTemporaryVariable(operationName: String? = nil, ref: Output, varName: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["var_name"] = varName
	let opspec = OpSpec(
		type: "DestroyTemporaryVariable",
		name: (operationName ?? "Type"),
		input: [ref],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///A Reader that outputs the entire contents of a file as a value.
///To use, enqueue filenames in a Queue.  The output of ReaderRead will
/// be a filename (key) and the contents of that file (value).
/// - Parameter container: If non-empty, this reader is placed in the given container.
/// Otherwise, a default container is used.
/// - Parameter sharedName: If non-empty, this reader is named in the given bucket
/// with this shared_name. Otherwise, the node name is used instead.
/// - Returns: 
///	reader_handle: The handle to reference the Reader.
public func wholeFileReader(operationName: String? = nil, container: String, sharedName: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["container"] = container
	attrs["shared_name"] = sharedName
	let opspec = OpSpec(
		type: "WholeFileReader",
		name: (operationName ?? "Type"),
		input: [],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Read `SparseTensors` from a `SparseTensorsMap` and concatenate them.
///The input `sparse_handles` must be an `int64` matrix of shape `[N, 1]` where
/// `N` is the minibatch size and the rows correspond to the output handles of
/// `AddSparseToTensorsMap` or `AddManySparseToTensorsMap`.  The ranks of the
/// original `SparseTensor` objects that went into the given input ops must all
/// match.  When the final `SparseTensor` is created, it has rank one
/// higher than the ranks of the incoming `SparseTensor` objects
/// (they have been concatenated along a new row dimension on the left).
/// 
/// The output `SparseTensor` object's shape values for all dimensions but the
/// first are the max across the input `SparseTensor` objects' shape values
/// for the corresponding dimensions.  Its first shape value is `N`, the minibatch
/// size.
/// 
/// The input `SparseTensor` objects' indices are assumed ordered in
/// standard lexicographic order.  If this is not the case, after this
/// step run `SparseReorder` to restore index ordering.
/// 
/// For example, if the handles represent an input, which is a `[2, 3]` matrix
/// representing two original `SparseTensor` objects:
/// 
/// ```
///     index = [ 0]
///             [10]
///             [20]
///     values = [1, 2, 3]
///     shape = [50]
/// ```
/// 
/// and
/// 
/// ```
///     index = [ 2]
///             [10]
///     values = [4, 5]
///     shape = [30]
/// ```
/// 
/// then the final `SparseTensor` will be:
/// 
/// ```
///     index = [0  0]
///             [0 10]
///             [0 20]
///             [1  2]
///             [1 10]
///     values = [1, 2, 3, 4, 5]
///     shape = [2 50]
/// ```
/// - Parameter sparseHandles: 1-D, The `N` serialized `SparseTensor` objects.
/// Shape: `[N]`.
/// - Parameter dtype: The `dtype` of the `SparseTensor` objects stored in the
/// `SparseTensorsMap`.
/// - Parameter container: The container name for the `SparseTensorsMap` read by this op.
/// - Parameter sharedName: The shared name for the `SparseTensorsMap` read by this op.
/// It should not be blank; rather the `shared_name` or unique Operation name
/// of the Op that created the original `SparseTensorsMap` should be used.
/// - Returns: 
///	sparse_indices: 2-D.  The `indices` of the minibatch `SparseTensor`.
///	sparse_values: 1-D.  The `values` of the minibatch `SparseTensor`.
///	sparse_shape: 1-D.  The `shape` of the minibatch `SparseTensor`.
public func takeManySparseFromTensorsMap(operationName: String? = nil, sparseHandles: Output, dtype: Any.Type, container: String, sharedName: String) throws -> (sparseIndices: Output, sparseValues: Output, sparseShape: Output) { 
	var attrs = [String : Any]()
	attrs["dtype"] = dtype
	attrs["container"] = container
	attrs["shared_name"] = sharedName
	let opspec = OpSpec(
		type: "TakeManySparseFromTensorsMap",
		name: (operationName ?? "Type"),
		input: [sparseHandles],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (sparseIndices: op.output(at: 0), sparseValues: op.output(at: 1), sparseShape: op.output(at: 2))
} 

///Applies a gradient to a given accumulator.
///Does not add if local_step is lesser than the accumulator's global_step.
/// - Parameter handle: The handle to a accumulator.
/// - Parameter localStep: The local_step value at which the gradient was computed.
/// - Parameter gradient: A tensor of the gradient to be accumulated.
/// - Parameter dtype: The data type of accumulated gradients. Needs to correspond to the type
/// of the accumulator.
public func accumulatorApplyGradient(operationName: String? = nil, handle: Output, localStep: Output, gradient: Output, dtype: Any.Type) throws -> Operation { 
	var attrs = [String : Any]()
	attrs["dtype"] = dtype
	let opspec = OpSpec(
		type: "AccumulatorApplyGradient",
		name: (operationName ?? "Type"),
		input: [handle, localStep, gradient],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///SpaceToBatch for N-D tensors of type T.
///This operation divides "spatial" dimensions `[1, ..., M]` of the input into a
/// grid of blocks of shape `block_shape`, and interleaves these blocks with the
/// "batch" dimension (0) such that in the output, the spatial dimensions
/// `[1, ..., M]` correspond to the position within the grid, and the batch
/// dimension combines both the position within a spatial block and the original
/// batch position.  Prior to division into blocks, the spatial dimensions of the
/// input are optionally zero padded according to `paddings`.  See below for a
/// precise description.
/// - Parameter input: N-D with shape `input_shape = [batch] + spatial_shape + remaining_shape`,
/// where spatial_shape has `M` dimensions.
/// - Parameter blockShape: 1-D with shape `[M]`, all values must be >= 1.
/// - Parameter paddings: 2-D with shape `[M, 2]`, all values must be >= 0.
///   `paddings[i] = [pad_start, pad_end]` specifies the padding for input dimension
///   `i + 1`, which corresponds to spatial dimension `i`.  It is required that
///   `block_shape[i]` divides `input_shape[i + 1] + pad_start + pad_end`.
/// 
/// This operation is equivalent to the following steps:
/// 
/// 1. Zero-pad the start and end of dimensions `[1, ..., M]` of the
///    input according to `paddings` to produce `padded` of shape `padded_shape`.
/// 
/// 2. Reshape `padded` to `reshaped_padded` of shape:
/// 
///      [batch] +
///      [padded_shape[1] / block_shape[0],
///        block_shape[0],
///       ...,
///       padded_shape[M] / block_shape[M-1],
///       block_shape[M-1]] +
///      remaining_shape
/// 
/// 3. Permute dimensions of `reshaped_padded` to produce
///    `permuted_reshaped_padded` of shape:
/// 
///      block_shape +
///      [batch] +
///      [padded_shape[1] / block_shape[0],
///       ...,
///       padded_shape[M] / block_shape[M-1]] +
///      remaining_shape
/// 
/// 4. Reshape `permuted_reshaped_padded` to flatten `block_shape` into the batch
///    dimension, producing an output tensor of shape:
/// 
///      [batch  *  prod(block_shape)] +
///      [padded_shape[1] / block_shape[0],
///       ...,
///       padded_shape[M] / block_shape[M-1]] +
///      remaining_shape
/// 
/// Some examples:
/// 
/// (1) For the following input of shape `[1, 2, 2, 1]`, `block_shape = [2, 2]`, and
///     `paddings = [[0, 0], [0, 0]]`:
/// 
/// ```
/// x = [[[[1], [2]], [[3], [4]]]]
/// ```
/// 
/// The output tensor has shape `[4, 1, 1, 1]` and value:
/// 
/// ```
/// [[[[1]]], [[[2]]], [[[3]]], [[[4]]]]
/// ```
/// 
/// (2) For the following input of shape `[1, 2, 2, 3]`, `block_shape = [2, 2]`, and
///     `paddings = [[0, 0], [0, 0]]`:
/// 
/// ```
/// x = [[[[1, 2, 3], [4, 5, 6]],
///       [[7, 8, 9], [10, 11, 12]]]]
/// ```
/// 
/// The output tensor has shape `[4, 1, 1, 3]` and value:
/// 
/// ```
/// [[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]], [[10, 11, 12]]]
/// ```
/// 
/// (3) For the following input of shape `[1, 4, 4, 1]`, `block_shape = [2, 2]`, and
///     `paddings = [[0, 0], [0, 0]]`:
/// 
/// ```
/// x = [[[[1],   [2],  [3],  [4]],
///       [[5],   [6],  [7],  [8]],
///       [[9],  [10], [11],  [12]],
///       [[13], [14], [15],  [16]]]]
/// ```
/// 
/// The output tensor has shape `[4, 2, 2, 1]` and value:
/// 
/// ```
/// x = [[[[1], [3]], [[9], [11]]],
///      [[[2], [4]], [[10], [12]]],
///      [[[5], [7]], [[13], [15]]],
///      [[[6], [8]], [[14], [16]]]]
/// ```
/// 
/// (4) For the following input of shape `[2, 2, 4, 1]`, block_shape = `[2, 2]`, and
///     paddings = `[[0, 0], [2, 0]]`:
/// 
/// ```
/// x = [[[[1],   [2],  [3],  [4]],
///       [[5],   [6],  [7],  [8]]],
///      [[[9],  [10], [11],  [12]],
///       [[13], [14], [15],  [16]]]]
/// ```
/// 
/// The output tensor has shape `[8, 1, 3, 1]` and value:
/// 
/// ```
/// x = [[[[0], [1], [3]]], [[[0], [9], [11]]],
///      [[[0], [2], [4]]], [[[0], [10], [12]]],
///      [[[0], [5], [7]]], [[[0], [13], [15]]],
///      [[[0], [6], [8]]], [[[0], [14], [16]]]]
/// ```
/// 
/// Among others, this operation is useful for reducing atrous convolution into
/// regular convolution.
/// - Parameter tblockShape: 
/// - Parameter tpaddings: 
/// - Returns: 
///	output: 
public func spaceToBatchND(operationName: String? = nil, input: Output, blockShape: Output, paddings: Output, tblockShape: Any.Type, tpaddings: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Tblock_shape"] = tblockShape
	attrs["Tpaddings"] = tpaddings
	let opspec = OpSpec(
		type: "SpaceToBatchND",
		name: (operationName ?? "Type"),
		input: [input, blockShape, paddings],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Adjust the hue of one or more images.
///`images` is a tensor of at least 3 dimensions.  The last dimension is
/// interpretted as channels, and must be three.
/// 
/// The input image is considered in the RGB colorspace. Conceptually, the RGB
/// colors are first mapped into HSV. A delta is then applied all the hue values,
/// and then remapped back to RGB colorspace.
/// - Parameter images: Images to adjust.  At least 3-D.
/// - Parameter delta: A float delta to add to the hue.
/// - Returns: 
///	output: The hue-adjusted image or images.
public func adjustHue(operationName: String? = nil, images: Output, delta: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "AdjustHue",
		name: (operationName ?? "Type"),
		input: [images, delta],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Performs max pooling on the input and outputs both max values and indices.
///The indices in `argmax` are flattened, so that a maximum value at position
/// `[b, y, x, c]` becomes flattened index
/// `((b  *  height + y)  *  width + x)  *  channels + c`.
/// 
/// The indices returned are always in `[0, height) x [0, width)` before flattening,
/// even if padding is involved and the mathematically correct answer is outside
/// (either negative or too large).  This is a bug, but fixing it is difficult to do
/// in a safe backwards compatible way, especially due to flattening.
/// - Parameter input: 4-D with shape `[batch, height, width, channels]`.  Input to pool over.
/// - Parameter ksize: The size of the window for each dimension of the input tensor.
/// - Parameter strides: The stride of the sliding window for each dimension of the
/// input tensor.
/// - Parameter targmax: 
/// - Parameter padding: The type of padding algorithm to use.
/// - Returns: 
///	output: The max pooled output tensor.
///	argmax: 4-D.  The flattened indices of the max values chosen for each output.
public func maxPoolWithArgmax(operationName: String? = nil, input: Output, ksize: [Int64], strides: [Int64], targmax: Any.Type, padding: String) throws -> (output: Output, argmax: Output) { 
	var attrs = [String : Any]()
	attrs["ksize"] = ksize
	attrs["strides"] = strides
	attrs["Targmax"] = targmax
	attrs["padding"] = padding
	let opspec = OpSpec(
		type: "MaxPoolWithArgmax",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (output: op.output(at: 0), argmax: op.output(at: 1))
} 

///Creates or finds a child frame, and makes `data` available to the child frame.
///The unique `frame_name` is used by the `Executor` to identify frames. If
/// `is_constant` is true, `output` is a constant in the child frame; otherwise
/// it may be changed in the child frame. At most `parallel_iterations` iterations
/// are run in parallel in the child frame.
/// - Parameter data: The tensor to be made available to the child frame.
/// - Parameter frameName: The name of the child frame.
/// - Parameter isConstant: If true, the output is constant within the child frame.
/// - Parameter parallelIterations: The number of iterations allowed to run in parallel.
/// - Returns: 
///	output: The same tensor as `data`.
public func refEnter(operationName: String? = nil, data: Output, frameName: String, isConstant: Bool, parallelIterations: UInt8) throws -> Output { 
	var attrs = [String : Any]()
	attrs["frame_name"] = frameName
	attrs["is_constant"] = isConstant
	attrs["parallel_iterations"] = parallelIterations
	let opspec = OpSpec(
		type: "RefEnter",
		name: (operationName ?? "Type"),
		input: [data],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///A queue that produces elements sorted by the first component value.
///Note that the PriorityQueue requires the first component of any element
/// to be a scalar int64, in addition to the other elements declared by
/// component_types.  Therefore calls to Enqueue and EnqueueMany (resp. Dequeue
/// and DequeueMany) on a PriorityQueue will all require (resp. output) one extra
/// entry in their input (resp. output) lists.
/// - Parameter componentTypes: The type of each component in a value.
/// - Parameter shapes: The shape of each component in a value. The length of this attr must
/// be either 0 or the same as the length of component_types. If the length of
/// this attr is 0, the shapes of queue elements are not constrained, and
/// only one element may be dequeued at a time.
/// - Parameter capacity: The upper bound on the number of elements in this queue.
/// Negative numbers mean no limit.
/// - Parameter container: If non-empty, this queue is placed in the given container.
/// Otherwise, a default container is used.
/// - Parameter sharedName: If non-empty, this queue will be shared under the given name
/// across multiple sessions.
/// - Returns: 
///	handle: The handle to the queue.
public func priorityQueueV2(operationName: String? = nil, componentTypes: [Any.Type], shapes: [Shape], capacity: UInt8, container: String, sharedName: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["component_types"] = componentTypes
	attrs["shapes"] = shapes
	attrs["capacity"] = capacity
	attrs["container"] = container
	attrs["shared_name"] = sharedName
	let opspec = OpSpec(
		type: "PriorityQueueV2",
		name: (operationName ?? "Type"),
		input: [],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Loads a 2-D (matrix) `Tensor` with name `old_tensor_name` from the checkpoint
///at `ckpt_path` and potentially reorders its rows and columns using the
/// specified remappings.
/// 
/// Most users should use one of the wrapper initializers (such as
/// `tf.contrib.framework.load_and_remap_matrix_initializer`) instead of this
/// function directly.
/// 
/// The remappings are 1-D tensors with the following properties:
/// 
///  *  `row_remapping` must have exactly `num_rows` entries. Row `i` of the output
///   matrix will be initialized from the row corresponding to index
///   `row_remapping[i]` in the old `Tensor` from the checkpoint.
///  *  `col_remapping` must have either 0 entries (indicating that no column
///   reordering is needed) or `num_cols` entries. If specified, column `j` of the
///   output matrix will be initialized from the column corresponding to index
///   `col_remapping[j]` in the old `Tensor` from the checkpoint.
///  *  A value of -1 in either of the remappings signifies a "missing" entry. In that
///   case, values from the `initializing_values` tensor will be used to fill that
///   missing row or column. If `row_remapping` has `r` missing entries and
///   `col_remapping` has `c` missing entries, then the following condition must be
///   true:
/// 
/// `(r  *  num_cols) + (c  *  num_rows) - (r  *  c) == len(initializing_values)`
/// 
/// The remapping tensors can be generated using the GenerateVocabRemapping op.
/// 
/// As an example, with row_remapping = [1, 0, -1], col_remapping = [0, 2, -1],
/// initializing_values = [0.5, -0.5, 0.25, -0.25, 42], and w(i, j) representing
/// the value from row i, column j of the old tensor in the checkpoint, the output
/// matrix will look like the following:
/// 
/// [[w(1, 0),  w(1, 2),  0.5],
///  [w(0, 0),  w(0, 2), -0.5],
///  [0.25,    -0.25,      42]]
/// - Parameter ckptPath: Path to the TensorFlow checkpoint (version 2, `TensorBundle`) from
/// which the old matrix `Tensor` will be loaded.
/// - Parameter oldTensorName: Name of the 2-D `Tensor` to load from checkpoint.
/// - Parameter rowRemapping: An int `Tensor` of row remappings (generally created by
/// `generate_vocab_remapping`).  Even if no row remapping is needed, this must
/// still be an index-valued Tensor (e.g. [0, 1, 2, ...]), or a shifted
/// index-valued `Tensor` (e.g. [8, 9, 10, ...], for partitioned `Variables`).
/// - Parameter colRemapping: An int `Tensor` of column remappings (generally created by
/// `generate_vocab_remapping`).  May be a size-0 `Tensor` if only row remapping
/// is to be done (e.g. column ordering is the same).
/// - Parameter initializingValues: A float `Tensor` containing  values to fill in for cells
/// in the output matrix that are not loaded from the checkpoint. Length must be
/// exactly the same as the number of missing / new cells.
/// - Parameter numRows: Number of rows (length of the 1st dimension) in the output matrix.
/// - Parameter numCols: Number of columns (length of the 2nd dimension) in the output matrix.
/// - Parameter maxRowsInMemory: The maximum number of rows to load from the checkpoint at
/// once. If less than or equal to 0, the entire matrix will be loaded into
/// memory. Setting this arg trades increased disk reads for lower memory usage.
/// - Returns: 
///	output_matrix: Output matrix containing existing values loaded from the
/// checkpoint, and with any missing values filled in from initializing_values.
public func loadAndRemapMatrix(operationName: String? = nil, ckptPath: Output, oldTensorName: Output, rowRemapping: Output, colRemapping: Output, initializingValues: Output, numRows: UInt8, numCols: UInt8, maxRowsInMemory: UInt8) throws -> Output { 
	var attrs = [String : Any]()
	attrs["num_rows"] = numRows
	attrs["num_cols"] = numCols
	attrs["max_rows_in_memory"] = maxRowsInMemory
	let opspec = OpSpec(
		type: "LoadAndRemapMatrix",
		name: (operationName ?? "Type"),
		input: [ckptPath, oldTensorName, rowRemapping, colRemapping, initializingValues],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Greedily selects a subset of bounding boxes in descending order of score,
///pruning away boxes that have high intersection-over-union (IOU) overlap
/// with previously selected boxes.  Bounding boxes are supplied as
/// [y1, x1, y2, x2], where (y1, x1) and (y2, x2) are the coordinates of any
/// diagonal pair of box corners and the coordinates can be provided as normalized
/// (i.e., lying in the interval [0, 1]) or absolute.  Note that this algorithm
/// is agnostic to where the origin is in the coordinate system.  Note that this
/// algorithm is invariant to orthogonal transformations and translations
/// of the coordinate system; thus translating or reflections of the coordinate
/// system result in the same boxes being selected by the algorithm.
/// 
/// The output of this operation is a set of integers indexing into the input
/// collection of bounding boxes representing the selected boxes.  The bounding
/// box coordinates corresponding to the selected indices can then be obtained
/// using the `tf.gather operation`.  For example:
/// 
///   selected_indices = tf.image.non_max_suppression_v2(
///       boxes, scores, max_output_size, iou_threshold)
///   selected_boxes = tf.gather(boxes, selected_indices)
/// - Parameter boxes: A 2-D float tensor of shape `[num_boxes, 4]`.
/// - Parameter scores: A 1-D float tensor of shape `[num_boxes]` representing a single
/// score corresponding to each box (each row of boxes).
/// - Parameter maxOutputSize: A scalar integer tensor representing the maximum number of
/// boxes to be selected by non max suppression.
/// - Parameter iouThreshold: A 0-D float tensor representing the threshold for deciding whether
/// boxes overlap too much with respect to IOU.
/// - Returns: 
///	selected_indices: A 1-D integer tensor of shape `[M]` representing the selected
/// indices from the boxes tensor, where `M <= max_output_size`.
public func nonMaxSuppressionV2(operationName: String? = nil, boxes: Output, scores: Output, maxOutputSize: Output, iouThreshold: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "NonMaxSuppressionV2",
		name: (operationName ?? "Type"),
		input: [boxes, scores, maxOutputSize, iouThreshold],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 


/// - Parameter handle: 
/// - Parameter value: 
/// - Parameter lengths: 
/// - Parameter flowIn: 
/// - Returns: 
///	flow_out: 
public func tensorArraySplit(operationName: String? = nil, handle: Output, value: Output, lengths: Output, flowIn: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "TensorArraySplit",
		name: (operationName ?? "Type"),
		input: [handle, value, lengths, flowIn],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Bucketizes 'input' based on 'boundaries'.
///For example, if the inputs are
///     boundaries = [0, 10, 100]
///     input = [[-5, 10000]
///              [150,   10]
///              [5,    100]]
/// 
/// then the output will be
///     output = [[0, 3]
///               [3, 2]
///               [1, 3]]
/// - Parameter input: Any shape of Tensor contains with int or float type.
/// - Parameter boundaries: A sorted list of floats gives the boundary of the buckets.
/// - Returns: 
///	output: Same shape with 'input', each value of input replaced with bucket index.
/// 
/// @compatibility(numpy)
/// Equivalent to np.digitize.
/// @end_compatibility
public func bucketize(operationName: String? = nil, input: Output, boundaries: [Float]) throws -> Output { 
	var attrs = [String : Any]()
	attrs["boundaries"] = boundaries
	let opspec = OpSpec(
		type: "Bucketize",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Dequantize the 'input' tensor into a float Tensor.
///[min_range, max_range] are scalar floats that specify the range for
/// the 'input' data. The 'mode' attribute controls exactly which calculations are
/// used to convert the float values to their quantized equivalents.
/// 
/// In 'MIN_COMBINED' mode, each value of the tensor will undergo the following:
/// 
/// ```
/// if T == qint8, in[i] += (range(T) + 1)/ 2.0
/// out[i] = min_range + (in[i] *  (max_range - min_range) / range(T))
/// ```
/// here `range(T) = numeric_limits<T>::max() - numeric_limits<T>::min()`
/// 
///  * MIN_COMBINED Mode Example * 
/// 
/// If the input comes from a QuantizedRelu6, the output type is
/// quint8 (range of 0-255) but the possible range of QuantizedRelu6 is
/// 0-6.  The min_range and max_range values are therefore 0.0 and 6.0.
/// Dequantize on quint8 will take each value, cast to float, and multiply
/// by 6 / 255.
/// Note that if quantizedtype is qint8, the operation will additionally add
/// each value by 128 prior to casting.
/// 
/// If the mode is 'MIN_FIRST', then this approach is used:
/// 
/// ```c++
/// number_of_steps = 1 << (# of bits in T)
/// range_adjust = number_of_steps / (number_of_steps - 1)
/// range = (range_max - range_min)  *  range_adjust
/// range_scale = range / number_of_steps
/// const double offset_input = static_cast<double>(input) - lowest_quantized;
/// result = range_min + ((input - numeric_limits<T>::min())  *  range_scale)
/// ```
/// 
///  * SCALED mode Example * 
/// 
/// `SCALED` mode matches the quantization approach used in
/// `QuantizeAndDequantize{V2|V3}`.
/// 
/// If the mode is `SCALED`, we do not use the full range of the output type,
/// choosing to elide the lowest possible value for symmetry (e.g., output range is
/// -127 to 127, not -128 to 127 for signed 8 bit quantization), so that 0.0 maps to
/// 0.
/// 
/// We first find the range of values in our tensor. The
/// range we use is always centered on 0, so we find m such that
/// ```c++
///   m = max(abs(input_min), abs(input_max))
/// ```
/// 
/// Our input tensor range is then `[-m, m]`.
/// 
/// Next, we choose our fixed-point quantization buckets, `[min_fixed, max_fixed]`.
/// If T is signed, this is
/// ```
///   num_bits = sizeof(T)  *  8
///   [min_fixed, max_fixed] =
///       [-(1 << (num_bits - 1) - 1), (1 << (num_bits - 1)) - 1]
/// ```
/// 
/// Otherwise, if T is unsigned, the fixed-point range is
/// ```
///   [min_fixed, max_fixed] = [0, (1 << num_bits) - 1]
/// ```
/// 
/// From this we compute our scaling factor, s:
/// ```c++
///   s = (2  *  m) / (max_fixed - min_fixed)
/// ```
/// 
/// Now we can dequantize the elements of our tensor:
/// ```c++
/// result = input  *  s
/// ```
/// - Parameter input: 
/// - Parameter minRange: The minimum scalar value possibly produced for the input.
/// - Parameter maxRange: The maximum scalar value possibly produced for the input.
/// - Parameter mode: 
/// - Returns: 
///	output: 
public func dequantize(operationName: String? = nil, input: Output, minRange: Output, maxRange: Output, mode: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["mode"] = mode
	let opspec = OpSpec(
		type: "Dequantize",
		name: (operationName ?? "Type"),
		input: [input, minRange, maxRange],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Draw bounding boxes on a batch of images.
///Outputs a copy of `images` but draws on top of the pixels zero or more bounding
/// boxes specified by the locations in `boxes`. The coordinates of the each
/// bounding box in `boxes` are encoded as `[y_min, x_min, y_max, x_max]`. The
/// bounding box coordinates are floats in `[0.0, 1.0]` relative to the width and
/// height of the underlying image.
/// 
/// For example, if an image is 100 x 200 pixels (height x width) and the bounding
/// box is `[0.1, 0.2, 0.5, 0.9]`, the upper-left and bottom-right coordinates of
/// the bounding box will be `(40, 10)` to `(100, 50)` (in (x,y) coordinates).
/// 
/// Parts of the bounding box may fall outside the image.
/// - Parameter images: 4-D with shape `[batch, height, width, depth]`. A batch of images.
/// - Parameter boxes: 3-D with shape `[batch, num_bounding_boxes, 4]` containing bounding
/// boxes.
/// - Returns: 
///	output: 4-D with the same shape as `images`. The batch of input images with
/// bounding boxes drawn on the images.
public func drawBoundingBoxes(operationName: String? = nil, images: Output, boxes: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "DrawBoundingBoxes",
		name: (operationName ?? "Type"),
		input: [images, boxes],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes the gradient of nearest neighbor interpolation.
/// - Parameter grads: 4-D with shape `[batch, height, width, channels]`.
/// - Parameter size: = A 1-D int32 Tensor of 2 elements: `orig_height, orig_width`. The
/// original input size.
/// - Parameter alignCorners: If true, rescale grads by (orig_height - 1) / (height - 1), which
/// exactly aligns the 4 corners of grads and original_image. If false, rescale by
/// orig_height / height. Treat similarly the width dimension.
/// - Returns: 
///	output: 4-D with shape `[batch, orig_height, orig_width, channels]`. Gradients
/// with respect to the input image.
public func resizeNearestNeighborGrad(operationName: String? = nil, grads: Output, size: Output, alignCorners: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["align_corners"] = alignCorners
	let opspec = OpSpec(
		type: "ResizeNearestNeighborGrad",
		name: (operationName ?? "Type"),
		input: [grads, size],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Returns x  *  y element-wise, working on quantized buffers.
/// - Parameter x: 
/// - Parameter y: 
/// - Parameter minX: The float value that the lowest quantized `x` value represents.
/// - Parameter maxX: The float value that the highest quantized `x` value represents.
/// - Parameter minY: The float value that the lowest quantized `y` value represents.
/// - Parameter maxY: The float value that the highest quantized `y` value represents.
/// - Parameter t1: 
/// - Parameter t2: 
/// - Parameter toutput: 
/// - Returns: 
///	z: 
///	min_z: The float value that the lowest quantized output value represents.
///	max_z: The float value that the highest quantized output value represents.
/// 
///  * NOTE * : `QuantizedMul` supports limited forms of broadcasting. More about
/// broadcasting [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
public func quantizedMul(operationName: String? = nil, x: Output, y: Output, minX: Output, maxX: Output, minY: Output, maxY: Output, t1: Any.Type, t2: Any.Type, toutput: Any.Type) throws -> (z: Output, minZ: Output, maxZ: Output) { 
	var attrs = [String : Any]()
	attrs["T1"] = t1
	attrs["T2"] = t2
	attrs["Toutput"] = toutput
	let opspec = OpSpec(
		type: "QuantizedMul",
		name: (operationName ?? "Type"),
		input: [x, y, minX, maxX, minY, maxY],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (z: op.output(at: 0), minZ: op.output(at: 1), maxZ: op.output(at: 2))
} 

///Generates labels for candidate sampling with a learned unigram distribution.
///See explanations of candidate sampling and the data formats at
/// go/candidate-sampling.
/// 
/// For each batch, this op picks a single set of sampled candidate labels.
/// 
/// The advantages of sampling candidates per-batch are simplicity and the
/// possibility of efficient dense matrix multiplication. The disadvantage is that
/// the sampled candidates must be chosen independently of the context and of the
/// true labels.
/// - Parameter trueClasses: A batch_size  *  num_true matrix, in which each row contains the
/// IDs of the num_true target_classes in the corresponding original label.
/// - Parameter numTrue: Number of true labels per context.
/// - Parameter numSampled: Number of candidates to produce.
/// - Parameter unique: If unique is true, we sample with rejection, so that all sampled
/// candidates in a batch are unique. This requires some approximation to
/// estimate the post-rejection sampling probabilities.
/// - Parameter seed: If either seed or seed2 are set to be non-zero, the random number
/// generator is seeded by the given seed.  Otherwise, it is seeded by a
/// random seed.
/// - Parameter seed2: An second seed to avoid seed collision.
/// - Returns: 
///	sampled_candidates: A vector of length num_sampled, in which each element is
/// the ID of a sampled candidate.
///	true_expected_count: A batch_size  *  num_true matrix, representing
/// the number of times each candidate is expected to occur in a batch
/// of sampled candidates. If unique=true, then this is a probability.
///	sampled_expected_count: A vector of length num_sampled, for each sampled
/// candidate representing the number of times the candidate is expected
/// to occur in a batch of sampled candidates.  If unique=true, then this is a
/// probability.
public func allCandidateSampler(operationName: String? = nil, trueClasses: Output, numTrue: UInt8, numSampled: UInt8, unique: Bool, seed: UInt8, seed2: UInt8) throws -> (sampledCandidates: Output, trueExpectedCount: Output, sampledExpectedCount: Output) { 
	var attrs = [String : Any]()
	attrs["num_true"] = numTrue
	attrs["num_sampled"] = numSampled
	attrs["unique"] = unique
	attrs["seed"] = seed
	attrs["seed2"] = seed2
	let opspec = OpSpec(
		type: "AllCandidateSampler",
		name: (operationName ?? "Type"),
		input: [trueClasses],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (sampledCandidates: op.output(at: 0), trueExpectedCount: op.output(at: 1), sampledExpectedCount: op.output(at: 2))
} 

///Enqueues a tuple of one or more tensors in the given queue.
///The components input has k elements, which correspond to the components of
/// tuples stored in the given queue.
/// 
/// N.B. If the queue is full, this operation will block until the given
/// element has been enqueued (or 'timeout_ms' elapses, if specified).
/// - Parameter handle: The handle to a queue.
/// - Parameter components: One or more tensors from which the enqueued tensors should be taken.
/// - Parameter tcomponents: 
/// - Parameter timeoutMs: If the queue is full, this operation will block for up to
/// timeout_ms milliseconds.
/// Note: This option is not supported yet.
public func queueEnqueueV2(operationName: String? = nil, handle: Output, components: Output, tcomponents: [Any.Type], timeoutMs: UInt8) throws -> Operation { 
	var attrs = [String : Any]()
	attrs["Tcomponents"] = tcomponents
	attrs["timeout_ms"] = timeoutMs
	let opspec = OpSpec(
		type: "QueueEnqueueV2",
		name: (operationName ?? "Type"),
		input: [handle, components],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///A queue that randomizes the order of elements.
/// - Parameter componentTypes: The type of each component in a value.
/// - Parameter shapes: The shape of each component in a value. The length of this attr must
/// be either 0 or the same as the length of component_types. If the length of
/// this attr is 0, the shapes of queue elements are not constrained, and
/// only one element may be dequeued at a time.
/// - Parameter capacity: The upper bound on the number of elements in this queue.
/// Negative numbers mean no limit.
/// - Parameter minAfterDequeue: Dequeue will block unless there would be this
/// many elements after the dequeue or the queue is closed. This
/// ensures a minimum level of mixing of elements.
/// - Parameter seed: If either seed or seed2 is set to be non-zero, the random number
/// generator is seeded by the given seed.  Otherwise, a random seed is used.
/// - Parameter seed2: A second seed to avoid seed collision.
/// - Parameter container: If non-empty, this queue is placed in the given container.
/// Otherwise, a default container is used.
/// - Parameter sharedName: If non-empty, this queue will be shared under the given name
/// across multiple sessions.
/// - Returns: 
///	handle: The handle to the queue.
public func randomShuffleQueueV2(operationName: String? = nil, componentTypes: [Any.Type], shapes: [Shape], capacity: UInt8, minAfterDequeue: UInt8, seed: UInt8, seed2: UInt8, container: String, sharedName: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["component_types"] = componentTypes
	attrs["shapes"] = shapes
	attrs["capacity"] = capacity
	attrs["min_after_dequeue"] = minAfterDequeue
	attrs["seed"] = seed
	attrs["seed2"] = seed2
	attrs["container"] = container
	attrs["shared_name"] = sharedName
	let opspec = OpSpec(
		type: "RandomShuffleQueueV2",
		name: (operationName ?? "Type"),
		input: [],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Forwards the value of an available tensor from `inputs` to `output`.
///`Merge` waits for at least one of the tensors in `inputs` to become available.
/// It is usually combined with `Switch` to implement branching.
/// 
/// `Merge` forwards the first tensor for become available to `output`, and sets
/// `value_index` to its index in `inputs`.
/// - Parameter inputs: The input tensors, exactly one of which will become available.
/// - Parameter n: 
/// - Returns: 
///	output: Will be set to the available input tensor.
///	value_index: The index of the chosen input tensor in `inputs`.
public func refMerge(operationName: String? = nil, inputs: [Output], n: UInt8) throws -> (output: Output, valueIndex: Output) { 
	var attrs = [String : Any]()
	attrs["N"] = n
	let opspec = OpSpec(
		type: "RefMerge",
		name: (operationName ?? "Type"),
		input: [inputs],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (output: op.output(at: 0), valueIndex: op.output(at: 1))
} 

///Forwards the value of an available tensor from `inputs` to `output`.
///`Merge` waits for at least one of the tensors in `inputs` to become available.
/// It is usually combined with `Switch` to implement branching.
/// 
/// `Merge` forwards the first tensor to become available to `output`, and sets
/// `value_index` to its index in `inputs`.
/// - Parameter inputs: The input tensors, exactly one of which will become available.
/// - Parameter n: 
/// - Returns: 
///	output: Will be set to the available input tensor.
///	value_index: The index of the chosen input tensor in `inputs`.
public func merge(operationName: String? = nil, inputs: [Output], n: UInt8) throws -> (output: Output, valueIndex: Output) { 
	var attrs = [String : Any]()
	attrs["N"] = n
	let opspec = OpSpec(
		type: "Merge",
		name: (operationName ?? "Type"),
		input: [inputs],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (output: op.output(at: 0), valueIndex: op.output(at: 1))
} 

///Creates a dataset that batches `batch_size` elements from `input_dataset`.
/// - Parameter inputDataset: 
/// - Parameter batchSize: A scalar representing the number of elements to accumulate in a
/// batch.
/// - Parameter outputTypes: 
/// - Parameter outputShapes: 
/// - Returns: 
///	handle: 
public func batchDataset(operationName: String? = nil, inputDataset: Output, batchSize: Output, outputTypes: [Any.Type], outputShapes: [Shape]) throws -> Output { 
	var attrs = [String : Any]()
	attrs["output_types"] = outputTypes
	attrs["output_shapes"] = outputShapes
	let opspec = OpSpec(
		type: "BatchDataset",
		name: (operationName ?? "Type"),
		input: [inputDataset, batchSize],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes the sum along sparse segments of a tensor.
///Read @{$math_ops#segmentation$the section on segmentation} for an explanation of
/// segments.
/// 
/// Like `SegmentSum`, but `segment_ids` can have rank less than `data`'s first
/// dimension, selecting a subset of dimension 0, specified by `indices`.
/// 
/// For example:
/// 
/// ```python
/// c = tf.constant([[1,2,3,4], [-1,-2,-3,-4], [5,6,7,8]])
/// 
/// # Select two rows, one segment.
/// tf.sparse_segment_sum(c, tf.constant([0, 1]), tf.constant([0, 0]))
/// # => [[0 0 0 0]]
/// 
/// # Select two rows, two segment.
/// tf.sparse_segment_sum(c, tf.constant([0, 1]), tf.constant([0, 1]))
/// # => [[ 1  2  3  4]
/// #     [-1 -2 -3 -4]]
/// 
/// # Select all rows, two segments.
/// tf.sparse_segment_sum(c, tf.constant([0, 1, 2]), tf.constant([0, 0, 1]))
/// # => [[0 0 0 0]
/// #     [5 6 7 8]]
/// 
/// # Which is equivalent to:
/// tf.segment_sum(c, tf.constant([0, 0, 1]))
/// ```
/// - Parameter data: 
/// - Parameter indices: A 1-D tensor. Has same rank as `segment_ids`.
/// - Parameter segmentIds: A 1-D tensor. Values should be sorted and can be repeated.
/// - Parameter tidx: 
/// - Returns: 
///	output: Has same shape as data, except for dimension 0 which
/// has size `k`, the number of segments.
public func sparseSegmentSum(operationName: String? = nil, data: Output, indices: Output, segmentIds: Output, tidx: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Tidx"] = tidx
	let opspec = OpSpec(
		type: "SparseSegmentSum",
		name: (operationName ?? "Type"),
		input: [data, indices, segmentIds],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Creates a dataset that emits the outputs of `input_dataset` `count` times.
/// - Parameter inputDataset: 
/// - Parameter count: A scalar representing the number of times that `input_dataset` should
/// be repeated. A value of `-1` indicates that it should be repeated infinitely.
/// - Parameter outputTypes: 
/// - Parameter outputShapes: 
/// - Returns: 
///	handle: 
public func repeatDataset(operationName: String? = nil, inputDataset: Output, count: Output, outputTypes: [Any.Type], outputShapes: [Shape]) throws -> Output { 
	var attrs = [String : Any]()
	attrs["output_types"] = outputTypes
	attrs["output_shapes"] = outputShapes
	let opspec = OpSpec(
		type: "RepeatDataset",
		name: (operationName ?? "Type"),
		input: [inputDataset, count],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Dequeues `n` tuples of one or more tensors from the given queue.
///If the queue is closed and there are fewer than `n` elements, then an
/// OutOfRange error is returned.
/// 
/// This operation concatenates queue-element component tensors along the
/// 0th dimension to make a single component tensor.  All of the components
/// in the dequeued tuple will have size `n` in the 0th dimension.
/// 
/// This operation has `k` outputs, where `k` is the number of components in
/// the tuples stored in the given queue, and output `i` is the ith
/// component of the dequeued tuple.
/// 
/// N.B. If the queue is empty, this operation will block until `n` elements
/// have been dequeued (or 'timeout_ms' elapses, if specified).
/// - Parameter handle: The handle to a queue.
/// - Parameter n: The number of tuples to dequeue.
/// - Parameter componentTypes: The type of each component in a tuple.
/// - Parameter timeoutMs: If the queue has fewer than n elements, this operation
/// will block for up to timeout_ms milliseconds.
/// Note: This option is not supported yet.
/// - Returns: 
///	components: One or more tensors that were dequeued as a tuple.
public func queueDequeueManyV2(operationName: String? = nil, handle: Output, n: Output, componentTypes: [Any.Type], timeoutMs: UInt8) throws -> Output { 
	var attrs = [String : Any]()
	attrs["component_types"] = componentTypes
	attrs["timeout_ms"] = timeoutMs
	let opspec = OpSpec(
		type: "QueueDequeueManyV2",
		name: (operationName ?? "Type"),
		input: [handle, n],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Fake-quantize the 'inputs' tensor of type float via global float scalars `min`
///and `max` to 'outputs' tensor of same shape as `inputs`.
/// 
/// `[min; max]` define the clamping range for the `inputs` data.
/// `inputs` values are quantized into the quantization range (`[0; 2// ^num_bits - 1]`
/// when `narrow_range` is false and `[1; 2// ^num_bits - 1]` when it is true) and
/// then de-quantized and output as floats in `[min; max]` interval.
/// `num_bits` is the bitwidth of the quantization; between 2 and 8, inclusive.
/// 
/// This operation has a gradient and thus allows for training `min` and `max`
/// values.
/// - Parameter inputs: 
/// - Parameter min: 
/// - Parameter max: 
/// - Parameter numBits: 
/// - Parameter narrowRange: 
/// - Returns: 
///	outputs: 
public func fakeQuantWithMinMaxVars(operationName: String? = nil, inputs: Output, min: Output, max: Output, numBits: UInt8, narrowRange: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["num_bits"] = numBits
	attrs["narrow_range"] = narrowRange
	let opspec = OpSpec(
		type: "FakeQuantWithMinMaxVars",
		name: (operationName ?? "Type"),
		input: [inputs, min, max],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes the number of incomplete elements in the given barrier.
/// - Parameter handle: The handle to a barrier.
/// - Returns: 
///	size: The number of incomplete elements (i.e. those with some of their value
/// components not set) in the barrier.
public func barrierIncompleteSize(operationName: String? = nil, handle: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "BarrierIncompleteSize",
		name: (operationName ?? "Type"),
		input: [handle],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Returns the truth value of NOT x element-wise.
/// - Parameter x: 
/// - Returns: 
///	y: 
public func logicalNot(operationName: String? = nil, x: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "LogicalNot",
		name: (operationName ?? "Type"),
		input: [x],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Update relevant entries in ' * var' and ' * accum' according to the adagrad scheme.
///That is for rows we have grad for, we update var and accum as follows:
/// accum += grad  *  grad
/// var -= lr  *  grad  *  (1 / sqrt(accum))
/// - Parameter `var`: Should be from a Variable().
/// - Parameter accum: Should be from a Variable().
/// - Parameter lr: Learning rate. Must be a scalar.
/// - Parameter grad: The gradient.
/// - Parameter indices: A vector of indices into the first dimension of var and accum.
/// - Parameter tindices: 
/// - Parameter useLocking: If `True`, updating of the var and accum tensors will be protected
/// by a lock; otherwise the behavior is undefined, but may exhibit less
/// contention.
/// - Returns: 
///	out: Same as "var".
public func sparseApplyAdagrad(operationName: String? = nil, `var`: Output, accum: Output, lr: Output, grad: Output, indices: Output, tindices: Any.Type, useLocking: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Tindices"] = tindices
	attrs["use_locking"] = useLocking
	let opspec = OpSpec(
		type: "SparseApplyAdagrad",
		name: (operationName ?? "Type"),
		input: [`var`, accum, lr, grad, indices],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes the number of elements in the given queue.
/// - Parameter handle: The handle to a queue.
/// - Returns: 
///	size: The number of elements in the given queue.
public func queueSize(operationName: String? = nil, handle: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "QueueSize",
		name: (operationName ?? "Type"),
		input: [handle],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Distributed version of Stochastic Dual Coordinate Ascent (SDCA) optimizer for
///linear models with L1 + L2 regularization. As global optimization objective is
/// strongly-convex, the optimizer optimizes the dual objective at each step. The
/// optimizer applies each update one example at a time. Examples are sampled
/// uniformly, and the optimizer is learning rate free and enjoys linear convergence
/// rate.
/// 
/// [Proximal Stochastic Dual Coordinate Ascent](http://arxiv.org/pdf/1211.2717v1.pdf).<br>
/// Shai Shalev-Shwartz, Tong Zhang. 2012
/// 
/// $$Loss Objective = \sum f_{i} (wx_{i}) + (l2 / 2)  *  |w|// ^2 + l1  *  |w|$$
/// 
/// [Adding vs. Averaging in Distributed Primal-Dual Optimization](http://arxiv.org/abs/1502.03508).<br>
/// Chenxin Ma, Virginia Smith, Martin Jaggi, Michael I. Jordan,
/// Peter Richtarik, Martin Takac. 2015
/// 
/// [Stochastic Dual Coordinate Ascent with Adaptive Probabilities](https://arxiv.org/abs/1502.08053).<br>
/// Dominik Csiba, Zheng Qu, Peter Richtarik. 2015
/// - Parameter sparseExampleIndices: a list of vectors which contain example indices.
/// - Parameter sparseFeatureIndices: a list of vectors which contain feature indices.
/// - Parameter sparseFeatureValues: a list of vectors which contains feature value
/// associated with each feature group.
/// - Parameter denseFeatures: a list of matrices which contains the dense feature values.
/// - Parameter exampleWeights: a vector which contains the weight associated with each
/// example.
/// - Parameter exampleLabels: a vector which contains the label/target associated with each
/// example.
/// - Parameter sparseIndices: a list of vectors where each value is the indices which has
/// corresponding weights in sparse_weights. This field maybe omitted for the
/// dense approach.
/// - Parameter sparseWeights: a list of vectors where each value is the weight associated with
/// a sparse feature group.
/// - Parameter denseWeights: a list of vectors where the values are the weights associated
/// with a dense feature group.
/// - Parameter exampleStateData: a list of vectors containing the example state data.
/// - Parameter lossType: Type of the primal loss. Currently SdcaSolver supports logistic,
/// squared and hinge losses.
/// - Parameter adaptative: Whether to use Adapative SDCA for the inner loop.
/// - Parameter numSparseFeatures: Number of sparse feature groups to train on.
/// - Parameter numSparseFeaturesWithValues: Number of sparse feature groups with values
/// associated with it, otherwise implicitly treats values as 1.0.
/// - Parameter numDenseFeatures: Number of dense feature groups to train on.
/// - Parameter l1: Symmetric l1 regularization strength.
/// - Parameter l2: Symmetric l2 regularization strength.
/// - Parameter numLossPartitions: Number of partitions of the global loss function.
/// - Parameter numInnerIterations: Number of iterations per mini-batch.
/// - Returns: 
///	out_example_state_data: a list of vectors containing the updated example state
/// data.
///	out_delta_sparse_weights: a list of vectors where each value is the delta
/// weights associated with a sparse feature group.
///	out_delta_dense_weights: a list of vectors where the values are the delta
/// weights associated with a dense feature group.
public func sdcaOptimizer(operationName: String? = nil, sparseExampleIndices: Output, sparseFeatureIndices: Output, sparseFeatureValues: Output, denseFeatures: Output, exampleWeights: Output, exampleLabels: Output, sparseIndices: Output, sparseWeights: Output, denseWeights: Output, exampleStateData: Output, lossType: String, adaptative: Bool, numSparseFeatures: UInt8, numSparseFeaturesWithValues: UInt8, numDenseFeatures: UInt8, l1: Float, l2: Float, numLossPartitions: UInt8, numInnerIterations: UInt8) throws -> (outExampleStateData: Output, outDeltaSparseWeights: Output, outDeltaDenseWeights: Output) { 
	var attrs = [String : Any]()
	attrs["loss_type"] = lossType
	attrs["adaptative"] = adaptative
	attrs["num_sparse_features"] = numSparseFeatures
	attrs["num_sparse_features_with_values"] = numSparseFeaturesWithValues
	attrs["num_dense_features"] = numDenseFeatures
	attrs["l1"] = l1
	attrs["l2"] = l2
	attrs["num_loss_partitions"] = numLossPartitions
	attrs["num_inner_iterations"] = numInnerIterations
	let opspec = OpSpec(
		type: "SdcaOptimizer",
		name: (operationName ?? "Type"),
		input: [sparseExampleIndices, sparseFeatureIndices, sparseFeatureValues, denseFeatures, exampleWeights, exampleLabels, sparseIndices, sparseWeights, denseWeights, exampleStateData],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (outExampleStateData: op.output(at: 0), outDeltaSparseWeights: op.output(at: 1), outDeltaDenseWeights: op.output(at: 2))
} 

///Inverse fast Fourier transform.
///Computes the inverse 1-dimensional discrete Fourier transform over the
/// inner-most dimension of `input`.
/// - Parameter input: A complex64 tensor.
/// - Returns: 
///	output: A complex64 tensor of the same shape as `input`. The inner-most
///   dimension of `input` is replaced with its inverse 1D Fourier transform.
/// 
/// @compatibility(numpy)
/// Equivalent to np.fft.ifft
/// @end_compatibility
public func ifft(operationName: String? = nil, input: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "IFFT",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes atan of x element-wise.
/// - Parameter x: 
/// - Returns: 
///	y: 
public func atan(operationName: String? = nil, x: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Atan",
		name: (operationName ?? "Type"),
		input: [x],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Does nothing. Serves as a control trigger for scheduling.
///Only useful as a placeholder for control edges.
public func controlTrigger(operationName: String? = nil) throws -> Operation { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "ControlTrigger",
		name: (operationName ?? "Type"),
		input: [],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Computes numerical negative value element-wise.
///I.e., \\(y = -x\\).
/// - Parameter x: 
/// - Returns: 
///	y: 
public func neg(operationName: String? = nil, x: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Neg",
		name: (operationName ?? "Type"),
		input: [x],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Compute gradients for a FakeQuantWithMinMaxArgs operation.
/// - Parameter gradients: Backpropagated gradients above the FakeQuantWithMinMaxArgs operation.
/// - Parameter inputs: Values passed as inputs to the FakeQuantWithMinMaxArgs operation.
/// - Parameter min: 
/// - Parameter max: 
/// - Parameter numBits: 
/// - Parameter narrowRange: 
/// - Returns: 
///	backprops: Backpropagated gradients below the FakeQuantWithMinMaxArgs operation:
/// `gradients  *  (inputs >= min && inputs <= max)`.
public func fakeQuantWithMinMaxArgsGradient(operationName: String? = nil, gradients: Output, inputs: Output, min: Float, max: Float, numBits: UInt8, narrowRange: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["min"] = min
	attrs["max"] = max
	attrs["num_bits"] = numBits
	attrs["narrow_range"] = narrowRange
	let opspec = OpSpec(
		type: "FakeQuantWithMinMaxArgsGradient",
		name: (operationName ?? "Type"),
		input: [gradients, inputs],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Outputs a `Summary` protocol buffer with scalar values.
///The input `tags` and `values` must have the same shape.  The generated summary
/// has a summary value for each tag-value pair in `tags` and `values`.
/// - Parameter tags: Tags for the summary.
/// - Parameter values: Same shape as `tags.  Values for the summary.
/// - Returns: 
///	summary: Scalar.  Serialized `Summary` protocol buffer.
public func scalarSummary(operationName: String? = nil, tags: Output, values: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "ScalarSummary",
		name: (operationName ?? "Type"),
		input: [tags, values],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Reads and outputs the entire contents of the input filename.
/// - Parameter filename: 
/// - Returns: 
///	contents: 
public func readFile(operationName: String? = nil, filename: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "ReadFile",
		name: (operationName ?? "Type"),
		input: [filename],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes the power of one value to another.
///Given a tensor `x` and a tensor `y`, this operation computes \\(x// ^y\\) for
/// corresponding elements in `x` and `y`. For example:
/// 
/// ```
/// # tensor 'x' is [[2, 2]], [3, 3]]
/// # tensor 'y' is [[8, 16], [2, 3]]
/// tf.pow(x, y) ==> [[256, 65536], [9, 27]]
/// ```
/// - Parameter x: 
/// - Parameter y: 
/// - Returns: 
///	z: 
public func pow(operationName: String? = nil, x: Output, y: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Pow",
		name: (operationName ?? "Type"),
		input: [x, y],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Forwards the input to the output.
///This operator represents the loop termination condition used by the
/// "pivot" switches of a loop.
/// - Parameter input: A boolean scalar, representing the branch predicate of the Switch op.
/// - Returns: 
///	output: The same tensor as `input`.
public func loopCond(operationName: String? = nil, input: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "LoopCond",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Exits the current frame to its parent frame.
///Exit makes its input `data` available to the parent frame.
/// - Parameter data: The tensor to be made available to the parent frame.
/// - Returns: 
///	output: The same tensor as `data`.
public func exit(operationName: String? = nil, data: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Exit",
		name: (operationName ?? "Type"),
		input: [data],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Updates the accumulator with a new value for global_step.
///Logs warning if the accumulator's value is already higher than
/// new_global_step.
/// - Parameter handle: The handle to an accumulator.
/// - Parameter newGlobalStep: The new global_step value to set.
public func accumulatorSetGlobalStep(operationName: String? = nil, handle: Output, newGlobalStep: Output) throws -> Operation { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "AccumulatorSetGlobalStep",
		name: (operationName ?? "Type"),
		input: [handle, newGlobalStep],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Computes the gradients of depthwise convolution with respect to the input.
/// - Parameter inputSizes: An integer vector representing the shape of `input`, based
/// on `data_format`.  For example, if `data_format` is 'NHWC' then
///  `input` is a 4-D `[batch, height, width, channels]` tensor.
/// - Parameter filter: 4-D with shape
/// `[filter_height, filter_width, in_channels, depthwise_multiplier]`.
/// - Parameter outBackprop: 4-D with shape  based on `data_format`.
/// For example, if `data_format` is 'NHWC' then
/// out_backprop shape is `[batch, out_height, out_width, out_channels]`.
/// Gradients w.r.t. the output of the convolution.
/// - Parameter strides: The stride of the sliding window for each dimension of the input
/// of the convolution.
/// - Parameter padding: The type of padding algorithm to use.
/// - Parameter dataFormat: Specify the data format of the input and output data. With the
/// default format "NHWC", the data is stored in the order of:
///     [batch, height, width, channels].
/// Alternatively, the format could be "NCHW", the data storage order of:
///     [batch, channels, height, width].
/// - Returns: 
///	output: 4-D with shape according to `data_format`.  For example, if
/// `data_format` is 'NHWC', output shape is `[batch, in_height,
/// in_width, in_channels]`.  Gradient w.r.t. the input of the
/// convolution.
public func depthwiseConv2dNativeBackpropInput(operationName: String? = nil, inputSizes: Output, filter: Output, outBackprop: Output, strides: [Int64], padding: String, dataFormat: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["strides"] = strides
	attrs["padding"] = padding
	attrs["data_format"] = dataFormat
	let opspec = OpSpec(
		type: "DepthwiseConv2dNativeBackpropInput",
		name: (operationName ?? "Type"),
		input: [inputSizes, filter, outBackprop],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Returns which elements of x are NaN.
///@compatibility(numpy)
/// Equivalent to np.isnan
/// @end_compatibility
/// - Parameter x: 
/// - Returns: 
///	y: 
public func isNan(operationName: String? = nil, x: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "IsNan",
		name: (operationName ?? "Type"),
		input: [x],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes the gradient of bicubic interpolation.
/// - Parameter grads: 4-D with shape `[batch, height, width, channels]`.
/// - Parameter originalImage: 4-D with shape `[batch, orig_height, orig_width, channels]`,
/// The image tensor that was resized.
/// - Parameter alignCorners: If true, rescale grads by (orig_height - 1) / (height - 1), which
/// exactly aligns the 4 corners of grads and original_image. If false, rescale by
/// orig_height / height. Treat similarly the width dimension.
/// - Returns: 
///	output: 4-D with shape `[batch, orig_height, orig_width, channels]`.
/// Gradients with respect to the input image. Input image must have been
/// float or double.
public func resizeBicubicGrad(operationName: String? = nil, grads: Output, originalImage: Output, alignCorners: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["align_corners"] = alignCorners
	let opspec = OpSpec(
		type: "ResizeBicubicGrad",
		name: (operationName ?? "Type"),
		input: [grads, originalImage],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Compute the cumulative product of the tensor `x` along `axis`.
///By default, this op performs an inclusive cumprod, which means that the first
/// element of the input is identical to the first element of the output:
/// 
/// ```python
/// tf.cumprod([a, b, c])  # => [a, a  *  b, a  *  b  *  c]
/// ```
/// 
/// By setting the `exclusive` kwarg to `True`, an exclusive cumprod is
/// performed instead:
/// 
/// ```python
/// tf.cumprod([a, b, c], exclusive=True)  # => [1, a, a  *  b]
/// ```
/// 
/// By setting the `reverse` kwarg to `True`, the cumprod is performed in the
/// opposite direction:
/// 
/// ```python
/// tf.cumprod([a, b, c], reverse=True)  # => [a  *  b  *  c, b  *  c, c]
/// ```
/// 
/// This is more efficient than using separate `tf.reverse` ops.
/// 
/// The `reverse` and `exclusive` kwargs can also be combined:
/// 
/// ```python
/// tf.cumprod([a, b, c], exclusive=True, reverse=True)  # => [b  *  c, c, 1]
/// ```
/// - Parameter x: A `Tensor`. Must be one of the following types: `float32`, `float64`,
/// `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`,
/// `complex128`, `qint8`, `quint8`, `qint32`, `half`.
/// - Parameter axis: A `Tensor` of type `int32` (default: 0). Must be in the range
/// `[-rank(x), rank(x))`.
/// - Parameter exclusive: If `True`, perform exclusive cumprod.
/// - Parameter reverse: A `bool` (default: False).
/// - Parameter tidx: 
/// - Returns: 
///	out: 
public func cumprod(operationName: String? = nil, x: Output, axis: Output, exclusive: Bool, reverse: Bool, tidx: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["exclusive"] = exclusive
	attrs["reverse"] = reverse
	attrs["Tidx"] = tidx
	let opspec = OpSpec(
		type: "Cumprod",
		name: (operationName ?? "Type"),
		input: [x, axis],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Returns the next record (key, value pair) produced by a Reader.
///Will dequeue from the input queue if necessary (e.g. when the
/// Reader needs to start reading from a new file since it has finished
/// with the previous file).
/// - Parameter readerHandle: Handle to a Reader.
/// - Parameter queueHandle: Handle to a Queue, with string work items.
/// - Returns: 
///	key: A scalar.
///	value: A scalar.
public func readerReadV2(operationName: String? = nil, readerHandle: Output, queueHandle: Output) throws -> (key: Output, value: Output) { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "ReaderReadV2",
		name: (operationName ?? "Type"),
		input: [readerHandle, queueHandle],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (key: op.output(at: 0), value: op.output(at: 1))
} 

///Forwards the `index`th element of `inputs` to `output`.
/// - Parameter index: A scalar that determines the input that gets selected.
/// - Parameter inputs: A list of ref tensors, one of which will be forwarded to `output`.
/// - Parameter n: 
/// - Returns: 
///	output: The forwarded tensor.
public func refSelect(operationName: String? = nil, index: Output, inputs: [Output], n: UInt8) throws -> Output { 
	var attrs = [String : Any]()
	attrs["N"] = n
	let opspec = OpSpec(
		type: "RefSelect",
		name: (operationName ?? "Type"),
		input: [index, inputs],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Update ' * var' according to the centered RMSProp algorithm.
///The centered RMSProp algorithm uses an estimate of the centered second moment
/// (i.e., the variance) for normalization, as opposed to regular RMSProp, which
/// uses the (uncentered) second moment. This often helps with training, but is
/// slightly more expensive in terms of computation and memory.
/// 
/// Note that in dense implementation of this algorithm, mg, ms, and mom will
/// update even if the grad is zero, but in this sparse implementation, mg, ms,
/// and mom will not update in iterations during which the grad is zero.
/// 
/// mean_square = decay  *  mean_square + (1-decay)  *  gradient  *  *  2
/// mean_grad = decay  *  mean_grad + (1-decay)  *  gradient
/// Delta = learning_rate  *  gradient / sqrt(mean_square + epsilon - mean_grad  *  *  2)
/// 
/// ms <- rho  *  ms_{t-1} + (1-rho)  *  grad  *  grad
/// mom <- momentum  *  mom_{t-1} + lr  *  grad / sqrt(ms + epsilon)
/// var <- var - mom
/// - Parameter `var`: Should be from a Variable().
/// - Parameter mg: Should be from a Variable().
/// - Parameter ms: Should be from a Variable().
/// - Parameter mom: Should be from a Variable().
/// - Parameter lr: Scaling factor. Must be a scalar.
/// - Parameter rho: Decay rate. Must be a scalar.
/// - Parameter momentum: 
/// - Parameter epsilon: Ridge term. Must be a scalar.
/// - Parameter grad: The gradient.
/// - Parameter indices: A vector of indices into the first dimension of var, ms and mom.
/// - Parameter tindices: 
/// - Parameter useLocking: If `True`, updating of the var, mg, ms, and mom tensors is
/// protected by a lock; otherwise the behavior is undefined, but may exhibit less
/// contention.
/// - Returns: 
///	out: Same as "var".
public func sparseApplyCenteredRMSProp(operationName: String? = nil, `var`: Output, mg: Output, ms: Output, mom: Output, lr: Output, rho: Output, momentum: Output, epsilon: Output, grad: Output, indices: Output, tindices: Any.Type, useLocking: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Tindices"] = tindices
	attrs["use_locking"] = useLocking
	let opspec = OpSpec(
		type: "SparseApplyCenteredRMSProp",
		name: (operationName ?? "Type"),
		input: [`var`, mg, ms, mom, lr, rho, momentum, epsilon, grad, indices],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Adds two `SparseTensor` objects to produce another `SparseTensor`.
///The input `SparseTensor` objects' indices are assumed ordered in standard
/// lexicographic order.  If this is not the case, before this step run
/// `SparseReorder` to restore index ordering.
/// 
/// By default, if two values sum to zero at some index, the output `SparseTensor`
/// would still include that particular location in its index, storing a zero in the
/// corresponding value slot.  To override this, callers can specify `thresh`,
/// indicating that if the sum has a magnitude strictly smaller than `thresh`, its
/// corresponding value and index would then not be included.  In particular,
/// `thresh == 0` (default) means everything is kept and actual thresholding happens
/// only for a positive value.
/// 
/// In the following shapes, `nnz` is the count after taking `thresh` into account.
/// - Parameter aIndices: 2-D.  The `indices` of the first `SparseTensor`, size `[nnz, ndims]` Matrix.
/// - Parameter aValues: 1-D.  The `values` of the first `SparseTensor`, size `[nnz]` Vector.
/// - Parameter aShape: 1-D.  The `shape` of the first `SparseTensor`, size `[ndims]` Vector.
/// - Parameter bIndices: 2-D.  The `indices` of the second `SparseTensor`, size `[nnz, ndims]` Matrix.
/// - Parameter bValues: 1-D.  The `values` of the second `SparseTensor`, size `[nnz]` Vector.
/// - Parameter bShape: 1-D.  The `shape` of the second `SparseTensor`, size `[ndims]` Vector.
/// - Parameter thresh: 0-D.  The magnitude threshold that determines if an output value/index
/// pair takes space.
/// - Parameter treal: 
/// - Returns: 
///	sum_indices: 
///	sum_values: 
///	sum_shape: 
public func sparseAdd(operationName: String? = nil, aIndices: Output, aValues: Output, aShape: Output, bIndices: Output, bValues: Output, bShape: Output, thresh: Output, treal: Any.Type) throws -> (sumIndices: Output, sumValues: Output, sumShape: Output) { 
	var attrs = [String : Any]()
	attrs["Treal"] = treal
	let opspec = OpSpec(
		type: "SparseAdd",
		name: (operationName ?? "Type"),
		input: [aIndices, aValues, aShape, bIndices, bValues, bShape, thresh],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (sumIndices: op.output(at: 0), sumValues: op.output(at: 1), sumShape: op.output(at: 2))
} 

///Reverses variable length slices.
///This op first slices `input` along the dimension `batch_dim`, and for each
/// slice `i`, reverses the first `seq_lengths[i]` elements along
/// the dimension `seq_dim`.
/// 
/// The elements of `seq_lengths` must obey `seq_lengths[i] <= input.dims[seq_dim]`,
/// and `seq_lengths` must be a vector of length `input.dims[batch_dim]`.
/// 
/// The output slice `i` along dimension `batch_dim` is then given by input
/// slice `i`, with the first `seq_lengths[i]` slices along dimension
/// `seq_dim` reversed.
/// 
/// For example:
/// 
/// ```
/// # Given this:
/// batch_dim = 0
/// seq_dim = 1
/// input.dims = (4, 8, ...)
/// seq_lengths = [7, 2, 3, 5]
/// 
/// # then slices of input are reversed on seq_dim, but only up to seq_lengths:
/// output[0, 0:7, :, ...] = input[0, 7:0:-1, :, ...]
/// output[1, 0:2, :, ...] = input[1, 2:0:-1, :, ...]
/// output[2, 0:3, :, ...] = input[2, 3:0:-1, :, ...]
/// output[3, 0:5, :, ...] = input[3, 5:0:-1, :, ...]
/// 
/// # while entries past seq_lens are copied through:
/// output[0, 7:, :, ...] = input[0, 7:, :, ...]
/// output[1, 2:, :, ...] = input[1, 2:, :, ...]
/// output[2, 3:, :, ...] = input[2, 3:, :, ...]
/// output[3, 2:, :, ...] = input[3, 2:, :, ...]
/// ```
/// 
/// In contrast, if:
/// 
/// ```
/// # Given this:
/// batch_dim = 2
/// seq_dim = 0
/// input.dims = (8, ?, 4, ...)
/// seq_lengths = [7, 2, 3, 5]
/// 
/// # then slices of input are reversed on seq_dim, but only up to seq_lengths:
/// output[0:7, :, 0, :, ...] = input[7:0:-1, :, 0, :, ...]
/// output[0:2, :, 1, :, ...] = input[2:0:-1, :, 1, :, ...]
/// output[0:3, :, 2, :, ...] = input[3:0:-1, :, 2, :, ...]
/// output[0:5, :, 3, :, ...] = input[5:0:-1, :, 3, :, ...]
/// 
/// # while entries past seq_lens are copied through:
/// output[7:, :, 0, :, ...] = input[7:, :, 0, :, ...]
/// output[2:, :, 1, :, ...] = input[2:, :, 1, :, ...]
/// output[3:, :, 2, :, ...] = input[3:, :, 2, :, ...]
/// output[2:, :, 3, :, ...] = input[2:, :, 3, :, ...]
/// ```
/// - Parameter input: The input to reverse.
/// - Parameter seqLengths: 1-D with length `input.dims(batch_dim)` and
/// `max(seq_lengths) <= input.dims(seq_dim)`
/// - Parameter seqDim: The dimension which is partially reversed.
/// - Parameter batchDim: The dimension along which reversal is performed.
/// - Parameter tlen: 
/// - Returns: 
///	output: The partially reversed input. It has the same shape as `input`.
public func reverseSequence(operationName: String? = nil, input: Output, seqLengths: Output, seqDim: UInt8, batchDim: UInt8, tlen: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["seq_dim"] = seqDim
	attrs["batch_dim"] = batchDim
	attrs["Tlen"] = tlen
	let opspec = OpSpec(
		type: "ReverseSequence",
		name: (operationName ?? "Type"),
		input: [input, seqLengths],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Gather specific elements from the TensorArray into output `value`.
///All elements selected by `indices` must have the same shape.
/// - Parameter handle: The handle to a TensorArray.
/// - Parameter indices: The locations in the TensorArray from which to read tensor elements.
/// - Parameter flowIn: A float scalar that enforces proper chaining of operations.
/// - Parameter dtype: The type of the elem that is returned.
/// - Parameter elementShape: The expected shape of an element, if known. Used to
/// validate the shapes of TensorArray elements. If this shape is not
/// fully specified, gathering zero-size TensorArrays is an error.
/// - Returns: 
///	value: All of the elements in the TensorArray, concatenated along a new
/// axis (the new dimension 0).
public func tensorArrayGatherV3(operationName: String? = nil, handle: Output, indices: Output, flowIn: Output, dtype: Any.Type, elementShape: Shape) throws -> Output { 
	var attrs = [String : Any]()
	attrs["dtype"] = dtype
	attrs["element_shape"] = elementShape
	let opspec = OpSpec(
		type: "TensorArrayGatherV3",
		name: (operationName ?? "Type"),
		input: [handle, indices, flowIn],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Update ' * var' according to the RMSProp algorithm.
///Note that in dense implementation of this algorithm, ms and mom will
/// update even if the grad is zero, but in this sparse implementation, ms
/// and mom will not update in iterations during which the grad is zero.
/// 
/// mean_square = decay  *  mean_square + (1-decay)  *  gradient  *  *  2
/// Delta = learning_rate  *  gradient / sqrt(mean_square + epsilon)
/// 
/// ms <- rho  *  ms_{t-1} + (1-rho)  *  grad  *  grad
/// mom <- momentum  *  mom_{t-1} + lr  *  grad / sqrt(ms + epsilon)
/// var <- var - mom
/// - Parameter `var`: Should be from a Variable().
/// - Parameter ms: Should be from a Variable().
/// - Parameter mom: Should be from a Variable().
/// - Parameter lr: Scaling factor. Must be a scalar.
/// - Parameter rho: Decay rate. Must be a scalar.
/// - Parameter momentum: 
/// - Parameter epsilon: Ridge term. Must be a scalar.
/// - Parameter grad: The gradient.
/// - Parameter useLocking: If `True`, updating of the var, ms, and mom tensors is protected
/// by a lock; otherwise the behavior is undefined, but may exhibit less
/// contention.
/// - Returns: 
///	out: Same as "var".
public func applyRMSProp(operationName: String? = nil, `var`: Output, ms: Output, mom: Output, lr: Output, rho: Output, momentum: Output, epsilon: Output, grad: Output, useLocking: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["use_locking"] = useLocking
	let opspec = OpSpec(
		type: "ApplyRMSProp",
		name: (operationName ?? "Type"),
		input: [`var`, ms, mom, lr, rho, momentum, epsilon, grad],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Push an element onto the stack.
/// - Parameter handle: The handle to a stack.
/// - Parameter elem: The tensor to be pushed onto the stack.
/// - Parameter swapMemory: Swap `elem` to CPU. Default to false.
/// - Returns: 
///	output: The same tensor as the input 'elem'.
public func stackPushV2(operationName: String? = nil, handle: Output, elem: Output, swapMemory: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["swap_memory"] = swapMemory
	let opspec = OpSpec(
		type: "StackPushV2",
		name: (operationName ?? "Type"),
		input: [handle, elem],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///A queue that produces elements sorted by the first component value.
///Note that the PriorityQueue requires the first component of any element
/// to be a scalar int64, in addition to the other elements declared by
/// component_types.  Therefore calls to Enqueue and EnqueueMany (resp. Dequeue
/// and DequeueMany) on a PriorityQueue will all require (resp. output) one extra
/// entry in their input (resp. output) lists.
/// - Parameter componentTypes: The type of each component in a value.
/// - Parameter shapes: The shape of each component in a value. The length of this attr must
/// be either 0 or the same as the length of component_types. If the length of
/// this attr is 0, the shapes of queue elements are not constrained, and
/// only one element may be dequeued at a time.
/// - Parameter capacity: The upper bound on the number of elements in this queue.
/// Negative numbers mean no limit.
/// - Parameter container: If non-empty, this queue is placed in the given container.
/// Otherwise, a default container is used.
/// - Parameter sharedName: If non-empty, this queue will be shared under the given name
/// across multiple sessions.
/// - Returns: 
///	handle: The handle to the queue.
public func priorityQueue(operationName: String? = nil, componentTypes: [Any.Type], shapes: [Shape], capacity: UInt8, container: String, sharedName: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["component_types"] = componentTypes
	attrs["shapes"] = shapes
	attrs["capacity"] = capacity
	attrs["container"] = container
	attrs["shared_name"] = sharedName
	let opspec = OpSpec(
		type: "PriorityQueue",
		name: (operationName ?? "Type"),
		input: [],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Initializes a table from a text file.
///It inserts one key-value pair into the table for each line of the file.
/// The key and value is extracted from the whole line content, elements from the
/// split line based on `delimiter` or the line number (starting from zero).
/// Where to extract the key and value from a line is specified by `key_index` and
/// `value_index`.
/// 
/// - A value of -1 means use the line number(starting from zero), expects `int64`.
/// - A value of -2 means use the whole line content, expects `string`.
/// - A value >= 0 means use the index (starting at zero) of the split line based
///   on `delimiter`.
/// - Parameter tableHandle: Handle to a table which will be initialized.
/// - Parameter filename: Filename of a vocabulary text file.
/// - Parameter keyIndex: Column index in a line to get the table `key` values from.
/// - Parameter valueIndex: Column index that represents information of a line to get the table
/// `value` values from.
/// - Parameter vocabSize: Number of elements of the file, use -1 if unknown.
/// - Parameter delimiter: Delimiter to separate fields in a line.
public func initializeTableFromTextFileV2(operationName: String? = nil, tableHandle: Output, filename: Output, keyIndex: UInt8, valueIndex: UInt8, vocabSize: UInt8, delimiter: String) throws -> Operation { 
	var attrs = [String : Any]()
	attrs["key_index"] = keyIndex
	attrs["value_index"] = valueIndex
	attrs["vocab_size"] = vocabSize
	attrs["delimiter"] = delimiter
	let opspec = OpSpec(
		type: "InitializeTableFromTextFileV2",
		name: (operationName ?? "Type"),
		input: [tableHandle, filename],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Randomly crop `image`.
///`size` is a 1-D int64 tensor with 2 elements representing the crop height and
/// width.  The values must be non negative.
/// 
/// This Op picks a random location in `image` and crops a `height` by `width`
/// rectangle from that location.  The random location is picked so the cropped
/// area will fit inside the original image.
/// - Parameter image: 3-D of shape `[height, width, channels]`.
/// - Parameter size: 1-D of length 2 containing: `crop_height`, `crop_width`..
/// - Parameter seed: If either seed or seed2 are set to be non-zero, the random number
/// generator is seeded by the given seed.  Otherwise, it is seeded by a
/// random seed.
/// - Parameter seed2: An second seed to avoid seed collision.
/// - Returns: 
///	output: 3-D of shape `[crop_height, crop_width, channels].`
public func randomCrop(operationName: String? = nil, image: Output, size: Output, seed: UInt8, seed2: UInt8) throws -> Output { 
	var attrs = [String : Any]()
	attrs["seed"] = seed
	attrs["seed2"] = seed2
	let opspec = OpSpec(
		type: "RandomCrop",
		name: (operationName ?? "Type"),
		input: [image, size],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Exits the current frame to its parent frame.
///Exit makes its input `data` available to the parent frame.
/// - Parameter data: The tensor to be made available to the parent frame.
/// - Returns: 
///	output: The same tensor as `data`.
public func refExit(operationName: String? = nil, data: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "RefExit",
		name: (operationName ?? "Type"),
		input: [data],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Returns the truth value of (x > y) element-wise.
/// * NOTE * : `Greater` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
/// - Parameter x: 
/// - Parameter y: 
/// - Returns: 
///	z: 
public func greater(operationName: String? = nil, x: Output, y: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Greater",
		name: (operationName ?? "Type"),
		input: [x, y],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Returns the number of work units this Reader has finished processing.
/// - Parameter readerHandle: Handle to a Reader.
/// - Returns: 
///	units_completed: 
public func readerNumWorkUnitsCompleted(operationName: String? = nil, readerHandle: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "ReaderNumWorkUnitsCompleted",
		name: (operationName ?? "Type"),
		input: [readerHandle],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Decode a 16-bit PCM WAV file to a float tensor.
///The -32768 to 32767 signed 16-bit values will be scaled to -1.0 to 1.0 in float.
/// 
/// When desired_channels is set, if the input contains fewer channels than this
/// then the last channel will be duplicated to give the requested number, else if
/// the input has more channels than requested then the additional channels will be
/// ignored.
/// 
/// If desired_samples is set, then the audio will be cropped or padded with zeroes
/// to the requested length.
/// 
/// The first output contains a Tensor with the content of the audio samples. The
/// lowest dimension will be the number of channels, and the second will be the
/// number of samples. For example, a ten-sample-long stereo WAV file should give an
/// output shape of [10, 2].
/// - Parameter contents: The WAV-encoded audio, usually from a file.
/// - Parameter desiredChannels: Number of sample channels wanted.
/// - Parameter desiredSamples: Length of audio requested.
/// - Returns: 
///	audio: 2-D with shape `[length, channels]`.
///	sample_rate: Scalar holding the sample rate found in the WAV header.
public func decodeWav(operationName: String? = nil, contents: Output, desiredChannels: UInt8, desiredSamples: UInt8) throws -> (audio: Output, sampleRate: Output) { 
	var attrs = [String : Any]()
	attrs["desired_channels"] = desiredChannels
	attrs["desired_samples"] = desiredSamples
	let opspec = OpSpec(
		type: "DecodeWav",
		name: (operationName ?? "Type"),
		input: [contents],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (audio: op.output(at: 0), sampleRate: op.output(at: 1))
} 

///Dequeues `n` tuples of one or more tensors from the given queue.
///This operation is not supported by all queues.  If a queue does not support
/// DequeueUpTo, then an Unimplemented error is returned.
/// 
/// If the queue is closed and there are more than 0 but less than `n`
/// elements remaining, then instead of returning an OutOfRange error like
/// QueueDequeueMany, less than `n` elements are returned immediately.  If
/// the queue is closed and there are 0 elements left in the queue, then
/// an OutOfRange error is returned just like in QueueDequeueMany.
/// Otherwise the behavior is identical to QueueDequeueMany:
/// 
/// This operation concatenates queue-element component tensors along the
/// 0th dimension to make a single component tensor.  All of the components
/// in the dequeued tuple will have size n in the 0th dimension.
/// 
/// This operation has `k` outputs, where `k` is the number of components in
/// the tuples stored in the given queue, and output `i` is the ith
/// component of the dequeued tuple.
/// - Parameter handle: The handle to a queue.
/// - Parameter n: The number of tuples to dequeue.
/// - Parameter componentTypes: The type of each component in a tuple.
/// - Parameter timeoutMs: If the queue has fewer than n elements, this operation
/// will block for up to timeout_ms milliseconds.
/// Note: This option is not supported yet.
/// - Returns: 
///	components: One or more tensors that were dequeued as a tuple.
public func queueDequeueUpToV2(operationName: String? = nil, handle: Output, n: Output, componentTypes: [Any.Type], timeoutMs: UInt8) throws -> Output { 
	var attrs = [String : Any]()
	attrs["component_types"] = componentTypes
	attrs["timeout_ms"] = timeoutMs
	let opspec = OpSpec(
		type: "QueueDequeueUpToV2",
		name: (operationName ?? "Type"),
		input: [handle, n],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Store the input tensor in the state of the current session.
/// - Parameter value: The tensor to be stored.
/// - Returns: 
///	handle: The handle for the tensor stored in the session state, represented
/// as a string.
public func getSessionHandle(operationName: String? = nil, value: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "GetSessionHandle",
		name: (operationName ?? "Type"),
		input: [value],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Component-wise multiplies a SparseTensor by a dense Tensor.
///The output locations corresponding to the implicitly zero elements in the sparse
/// tensor will be zero (i.e., will not take up storage space), regardless of the
/// contents of the dense tensor (even if it's +/-INF and that INF * 0 == NaN).
/// 
///  * Limitation * : this Op only broadcasts the dense side to the sparse side, but not
/// the other direction.
/// - Parameter spIndices: 2-D.  `N x R` matrix with the indices of non-empty values in a
/// SparseTensor, possibly not in canonical ordering.
/// - Parameter spValues: 1-D.  `N` non-empty values corresponding to `sp_indices`.
/// - Parameter spShape: 1-D.  Shape of the input SparseTensor.
/// - Parameter dense: `R`-D.  The dense Tensor operand.
/// - Returns: 
///	output: 1-D.  The `N` values that are operated on.
public func sparseDenseCwiseMul(operationName: String? = nil, spIndices: Output, spValues: Output, spShape: Output, dense: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "SparseDenseCwiseMul",
		name: (operationName ?? "Type"),
		input: [spIndices, spValues, spShape, dense],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Creates a dataset that contains the elements of `input_dataset` ignoring errors.
/// - Parameter inputDataset: 
/// - Parameter outputTypes: 
/// - Parameter outputShapes: 
/// - Returns: 
///	handle: 
public func ignoreErrorsDataset(operationName: String? = nil, inputDataset: Output, outputTypes: [Any.Type], outputShapes: [Shape]) throws -> Output { 
	var attrs = [String : Any]()
	attrs["output_types"] = outputTypes
	attrs["output_shapes"] = outputShapes
	let opspec = OpSpec(
		type: "IgnoreErrorsDataset",
		name: (operationName ?? "Type"),
		input: [inputDataset],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Update ' * var' and ' * accum' according to FOBOS with Adagrad learning rate.
///accum += grad  *  grad
/// prox_v = var - lr  *  grad  *  (1 / sqrt(accum))
/// var = sign(prox_v)/(1+lr * l2)  *  max{|prox_v|-lr * l1,0}
/// - Parameter `var`: Should be from a Variable().
/// - Parameter accum: Should be from a Variable().
/// - Parameter lr: Scaling factor. Must be a scalar.
/// - Parameter l1: L1 regularization. Must be a scalar.
/// - Parameter l2: L2 regularization. Must be a scalar.
/// - Parameter grad: The gradient.
/// - Parameter useLocking: If True, updating of the var and accum tensors will be protected by
/// a lock; otherwise the behavior is undefined, but may exhibit less contention.
/// - Returns: 
///	out: Same as "var".
public func applyProximalAdagrad(operationName: String? = nil, `var`: Output, accum: Output, lr: Output, l1: Output, l2: Output, grad: Output, useLocking: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["use_locking"] = useLocking
	let opspec = OpSpec(
		type: "ApplyProximalAdagrad",
		name: (operationName ?? "Type"),
		input: [`var`, accum, lr, l1, l2, grad],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Enqueues zero or more tuples of one or more tensors in the given queue.
///This operation slices each component tensor along the 0th dimension to
/// make multiple queue elements. All of the tuple components must have the
/// same size in the 0th dimension.
/// 
/// The components input has k elements, which correspond to the components of
/// tuples stored in the given queue.
/// 
/// N.B. If the queue is full, this operation will block until the given
/// elements have been enqueued (or 'timeout_ms' elapses, if specified).
/// - Parameter handle: The handle to a queue.
/// - Parameter components: One or more tensors from which the enqueued tensors should
/// be taken.
/// - Parameter tcomponents: 
/// - Parameter timeoutMs: If the queue is too full, this operation will block for up
/// to timeout_ms milliseconds.
/// Note: This option is not supported yet.
public func queueEnqueueMany(operationName: String? = nil, handle: Output, components: Output, tcomponents: [Any.Type], timeoutMs: UInt8) throws -> Operation { 
	var attrs = [String : Any]()
	attrs["Tcomponents"] = tcomponents
	attrs["timeout_ms"] = timeoutMs
	let opspec = OpSpec(
		type: "QueueEnqueueMany",
		name: (operationName ?? "Type"),
		input: [handle, components],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op
} 

///Returns the index with the smallest value across dimensions of a tensor.
///Note that in case of ties the identity of the return value is not guaranteed.
/// - Parameter input: 
/// - Parameter dimension: int32 or int64, must be in the range `[-rank(input), rank(input))`.
/// Describes which dimension of the input Tensor to reduce across. For vectors,
/// use dimension = 0.
/// - Parameter tidx: 
/// - Parameter outputType: 
/// - Returns: 
///	output: 
public func argMin(operationName: String? = nil, input: Output, dimension: Output, tidx: Any.Type, outputType: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Tidx"] = tidx
	attrs["output_type"] = outputType
	let opspec = OpSpec(
		type: "ArgMin",
		name: (operationName ?? "Type"),
		input: [input, dimension],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Creates a dataset that computes a windowed group-by on `input_dataset`.
///// TODO(mrry): Support non-int64 keys.
/// - Parameter inputDataset: 
/// - Parameter keyFuncOtherArguments: 
/// - Parameter reduceFuncOtherArguments: 
/// - Parameter windowSizeFuncOtherArguments: 
/// - Parameter keyFunc: A function mapping an element of `input_dataset`, concatenated
/// with `key_func_other_arguments` to a scalar value of type DT_INT64.
/// - Parameter reduceFunc: 
/// - Parameter windowSizeFunc: 
/// - Parameter tkeyFuncOtherArguments: 
/// - Parameter treduceFuncOtherArguments: 
/// - Parameter twindowSizeFuncOtherArguments: 
/// - Parameter outputTypes: 
/// - Parameter outputShapes: 
/// - Returns: 
///	handle: 
public func groupByWindowDataset(operationName: String? = nil, inputDataset: Output, keyFuncOtherArguments: Output, reduceFuncOtherArguments: Output, windowSizeFuncOtherArguments: Output, keyFunc: Tensorflow_NameAttrList, reduceFunc: Tensorflow_NameAttrList, windowSizeFunc: Tensorflow_NameAttrList, tkeyFuncOtherArguments: [Any.Type], treduceFuncOtherArguments: [Any.Type], twindowSizeFuncOtherArguments: [Any.Type], outputTypes: [Any.Type], outputShapes: [Shape]) throws -> Output { 
	var attrs = [String : Any]()
	attrs["key_func"] = keyFunc
	attrs["reduce_func"] = reduceFunc
	attrs["window_size_func"] = windowSizeFunc
	attrs["Tkey_func_other_arguments"] = tkeyFuncOtherArguments
	attrs["Treduce_func_other_arguments"] = treduceFuncOtherArguments
	attrs["Twindow_size_func_other_arguments"] = twindowSizeFuncOtherArguments
	attrs["output_types"] = outputTypes
	attrs["output_shapes"] = outputShapes
	let opspec = OpSpec(
		type: "GroupByWindowDataset",
		name: (operationName ?? "Type"),
		input: [inputDataset, keyFuncOtherArguments, reduceFuncOtherArguments, windowSizeFuncOtherArguments],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 


/// - Parameter handle: 
/// - Parameter flowIn: 
/// - Returns: 
///	size: 
public func tensorArraySize(operationName: String? = nil, handle: Output, flowIn: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "TensorArraySize",
		name: (operationName ?? "Type"),
		input: [handle, flowIn],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Computes the sum of elements across dimensions of a SparseTensor.
///This Op takes a SparseTensor and is the sparse counterpart to
/// `tf.reduce_sum()`.  In particular, this Op also returns a dense `Tensor`
/// instead of a sparse one.
/// 
/// Reduces `sp_input` along the dimensions given in `reduction_axes`.  Unless
/// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
/// `reduction_axes`. If `keep_dims` is true, the reduced dimensions are retained
/// with length 1.
/// 
/// If `reduction_axes` has no entries, all dimensions are reduced, and a tensor
/// with a single element is returned.  Additionally, the axes can be negative,
/// which are interpreted according to the indexing rules in Python.
/// - Parameter inputIndices: 2-D.  `N x R` matrix with the indices of non-empty values in a
/// SparseTensor, possibly not in canonical ordering.
/// - Parameter inputValues: 1-D.  `N` non-empty values corresponding to `input_indices`.
/// - Parameter inputShape: 1-D.  Shape of the input SparseTensor.
/// - Parameter reductionAxes: 1-D.  Length-`K` vector containing the reduction axes.
/// - Parameter keepDims: If true, retain reduced dimensions with length 1.
/// - Returns: 
///	output: `R-K`-D.  The reduced Tensor.
public func sparseReduceSum(operationName: String? = nil, inputIndices: Output, inputValues: Output, inputShape: Output, reductionAxes: Output, keepDims: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["keep_dims"] = keepDims
	let opspec = OpSpec(
		type: "SparseReduceSum",
		name: (operationName ?? "Type"),
		input: [inputIndices, inputValues, inputShape, reductionAxes],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Gather slices from `params` according to `indices`.
///`indices` must be an integer tensor of any dimension (usually 0-D or 1-D).
/// Produces an output tensor with shape `indices.shape + params.shape[1:]` where:
/// 
/// ```python
///     # Scalar indices
///     output[:, ..., :] = params[indices, :, ... :]
/// 
///     # Vector indices
///     output[i, :, ..., :] = params[indices[i], :, ... :]
/// 
///     # Higher rank indices
///     output[i, ..., j, :, ... :] = params[indices[i, ..., j], :, ..., :]
/// ```
/// 
/// If `indices` is a permutation and `len(indices) == params.shape[0]` then
/// this operation will permute `params` accordingly.
/// 
/// `validate_indices`: DEPRECATED. If this operation is assigned to CPU, values in
/// `indices` are always validated to be within range. If assigned to GPU,
/// out-of-bound indices result in safe but unspecified behavior, which may include
/// raising an error.
/// 
/// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
/// <img style="width:100%" src="https://www.tensorflow.org/images/Gather.png" alt>
/// </div>
/// - Parameter params: 
/// - Parameter indices: 
/// - Parameter validateIndices: 
/// - Parameter tparams: 
/// - Parameter tindices: 
/// - Returns: 
///	output: 
public func gather(operationName: String? = nil, params: Output, indices: Output, validateIndices: Bool, tparams: Any.Type, tindices: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["validate_indices"] = validateIndices
	attrs["Tparams"] = tparams
	attrs["Tindices"] = tindices
	let opspec = OpSpec(
		type: "Gather",
		name: (operationName ?? "Type"),
		input: [params, indices],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Generates labels for candidate sampling with a uniform distribution.
///See explanations of candidate sampling and the data formats at
/// go/candidate-sampling.
/// 
/// For each batch, this op picks a single set of sampled candidate labels.
/// 
/// The advantages of sampling candidates per-batch are simplicity and the
/// possibility of efficient dense matrix multiplication. The disadvantage is that
/// the sampled candidates must be chosen independently of the context and of the
/// true labels.
/// - Parameter trueClasses: A batch_size  *  num_true matrix, in which each row contains the
/// IDs of the num_true target_classes in the corresponding original label.
/// - Parameter numTrue: Number of true labels per context.
/// - Parameter numSampled: Number of candidates to randomly sample.
/// - Parameter unique: If unique is true, we sample with rejection, so that all sampled
/// candidates in a batch are unique. This requires some approximation to
/// estimate the post-rejection sampling probabilities.
/// - Parameter rangeMax: The sampler will sample integers from the interval [0, range_max).
/// - Parameter seed: If either seed or seed2 are set to be non-zero, the random number
/// generator is seeded by the given seed.  Otherwise, it is seeded by a
/// random seed.
/// - Parameter seed2: An second seed to avoid seed collision.
/// - Returns: 
///	sampled_candidates: A vector of length num_sampled, in which each element is
/// the ID of a sampled candidate.
///	true_expected_count: A batch_size  *  num_true matrix, representing
/// the number of times each candidate is expected to occur in a batch
/// of sampled candidates. If unique=true, then this is a probability.
///	sampled_expected_count: A vector of length num_sampled, for each sampled
/// candidate representing the number of times the candidate is expected
/// to occur in a batch of sampled candidates.  If unique=true, then this is a
/// probability.
public func uniformCandidateSampler(operationName: String? = nil, trueClasses: Output, numTrue: UInt8, numSampled: UInt8, unique: Bool, rangeMax: UInt8, seed: UInt8, seed2: UInt8) throws -> (sampledCandidates: Output, trueExpectedCount: Output, sampledExpectedCount: Output) { 
	var attrs = [String : Any]()
	attrs["num_true"] = numTrue
	attrs["num_sampled"] = numSampled
	attrs["unique"] = unique
	attrs["range_max"] = rangeMax
	attrs["seed"] = seed
	attrs["seed2"] = seed2
	let opspec = OpSpec(
		type: "UniformCandidateSampler",
		name: (operationName ?? "Type"),
		input: [trueClasses],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (sampledCandidates: op.output(at: 0), trueExpectedCount: op.output(at: 1), sampledExpectedCount: op.output(at: 2))
} 

///Computes the reciprocal of x element-wise.
///I.e., \\(y = 1 / x\\).
/// - Parameter x: 
/// - Returns: 
///	y: 
public func reciprocal(operationName: String? = nil, x: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Reciprocal",
		name: (operationName ?? "Type"),
		input: [x],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Returns the number of work units this Reader has finished processing.
/// - Parameter readerHandle: Handle to a Reader.
/// - Returns: 
///	units_completed: 
public func readerNumWorkUnitsCompletedV2(operationName: String? = nil, readerHandle: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "ReaderNumWorkUnitsCompletedV2",
		name: (operationName ?? "Type"),
		input: [readerHandle],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Given a path to new and old vocabulary files, returns a remapping Tensor of
///length `num_new_vocab`, where `remapping[i]` contains the row number in the old
/// vocabulary that corresponds to row `i` in the new vocabulary (starting at line
/// `new_vocab_offset` and up to `num_new_vocab` entities), or `-1` if entry `i`
/// in the new vocabulary is not in the old vocabulary.  `num_vocab_offset` enables
/// use in the partitioned variable case, and should generally be set through
/// examining partitioning info.  The format of the files should be a text file,
/// with each line containing a single entity within the vocabulary.
/// 
/// For example, with `new_vocab_file` a text file containing each of the following
/// elements on a single line: `[f0, f1, f2, f3]`, old_vocab_file = [f1, f0, f3],
/// `num_new_vocab = 3, new_vocab_offset = 1`, the returned remapping would be
/// `[0, -1, 2]`.
/// 
/// The op also returns a count of how many entries in the new vocabulary
/// were present in the old vocabulary, which is used to calculate the number of
/// values to initialize in a weight matrix remapping
/// 
/// This functionality can be used to remap both row vocabularies (typically,
/// features) and column vocabularies (typically, classes) from TensorFlow
/// checkpoints.  Note that the partitioning logic relies on contiguous vocabularies
/// corresponding to div-partitioned variables.  Moreover, the underlying remapping
/// uses an IndexTable (as opposed to an inexact CuckooTable), so client code should
/// use the corresponding index_table_from_file() as the FeatureColumn framework
/// does (as opposed to tf.feature_to_id(), which uses a CuckooTable).
/// - Parameter newVocabFile: Path to the new vocab file.
/// - Parameter oldVocabFile: Path to the old vocab file.
/// - Parameter newVocabOffset: How many entries into the new vocab file to start reading.
/// - Parameter numNewVocab: Number of entries in the new vocab file to remap.
/// - Returns: 
///	remapping: A Tensor of length num_new_vocab where the element at index i
/// is equal to the old ID that maps to the new ID i.  This element is -1 for any
/// new ID that is not found in the old vocabulary.
///	num_present: Number of new vocab entries found in old vocab.
public func generateVocabRemapping(operationName: String? = nil, newVocabFile: Output, oldVocabFile: Output, newVocabOffset: UInt8, numNewVocab: UInt8) throws -> (remapping: Output, numPresent: Output) { 
	var attrs = [String : Any]()
	attrs["new_vocab_offset"] = newVocabOffset
	attrs["num_new_vocab"] = numNewVocab
	let opspec = OpSpec(
		type: "GenerateVocabRemapping",
		name: (operationName ?? "Type"),
		input: [newVocabFile, oldVocabFile],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (remapping: op.output(at: 0), numPresent: op.output(at: 1))
} 

///Checks whether a resource handle-based variable has been initialized.
/// - Parameter resource: the input resource handle.
/// - Returns: 
///	is_initialized: a scalar boolean which is true if the variable has been
/// initialized.
public func varIsInitializedOp(operationName: String? = nil, resource: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "VarIsInitializedOp",
		name: (operationName ?? "Type"),
		input: [resource],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Performs a resize and padding as a preprocess during a convolution.
///It's often possible to do spatial transformations more efficiently as part of
/// the packing stage of a convolution, so this op allows for an optimized
/// implementation where these stages are fused together. This prevents the need to
/// write out the intermediate results as whole tensors, reducing memory pressure,
/// and we can get some latency gains by merging the transformation calculations.
/// The data_format attribute for Conv2D isn't supported by this op, and defaults to
/// 'NHWC' order.
/// Internally this op uses a single per-graph scratch buffer, which means that it
/// will block if multiple versions are being run in parallel. This is because this
/// operator is primarily an optimization to minimize memory usage.
/// - Parameter input: 4-D with shape `[batch, in_height, in_width, in_channels]`.
/// - Parameter size: A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
/// new size for the images.
/// - Parameter paddings: A two-column matrix specifying the padding sizes. The number of
/// rows must be the same as the rank of `input`.
/// - Parameter filter: 4-D with shape
/// `[filter_height, filter_width, in_channels, out_channels]`.
/// - Parameter resizeAlignCorners: If true, rescale input by (new_height - 1) / (height - 1),
/// which exactly aligns the 4 corners of images and resized images. If false, rescale
/// by new_height / height. Treat similarly the width dimension.
/// - Parameter mode: 
/// - Parameter strides: 1-D of length 4.  The stride of the sliding window for each dimension
/// of `input`. Must be in the same order as the dimension specified with format.
/// - Parameter padding: The type of padding algorithm to use.
/// - Returns: 
///	output: 
public func fusedResizeAndPadConv2D(operationName: String? = nil, input: Output, size: Output, paddings: Output, filter: Output, resizeAlignCorners: Bool, mode: String, strides: [Int64], padding: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["resize_align_corners"] = resizeAlignCorners
	attrs["mode"] = mode
	attrs["strides"] = strides
	attrs["padding"] = padding
	let opspec = OpSpec(
		type: "FusedResizeAndPadConv2D",
		name: (operationName ?? "Type"),
		input: [input, size, paddings, filter],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Returns x - y element-wise.
/// * NOTE * : `Sub` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
/// - Parameter x: 
/// - Parameter y: 
/// - Returns: 
///	z: 
public func sub(operationName: String? = nil, x: Output, y: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "Sub",
		name: (operationName ?? "Type"),
		input: [x, y],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Transforms a scalar brain.SequenceExample proto (as strings) into typed tensors.
/// - Parameter serialized: A scalar containing a binary serialized SequenceExample proto.
/// - Parameter featureListDenseMissingAssumedEmpty: A vector listing the
/// FeatureList keys which may be missing from the SequenceExample.  If the
/// associated FeatureList is missing, it is treated as empty.  By default,
/// any FeatureList not listed in this vector must exist in the SequenceExample.
/// - Parameter contextSparseKeys: A list of Ncontext_sparse string Tensors (scalars).
/// The keys expected in the Examples' features associated with context_sparse
/// values.
/// - Parameter contextDenseKeys: A list of Ncontext_dense string Tensors (scalars).
/// The keys expected in the SequenceExamples' context features associated with
/// dense values.
/// - Parameter featureListSparseKeys: A list of Nfeature_list_sparse string Tensors
/// (scalars).  The keys expected in the FeatureLists associated with sparse
/// values.
/// - Parameter featureListDenseKeys: A list of Nfeature_list_dense string Tensors (scalars).
/// The keys expected in the SequenceExamples' feature_lists associated
/// with lists of dense values.
/// - Parameter contextDenseDefaults: A list of Ncontext_dense Tensors (some may be empty).
/// context_dense_defaults[j] provides default values
/// when the SequenceExample's context map lacks context_dense_key[j].
/// If an empty Tensor is provided for context_dense_defaults[j],
/// then the Feature context_dense_keys[j] is required.
/// The input type is inferred from context_dense_defaults[j], even when it's
/// empty.  If context_dense_defaults[j] is not empty, its shape must match
/// context_dense_shapes[j].
/// - Parameter debugName: A scalar containing the name of the serialized proto.
/// May contain, for example, table key (descriptive) name for the
/// corresponding serialized proto.  This is purely useful for debugging
/// purposes, and the presence of values here has no effect on the output.
/// May also be an empty scalar if no name is available.
/// - Parameter ncontextSparse: 
/// - Parameter ncontextDense: 
/// - Parameter nfeatureListSparse: 
/// - Parameter nfeatureListDense: 
/// - Parameter contextSparseTypes: A list of Ncontext_sparse types; the data types of data in
/// each context Feature given in context_sparse_keys.
/// Currently the ParseSingleSequenceExample supports DT_FLOAT (FloatList),
/// DT_INT64 (Int64List), and DT_STRING (BytesList).
/// - Parameter tcontextDense: 
/// - Parameter featureListDenseTypes: 
/// - Parameter contextDenseShapes: A list of Ncontext_dense shapes; the shapes of data in
/// each context Feature given in context_dense_keys.
/// The number of elements in the Feature corresponding to context_dense_key[j]
/// must always equal context_dense_shapes[j].NumEntries().
/// The shape of context_dense_values[j] will match context_dense_shapes[j].
/// - Parameter featureListSparseTypes: A list of Nfeature_list_sparse types; the data types
/// of data in each FeatureList given in feature_list_sparse_keys.
/// Currently the ParseSingleSequenceExample supports DT_FLOAT (FloatList),
/// DT_INT64 (Int64List), and DT_STRING (BytesList).
/// - Parameter featureListDenseShapes: A list of Nfeature_list_dense shapes; the shapes of
/// data in each FeatureList given in feature_list_dense_keys.
/// The shape of each Feature in the FeatureList corresponding to
/// feature_list_dense_key[j] must always equal
/// feature_list_dense_shapes[j].NumEntries().
/// - Returns: 
///	context_sparse_indices: 
///	context_sparse_values: 
///	context_sparse_shapes: 
///	context_dense_values: 
///	feature_list_sparse_indices: 
///	feature_list_sparse_values: 
///	feature_list_sparse_shapes: 
///	feature_list_dense_values: 
public func parseSingleSequenceExample(operationName: String? = nil, serialized: Output, featureListDenseMissingAssumedEmpty: Output, contextSparseKeys: Output, contextDenseKeys: Output, featureListSparseKeys: Output, featureListDenseKeys: Output, contextDenseDefaults: Output, debugName: Output, ncontextSparse: UInt8, ncontextDense: UInt8, nfeatureListSparse: UInt8, nfeatureListDense: UInt8, contextSparseTypes: [Any.Type], tcontextDense: [Any.Type], featureListDenseTypes: [Any.Type], contextDenseShapes: [Shape], featureListSparseTypes: [Any.Type], featureListDenseShapes: [Shape]) throws -> (contextSparseIndices: Output, contextSparseValues: Output, contextSparseShapes: Output, contextDenseValues: Output, featureListSparseIndices: Output, featureListSparseValues: Output, featureListSparseShapes: Output, featureListDenseValues: Output) { 
	var attrs = [String : Any]()
	attrs["Ncontext_sparse"] = ncontextSparse
	attrs["Ncontext_dense"] = ncontextDense
	attrs["Nfeature_list_sparse"] = nfeatureListSparse
	attrs["Nfeature_list_dense"] = nfeatureListDense
	attrs["context_sparse_types"] = contextSparseTypes
	attrs["Tcontext_dense"] = tcontextDense
	attrs["feature_list_dense_types"] = featureListDenseTypes
	attrs["context_dense_shapes"] = contextDenseShapes
	attrs["feature_list_sparse_types"] = featureListSparseTypes
	attrs["feature_list_dense_shapes"] = featureListDenseShapes
	let opspec = OpSpec(
		type: "ParseSingleSequenceExample",
		name: (operationName ?? "Type"),
		input: [serialized, featureListDenseMissingAssumedEmpty, contextSparseKeys, contextDenseKeys, featureListSparseKeys, featureListDenseKeys, contextDenseDefaults, debugName],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (contextSparseIndices: op.output(at: 0), contextSparseValues: op.output(at: 1), contextSparseShapes: op.output(at: 2), contextDenseValues: op.output(at: 3), featureListSparseIndices: op.output(at: 4), featureListSparseValues: op.output(at: 5), featureListSparseShapes: op.output(at: 6), featureListDenseValues: op.output(at: 7))
} 

///Runs function `f` on a remote device indicated by `target`.
/// - Parameter target: A fully specified device name where we want to run the function.
/// - Parameter args: A list of arguments for the function.
/// - Parameter tin: The type list for the arguments.
/// - Parameter tout: The type list for the return values.
/// - Parameter f: The function to run remotely.
/// - Returns: 
///	output: A list of return values.
public func remoteCall(operationName: String? = nil, target: Output, args: Output, tin: [Any.Type], tout: [Any.Type], f: Tensorflow_NameAttrList) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Tin"] = tin
	attrs["Tout"] = tout
	attrs["f"] = f
	let opspec = OpSpec(
		type: "RemoteCall",
		name: (operationName ?? "Type"),
		input: [target, args],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Returns the argument of a complex number.
///Given a tensor `input` of complex numbers, this operation returns a tensor of
/// type `float` that is the argument of each element in `input`. All elements in
/// `input` must be complex numbers of the form \\(a + bj\\), where  * a * 
/// is the real part and  * b *  is the imaginary part.
/// 
/// The argument returned by this operation is of the form \\(atan2(b, a)\\).
/// 
/// For example:
/// 
/// ```
/// # tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
/// tf.angle(input) ==> [2.0132, 1.056]
/// ```
/// 
/// @compatibility(numpy)
/// Equivalent to np.angle.
/// @end_compatibility
/// - Parameter input: 
/// - Parameter tout: 
/// - Returns: 
///	output: 
public func angle(operationName: String? = nil, input: Output, tout: Any.Type) throws -> Output { 
	var attrs = [String : Any]()
	attrs["Tout"] = tout
	let opspec = OpSpec(
		type: "Angle",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///3D real-valued fast Fourier transform.
///Computes the 3-dimensional discrete Fourier transform of a real-valued signal
/// over the inner-most 3 dimensions of `input`.
/// 
/// Since the DFT of a real signal is Hermitian-symmetric, `RFFT3D` only returns the
/// `fft_length / 2 + 1` unique components of the FFT for the inner-most dimension
/// of `output`: the zero-frequency term, followed by the `fft_length / 2`
/// positive-frequency terms.
/// 
/// Along each axis `RFFT3D` is computed on, if `fft_length` is smaller than the
/// corresponding dimension of `input`, the dimension is cropped. If it is larger,
/// the dimension is padded with zeros.
/// - Parameter input: A float32 tensor.
/// - Parameter fftLength: An int32 tensor of shape [3]. The FFT length for each dimension.
/// - Returns: 
///	output: A complex64 tensor of the same rank as `input`. The inner-most 3
///   dimensions of `input` are replaced with the their 3D Fourier transform. The
///   inner-most dimension contains `fft_length / 2 + 1` unique frequency
///   components.
/// 
/// @compatibility(numpy)
/// Equivalent to np.fft.rfftn with 3 dimensions.
/// @end_compatibility
public func rfft3D(operationName: String? = nil, input: Output, fftLength: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "RFFT3D",
		name: (operationName ?? "Type"),
		input: [input, fftLength],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///A queue that produces elements in first-in first-out order.
/// - Parameter componentTypes: The type of each component in a value.
/// - Parameter shapes: The shape of each component in a value. The length of this attr must
/// be either 0 or the same as the length of component_types. If the length of
/// this attr is 0, the shapes of queue elements are not constrained, and
/// only one element may be dequeued at a time.
/// - Parameter capacity: The upper bound on the number of elements in this queue.
/// Negative numbers mean no limit.
/// - Parameter container: If non-empty, this queue is placed in the given container.
/// Otherwise, a default container is used.
/// - Parameter sharedName: If non-empty, this queue will be shared under the given name
/// across multiple sessions.
/// - Returns: 
///	handle: The handle to the queue.
public func fIFOQueueV2(operationName: String? = nil, componentTypes: [Any.Type], shapes: [Shape], capacity: UInt8, container: String, sharedName: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["component_types"] = componentTypes
	attrs["shapes"] = shapes
	attrs["capacity"] = capacity
	attrs["container"] = container
	attrs["shared_name"] = sharedName
	let opspec = OpSpec(
		type: "FIFOQueueV2",
		name: (operationName ?? "Type"),
		input: [],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 


/// - Parameter handle: 
/// - Parameter value: 
/// - Parameter flowIn: 
/// - Returns: 
///	flow_out: 
public func tensorArrayUnpack(operationName: String? = nil, handle: Output, value: Output, flowIn: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "TensorArrayUnpack",
		name: (operationName ?? "Type"),
		input: [handle, value, flowIn],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Decode and Crop a JPEG-encoded image to a uint8 tensor.
///The attr `channels` indicates the desired number of color channels for the
/// decoded image.
/// 
/// Accepted values are:
/// 
///  *    0: Use the number of channels in the JPEG-encoded image.
///  *    1: output a grayscale image.
///  *    3: output an RGB image.
/// 
/// If needed, the JPEG-encoded image is transformed to match the requested number
/// of color channels.
/// 
/// The attr `ratio` allows downscaling the image by an integer factor during
/// decoding.  Allowed values are: 1, 2, 4, and 8.  This is much faster than
/// downscaling the image later.
/// 
/// 
/// It is equivalent to a combination of decode and crop, but much faster by only
/// decoding partial jpeg image.
/// - Parameter contents: 0-D.  The JPEG-encoded image.
/// - Parameter cropWindow: 1-D.  The crop window: [crop_y, crop_x, crop_height, crop_width].
/// - Parameter channels: Number of color channels for the decoded image.
/// - Parameter ratio: Downscaling ratio.
/// - Parameter fancyUpscaling: If true use a slower but nicer upscaling of the
/// chroma planes (yuv420/422 only).
/// - Parameter tryRecoverTruncated: If true try to recover an image from truncated input.
/// - Parameter acceptableFraction: The minimum required fraction of lines before a truncated
/// input is accepted.
/// - Parameter dctMethod: string specifying a hint about the algorithm used for
/// decompression.  Defaults to "" which maps to a system-specific
/// default.  Currently valid values are ["INTEGER_FAST",
/// "INTEGER_ACCURATE"].  The hint may be ignored (e.g., the internal
/// jpeg library changes to a version that does not have that specific
/// option.)
/// - Returns: 
///	image: 3-D with shape `[height, width, channels]`..
public func decodeAndCropJpeg(operationName: String? = nil, contents: Output, cropWindow: Output, channels: UInt8, ratio: UInt8, fancyUpscaling: Bool, tryRecoverTruncated: Bool, acceptableFraction: Float, dctMethod: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["channels"] = channels
	attrs["ratio"] = ratio
	attrs["fancy_upscaling"] = fancyUpscaling
	attrs["try_recover_truncated"] = tryRecoverTruncated
	attrs["acceptable_fraction"] = acceptableFraction
	attrs["dct_method"] = dctMethod
	let opspec = OpSpec(
		type: "DecodeAndCropJpeg",
		name: (operationName ?? "Type"),
		input: [contents, cropWindow],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Receives the named tensor from send_device on recv_device.
/// - Parameter tensorType: 
/// - Parameter tensorName: The name of the tensor to receive.
/// - Parameter sendDevice: The name of the device sending the tensor.
/// - Parameter sendDeviceIncarnation: The current incarnation of send_device.
/// - Parameter recvDevice: The name of the device receiving the tensor.
/// - Parameter clientTerminated: If set to true, this indicates that the node was added
/// to the graph as a result of a client-side feed or fetch of Tensor data,
/// in which case the corresponding send or recv is expected to be managed
/// locally by the caller.
/// - Returns: 
///	tensor: The tensor to receive.
public func recv(operationName: String? = nil, tensorType: Any.Type, tensorName: String, sendDevice: String, sendDeviceIncarnation: UInt8, recvDevice: String, clientTerminated: Bool) throws -> Output { 
	var attrs = [String : Any]()
	attrs["tensor_type"] = tensorType
	attrs["tensor_name"] = tensorName
	attrs["send_device"] = sendDevice
	attrs["send_device_incarnation"] = sendDeviceIncarnation
	attrs["recv_device"] = recvDevice
	attrs["client_terminated"] = clientTerminated
	let opspec = OpSpec(
		type: "_Recv",
		name: (operationName ?? "Type"),
		input: [],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Converts the given `resource_handle` representing an iterator to a string.
/// - Parameter resourceHandle: A handle to an iterator resource.
/// - Returns: 
///	string_handle: A string representation of the given handle.
public func iteratorToStringHandle(operationName: String? = nil, resourceHandle: Output) throws -> Output { 
	let attrs = [String : Any]()
	let opspec = OpSpec(
		type: "IteratorToStringHandle",
		name: (operationName ?? "Type"),
		input: [resourceHandle],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 

///Performs fractional average pooling on the input.
///Fractional average pooling is similar to Fractional max pooling in the pooling
/// region generation step. The only difference is that after pooling regions are
/// generated, a mean operation is performed instead of a max operation in each
/// pooling region.
/// - Parameter value: 4-D with shape `[batch, height, width, channels]`.
/// - Parameter poolingRatio: Pooling ratio for each dimension of `value`, currently only
/// supports row and col dimension and should be >= 1.0. For example, a valid
/// pooling ratio looks like [1.0, 1.44, 1.73, 1.0]. The first and last elements
/// must be 1.0 because we don't allow pooling on batch and channels
/// dimensions. 1.44 and 1.73 are pooling ratio on height and width dimensions
/// respectively.
/// - Parameter pseudoRandom: When set to True, generates the pooling sequence in a
/// pseudorandom fashion, otherwise, in a random fashion. Check paper [Benjamin
/// Graham, Fractional Max-Pooling](http://arxiv.org/abs/1412.6071) for
/// difference between pseudorandom and random.
/// - Parameter overlapping: When set to True, it means when pooling, the values at the boundary
/// of adjacent pooling cells are used by both cells. For example:
/// 
/// `index  0  1  2  3  4`
/// 
/// `value  20 5  16 3  7`
/// 
/// If the pooling sequence is [0, 2, 4], then 16, at index 2 will be used twice.
/// The result would be [41/3, 26/3] for fractional avg pooling.
/// - Parameter deterministic: When set to True, a fixed pooling region will be used when
/// iterating over a FractionalAvgPool node in the computation graph. Mainly used
/// in unit test to make FractionalAvgPool deterministic.
/// - Parameter seed: If either seed or seed2 are set to be non-zero, the random number
/// generator is seeded by the given seed.  Otherwise, it is seeded by a
/// random seed.
/// - Parameter seed2: An second seed to avoid seed collision.
/// - Returns: 
///	output: output tensor after fractional avg pooling.
///	row_pooling_sequence: row pooling sequence, needed to calculate gradient.
///	col_pooling_sequence: column pooling sequence, needed to calculate gradient.
public func fractionalAvgPool(operationName: String? = nil, value: Output, poolingRatio: [Float], pseudoRandom: Bool, overlapping: Bool, deterministic: Bool, seed: UInt8, seed2: UInt8) throws -> (output: Output, rowPoolingSequence: Output, colPoolingSequence: Output) { 
	var attrs = [String : Any]()
	attrs["pooling_ratio"] = poolingRatio
	attrs["pseudo_random"] = pseudoRandom
	attrs["overlapping"] = overlapping
	attrs["deterministic"] = deterministic
	attrs["seed"] = seed
	attrs["seed2"] = seed2
	let opspec = OpSpec(
		type: "FractionalAvgPool",
		name: (operationName ?? "Type"),
		input: [value],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (output: op.output(at: 0), rowPoolingSequence: op.output(at: 1), colPoolingSequence: op.output(at: 2))
} 

///Gradient for batch normalization.
///Note that the size of 4D Tensors are defined by either "NHWC" or "NCHW".
/// The size of 1D Tensors matches the dimension C of the 4D Tensors.
/// - Parameter yBackprop: A 4D Tensor for the gradient with respect to y.
/// - Parameter x: A 4D Tensor for input data.
/// - Parameter scale: A 1D Tensor for scaling factor, to scale the normalized x.
/// - Parameter reserveSpace1: When is_training is True, a 1D Tensor for the computed batch
/// mean to be reused in gradient computation. When is_training is
/// False, a 1D Tensor for the population mean to be reused in both
/// 1st and 2nd order gradient computation.
/// - Parameter reserveSpace2: When is_training is True, a 1D Tensor for the computed batch
/// variance (inverted variance in the cuDNN case) to be reused in
/// gradient computation. When is_training is False, a 1D Tensor
/// for the population variance to be reused in both 1st and 2nd
/// order gradient computation.
/// - Parameter u: The data type for the scale, offset, mean, and variance.
/// - Parameter epsilon: A small float number added to the variance of x.
/// - Parameter dataFormat: The data format for y_backprop, x, x_backprop.
/// Either "NHWC" (default) or "NCHW".
/// - Parameter isTraining: A bool value to indicate the operation is for training (default)
/// or inference.
/// - Returns: 
///	x_backprop: A 4D Tensor for the gradient with respect to x.
///	scale_backprop: A 1D Tensor for the gradient with respect to scale.
///	offset_backprop: A 1D Tensor for the gradient with respect to offset.
///	reserve_space_3: Unused placeholder to match the mean input in FusedBatchNorm.
///	reserve_space_4: Unused placeholder to match the variance input
/// in FusedBatchNorm.
public func fusedBatchNormGradV2(operationName: String? = nil, yBackprop: Output, x: Output, scale: Output, reserveSpace1: Output, reserveSpace2: Output, u: Any.Type, epsilon: Float, dataFormat: String, isTraining: Bool) throws -> (xBackprop: Output, scaleBackprop: Output, offsetBackprop: Output, reserveSpace3: Output, reserveSpace4: Output) { 
	var attrs = [String : Any]()
	attrs["U"] = u
	attrs["epsilon"] = epsilon
	attrs["data_format"] = dataFormat
	attrs["is_training"] = isTraining
	let opspec = OpSpec(
		type: "FusedBatchNormGradV2",
		name: (operationName ?? "Type"),
		input: [yBackprop, x, scale, reserveSpace1, reserveSpace2],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (xBackprop: op.output(at: 0), scaleBackprop: op.output(at: 1), offsetBackprop: op.output(at: 2), reserveSpace3: op.output(at: 3), reserveSpace4: op.output(at: 4))
} 

///Extracts the average sparse gradient in a SparseConditionalAccumulator.
///The op will blocks until sufficient (i.e., more than num_required)
/// gradients have been accumulated. If the accumulator has already
/// aggregated more than num_required gradients, it will return its
/// average of the accumulated gradients.  Also automatically increments
/// the recorded global_step in the accumulator by 1, and resets the
/// aggregate to 0.
/// - Parameter handle: The handle to a SparseConditionalAccumulator.
/// - Parameter numRequired: Number of gradients required before we return an aggregate.
/// - Parameter dtype: The data type of accumulated gradients. Needs to correspond to the type
/// of the accumulator.
/// - Returns: 
///	indices: Indices of the average of the accumulated sparse gradients.
///	values: Values of the average of the accumulated sparse gradients.
///	shape: Shape of the average of the accumulated sparse gradients.
public func sparseAccumulatorTakeGradient(operationName: String? = nil, handle: Output, numRequired: Output, dtype: Any.Type) throws -> (indices: Output, values: Output, shape: Output) { 
	var attrs = [String : Any]()
	attrs["dtype"] = dtype
	let opspec = OpSpec(
		type: "SparseAccumulatorTakeGradient",
		name: (operationName ?? "Type"),
		input: [handle, numRequired],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (indices: op.output(at: 0), values: op.output(at: 1), shape: op.output(at: 2))
} 

///Finds values and indices of the `k` largest elements for the last dimension.
///If the input is a vector (rank-1), finds the `k` largest entries in the vector
/// and outputs their values and indices as vectors.  Thus `values[j]` is the
/// `j`-th largest entry in `input`, and its index is `indices[j]`.
/// 
/// For matrices (resp. higher rank input), computes the top `k` entries in each
/// row (resp. vector along the last dimension).  Thus,
/// 
///     values.shape = indices.shape = input.shape[:-1] + [k]
/// 
/// If two elements are equal, the lower-index element appears first.
/// 
/// If `k` varies dynamically, use `TopKV2` below.
/// - Parameter input: 1-D or higher with last dimension at least `k`.
/// - Parameter k: Number of top elements to look for along the last dimension (along each
/// row for matrices).
/// - Parameter sorted: If true the resulting `k` elements will be sorted by the values in
/// descending order.
/// - Returns: 
///	values: The `k` largest elements along each last dimensional slice.
///	indices: The indices of `values` within the last dimension of `input`.
public func topK(operationName: String? = nil, input: Output, k: UInt8, sorted: Bool) throws -> (values: Output, indices: Output) { 
	var attrs = [String : Any]()
	attrs["k"] = k
	attrs["sorted"] = sorted
	let opspec = OpSpec(
		type: "TopK",
		name: (operationName ?? "Type"),
		input: [input],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return (values: op.output(at: 0), indices: op.output(at: 1))
} 

///Op peeks at the values at the specified key.  If the
///underlying container does not contain this key
/// this op will block until it does.
/// - Parameter key: 
/// - Parameter indices: 
/// - Parameter capacity: 
/// - Parameter memoryLimit: 
/// - Parameter dtypes: 
/// - Parameter container: 
/// - Parameter sharedName: 
/// - Returns: 
///	values: 
public func mapPeek(operationName: String? = nil, key: Output, indices: Output, capacity: UInt8, memoryLimit: UInt8, dtypes: [Any.Type], container: String, sharedName: String) throws -> Output { 
	var attrs = [String : Any]()
	attrs["capacity"] = capacity
	attrs["memory_limit"] = memoryLimit
	attrs["dtypes"] = dtypes
	attrs["container"] = container
	attrs["shared_name"] = sharedName
	let opspec = OpSpec(
		type: "MapPeek",
		name: (operationName ?? "Type"),
		input: [key, indices],
		attrs: attrs
	)
	let op = try self.addOperation(specification: opspec)
	return op.output(at: 0)
} 
} 
