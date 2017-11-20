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

import Foundation
import CTensorFlow
import Proto
import CAPI

public enum SummaryError: Error {
    case notReady
    case incorrectImageSize
}

///Summaries provide a way to export condensed information about a model, which is then accessible in tools such as [TensorBoard](https://www.tensorflow.org/get_started/summaries_and_tensorboard).
/// TensorBoard operates by reading TensorFlow events files,
/// which contain summary data that you can generate when running TensorFlow.
/// Here's the general lifecycle for summary data within TensorBoard.
public class Summary {
	
    /// Storage for outputs added to summary.
    var values = [Output]()
    
    let scope: Scope
    
    /// Associated scope
    public init(scope: Scope) {
        self.scope = scope
    }
    
    /// Add scalar value output to `Summary`.
    public func scalar(output: Output, key: String) throws {
        let tensor = try Tensor(dimensions: [Int64](), values: [key])
        let const = try scope.addConst(tensor: tensor, as: key + "TagConst").defaultOutput
        let operation = try scope.scalarSummary(operationName: key, tags: const, values: output)
        self.values.append(operation)
    }
	
    /// Add histogram value output to `Summary`.
	public func histogram(output: Output, key: String) throws {
		let tensor = try Tensor(dimensions: [Int64](), values: [key])
		let const = try scope.addConst(tensor: tensor, as: key + "TagConst").defaultOutput
		let operation = try scope.histogramSummary(operationName: key, tag: const, values: output)
		self.values.append(operation)
	}
    
    public enum Channel: Int {
        case grayscale = 1
        case rgb = 3
        case rgba = 4
        
        var value: Int {
            return rawValue
        }
    }
    
    /// Represents BadColor options for `Summary` image `Tensor`.
    public struct BadColor {
        public let channel: Channel
        public let colorComponents: [UInt8]
        
        public init(channel: Channel, colorComponents: [UInt8]) {
            self.channel = channel
            self.colorComponents = colorComponents
        }
        
        public static let `default` = BadColor(channel: .grayscale, colorComponents: [UInt8(255)])
    }
    
    /// Represents Image size for `Summary` image `Tensor`.
    public struct ImageSize {
        public let width: Int
        public let height: Int
        
        public var points: Int {
            return width * height
        }
        
        public init(width: Int, height: Int) {
            self.height = height
            self.width = width
        }
    }
    
    /// Add image to `Summary`
    public func images(name: String, batchSize: Int, size: ImageSize, values: [Float], maxImages: UInt8 = 255, badColor: BadColor = BadColor.default) throws {
        guard batchSize * size.points * badColor.channel.value == values.count else {
            throw SummaryError.incorrectImageSize
        }
        
        let imageTensor = try Tensor(dimensions: [batchSize, size.width, size.height, badColor.channel.value], values: values)
        let const = try scope.addConst(tensor: imageTensor, as: "Image-\(name)-\(UUID().uuidString)-Const")
        try images(name: name, output: const.defaultOutput, maxImages: maxImages, badColor: badColor)
    }
    
    ///Add images to `Summary`.
    /// Outputs a `Summary` protocol buffer with images.
    /// The summary has up to `max_images` summary values containing images. The
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
    ///    summary: Scalar. Serialized `Summary` protocol buffer.
    public func images(name: String, output: Output, maxImages: UInt8 = 255, badColor: BadColor = BadColor.default) throws {
        
        let tensor = try Tensor(dimensions: [Int64](), values: [name])
        let const = try scope.addConst(tensor: tensor, as:  "Tag-\(name)-\(UUID().uuidString)-Const").defaultOutput

        let badColorTensor = try Tensor(dimensions: [Int64(badColor.channel.value)], values: badColor.colorComponents)

        let operation = try scope.imageSummary(operationName: name + UUID().uuidString, tag: const, tensor: output, maxImages: maxImages, badColor: badColorTensor)
        self.values.append(operation)
    }
    
    /// It merge all outputs and provide them as one output.
    ///     This op creates a [`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
    ///     protocol buffer that contains the union of all the values in the input
    ///     summaries.
    public func merged(identifier: String = UUID().uuidString) throws -> Output {
        let output = try scope.mergeSummary(operationName: "MergeSummary-\(identifier)", inputs: self.values, n: UInt8(self.values.count))
        self.values.removeAll()
        return output
    }
    

    
}
