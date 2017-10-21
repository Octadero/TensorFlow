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
import Proto
import CAPI

public typealias TensorflowNameAttrList = Tensorflow_NameAttrList

/// Container for operation attributes.
struct MutableAttrDef {
	
	var name: String
	var type: String
	var defaultValue: Tensorflow_AttrValue
	var hasDefaultValue: Bool
	var description_p: String
	var hasMinimum_p: Bool
	var minimum: Int64
	var allowedValues: Tensorflow_AttrValue
	
	init(att:Tensorflow_OpDef.AttrDef) {
		self.name = att.name
		self.type = att.type
		self.defaultValue = att.defaultValue
		self.hasDefaultValue = att.hasDefaultValue
		self.description_p = att.description_p
		self.hasMinimum_p = att.hasMinimum_p
		self.minimum = att.minimum
		self.allowedValues = att.allowedValues
	}
}
/// Container for operation arguments.
struct MutableArgDef {
	
	var name: String = String()
	var description_p: String = String()
	var type: Tensorflow_DataType = Tensorflow_DataType.dtInvalid
	var typeAttr: String = String()
	var numberAttr: String = String()
	var typeListAttr: String = String()
	var isRef: Bool = false
	
	init(arg:Tensorflow_OpDef.ArgDef) {
		self.name = arg.name
		self.description_p = arg.description_p
		self.type = arg.type
		self.typeAttr = arg.typeAttr
		self.numberAttr = arg.numberAttr
		self.typeListAttr = arg.typeListAttr
		self.isRef = arg.isRef
	}
}

/// Container for operation.
struct MutableTensorflow_OpDef{
	var jsonString: String? = String()
	var name: String = String()
	var inputArg: [MutableArgDef] = []
	var outputArg: [MutableArgDef] = []
	var attr: [MutableAttrDef] = []
	
	var summary: String = String()
	var description_p: String = String()
	var isCommutative: Bool = false
	var isAggregate: Bool = false
	var isStateful: Bool = false
	var allowsUninitializedInput: Bool = false
	
	var hasOutputArgs:Bool = false
	var hasOneOutputArg:Bool = false
	var hasNoOutputArg:Bool = false
	
	var hasAttributeOrInputArgs:Bool = false
	
	init(op:Tensorflow_OpDef) {
		self.jsonString = try?  op.jsonString()
		
		self.name = op.name
		
		var inputArrayArgs = Array<MutableArgDef>()
		
		for arg in op.inputArg{
			let mArg = MutableArgDef.init(arg:arg )
			inputArrayArgs.append(mArg)
		}
		
		self.inputArg = inputArrayArgs
		
		var outputArrayArgs = Array<MutableArgDef>()
		for arg in op.outputArg{
			let mArg = MutableArgDef.init(arg:arg )
			outputArrayArgs.append(mArg)
		}
		
		self.outputArg = outputArrayArgs
		
		var attArray = Array<MutableAttrDef>()
		for att in op.attr {
			let mAttr = MutableAttrDef.init(att:att )
			/*
			if(att.type == "type") {
				print("SKIPPING ->>>> ", att.allowedValues.list)
			} else if(att.type == "list(type)") {
				print("SKIPPING list ->>>> ", att.allowedValues.list)
			} else{
				attArray.append(mAttr)
			}
			*/
			attArray.append(mAttr)
		}
		
		self.attr = attArray
		
		if(self.attr.count > 0) {
			hasAttributeOrInputArgs = true
		}
		if(self.inputArg.count > 0) {
			hasAttributeOrInputArgs = true
		}
		
		
		self.summary = op.summary
		self.description_p = op.description_p
		self.isCommutative = op.isCommutative
		self.isAggregate = op.isAggregate
		self.isStateful = op.isStateful
		self.allowsUninitializedInput = op.allowsUninitializedInput
		
		
		if(self.outputArg.count > 0) {
			if(self.outputArg.count == 1) {
				self.hasOneOutputArg = true
			} else{
				self.hasOutputArgs = true
			}
		} else{
			self.hasNoOutputArg = true
		}
	}
}
