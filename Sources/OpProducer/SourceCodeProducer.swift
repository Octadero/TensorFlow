//
//  SourceCodeProducer.swift
//  TensorFlowPackageDescription
//
//  Created by Volodymyr Pavliukevych on 9/28/17.
//

import Foundation
import Proto

extension String {
    /// Makes first letter to lowercase.
	internal func lowercasedFirstLetter() -> String {
		return self.prefix(1).lowercased() + self.dropFirst()
	}
	/* https://github.com/SwiftGen/StencilSwiftKit/blob/b6841ced0679a7d5082419709f6ec971f6bee776/Sources/Filters%2BStrings%2BLettercase.swift
	static func snakeToCamelCase(_ string: String, stripLeading: Bool) throws -> String {
		let unprefixed: String
		if try containsAnyLowercasedChar(string) {
			let comps = string.components(separatedBy: "_")
			unprefixed = comps.map { upperFirstLetter($0) }.joined(separator: "")
		} else {
			let comps = try snakecase(string).components(separatedBy: "_")
			unprefixed = comps.map { $0.capitalized }.joined(separator: "")
		}
		
		// only if passed true, strip the prefix underscores
		var prefixUnderscores = ""
		var result: String { return prefixUnderscores + unprefixed }
		if stripLeading {
			return result
		}
		for scalar in string.unicodeScalars {
			guard scalar == "_" else { break }
			prefixUnderscores += "_"
		}
		return result
	}
	*/
}

/// Producer of source code.
///     Processing `MutableTensorflow_OpDef` elements and produce functions.
class SourceCodeProducer {
    /// Container of `MutableTensorflow_OpDef` operations, array.
	var operations: [MutableTensorflow_OpDef] = []
	
    /// Variable to hold access lefel in produced source code.
	let accessLevel = Template.Access.public
	/// Preparing operations function.
	func updateOps() {
		
		for (opIndex, op) in  operations.enumerated() {
			
			var str = op.description_p.replacingOccurrences(of: "*", with: " * ")
			//str = str.replacingOccurrences(of: "\n", with: "\n// ")
			str = str.replacingOccurrences(of: "^", with: "// ^")
			operations[opIndex].description_p = str
			
			if (op.name.lowercased() == "where") {
				operations[opIndex].name = "where_p"
			} else  if (op.name.lowercased() == "switch") {
				operations[opIndex].name =  "switch_p"
			}
			
			for (attributeIndex, att) in op.attr.enumerated() {
				if (att.name == "T") {
					if (att.type == "type") {
						operations[opIndex].attr[attributeIndex].type = "Tensorflow_DataType"
					}
				} else if (att.name == "dtype") {
					if (att.type == "type") {
						operations[opIndex].attr[attributeIndex].type = "Tensorflow_DataType"
					} else if(att.allowedValues.list.type.count > 0) {
						operations[opIndex].attr[attributeIndex].type = "[Any]"
					}
				}else if (att.name == "type") {
					operations[opIndex].attr[attributeIndex].type = "Tensorflow_DataType"
				}
				if (att.type == "func") {
					operations[opIndex].attr[attributeIndex].type = "Tensorflow_NameAttrList"
				} else if (att.type == "int") {
					operations[opIndex].attr[attributeIndex].type = "UInt8"
				} else if (att.type == "bool") {
					operations[opIndex].attr[attributeIndex].type = "Bool"
				} else if (att.type == "list(string)") {
					operations[opIndex].attr[attributeIndex].type = "[Data]"
				} else if (att.type == "string") {
					operations[opIndex].attr[attributeIndex].type = "String"
				} else if (att.type == "list(tensor)") {
					operations[opIndex].attr[attributeIndex].type = "[Tensorflow_TensorProto]"
				} else if(att.type == "list(bool)") {
					operations[opIndex].attr[attributeIndex].type = "[Bool]"
				} else if(att.type == "list(float)") {
					operations[opIndex].attr[attributeIndex].type = "[Float]"
				} else if (att.type == "list(attr)") {
					operations[opIndex].attr[attributeIndex].type = "[Tensorflow_NameAttrList]"
				} else if(att.type == "list(int)") {
					operations[opIndex].attr[attributeIndex].type = "[Int64]"
				} else if (att.type == "list(type)") {
					operations[opIndex].attr[attributeIndex].type = "[Tensorflow_DataType]"
				} else if(att.type == "list(shape)") {
					operations[opIndex].attr[attributeIndex].type = "[Shape]"
				}
			}
		}
	}

	func process(operations: [MutableTensorflow_OpDef]) throws {
		self.operations = operations
		updateOps()
	}
	
    /// Writer
	func write(`in` filePath: String) throws {
		add("write file to: \(filePath)")
		add(Template.header, terminator: Template.newLine)
		add(Template.import, terminator: Template.newLine)
		
		for operation in operations {
			var funcArgs: [(name: String, description: String, type: String)] = operation.inputArg.map {(name: $0.name, description: $0.description_p, type: $0.typeAttr)}
			funcArgs.append(contentsOf: operation.attr.map{ (name: $0.name, description: $0.description_p, type: $0.type) })

			if !operation.summary.isEmpty || !operation.description_p.isEmpty || funcArgs.count != 0 || operation.outputArg.count != 0 {
				add(Template.newLine)
				add(Template.openCommentBracket, terminator: Template.newLine)
				
				add(operation.summary, terminator: Template.newLine)
				add(operation.description_p, terminator: Template.newLine)
				
				funcArgs.forEach({ (argument) in
					add("- Parameter ")
					add(argument.name)
					add(": ")
					add(argument.description, terminator: Template.newLine)
				})
				
				if operation.outputArg.count != 0 {
					add("- Returns: ", terminator: Template.newLine)
					operation.outputArg.forEach({ (output) in
						add("\t")
						add(output.name)
						add(": ")
						add(output.description_p, terminator: Template.newLine)
					})
					add(Template.newLine)
				}
				
				add(Template.closeCommentBracket, terminator: Template.newLine)
			}
			
			add(Template.newLine)
			
			add(accessLevel.label)
			add(Template.function)
			add(operation.name.lowercasedFirstLetter())
			add(Template.openRoundBracket)
			
			//TODO: - Improve names, snakeToCamelCase
			if operation.hasAttributeOrInputArgs {
				for (index, funcArgument) in operation.inputArg.enumerated() {
					add(funcArgument.name)
					add(": ")
					add("Operation")
					if index < (operation.inputArg.count - 1) || operation.attr.count != 0 {
						add(Template.commaMark)
					}
				}
				
				for (index, funcArgument) in operation.attr.enumerated() {
					add(funcArgument.name)
					add(": ")
					add(funcArgument.type)
					if index < (operation.attr.count - 1) {
						add(Template.commaMark)
					}
				}
			}
	
			add(Template.closeRoundBracket)
			
			add(Template.throwsMark)
			add(Template.returnMark)
			
			if operation.hasOutputArgs {
				add(Template.openRoundBracket)
				for (index, outputArg) in operation.outputArg.enumerated() {
					add(outputArg.name)
					add(": Output")
					if index < (operation.outputArg.count - 1) {
						add(Template.commaMark)
					}
				}
				add(Template.closeRoundBracket)
			}
			
			if operation.hasOneOutputArg {
				add("Output")
			}
			
			if operation.hasNoOutputArg {
				add("Operation")
			}
			
			add(Template.openCurlyBracket, terminator: Template.newLine)
			
			add("if let error = scope.error { \n\tthrow error \n}\n\n")
			if operation.attr.count != 0 {
				add("var attrs = [String : Any]()", terminator: Template.newLine)
				
				operation.attr.forEach({ (attribute) in
					//FIXME: attribute.name snakeToCamelCase
					add("attrs[\"\(attribute.name)\"] = \(attribute.name)", terminator: Template.newLine)
				})
			} else {
				add("let attrs = [String : Any]()", terminator: "\n")
			}
			
			add("\tlet opspec = OpSpec(", terminator: Template.newLine)
			add("\ttype: \"\(operation.name)\",", terminator: Template.newLine)
			add("\tname: \"Type\",", terminator: Template.newLine)
			add("\tinput: [")
			//FIXME: attribute.name snakeToCamelCase
			for (index, inputArg) in operation.inputArg.enumerated() {
				add(inputArg.name)
				if index < (operation.inputArg.count - 1) {
					add(Template.commaMark)
				}
			}
			add("],", terminator: Template.newLine)

			add("\t\t\tattrs: attrs", terminator: Template.newLine)
			add("\t)", terminator: Template.newLine)

			add("let op = try scope.addOperation(specification: opspec)", terminator: Template.newLine)
			if operation.hasOutputArgs {
				add("return ")
				add(Template.openRoundBracket)
				for (index, outputArg) in operation.outputArg.enumerated() {
					add("\(outputArg.name): op.output(at: \(index))")
					if index < (operation.outputArg.count - 1) {
						add(Template.commaMark)
					}
				}
				add(Template.closeRoundBracket, terminator: Template.newLine)
			}
			
			if operation.hasOneOutputArg {
				add("return op.output(at: 0)", terminator: Template.newLine)
			}
			
			if operation.hasNoOutputArg {
				add("return op", terminator: Template.newLine)
			}
			
			add(Template.closeCurlyBracket, terminator: Template.newLine)
		}
	}
	
	internal func add(_ string: String, terminator: String? = nil) {
		print(string, separator: "", terminator: terminator ?? "")
	}
	
    /// List of settings and templates.
	internal struct Template {
		
		enum Access: String {
			case `public`
			case `internal`
			case `private`
			
			var label: String  {
				return self.rawValue
			}
		}
		
		static let header = "/* HEADER */"
		static let openCommentBracket = "/*"
		static let closeCommentBracket = "*/"
		static let openRoundBracket = "("
		static let closeRoundBracket = ")"
		static let openCurlyBracket = " { "
		static let closeCurlyBracket = "} "
		static let `import` = "import Foundation"
		static let newLine = "\n"
		static let function = " func "
		static let returnMark = "-> "
		static let throwsMark = " throws "
		static let commaMark = ", "
	}
}
