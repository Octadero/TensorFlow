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

public struct SwiftReserved {
    public static let keywords = ["associatedtype", "class", "deinit", "enum", "extension", "fileprivate", "func", "import", "init", "inout", "internal", "let", "open", "operator", "private", "protocol", "public", "static", "struct", "subscript", "typealias", "var", "break", "case", "continue", "default", "defer", "do", "else", "fallthrough", "for", "guard", "if", "in", "repeat", "return", "switch", "where", "while", "as", "Any", "catch", "false", "is", "nil", "rethrows", "super", "self", "Self", "throw", "throws", "true", "try"]
}

/// Producer of source code.
///     Processing `MutableTensorflow_OpDef` elements and produce functions.
class SourceCodeProducer {
    /// Container of `MutableTensorflow_OpDef` operations, array.
	var operations: [MutableTensorflow_OpDef] = []
	
    /// Variable to hold access lefel in produced source code.
	let accessLevel = Template.Access.public
    
    /// Data for writer
    var fileContent = Data()
    var filePath: String? = nil
    
    func namingConventionsRecommendation(argumentName: String) -> String? {
        if SwiftReserved.keywords.contains(argumentName) {
            return "`" + argumentName + "`"
        }
        return nil
    }
    
    func process(description input: String, firstLine: Bool) -> String {
        var description = input.replacingOccurrences(of: "*", with: " * ")
        description = description.replacingOccurrences(of: "\n", with: "\n/// ")
        description = description.replacingOccurrences(of: "^", with: "// ^")
        return (description.isEmpty || !firstLine ? "" : "///") + description
    }
    
	/// Preparing operations function.
	func updateOps() {
		for (opIndex, op) in  operations.enumerated() {
			
            operations[opIndex].description_p = process(description: op.description_p, firstLine: true)
            operations[opIndex].summary = process(description: op.summary, firstLine: true)

            for (inputArgumentIndex, inputArgument) in op.inputArg.enumerated() {
                operations[opIndex].inputArg[inputArgumentIndex].description_p = process(description: inputArgument.description_p, firstLine: false)
            }

            for (attributeIndex, attribute) in op.attr.enumerated() {
                operations[opIndex].attr[attributeIndex].description_p = process(description: attribute.description_p, firstLine: false)
            }

            for (outputIndex, output) in op.outputArg.enumerated() {
                operations[opIndex].outputArg[outputIndex].description_p = process(description: output.description_p, firstLine: false)
            }
            
			if (op.name.lowercased() == "where") {
				operations[opIndex].name = "`where`"
			} else  if (op.name.lowercased() == "switch") {
				operations[opIndex].name =  "`switch`"
			}
			
            for (argumentIndex, argument) in op.inputArg.enumerated() {
                if let recommendation = namingConventionsRecommendation(argumentName: argument.name) {
                    operations[opIndex].inputArg[argumentIndex].name = recommendation
                }
            }
            
            var attributeRemoveIndexes = [Int]()
            
			for (attributeIndex, att) in op.attr.enumerated() {
				if (att.name == "T") {
					if (att.type == "type") {
                        // operations[opIndex].attr[attributeIndex].type = "Any.Type"
                        // FIXME: waiting for issue with that renaming.
                        attributeRemoveIndexes.append(attributeIndex)
					}
				} else if (att.name == "dtype") {
					if (att.type == "type") {
						operations[opIndex].attr[attributeIndex].type = "Any.Type"
					} else if(att.allowedValues.list.type.count > 0) {
						operations[opIndex].attr[attributeIndex].type = "[Any.Type]"
					}
				} else if (att.name == "type") {
					operations[opIndex].attr[attributeIndex].type = "Any.Type"
				}
                
				if (att.type == "func") {
					operations[opIndex].attr[attributeIndex].type = "Tensorflow_NameAttrList"
				} else if (att.type == "int") {
					operations[opIndex].attr[attributeIndex].type = "UInt8"
				} else if (att.type == "bool") {
					operations[opIndex].attr[attributeIndex].type = "Bool"
				} else if (att.type == "list(string)") {
					operations[opIndex].attr[attributeIndex].type = "[String]"
				} else if (att.type == "string") {
					operations[opIndex].attr[attributeIndex].type = "String"
                } else if (att.type == "tensor") {
                    operations[opIndex].attr[attributeIndex].type = "Tensor"
                } else if (att.type == "list(tensor)") {
					operations[opIndex].attr[attributeIndex].type = "[Tensor]"
				} else if(att.type == "list(bool)") {
					operations[opIndex].attr[attributeIndex].type = "[Bool]"
				} else if(att.type == "float") {
					operations[opIndex].attr[attributeIndex].type = "Float"
                } else if(att.type == "list(float)") {
                    operations[opIndex].attr[attributeIndex].type = "[Float]"
                } else if (att.type == "list(attr)") {
					operations[opIndex].attr[attributeIndex].type = "[Tensorflow_NameAttrList]"
				} else if(att.type == "list(int)") {
					operations[opIndex].attr[attributeIndex].type = "[Int64]"
                } else if (att.type == "type") {
                    operations[opIndex].attr[attributeIndex].type = "Any.Type"
                } else if (att.type == "list(type)") {
					operations[opIndex].attr[attributeIndex].type = "[Any.Type]"
                } else if (att.type == "shape") {
                    operations[opIndex].attr[attributeIndex].type = "Shape"
                } else if(att.type == "list(shape)") {
					operations[opIndex].attr[attributeIndex].type = "[Shape]"
				}
			}
            for index in attributeRemoveIndexes.reversed() {
                operations[opIndex].attr.remove(at: index)
            }
        }
	}

	func process(operations: [MutableTensorflow_OpDef]) throws {
		self.operations = operations
		updateOps()
	}
	
    /// Writer
	func write(`in` filePath: String) throws {
		self.fileContent = Data()
        self.filePath = filePath
        
		add(Template.header, terminator: Template.newLine)
		add(Template.import, terminator: Template.newLine)
		
        add("extension Scope {", terminator: Template.newLine)
        
		for operation in operations {
			var funcArgs: [(name: String, description: String, type: String)] = operation.inputArg.map {(name: $0.name,
                                                                                                         description: $0.description_p,
                                                                                                         type: $0.typeAttr)}
            
			funcArgs.append(contentsOf: operation.attr.map{ (name: $0.name,
                                                             description: $0.description_p,
                                                             type: $0.type) })

			if !operation.summary.isEmpty || !operation.description_p.isEmpty || funcArgs.count != 0 || operation.outputArg.count != 0 {
				add(Template.newLine)
				
                add(operation.summary, terminator: (operation.description_p.isEmpty ? "" : Template.newLine))
				add(operation.description_p, terminator: Template.newLine)
				
                try funcArgs.forEach({ (argument) in
                    add("/// - Parameter ")
                    add(try String.snakeToCamelCase(argument.name).lowercasedFirstLetter())
                    add(": ")
                    add(argument.description, terminator: Template.newLine)
                })
				
				if operation.outputArg.count != 0 {
					add("/// - Returns: ", terminator: Template.newLine)
					operation.outputArg.forEach({ (output) in
						add("///\t")
						add(output.name)
						add(": ")
						add(output.description_p, terminator: Template.newLine)
					})
				}
			}
						
			add(accessLevel.label)
			add(Template.function)
            
			add((try String.snakeToCamelCase(operation.name)).lowercasedFirstLetter())
			add(Template.openRoundBracket)
			
            add("operationName: String? = nil")
            
			if operation.hasAttributeOrInputArgs {
                add(Template.commaMark)
                
				for (index, funcArgument) in operation.inputArg.enumerated() {
					add(try String.snakeToCamelCase(funcArgument.name).lowercasedFirstLetter())
					add(": Output")
					if index < (operation.inputArg.count - 1) || operation.attr.count != 0 {
						add(Template.commaMark)
					}
				}
				
				for (index, funcArgument) in operation.attr.enumerated() {
					add(try String.snakeToCamelCase(funcArgument.name).lowercasedFirstLetter())
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
					add(try String.snakeToCamelCase(outputArg.name).lowercasedFirstLetter())
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
			
			if operation.attr.count != 0 {
				add("\tvar attrs = [String : Any]()", terminator: Template.newLine)
				
                try operation.attr.forEach({ (attribute) in
                    add("\tattrs[\"\(attribute.name)\"] = \(try String.snakeToCamelCase(attribute.name).lowercasedFirstLetter())", terminator: Template.newLine)
                })
                
			} else {
				add("\tlet attrs = [String : Any]()", terminator: "\n")
			}
			
			add("\tlet opspec = OpSpec(", terminator: Template.newLine)
			add("\t\ttype: \"\(operation.name)\",", terminator: Template.newLine)
            
			add("\t\tname: (operationName ?? \"Type\"),", terminator: Template.newLine)
			add("\t\tinput: [")
			for (index, inputArg) in operation.inputArg.enumerated() {
				add(try String.snakeToCamelCase(inputArg.name).lowercasedFirstLetter())
				if index < (operation.inputArg.count - 1) {
					add(Template.commaMark)
				}
			}
			add("],", terminator: Template.newLine)

			add("\t\tattrs: attrs", terminator: Template.newLine)
			add("\t)", terminator: Template.newLine)

			add("\tlet op = try self.addOperation(specification: opspec)", terminator: Template.newLine)
			if operation.hasOutputArgs {
				add("\treturn ")
				add(Template.openRoundBracket)
				for (index, outputArg) in operation.outputArg.enumerated() {
					add("\(try String.snakeToCamelCase(outputArg.name).lowercasedFirstLetter()): op.output(at: \(index))")
					if index < (operation.outputArg.count - 1) {
						add(Template.commaMark)
					}
				}
				add(Template.closeRoundBracket, terminator: Template.newLine)
			}
			
			if operation.hasOneOutputArg {
				add("\treturn op.output(at: 0)", terminator: Template.newLine)
			}
			
			if operation.hasNoOutputArg {
				add("\treturn op", terminator: Template.newLine)
			}
			
			add(Template.closeCurlyBracket, terminator: Template.newLine)
		}
        
        /// Close extension bracket.
        add(Template.closeCurlyBracket, terminator: Template.newLine)
        try fileContent.write(to: URL(fileURLWithPath: filePath))
    }
	
	internal func add(_ string: String, terminator: String? = nil) {
        #if DEBUG
		print(string, separator: "", terminator: terminator ?? "")
        #endif
        
        if let data = string.data(using: .utf8) {
            fileContent.append(data)
            if let terminator = terminator {
                if let terminatorData = terminator.data(using: .utf8) {
                    fileContent.append(terminatorData)
                }
            }
        }
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
		static let `import` = "import Foundation\nimport Proto"
		static let newLine = "\n"
		static let function = " func "
		static let returnMark = "-> "
		static let throwsMark = " throws "
		static let commaMark = ", "
	}
}
