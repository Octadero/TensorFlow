//
//  String.swift
//  TensorFlowPackageDescription
//
//  Created by Volodymyr Pavliukevych on 10/24/17.
//

import Foundation

extension String {
    /// Makes first letter to lowercase.
    internal func lowercasedFirstLetter() -> String {
        return self.prefix(1).lowercased() + self.dropFirst()
    }
    
    /// Makes first letter to uppercased.
    internal func uppercasedFirstLetter() -> String {
        return self.prefix(1).uppercased() + self.dropFirst()
    }
    /// Makes "camel case" style
    static func snakeToCamelCase(_ string: String) throws -> String {
        let unprefixed: String
        if try containsAnyLowercasedChar(string) {
            let comps = string.components(separatedBy: "_")
            unprefixed = comps.map { $0.uppercasedFirstLetter() }.joined(separator: "")
        } else {
            let comps = try snakecase(string).components(separatedBy: "_")
            unprefixed = comps.map { $0.capitalized }.joined(separator: "")
        }
        
        var prefixUnderscores = ""
        var result: String { return prefixUnderscores + unprefixed }
        return result
    }
    
    private static func containsAnyLowercasedChar(_ string: String) throws -> Bool {
        let lowercaseCharRegex = try NSRegularExpression(pattern: "[a-z]", options: .dotMatchesLineSeparators)
        let fullRange = NSRange(location: 0, length: string.unicodeScalars.count)
        return lowercaseCharRegex.firstMatch(in: string, options: .reportCompletion, range: fullRange) != nil
    }
    
    
    /// This returns the snake cased variant of the string.
    ///
    /// - Parameter string: The string to snake_case
    /// - Returns: The string snake cased from either snake_cased or camelCased string.
    private static func snakecase(_ string: String) throws -> String {
        let longUpper = try NSRegularExpression(pattern: "([A-Z\\d]+)([A-Z][a-z])", options: .dotMatchesLineSeparators)
        let camelCased = try NSRegularExpression(pattern: "([a-z\\d])([A-Z])", options: .dotMatchesLineSeparators)
        
        let fullRange = NSRange(location: 0, length: string.unicodeScalars.count)
        var result = longUpper.stringByReplacingMatches(in: string,
                                                        options: .reportCompletion,
                                                        range: fullRange,
                                                        withTemplate: "$1_$2")
        result = camelCased.stringByReplacingMatches(in: result,
                                                     options: .reportCompletion,
                                                     range: fullRange,
                                                     withTemplate: "$1_$2")
        return result.replacingOccurrences(of: "-", with: "_")
    }
}
