//
//  MTLLibraryCompiler.swift
//  FlashAttention
//
//  Created for macOS 15+ compatibility.
//

import Foundation
import Metal

/// Compiles Metal shader source code to an MTLLibrary.
///
/// On macOS 15 (Sequoia) and later, Apple restricted the use of `__asm` directives
/// in runtime-compiled Metal shaders. This affects the simdgroup async copy intrinsics
/// used by FlashAttention for optimal performance.
///
/// This compiler automatically detects the failure and falls back to using the
/// command-line `xcrun metal` toolchain, which still supports these intrinsics.
public struct MTLLibraryCompiler {

    /// Error types for library compilation
    public enum CompilerError: Error, LocalizedError {
        case runtimeCompilationFailed(String)
        case cliCompilationFailed(String)
        case metalToolchainNotFound
        case tempFileCreationFailed

        public var errorDescription: String? {
            switch self {
            case .runtimeCompilationFailed(let msg):
                return "Runtime Metal compilation failed: \(msg)"
            case .cliCompilationFailed(let msg):
                return "CLI Metal compilation failed: \(msg)"
            case .metalToolchainNotFound:
                return "Metal toolchain (xcrun metal) not found"
            case .tempFileCreationFailed:
                return "Failed to create temporary files for compilation"
            }
        }
    }

    /// Compile Metal source code to a library.
    ///
    /// First attempts runtime compilation via `MTLDevice.makeLibrary(source:)`.
    /// If that fails (e.g., due to `__asm` restrictions on macOS 15+), falls back
    /// to using the command-line Metal compiler.
    ///
    /// - Parameters:
    ///   - source: The Metal shader source code
    ///   - device: The Metal device to compile for
    ///   - options: Optional compile options (used for runtime compilation only)
    /// - Returns: The compiled MTLLibrary
    /// - Throws: CompilerError if compilation fails
    public static func makeLibrary(
        source: String,
        device: MTLDevice,
        options: MTLCompileOptions? = nil
    ) throws -> MTLLibrary {
        // First, try runtime compilation (faster when it works)
        do {
            return try device.makeLibrary(source: source, options: options)
        } catch let runtimeError {
            // Check if this is the __asm restriction error
            let errorString = String(describing: runtimeError)
            if errorString.contains("illegal string literal in 'asm'") ||
               errorString.contains("__metal_simdgroup_async_copy") {
                // Fall back to CLI compilation
                return try makeLibraryViaCLI(source: source, device: device)
            } else {
                // Some other error, re-throw
                throw CompilerError.runtimeCompilationFailed(errorString)
            }
        }
    }

    /// Compile Metal source using the command-line toolchain.
    ///
    /// This method writes the source to a temp file, compiles it with `xcrun metal`,
    /// links it with `xcrun metallib`, and loads the result.
    private static func makeLibraryViaCLI(
        source: String,
        device: MTLDevice
    ) throws -> MTLLibrary {
        let fileManager = FileManager.default
        let tempDir = fileManager.temporaryDirectory
        let uuid = UUID().uuidString

        let sourceFile = tempDir.appendingPathComponent("mfa_\(uuid).metal")
        let airFile = tempDir.appendingPathComponent("mfa_\(uuid).air")
        let metallibFile = tempDir.appendingPathComponent("mfa_\(uuid).metallib")

        defer {
            // Cleanup temp files
            try? fileManager.removeItem(at: sourceFile)
            try? fileManager.removeItem(at: airFile)
            try? fileManager.removeItem(at: metallibFile)
        }

        // Write source to temp file
        do {
            try source.write(to: sourceFile, atomically: true, encoding: .utf8)
        } catch {
            throw CompilerError.tempFileCreationFailed
        }

        // Compile to AIR
        let compileResult = shell("xcrun metal -c '\(sourceFile.path)' -o '\(airFile.path)' 2>&1")
        if compileResult.status != 0 {
            throw CompilerError.cliCompilationFailed(
                "metal compile failed: \(compileResult.output)"
            )
        }

        // Link to metallib
        let linkResult = shell("xcrun metallib '\(airFile.path)' -o '\(metallibFile.path)' 2>&1")
        if linkResult.status != 0 {
            throw CompilerError.cliCompilationFailed(
                "metallib link failed: \(linkResult.output)"
            )
        }

        // Load the compiled library
        return try device.makeLibrary(URL: metallibFile)
    }

    /// Execute a shell command and return output + status
    private static func shell(_ command: String) -> (output: String, status: Int32) {
        let task = Process()
        let pipe = Pipe()

        task.standardOutput = pipe
        task.standardError = pipe
        task.launchPath = "/bin/bash"
        task.arguments = ["-c", command]

        do {
            try task.run()
            task.waitUntilExit()
        } catch {
            return ("Failed to launch process: \(error)", -1)
        }

        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        let output = String(data: data, encoding: .utf8) ?? ""
        return (output, task.terminationStatus)
    }
}

// MARK: - Convenience extension for MTLDevice

public extension MTLDevice {
    /// Compile Metal source with automatic fallback for macOS 15+ compatibility.
    ///
    /// Use this instead of `makeLibrary(source:options:)` when your shader
    /// uses simdgroup async copy intrinsics.
    func makeLibraryWithFallback(
        source: String,
        options: MTLCompileOptions? = nil
    ) throws -> MTLLibrary {
        try MTLLibraryCompiler.makeLibrary(
            source: source,
            device: self,
            options: options
        )
    }
}
