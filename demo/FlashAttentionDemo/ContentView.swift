import SwiftUI

struct ContentView: View {
    @StateObject private var runner = MetalRunner()

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                // Header
                VStack(alignment: .leading, spacing: 4) {
                    Text("FlashAttention")
                        .font(.system(size: 28, weight: .bold, design: .monospaced))
                        .foregroundStyle(.white)
                    Text("Metal GPU Compute Demo")
                        .font(.system(size: 14, weight: .medium, design: .monospaced))
                        .foregroundStyle(.gray)
                }

                // Device card
                HStack(spacing: 12) {
                    Image(systemName: "cpu")
                        .font(.title2)
                        .foregroundStyle(.cyan)
                    VStack(alignment: .leading, spacing: 2) {
                        Text("GPU")
                            .font(.system(size: 11, weight: .medium, design: .monospaced))
                            .foregroundStyle(.gray)
                        Text(runner.deviceName)
                            .font(.system(size: 15, weight: .semibold, design: .monospaced))
                            .foregroundStyle(.white)
                    }
                    Spacer()
                }
                .padding(14)
                .background(Color.white.opacity(0.06))
                .clipShape(RoundedRectangle(cornerRadius: 10))

                // Run button
                Button(action: { runner.runAll() }) {
                    HStack(spacing: 8) {
                        if runner.isRunning {
                            ProgressView()
                                .controlSize(.small)
                                .tint(.black)
                        } else {
                            Image(systemName: "play.fill")
                        }
                        Text(runner.isRunning ? "Running..." : "Run Benchmarks")
                            .font(.system(size: 14, weight: .semibold, design: .monospaced))
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 12)
                    .background(runner.isRunning ? Color.gray : Color.cyan)
                    .foregroundStyle(.black)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                }
                .disabled(runner.isRunning)

                // Results
                if !runner.results.isEmpty {
                    VStack(alignment: .leading, spacing: 0) {
                        Text("RESULTS")
                            .font(.system(size: 11, weight: .bold, design: .monospaced))
                            .foregroundStyle(.gray)
                            .padding(.bottom, 8)

                        ForEach(runner.results) { result in
                            ResultRow(result: result)
                            if result.id != runner.results.last?.id {
                                Divider().overlay(Color.white.opacity(0.08))
                            }
                        }
                    }
                }

                Spacer()
            }
            .padding(20)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(red: 0.08, green: 0.08, blue: 0.10))
        .onAppear { runner.runAll() }
    }
}

struct ResultRow: View {
    let result: BenchmarkResult

    var statusIcon: String {
        switch result.status {
        case .pass: return "checkmark.circle.fill"
        case .fail: return "xmark.circle.fill"
        case .running: return "circle.dotted"
        }
    }

    var statusColor: Color {
        switch result.status {
        case .pass: return .green
        case .fail: return .red
        case .running: return .yellow
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Image(systemName: statusIcon)
                    .foregroundStyle(statusColor)
                    .font(.system(size: 14))
                Text(result.name)
                    .font(.system(size: 14, weight: .semibold, design: .monospaced))
                    .foregroundStyle(.white)
                Spacer()
                Text(String(format: "%.2f ms", result.timeMs))
                    .font(.system(size: 12, weight: .medium, design: .monospaced))
                    .foregroundStyle(.cyan)
            }
            Text(result.detail)
                .font(.system(size: 11, weight: .regular, design: .monospaced))
                .foregroundStyle(.gray)
        }
        .padding(.vertical, 10)
    }
}

#Preview {
    ContentView()
}
