// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "local-gguf-engine",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .library(
            name: "LocalGGUFEngine",
            targets: ["LocalGGUFEngine"]
        )
    ],
    targets: [
        .target(
            name: "LocalGGUFEngine"
        ),
        .testTarget(
            name: "LocalGGUFEngineTests",
            dependencies: ["LocalGGUFEngine"]
        )
    ]
)
