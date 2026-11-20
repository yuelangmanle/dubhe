// swift-tools-version: 5.9
import PackageDescription

let sharedSources = [
    "gguf.cpp", "quant.cpp", "quant_q8.cpp", "tokenizer.cpp", "model.cpp",
    "engine.cpp", "mmproj.cpp", "mel.cpp", "audiomm.cpp",
]
#if os(macOS)
let platformSources = ["metal.mm", "image.mm", "audiofile.mm", "vit_gpu.mm"]
let cxxFlags: [CXXSetting] = [.unsafeFlags(["-fobjc-arc"], .when(platforms: [.macOS]))]
let linkerFlags: [LinkerSetting] = [
    "Metal", "Foundation", "ImageIO", "CoreGraphics", "CoreFoundation",
    "AVFoundation", "CoreMedia",
].map { .linkedFramework($0, .when(platforms: [.macOS])) }
#else
let platformSources = ["metal_stub.cpp", "image_stub.cpp", "audiofile_stub.cpp", "vit_gpu_stub.cpp"]
let cxxFlags: [CXXSetting] = []
let linkerFlags: [LinkerSetting] = []
#endif

let package = Package(
    name: "DubheKit",
    platforms: [.macOS(.v13)],
    products: [.library(name: "DubheKit", targets: ["DubheKit"])],
    targets: [
        .target(
            name: "CDubhe",
            sources: (sharedSources + platformSources).map { "engine/src/\($0)" },
            publicHeadersPath: "include",
            cxxSettings: [.headerSearchPath("engine/include")] + cxxFlags,
            linkerSettings: linkerFlags
        ),
        .target(name: "DubheKit", dependencies: ["CDubhe"]),
        .testTarget(name: "DubheKitTests", dependencies: ["DubheKit"]),
    ],
    cxxLanguageStandard: .cxx20
)
