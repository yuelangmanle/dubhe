// Byte-fallback tokens can end partway through a Unicode character.
struct UTF8Stream {
    private var pending: [UInt8] = []

    mutating func append(_ bytes: some Sequence<UInt8>) -> String {
        pending.append(contentsOf: bytes)
        var boundary = pending.endIndex
        var lead = boundary
        for byte in pending.reversed() {
            lead -= 1
            if byte & 0xC0 == 0x80 { continue }
            let width: Int
            switch byte {
            case 0xC2...0xDF: width = 2
            case 0xE0...0xEF: width = 3
            case 0xF0...0xF4: width = 4
            default: width = 1
            }
            if boundary - lead < width { boundary = lead }
            break
        }
        let text = String(decoding: pending[..<boundary], as: UTF8.self)
        pending.removeFirst(boundary)
        return text
    }

    mutating func finish() -> String {
        let text = String(decoding: pending, as: UTF8.self)
        pending.removeAll()
        return text
    }
}
