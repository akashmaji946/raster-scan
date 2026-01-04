// Compact Index Query - Graphics Pipeline Fragment Shader
// Uses validCount buffer for efficient iteration (only valid entries, not allocated capacity)

#version 450

#define MAX_ENTRIES_PER_BIN 100000

layout(push_constant) uniform ConstantBlock {
    uvec3 minVal;
    uint resolution;
    uvec3 binWidth;
    uint nqueries;
} consts;

// Binding 0: startAddrBuffer - prefix sum offsets
layout (std430, binding = 0) buffer StartAddrBuffer {
    uint startAddr[];
};

// Binding 2: dataBuffer - CompactEntry array
layout (std430, binding = 2) buffer DataBuffer {
    uvec4 data[];
};

// Binding 3: resultBuffer - bitmap
layout (std430, binding = 3) buffer ResultBuffer {
    uint result[];
};

// Binding 7: extentBuffer - highest index written per bin
// On build: extent[bin] = original_count
// On insert: extent[bin] = max(extent[bin], new_offset + 1)
// On delete: unchanged (entries not shifted, just marked invalid)
layout (std430, binding = 7) buffer ExtentBuffer {
    uint extent[];
};

layout (location = 0) flat in uint qind;
layout (location = 1) flat in uvec4 qrange;
layout (location = 2) flat in uvec2 zrange;

layout (location = 0) out vec4 fragColor;

void main() {
    uvec2 coord = uvec2(gl_FragCoord.xy);
    uint binIdx = coord.x + coord.y * consts.resolution;
    
    uint start = startAddr[binIdx];
    // Use extent instead of (startAddr[binIdx+1] - startAddr[binIdx])
    // extent[bin] = highest index written, so we iterate [0, extent) and skip invalid entries
    uint cnt = extent[binIdx];
    
    // Safety clamp
    cnt = min(cnt, MAX_ENTRIES_PER_BIN);
    
    // Iterate through all entries in this bin
    for (uint i = 0; i < cnt; i++) {
        // Each CompactEntry is 1 uvec4 (4 uints)
        uvec4 entry = data[start + i];
        
        uint x = entry.x;
        uint y = entry.y;
        uint z = entry.z;
        uint rowIdWithValid = entry.w;
        
        // Check validity (MSB of rowId)
        if ((rowIdWithValid & 0x80000000u) == 0u) {
            continue;  // Deleted entry
        }
        
        uint rowId = rowIdWithValid & 0x7FFFFFFFu;
        
        // Check if point is in query range
        // qrange = (x1, y1, x2, y2), zrange = (z1, z2)
        if (x >= qrange.x && x <= qrange.z &&
            y >= qrange.y && y <= qrange.w &&
            z >= zrange.x && z <= zrange.y) {
            // Set bit in result bitmap
            uint wordIdx = rowId >> 5;        // rowId / 32
            uint bit = 1u << (rowId & 0x1fu); // 1 << (rowId % 32)
            atomicOr(result[wordIdx], bit);
        }
    }
    
    discard;
}
