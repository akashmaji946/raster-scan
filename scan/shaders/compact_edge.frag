// Compact Index Query - Pass 2: Edge Fragment Shader
// Each fragment checks ONE entry - no loops!
// This is the key to matching RasterScan2D performance

#version 450

#define VALID_BIT_MASK 0x80000000u
#define ROWID_MASK 0x7FFFFFFFu

layout(push_constant) uniform ConstantBlock {
    uint texSize;    // ceil(sqrt(maxCount))
    uint ncols;      // number of columns
} consts;

// Binding 0: Data buffer - CompactEntry structs (4 uints each = uvec4)
layout (std430, binding = 0) buffer DataBuffer {
    uvec4 data[];  // Each entry is 1 uvec4: x, y, z, rowId (MSB = valid)
};

// Binding 1: Result buffer - bitmap
layout (std430, binding = 1) buffer ResultBuffer {
    uint result[];
};

// Binding 2: Query range buffer [x1, x2, y1, y2, z1, z2]
layout (std430, binding = 2) buffer QueryBuffer {
    uint qrange[];
};

layout (location = 0) flat in uint stpos;
layout (location = 1) flat in uvec2 binrange;  // st, en

layout (location = 0) out vec4 fragColor;

void main() {
    uvec2 coord = uvec2(gl_FragCoord.xy);
    
    // Convert 2D texture coord to 1D index within this bin's range
    uint localIdx = coord.x + coord.y * consts.texSize;
    
    // Global entry index
    uint entryIdx = binrange.x + localIdx;
    
    // Check if within this bin's range
    if (entryIdx >= binrange.y) {
        discard;
    }
    
    // Read entry (single uvec4)
    uvec4 entry = data[entryIdx];
    
    uint x = entry.x;
    uint y = entry.y;
    uint z = entry.z;
    uint rowIdWithValid = entry.w;
    
    // Check validity (MSB of rowId)
    if ((rowIdWithValid & VALID_BIT_MASK) == 0u) {
        discard;  // Deleted entry
    }
    
    uint rowId = rowIdWithValid & ROWID_MASK;
    
    // Query format: [x1, x2, y1, y2, z1, z2]
    uint x1 = qrange[0];
    uint x2 = qrange[1];
    uint y1 = qrange[2];
    uint y2 = qrange[3];
    uint z1 = qrange[4];
    uint z2 = qrange[5];
    
    // Check if point is in query range
    if (x >= x1 && x <= x2 &&
        y >= y1 && y <= y2 &&
        z >= z1 && z <= z2) {
        // Set bit in result bitmap
        uint wordIdx = rowId >> 5;        // rowId / 32
        uint bit = 1u << (rowId & 0x1fu); // 1 << (rowId % 32)
        atomicOr(result[wordIdx], bit);
    }
    
    discard;
}
