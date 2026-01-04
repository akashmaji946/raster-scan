// Compact Index Query - Pass 2: Edge Fragment Shader
// Like edge2D.frag in RasterScan2D
// Each fragment checks ONE entry - no loops!

#version 450

#define VALID_BIT_MASK 0x80000000u
#define ROWID_MASK 0x7FFFFFFFu

layout(push_constant) uniform ConstantBlock {
    uint res;    // INDEX_RESOLUTION
    uint ncols;  // number of columns
} consts;

// Binding 0: Data buffer - CompactEntry structs (4 uints each = uvec4)
layout (std430, binding = 0) buffer DataBuffer {
    uvec4 data[];  // Each entry is 1 uvec4: x, y, z, rowId (MSB = valid)
};

// Binding 1: Result buffer - bitmap
layout (std430, binding = 1) buffer ResultBuffer {
    uint res[];
};

// Binding 2: Query range buffer [x1, x2, y1, y2, z1, z2]
layout (std430, binding = 2) buffer QueryBuffer {
    uint qrange[];
};

layout (location = 0) flat in uint qind;
layout (location = 1) flat in uint stpos;
layout (location = 2) flat in uvec2 binrange;  // st, en

layout (location = 0) out vec4 fragColor;

void main() {
    uvec2 coord = uvec2(gl_FragCoord.xy);
    uint binid = coord.x + coord.y * consts.res;
    
    // Check if within this bin's range [st, en)
    if (binid + binrange.x <= binrange.y) {
        uint i = binid + binrange.x;
        
        // Read entry
        uvec4 entry = data[i];
        uint x = entry.x;
        uint y = entry.y;
        uint z = entry.z;
        uint rowIdWithValid = entry.w;
        
        // Check validity (MSB of rowId) - skip deleted entries
        if ((rowIdWithValid & VALID_BIT_MASK) != 0u) {
            uint rowId = rowIdWithValid & ROWID_MASK;
            
            bool flag = false;
            // Check if point is in query range
            if (x >= qrange[0] && y >= qrange[2] &&
                x <= qrange[1] && y <= qrange[3]) {
                if (consts.ncols == 2) {
                    flag = true;
                } else if (z >= qrange[4] && z <= qrange[5]) {
                    flag = true;
                }
            }
            
            if (flag) {
                // Set bit in result bitmap
                uint ind = rowId >> 5;        // rowId / 32
                uint bit = 1u << (rowId & 0x1fu); // 1 << (rowId % 32)
                atomicOr(res[ind], bit);
            }
        }
    }
    
    discard;
}
