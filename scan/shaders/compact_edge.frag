// Compact Index Query - Pass 2: Edge Fragment Shader
// Like edge2D.frag in RasterScan2D
// Each fragment checks ONE entry - no loops!

#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference_uvec2 : require
#extension GL_ARB_gpu_shader_int64 : require

#define VALID_BIT_MASK 0x80000000u
#define ROWID_MASK 0x7FFFFFFFu

// Buffer reference for >4GB data buffer access - single element for 64-bit pointer arithmetic
layout(buffer_reference, std430, buffer_reference_align = 16) readonly buffer DataBufferRef {
    uvec4 data;  // Single element, we'll use pointer arithmetic for 64-bit indexing
};

layout(push_constant) uniform ConstantBlock {
    uint res;    // INDEX_RESOLUTION
    uint ncols;  // number of columns
    uvec2 dataBufferAddr;  // 64-bit buffer device address (split into 2x32-bit)
} consts;

// Binding 0: Result buffer - bitmap (data buffer now uses buffer reference)
layout (std430, binding = 0) buffer ResultBuffer {
    uint res[];
};

// Binding 1: Query range buffer [x1, x2, y1, y2, z1, z2]
layout (std430, binding = 1) buffer QueryBuffer {
    uint qrange[];
};

layout (location = 0) flat in uint qind;
layout (location = 1) flat in uint stpos;
layout (location = 2) flat in uvec2 binrange;  // st, en

layout (location = 0) out vec4 fragColor;

void main() {
    uvec2 coord = uvec2(gl_FragCoord.xy);
    uint binid = coord.x + coord.y * consts.res;
    
    // Reconstruct 64-bit buffer address from push constants
    uint64_t baseAddr = uint64_t(consts.dataBufferAddr.x) | (uint64_t(consts.dataBufferAddr.y) << 32);
    
    // Check if within this bin's range [st, en)
    if (binid + binrange.x <= binrange.y) {
        uint i = binid + binrange.x;
        
        // Use 64-bit pointer arithmetic to bypass 4GB indexing limit
        // Each entry is 16 bytes (uvec4), so byte offset = i * 16
        uint64_t byteOffset = uint64_t(i) * 16u;
        uint64_t elementAddr = baseAddr + byteOffset;
        DataBufferRef dataRef = DataBufferRef(elementAddr);
        
        // Read entry via buffer reference (bypasses 4GB limit)
        uvec4 entry = dataRef.data;
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
