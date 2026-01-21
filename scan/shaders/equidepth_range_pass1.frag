// Equi-Depth Index Query - Pass 1: Range Fragment Shader
// Outputs [start, end) pairs for bins that intersect the query
// Similar to compact_query_gfx.frag but outputs to edgeBuffer instead of iterating
#version 450

layout(push_constant) uniform ConstantBlock {
    uint resolution;
    uint nqueries;
} consts;

// Binding 0: Start addresses (offsets) per bin
layout(std430, binding = 0) readonly buffer StartAddrBuffer {
    uint startAddr[];
};

// Binding 1: Extent per bin (count of valid entries)
layout(std430, binding = 1) readonly buffer ExtentBuffer {
    uint extent[];
};

// Binding 2: maxBuffer - [0]=vertexCount for drawIndirect
layout(std430, binding = 2) buffer MaxBuffer {
    uint maxBuf[];
};

// Binding 3: edgeBuffer - output [st, en) pairs
layout(std430, binding = 3) buffer EdgeBuffer {
    uvec2 edgeBuf[];
};

layout(location = 0) flat in uint qind;
layout(location = 1) flat in uvec4 qrange;  // x1, y1, x2, y2
layout(location = 2) flat in uvec2 zrange;  // z1, z2

layout(location = 0) out vec4 fragColor;

void main() {
    uvec2 coord = uvec2(gl_FragCoord.xy);
    uint binIdx = coord.x + coord.y * consts.resolution;
    
    uint start = startAddr[binIdx];
    uint cnt = extent[binIdx];
    
    // Only output if bin has entries
    if (cnt > 0) {
        // Atomically get slot in edgeBuffer
        uint slot = atomicAdd(maxBuf[0], 1);
        
        // Store [start, start + count) pair
        edgeBuf[slot] = uvec2(start, start + cnt);
    }
    
    discard;
}
