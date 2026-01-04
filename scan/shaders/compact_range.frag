// Compact Index Query - Pass 1: Range Fragment Shader
// For each bin in query range, stores [startAddr, startAddr+count) to result buffer
// This is similar to range2D.frag in RasterScan2D

#version 450

layout(push_constant) uniform ConstantBlock {
    uvec3 minVal;
    uint resolution;
    uvec3 binWidth;
    uint nqueries;
} consts;

// Binding 0: Start addresses (offsets) per bin
layout (std430, binding = 0) buffer StartAddrBuffer {
    uint startAddr[];
};

// Binding 1: Extent per bin (highest index written, used for iteration)
layout (std430, binding = 1) buffer ExtentBuffer {
    uint extent[];
};

// Binding 2: Result count buffer - VkDrawIndirectCommand format
// [0] = vertexCount (numRanges), [1] = instanceCount (1), [2] = firstVertex (0), [3] = firstInstance (0)
layout (std430, binding = 2) buffer ResCountBuffer {
    uint resct[];
};

// Binding 3: Result buffer - stores [st, en) pairs for each bin in query range
layout (std430, binding = 3) buffer ResultBuffer {
    uint result[];
};

layout (location = 0) flat in uint qind;
layout (location = 1) flat in uvec4 qrange;
layout (location = 2) flat in uvec2 zrange;

layout (location = 0) out vec4 fragColor;

void main() {
    uvec2 coord = uvec2(gl_FragCoord.xy);
    uint binIdx = coord.x + coord.y * consts.resolution;
    
    uint st = startAddr[binIdx];
    uint ext = extent[binIdx];
    uint en = st + ext;
    
    if (ext > 0) {
        // Atomically get index for this bin's result
        // resct[0] = vertexCount for VkDrawIndirectCommand
        uint ind = atomicAdd(resct[0], 1);
        
        // Store [st, en) pair
        result[ind * 2] = st;
        result[ind * 2 + 1] = en;
        
        // Set instanceCount = 1 (only need to set once, but atomic ensures it's set)
        atomicMax(resct[1], 1u);
    }
    
    discard;
}
