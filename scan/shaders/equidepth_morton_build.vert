// Equi-Depth Index - Round-Robin Build Vertex Shader
// Inserts points into data buffer using round-robin bin assignment
// This guarantees perfect equi-depth: each bin gets npoints/totalBins entries
#version 450
#extension GL_ARB_gpu_shader_int64 : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference_uvec2 : require

layout(push_constant) uniform ConstantBlock {
    uint resolution;    // INDEX_RESOLUTION
    uint npoints;
    uint minX;          // Not used for round-robin, kept for interface compatibility
    uint maxX;
    uint minY;
    uint maxY;
    uvec2 dataBufferAddr;  // 64-bit buffer device address
} consts;

layout(buffer_reference, std430, buffer_reference_align = 16) writeonly buffer DataBufferRef {
    uvec4 data;
};

// Binding 0: Quantiles buffer (not used for round-robin, kept for interface compatibility)
layout(std430, binding = 0) readonly buffer Quantiles {
    uint quantiles[];
};

// Binding 1: Start address buffer (prefix sum of counts)
layout(std430, binding = 1) readonly buffer StartAddr {
    uint startAddr[];
};

// Binding 2: Count buffer (atomically incremented for local offset)
layout(std430, binding = 2) buffer CountBuffer {
    uint counts[];
};

// Vertex inputs: X, Y, Z coordinates
layout(location = 0) in uint inX;
layout(location = 1) in uint inY;
layout(location = 2) in uint inZ;

out gl_PerVertex {
    vec4 gl_Position;
    float gl_PointSize;
};

void main() {
    uint totalBins = consts.resolution * consts.resolution;
    
    // Round-robin bin assignment: gl_VertexIndex % totalBins
    // This guarantees perfect equi-depth distribution
    uint bin = gl_VertexIndex % totalBins;
    
    // Get local offset within bin
    uint localOffset = atomicAdd(counts[bin], 1);
    
    // Compute global offset
    uint globalOffset = startAddr[bin] + localOffset;
    
    // Write entry to data buffer using buffer device address
    uint64_t baseAddr = uint64_t(consts.dataBufferAddr.x) | (uint64_t(consts.dataBufferAddr.y) << 32);
    uint64_t entryAddr = baseAddr + uint64_t(globalOffset) * 16u;  // 16 bytes per entry
    
    DataBufferRef dataRef = DataBufferRef(entryAddr);
    
    // Store entry: x, y, z, rowId (with valid bit)
    uint rowId = gl_VertexIndex | 0x80000000u;  // Set valid bit
    dataRef.data = uvec4(inX, inY, inZ, rowId);
    
    // Discard vertex (no rendering)
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    gl_PointSize = 1.0;
}
