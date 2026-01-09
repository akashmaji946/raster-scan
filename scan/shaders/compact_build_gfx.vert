// Compact Index Build Pass - like RasterScan2D's build2D.vert
#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference_uvec2 : require
#extension GL_ARB_gpu_shader_int64 : require

// Buffer reference for >4GB data buffer access - single element for pointer arithmetic
layout(buffer_reference, std430, buffer_reference_align = 16) buffer DataBufferRef {
    uvec4 data;  // Single element, we'll use pointer arithmetic for 64-bit indexing
};

layout(push_constant) uniform ConstantBlock {
    uvec2 minVal;
    uvec2 binRange;
    uint res;
    uint pad;  // padding for alignment
    uvec2 dataBufferAddr;  // 64-bit buffer device address
} consts;

layout (binding = 0) buffer countBuffer {
    int count[];
};

layout (binding = 1) buffer startAddrBuffer {
    uint startAddr[];
};

layout (location = 0) in uint valx;
layout (location = 1) in uint valy;
layout (location = 2) in uint valz;

out gl_PerVertex {
    vec4 gl_Position;
    float gl_PointSize;
};

void main() {
    // Reconstruct 64-bit buffer address from push constants
    uint64_t baseAddr = uint64_t(consts.dataBufferAddr.x) | (uint64_t(consts.dataBufferAddr.y) << 32);
    
    uvec2 val = uvec2(valx, valy);
    uvec2 binid = (val - consts.minVal) / consts.binRange;
    uint bin = binid.x + binid.y * consts.res;
    uint localPos = atomicAdd(count[bin], 1);
    // Global position = startAddr[bin] + local position within bin
    uint globalPos = startAddr[bin] + localPos;
    
    // Use 64-bit pointer arithmetic to bypass 4GB indexing limit
    // Each entry is 16 bytes (uvec4), so byte offset = globalPos * 16
    uint64_t byteOffset = uint64_t(globalPos) * 16u;
    uint64_t elementAddr = baseAddr + byteOffset;
    DataBufferRef dataRef = DataBufferRef(elementAddr);
    
    // Store with valid bit in MSB of rowId via buffer reference (bypasses 4GB limit)
    dataRef.data = uvec4(val, valz, uint(gl_VertexIndex) | 0x80000000u);

    gl_Position = vec4(-5, -5, 0, 1);
    gl_PointSize = 1;
}
