// Equi-Depth Binning: Build Pass - Graphics Pipeline Vertex Shader
// Uses quantile boundaries to assign equi-depth bins and insert points into dataBuffer
#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference_uvec2 : require
#extension GL_ARB_gpu_shader_int64 : require

// Buffer reference for >4GB data buffer access
layout(buffer_reference, std430, buffer_reference_align = 16) buffer DataBufferRef {
    uvec4 data;
};

layout(push_constant) uniform ConstantBlock {
    uint resolution;      // INDEX_RESOLUTION (e.g., 1024)
    uint npoints;
    uint pad0;
    uint pad1;
    uvec2 dataBufferAddr; // 64-bit buffer device address
} consts;

// Quantile boundaries (resolution + 1 values each)
layout(std430, binding = 0) readonly buffer QuantilesX {
    uint quantileX[];
};

layout(std430, binding = 1) readonly buffer QuantilesY {
    uint quantileY[];
};

// Count buffer (for atomicAdd to get local position)
layout(std430, binding = 2) buffer CountBuffer {
    int count[];
};

// Start address buffer (prefix sum of counts)
layout(std430, binding = 3) readonly buffer StartAddrBuffer {
    uint startAddr[];
};

layout(location = 0) in uint valx;
layout(location = 1) in uint valy;
layout(location = 2) in uint valz;

out gl_PerVertex {
    vec4 gl_Position;
    float gl_PointSize;
};

// Binary search to find bin index: largest i such that quantile[i] <= value
uint findBin(uint value, bool isX) {
    uint lo = 0;
    uint hi = consts.resolution;
    
    while (lo < hi) {
        uint mid = (lo + hi + 1) / 2;
        uint boundary;
        if (isX) {
            boundary = quantileX[mid];
        } else {
            boundary = quantileY[mid];
        }
        
        if (boundary <= value) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    return min(lo, consts.resolution - 1);
}

void main() {
    // Reconstruct 64-bit buffer address
    uint64_t baseAddr = uint64_t(consts.dataBufferAddr.x) | (uint64_t(consts.dataBufferAddr.y) << 32);
    
    uint binX = findBin(valx, true);
    uint binY = findBin(valy, false);
    uint bin = binX + binY * consts.resolution;
    
    uint localPos = atomicAdd(count[bin], 1);
    uint globalPos = startAddr[bin] + localPos;
    
    // Use 64-bit pointer arithmetic for >4GB support
    uint64_t byteOffset = uint64_t(globalPos) * 16u;
    uint64_t elementAddr = baseAddr + byteOffset;
    DataBufferRef dataRef = DataBufferRef(elementAddr);
    
    // Store (x, y, z, rowId with valid bit)
    dataRef.data = uvec4(valx, valy, valz, uint(gl_VertexIndex) | 0x80000000u);
    
    gl_Position = vec4(-5, -5, 0, 1);
    gl_PointSize = 1;
}
