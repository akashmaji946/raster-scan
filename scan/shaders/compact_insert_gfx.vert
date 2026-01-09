// Compact Index Insert Pass (Graphics) - avoids 4GB descriptor limit by using buffer device address
#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference_uvec2 : require
#extension GL_ARB_gpu_shader_int64 : require

// Buffer reference for >4GB data buffer access - single element for pointer arithmetic
layout(buffer_reference, std430, buffer_reference_align = 16) buffer DataBufferRef {
    uvec4 data;
};

layout(push_constant) uniform ConstantBlock {
    uvec2 minVal;
    uvec2 binWidth;
    uint res;
    uint scaleFactor;
    uvec2 dataBufferAddr;  // 64-bit buffer device address (split into 2x32-bit)
} consts;

// Binding 0: start addresses per bin
layout (binding = 0) buffer startAddrBuffer {
    uint startAddr[];
};

// Binding 1: extent per bin (atomicAdd allocation)
layout (binding = 1) buffer extentBuffer {
    uint extent[];
};

// Binding 2: original counts per bin (capacity base), actual capacity = origCount * scaleFactor
layout (binding = 2) buffer capacityBuffer {
    uint capacity[];
};

layout (location = 0) in uint valx;
layout (location = 1) in uint valy;
layout (location = 2) in uint valz;

out gl_PerVertex {
    vec4 gl_Position;
    float gl_PointSize;
};

void main() {
    // Reconstruct 64-bit buffer base address from push constants
    uint64_t baseAddr = uint64_t(consts.dataBufferAddr.x) | (uint64_t(consts.dataBufferAddr.y) << 32);

    uvec2 val = uvec2(valx, valy);
    uvec2 binid = (val - consts.minVal) / consts.binWidth;
    uint bin = binid.x + binid.y * consts.res;

    uint origCount = capacity[bin];
    uint binCapacity = origCount * consts.scaleFactor;

    // Allocate within bin using extent as next-free offset
    uint offsetInBin = atomicAdd(extent[bin], 1);
    if (offsetInBin < binCapacity) {
        uint globalPos = startAddr[bin] + offsetInBin;

        // 16 bytes per entry
        uint64_t byteOffset = uint64_t(globalPos) * 16u;
        uint64_t elementAddr = baseAddr + byteOffset;
        DataBufferRef dataRef = DataBufferRef(elementAddr);

        // Store with valid bit in MSB of rowId; use gl_VertexIndex as rowId within insert batch
        dataRef.data = uvec4(valx, valy, valz, uint(gl_VertexIndex) | 0x80000000u);
    }

    gl_Position = vec4(-5, -5, 0, 1);
    gl_PointSize = 1;
}
