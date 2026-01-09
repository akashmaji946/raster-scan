// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#version 450

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference_uvec2 : require
#extension GL_ARB_gpu_shader_int64 : require

layout(push_constant) uniform ConstantBlock {
    uvec2 minVal;
    uvec2 binRange;
    uint res;
    uint pad;
    uvec2 indexBufferAddr;
} consts;

layout (binding = 0) buffer countBuffer
{
    int count[];
};

layout(buffer_reference, std430, buffer_reference_align = 16) buffer IndexBufferRef {
    uvec4 index;
};

// TODO For now vertex position is RowID
layout (location = 0) in uint valx;
layout (location = 1) in uint valy;
layout (location = 2) in uint valz;

out gl_PerVertex {
    vec4 gl_Position;
    float gl_PointSize;
};

void main() {
    uint64_t baseAddr = uint64_t(consts.indexBufferAddr.x) | (uint64_t(consts.indexBufferAddr.y) << 32);

    uvec2 val = uvec2(valx,valy);
    uvec2 binid = (val - consts.minVal) / consts.binRange;
    uint bin = binid.x + binid.y * consts.res;
    uint pos = atomicAdd(count[bin],1);

    uint64_t byteOffset = uint64_t(pos) * 16u;
    IndexBufferRef indexRef = IndexBufferRef(baseAddr + byteOffset);
    indexRef.index = uvec4(val, valz, gl_VertexIndex);

    gl_Position = vec4(-5,-5,0,1);
    gl_PointSize = 1;
}
