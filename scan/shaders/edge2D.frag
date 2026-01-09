// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#version 450

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference_uvec2 : require
#extension GL_ARB_gpu_shader_int64 : require

layout(push_constant) uniform ConstantBlock {
    uint res;
    uint ncols;
    uvec2 indexBufferAddr;
} consts;

layout(buffer_reference, std430, buffer_reference_align = 16) buffer IndexBufferRef {
    uvec4 index;
};

layout (binding = 0) buffer resBuffer
{
    uint res[];
};

layout (binding = 1) buffer rangeBuffer
{
    uint qrange[];
};

layout (location = 0) flat in uint qind;
layout (location = 1) flat in uint stpos;
layout (location = 2) flat in uvec2 binrange;

layout (location = 0) out vec4 fragColor;

void main() {
    uint64_t baseAddr = uint64_t(consts.indexBufferAddr.x) | (uint64_t(consts.indexBufferAddr.y) << 32);

    uvec2 coord = uvec2(gl_FragCoord.xy);
    uint binid = coord.x + coord.y * consts.res;
    if(binid + binrange.x <= binrange.y) {
        uint i = binid + binrange.x;

        uint64_t byteOffset = uint64_t(i) * 16u;
        IndexBufferRef idxRef = IndexBufferRef(baseAddr + byteOffset);
        uvec4 idx = idxRef.index;

        bool flag = false;
        if(idx.x >= qrange[0] && idx.y >= qrange[1]
                && idx.x <= qrange[2] && idx.y <= qrange[3]) {
            if(consts.ncols == 2) {
                flag = true;
            } else if(idx.z >= qrange[4] && idx.z <= qrange[5]) {
                flag = true;
            }
        }
        if(flag) {
            uint ind = idx.w >> 5;
            uint bit = 1 << (idx.w & 0x1f);
            atomicOr(res[ind],bit);
        }
    }
    discard;
}

