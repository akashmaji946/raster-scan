// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#version 450

layout(push_constant) uniform ConstantBlock {
    uint res;
    uint ncols;
} consts;

layout (binding = 0) buffer indexBuffer
{
    uvec4 index[];
};

layout (binding = 1) buffer resBuffer
{
    uint res[];
};

layout (binding = 2) buffer rangeBuffer
{
    uint qrange[];
};

layout (location = 0) flat in uint qind;
layout (location = 1) flat in uint stpos;
layout (location = 2) flat in uvec2 binrange;

layout (location = 0) out vec4 fragColor;

void main() {
    uvec2 coord = uvec2(gl_FragCoord.xy);
    uint binid = coord.x + coord.y * consts.res;
    if(binid + binrange.x <= binrange.y) {
        uint i = binid + binrange.x;
        bool flag = false;
        if(index[i].x >= qrange[0] && index[i].y >= qrange[1]
                && index[i].x <= qrange[2] && index[i].y <= qrange[3]) {
            if(consts.ncols == 2) {
                flag = true;
            } else if(index[i].z >= qrange[4] && index[i].z <= qrange[5]) {
                flag = true;
            }
        }
        if(flag) {
            uint ind = index[i].w >> 5;
            uint bit = 1 << (index[i].w & 0x1f);
            atomicOr(res[ind],bit);
        }
    }
    discard;
}

