// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#version 450

layout(push_constant) uniform ConstantBlock {
    uvec2 minVal;
    uvec2 binRange;
    uint res;
} consts;

layout (binding = 0) buffer stCountBuffer
{
    uint stcount[];
};

layout (binding = 1) buffer enCountBuffer
{
    uint encount[];
};

layout (binding = 2) buffer indexBuffer
{
    uvec4 index[];
};

layout (binding = 2) buffer resCountBuffer
{
    uint resct[];
};

layout (binding = 3) buffer resBuffer
{
    uint res[];
};

layout (location = 0) flat in uint qind;
layout (location = 1) flat in uvec4 qrange;
layout (location = 2) flat in uvec2 zrange;

layout (location = 0) out vec4 fragColor;

void main() {
    uvec2 coord = uvec2(gl_FragCoord.xy);
    uint binid = coord.x + coord.y * consts.res;
    uint st = stcount[binid];
    uint en = encount[binid];
    
    // Only process bins that have points (en > st means bin has data)
    if (en > st) {
        uint ind = atomicAdd(resct[0], 1);
        res[ind * 2] = st;
        res[ind * 2 + 1] = en;
        resct[1] = 1;
    }

    discard;
}
