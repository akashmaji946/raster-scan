// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#version 450

layout(push_constant) uniform ConstantBlock {
    uvec2 minVal;
    uvec2 binRange;
    uint res;
} consts;

layout (binding = 0) buffer countBuffer
{
    int count[];
};

layout (binding = 1) buffer indexBuffer
{
    uvec4 index[];
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
    uvec2 val = uvec2(valx,valy);
    uvec2 binid = (val - consts.minVal) / consts.binRange;
    uint bin = binid.x + binid.y * consts.res;
    uint pos = atomicAdd(count[bin],1);
    index[pos] = uvec4(val,valz,gl_VertexIndex);

    gl_Position = vec4(-5,-5,0,1);
    gl_PointSize = 1;
}
