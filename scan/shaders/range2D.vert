// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#version 450

layout(push_constant) uniform ConstantBlock {
    uvec2 minVal;
    uvec2 binRange;
    uint res;
} consts;

// (x1,y1,x2,y2)
layout (location = 0) in uvec4 qrange;
// (z1,z2) in case it is 3D
layout (location = 1) in uvec2 zrange;

layout (location = 0) flat out uint qind;
layout (location = 1) flat out uvec4 range;
layout (location = 2) flat out uvec2 zrangeOut;

out gl_PerVertex {
    vec4 gl_Position;
    float gl_PointSize;
};

float getcoord(uint ind, bool floor) {
    float pos = float(ind);
    if(!floor) {
        pos ++;
    }
    float rb2 = float(consts.res) / 2;
    float loc = float(pos - rb2) / rb2;
    return loc;
}


void main() {

    uvec2 stbin = (qrange.xy - consts.minVal) / consts.binRange;
    uvec2 enbin = (qrange.zw - consts.minVal) / consts.binRange;

    float x1,y1,x2,y2;

    x1 = getcoord(stbin.x,true);
    y1 = getcoord(stbin.y,true);
    x2 = getcoord(enbin.x,false);
    y2 = getcoord(enbin.y,false);

    qind = gl_VertexIndex;
    range = qrange;
    zrangeOut = zrange;

    gl_Position = vec4(x1,y1,x2,y2);
    gl_PointSize = 1;
}

