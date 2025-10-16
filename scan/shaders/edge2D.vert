// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#version 450

layout(push_constant) uniform ConstantBlock {
    uint res;
} consts;

layout (location = 0) in uvec2 erange;


layout (location = 0) flat out uint qind;
layout (location = 1) flat out uint stpos;
layout (location = 2) flat out uvec2 binrange;

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
    stpos = erange.x;
    uint stbin = (erange.x - stpos);
    uint enbin = (erange.y - stpos);

    uint sxbin = stbin % consts.res;
    uint exbin = enbin % consts.res;

    uint sybin = stbin / consts.res;
    uint eybin = enbin / consts.res;

    float x1,y1,x2,y2;

    if(sybin == eybin) {
        // edge
        x1 = getcoord(sxbin,true);
        x2 = getcoord(exbin,false);
    } else {
        // quad
        x1 = -1;
        x2 = 1;
    }
    y1 = getcoord(sybin,true);
    y2 = getcoord(eybin,false);

    qind = gl_VertexIndex;
    binrange = erange;

    gl_Position = vec4(x1,y1,x2,y2);
    gl_PointSize = 1;
}

