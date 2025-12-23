// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// Query Page Linked List - Vertex Shader
// Processes page pointers from the texture query stage

#version 450

#define NULL_PAGE_PTR 0xFFFFFFFF

layout(push_constant) uniform ConstantBlock {
    uint res;
    uint ncols;
    uint pageDataSize;
} consts;

// Input: page pointer from previous stage
layout (location = 0) in uint pagePtr;

layout (location = 0) flat out uint pagePtrOut;

out gl_PerVertex {
    vec4 gl_Position;
    float gl_PointSize;
};

float getcoord(uint ind, bool floor) {
    float pos = float(ind);
    if (!floor) {
        pos++;
    }
    float rb2 = float(consts.res) / 2;
    float loc = float(pos - rb2) / rb2;
    return loc;
}

void main() {
    pagePtrOut = pagePtr;
    
    // Create a full-screen quad for this page
    // The fragment shader will handle the actual data checking
    gl_Position = vec4(-1, -1, 1, 1);
    gl_PointSize = 1;
}
