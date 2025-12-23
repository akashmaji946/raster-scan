// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// Query Page Linked List - Geometry Shader
// Expands point to a quad for processing

#version 450

layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

in gl_PerVertex {
    vec4 gl_Position;
    float gl_PointSize;
} gl_in[];

out gl_PerVertex {
    vec4 gl_Position;
};

layout (location = 0) flat in uint pagePtrIn[];

layout (location = 0) flat out uint pagePtr;

void main() {
    vec4 diag = gl_in[0].gl_Position;

    pagePtr = pagePtrIn[0];
    gl_Position = vec4(diag.xy, 0, 1);
    EmitVertex();

    pagePtr = pagePtrIn[0];
    gl_Position = vec4(diag.xw, 0, 1);
    EmitVertex();

    pagePtr = pagePtrIn[0];
    gl_Position = vec4(diag.zy, 0, 1);
    EmitVertex();

    pagePtr = pagePtrIn[0];
    gl_Position = vec4(diag.zw, 0, 1);
    EmitVertex();

    EndPrimitive();
}
