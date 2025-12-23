// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// Query Texture Linked List - Geometry Shader
// Expands point to triangle strip covering the query rectangle

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

layout (location = 0) flat in uint qindIn[];
layout (location = 1) flat in uvec4 rangeIn[];
layout (location = 2) flat in uvec2 zrangeIn[];

layout (location = 0) flat out uint qind;
layout (location = 1) flat out uvec4 qrange;
layout (location = 2) flat out uvec2 zrange;

void main() {
    // Input is two vertices of the diagonal stored in gl_Position
    vec4 diag = gl_in[0].gl_Position;

    // Emit 4 vertices for triangle strip
    qind = qindIn[0];
    qrange = rangeIn[0];
    zrange = zrangeIn[0];
    gl_Position = vec4(diag.xy, 0, 1);
    EmitVertex();

    qind = qindIn[0];
    qrange = rangeIn[0];
    zrange = zrangeIn[0];
    gl_Position = vec4(diag.xw, 0, 1);
    EmitVertex();

    qind = qindIn[0];
    qrange = rangeIn[0];
    zrange = zrangeIn[0];
    gl_Position = vec4(diag.zy, 0, 1);
    EmitVertex();

    qind = qindIn[0];
    qrange = rangeIn[0];
    zrange = zrangeIn[0];
    gl_Position = vec4(diag.zw, 0, 1);
    EmitVertex();

    EndPrimitive();
}
