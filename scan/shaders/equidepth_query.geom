// Equi-Depth Index Query - Geometry Shader
// Expands point to quad covering the bin range
#version 450

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

layout(location = 0) flat in uint qind[];
layout(location = 1) flat in uvec4 range[];
layout(location = 2) flat in uvec2 zrangeIn[];

layout(location = 0) flat out uint qindOut;
layout(location = 1) flat out uvec4 qrange;
layout(location = 2) flat out uvec2 zrange;

void main() {
    vec4 diag = gl_in[0].gl_Position;  // (x1, y1, x2, y2)
    
    // Emit quad vertices - must set flat outputs before each EmitVertex
    qindOut = qind[0];
    qrange = range[0];
    zrange = zrangeIn[0];
    gl_Position = vec4(diag.x, diag.y, 0, 1);
    EmitVertex();
    
    qindOut = qind[0];
    qrange = range[0];
    zrange = zrangeIn[0];
    gl_Position = vec4(diag.z, diag.y, 0, 1);
    EmitVertex();
    
    qindOut = qind[0];
    qrange = range[0];
    zrange = zrangeIn[0];
    gl_Position = vec4(diag.x, diag.w, 0, 1);
    EmitVertex();
    
    qindOut = qind[0];
    qrange = range[0];
    zrange = zrangeIn[0];
    gl_Position = vec4(diag.z, diag.w, 0, 1);
    EmitVertex();
    
    EndPrimitive();
}
