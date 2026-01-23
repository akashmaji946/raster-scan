// Equi-Depth Index Query - Pass 2: Edge Geometry Shader
// Expands point to quad covering the entry range
#version 450

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

in gl_PerVertex {
    vec4 gl_Position;
    float gl_PointSize;
} gl_in[];

out gl_PerVertex {
    vec4 gl_Position;
};

layout(location = 0) flat in uint qind[];
layout(location = 1) flat in uint stpos[];
layout(location = 2) flat in uvec2 binrange[];

layout(location = 0) flat out uint qindOut;
layout(location = 1) flat out uint stposOut;
layout(location = 2) flat out uvec2 binrangeOut;

void main() {
    vec4 diag = gl_in[0].gl_Position;  // (x1, y1, x2, y2)
    
    // Emit quad vertices - must set flat outputs before each EmitVertex
    qindOut = qind[0];
    stposOut = stpos[0];
    binrangeOut = binrange[0];
    gl_Position = vec4(diag.x, diag.y, 0, 1);
    EmitVertex();
    
    qindOut = qind[0];
    stposOut = stpos[0];
    binrangeOut = binrange[0];
    gl_Position = vec4(diag.z, diag.y, 0, 1);
    EmitVertex();
    
    qindOut = qind[0];
    stposOut = stpos[0];
    binrangeOut = binrange[0];
    gl_Position = vec4(diag.x, diag.w, 0, 1);
    EmitVertex();
    
    qindOut = qind[0];
    stposOut = stpos[0];
    binrangeOut = binrange[0];
    gl_Position = vec4(diag.z, diag.w, 0, 1);
    EmitVertex();
    
    EndPrimitive();
}
