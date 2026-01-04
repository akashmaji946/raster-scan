// Compact Index Query - Pass 2: Edge Geometry Shader
// Expands point to full-screen quad for texture-based parallel lookup

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

layout (location = 0) flat in uint stposIn[];
layout (location = 1) flat in uvec2 binrangeIn[];

layout (location = 0) flat out uint stpos;
layout (location = 1) flat out uvec2 binrange;

void main() {
    // Emit full-screen quad
    stpos = stposIn[0];
    binrange = binrangeIn[0];
    gl_Position = vec4(-1.0, -1.0, 0.0, 1.0);
    EmitVertex();
    
    stpos = stposIn[0];
    binrange = binrangeIn[0];
    gl_Position = vec4(-1.0, 1.0, 0.0, 1.0);
    EmitVertex();
    
    stpos = stposIn[0];
    binrange = binrangeIn[0];
    gl_Position = vec4(1.0, -1.0, 0.0, 1.0);
    EmitVertex();
    
    stpos = stposIn[0];
    binrange = binrangeIn[0];
    gl_Position = vec4(1.0, 1.0, 0.0, 1.0);
    EmitVertex();
    
    EndPrimitive();
}
