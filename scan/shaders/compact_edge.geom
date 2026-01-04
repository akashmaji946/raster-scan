// Compact Index Query - Pass 2: Edge Geometry Shader
// Like edge2D.geom in RasterScan2D
// Expands point to properly sized quad based on diagonal from vertex shader

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
layout (location = 1) flat in uint stposIn[];
layout (location = 2) flat in uvec2 binrangeIn[];

layout (location = 0) flat out uint qind;
layout (location = 1) flat out uint stpos;
layout (location = 2) flat out uvec2 binrange;

void main() {
    // Input is diagonal packed in gl_Position: (x1, y1, x2, y2)
    vec4 diag = gl_in[0].gl_Position;
    
    // Emit quad with proper size
    qind = qindIn[0];
    stpos = stposIn[0];
    binrange = binrangeIn[0];
    gl_Position = vec4(diag.xy, 0, 1);
    EmitVertex();
    
    qind = qindIn[0];
    stpos = stposIn[0];
    binrange = binrangeIn[0];
    gl_Position = vec4(diag.xw, 0, 1);
    EmitVertex();
    
    qind = qindIn[0];
    stpos = stposIn[0];
    binrange = binrangeIn[0];
    gl_Position = vec4(diag.zy, 0, 1);
    EmitVertex();
    
    qind = qindIn[0];
    stpos = stposIn[0];
    binrange = binrangeIn[0];
    gl_Position = vec4(diag.zw, 0, 1);
    EmitVertex();
    
    EndPrimitive();
}
