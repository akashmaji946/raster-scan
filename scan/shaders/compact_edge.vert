// Compact Index Query - Pass 2: Edge Vertex Shader
// Like edge2D.vert in RasterScan2D
// Each vertex represents one [st, en) range from pass 1

#version 450

layout(push_constant) uniform ConstantBlock {
    uint texSize;    // ceil(sqrt(maxCount)) - texture dimension
    uint ncols;      // number of columns (unused, for compatibility)
} consts;

// Input: [st, en) pairs from pass 1 (edgeBuffer)
layout (location = 0) in uint stpos_in;
layout (location = 1) in uint enpos_in;

// Output to geometry shader
layout (location = 0) flat out uint stpos;
layout (location = 1) flat out uvec2 binrange;

out gl_PerVertex {
    vec4 gl_Position;
    float gl_PointSize;
};

float getcoord(uint ind, bool floor) {
    float pos = float(ind);
    if (!floor) {
        pos++;
    }
    float rb2 = float(consts.texSize) / 2.0;
    float loc = (pos - rb2) / rb2;
    return loc;
}

void main() {
    stpos = stpos_in;
    binrange = uvec2(stpos_in, enpos_in);
    
    // Calculate how many entries in this range
    uint cnt = enpos_in - stpos_in;
    
    // Position covers a square of size ceil(sqrt(cnt)) x ceil(sqrt(cnt))
    // But we use the global texSize for simplicity (full screen quad)
    gl_Position = vec4(-1.0, -1.0, 1.0, 1.0);  // Diagonal for geometry shader
    gl_PointSize = 1.0;
}
