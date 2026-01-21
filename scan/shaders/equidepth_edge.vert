// Equi-Depth Index Query - Pass 2: Edge Vertex Shader
// Like compact_edge.vert - reads [st, en) pairs and computes quad for rasterization
#version 450

layout(push_constant) uniform ConstantBlock {
    uint res;    // INDEX_RESOLUTION (texture dimension)
    uint ncols;  // number of columns
    uvec2 dataBufferAddr;  // 64-bit buffer device address
} consts;

// Input: [st, en) pairs from pass 1 (edgeBuffer)
layout(location = 0) in uvec2 erange;

// Output to geometry shader
layout(location = 0) flat out uint qind;
layout(location = 1) flat out uint stpos;
layout(location = 2) flat out uvec2 binrange;

out gl_PerVertex {
    vec4 gl_Position;
    float gl_PointSize;
};

float getcoord(uint ind, bool floor) {
    float pos = float(ind);
    if (!floor) {
        pos++;
    }
    float rb2 = float(consts.res) / 2.0;
    float loc = (pos - rb2) / rb2;
    return loc;
}

void main() {
    stpos = erange.x;
    uint stbin = 0;  // Relative start
    uint enbin = erange.y - erange.x;  // Count of entries
    
    uint sxbin = stbin % consts.res;
    uint exbin = enbin % consts.res;
    
    uint sybin = stbin / consts.res;
    uint eybin = enbin / consts.res;
    
    float x1, y1, x2, y2;
    
    if (sybin == eybin) {
        // Single row - edge case
        x1 = getcoord(sxbin, true);
        x2 = getcoord(exbin, false);
    } else {
        // Multiple rows - full width quad
        x1 = -1.0;
        x2 = 1.0;
    }
    y1 = getcoord(sybin, true);
    y2 = getcoord(eybin, false);
    
    qind = gl_VertexIndex;
    binrange = erange;
    
    // Pack diagonal into gl_Position for geometry shader
    gl_Position = vec4(x1, y1, x2, y2);
    gl_PointSize = 1.0;
}
