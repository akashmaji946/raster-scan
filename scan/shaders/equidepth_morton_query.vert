// Equi-Depth Index Query - Graphics Pipeline Vertex Shader
// For round-robin binning, we scan all bins since points are distributed
// across bins without spatial locality
#version 450

layout(push_constant) uniform ConstantBlock {
    uint resolution;
    uint nqueries;
    uvec2 dataBufferAddr;  // Not used in vertex shader but needed for layout
} consts;

// Input: query range format is [x1, x2, y1, y2, z1, z2]
layout(location = 0) in uvec3 qpart1;  // x1, x2, y1
layout(location = 1) in uvec3 qpart2;  // y2, z1, z2

// Output to geometry shader
layout(location = 0) flat out uint qind;
layout(location = 1) flat out uvec4 range;      // Original query range (x1, y1, x2, y2)
layout(location = 2) flat out uvec2 zrangeOut;  // z1, z2

out gl_PerVertex {
    vec4 gl_Position;
    float gl_PointSize;
};

void main() {
    // Reorder from [x1, x2, y1, y2, z1, z2] to qrange=(x1,y1,x2,y2), zrange=(z1,z2)
    uvec4 qrange = uvec4(qpart1.x, qpart1.z, qpart1.y, qpart2.x);  // x1, y1, x2, y2
    uvec2 zrange = uvec2(qpart2.y, qpart2.z);  // z1, z2
    
    // For round-robin binning, we must scan ALL bins because:
    // - Points are assigned to bins based on their index, not spatial position
    // - The fragment shader will check each point against the actual query range
    
    qind = gl_VertexIndex;
    range = qrange;
    zrangeOut = zrange;
    
    // Full-screen quad covering all bins
    gl_Position = vec4(-1.0, -1.0, 1.0, 1.0);
    gl_PointSize = 1.0;
}
