// Compact Index Query - Graphics Pipeline Vertex Shader
// Converts query range to screen-space quad covering the bins

#version 450

layout(push_constant) uniform ConstantBlock {
    uvec3 minVal;
    uint resolution;
    uvec3 binWidth;
    uint nqueries;
} consts;

// Input: query range format is [x1, x2, y1, y2, z1, z2]
// We read as two uvec3s and reorder
layout (location = 0) in uvec3 qpart1;  // x1, x2, y1
layout (location = 1) in uvec3 qpart2;  // y2, z1, z2

// Output to geometry shader
layout (location = 0) flat out uint qind;
layout (location = 1) flat out uvec4 range;
layout (location = 2) flat out uvec2 zrangeOut;

out gl_PerVertex {
    vec4 gl_Position;
    float gl_PointSize;
};

float getcoord(uint ind, bool floor) {
    float pos = float(ind);
    if(!floor) {
        pos++;
    }
    float rb2 = float(consts.resolution) / 2.0;
    float loc = (pos - rb2) / rb2;
    return loc;
}

void main() {
    // Reorder from [x1, x2, y1, y2, z1, z2] to qrange=(x1,y1,x2,y2), zrange=(z1,z2)
    uvec4 qrange = uvec4(qpart1.x, qpart1.z, qpart1.y, qpart2.x);  // x1, y1, x2, y2
    uvec2 zrange = uvec2(qpart2.y, qpart2.z);  // z1, z2
    
    // Compute bin indices for query range
    uvec2 stbin = (qrange.xy - consts.minVal.xy) / consts.binWidth.xy;
    uvec2 enbin = (qrange.zw - consts.minVal.xy) / consts.binWidth.xy;
    
    // Clamp to resolution
    stbin = min(stbin, uvec2(consts.resolution - 1));
    enbin = min(enbin, uvec2(consts.resolution - 1));
    
    // Convert to normalized device coordinates
    float x1 = getcoord(stbin.x, true);
    float y1 = getcoord(stbin.y, true);
    float x2 = getcoord(enbin.x, false);
    float y2 = getcoord(enbin.y, false);
    
    qind = gl_VertexIndex;
    range = qrange;
    zrangeOut = zrange;
    
    // Pack diagonal into gl_Position (x1, y1, x2, y2)
    gl_Position = vec4(x1, y1, x2, y2);
    gl_PointSize = 1.0;
}
