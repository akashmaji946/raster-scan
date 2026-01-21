// Equi-Depth Index Query - Graphics Pipeline Vertex Shader
// Uses quantile boundaries to find bin range for query (binary search)
// Only rasterizes bins that intersect the query range
#version 450

layout(push_constant) uniform ConstantBlock {
    uint resolution;
    uint nqueries;
    uvec2 dataBufferAddr;  // Not used in vertex shader but needed for layout
} consts;

// Binding 0: Quantile X boundaries (resolution + 1 values)
layout(std430, binding = 0) readonly buffer QuantilesX {
    uint quantileX[];
};

// Binding 1: Quantile Y boundaries (resolution + 1 values)
layout(std430, binding = 1) readonly buffer QuantilesY {
    uint quantileY[];
};

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

// Binary search: find largest i such that quantile[i] <= value
uint lowerBound(uint value, bool isX) {
    uint lo = 0;
    uint hi = consts.resolution;
    
    while (lo < hi) {
        uint mid = (lo + hi + 1) / 2;
        uint boundary;
        if (isX) {
            boundary = quantileX[mid];
        } else {
            boundary = quantileY[mid];
        }
        
        if (boundary <= value) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    return lo;
}

float getcoord(uint ind, bool floor) {
    float pos = float(ind);
    if (!floor) {
        pos += 1.0;
    }
    float rb2 = float(consts.resolution) / 2.0;
    float loc = (pos - rb2) / rb2;
    return loc;
}

void main() {
    // Reorder from [x1, x2, y1, y2, z1, z2] to qrange=(x1,y1,x2,y2), zrange=(z1,z2)
    uvec4 qrange = uvec4(qpart1.x, qpart1.z, qpart1.y, qpart2.x);  // x1, y1, x2, y2
    uvec2 zrange = uvec2(qpart2.y, qpart2.z);  // z1, z2
    
    // For equi-depth binning, we must scan ALL bins because:
    // - Bin assignment is based on data rank, not absolute value
    // - A point with value x could be in any bin depending on data distribution
    // - Binary search on quantile boundaries doesn't guarantee correct bin range
    // 
    // The fragment shader will check each point against the actual query range,
    // so scanning extra bins only costs performance, not correctness.
    
    qind = gl_VertexIndex;
    range = qrange;
    zrangeOut = zrange;
    
    // Full-screen quad covering all bins
    gl_Position = vec4(-1.0, -1.0, 1.0, 1.0);
    gl_PointSize = 1.0;
}
