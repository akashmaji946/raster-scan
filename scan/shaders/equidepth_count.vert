// Equi-Depth Binning: Count Pass - Graphics Pipeline Vertex Shader
// Uses quantile boundaries to assign equi-depth bins and count points per bin
#version 450

layout(push_constant) uniform ConstantBlock {
    uint resolution;      // INDEX_RESOLUTION (e.g., 1024)
    uint npoints;
} consts;

// Quantile boundaries (resolution + 1 values each)
layout(std430, binding = 0) readonly buffer QuantilesX {
    uint quantileX[];
};

layout(std430, binding = 1) readonly buffer QuantilesY {
    uint quantileY[];
};

// Output: Bin counts
layout(std430, binding = 2) buffer CountBuffer {
    int count[];
};

layout(location = 0) in uint valx;
layout(location = 1) in uint valy;

out gl_PerVertex {
    vec4 gl_Position;
    float gl_PointSize;
};

// Binary search to find bin index: largest i such that quantile[i] <= value
uint findBin(uint value, bool isX) {
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
    return min(lo, consts.resolution - 1);
}

void main() {
    uint binX = findBin(valx, true);
    uint binY = findBin(valy, false);
    uint bin = binX + binY * consts.resolution;
    
    atomicAdd(count[bin], 1);
    
    gl_Position = vec4(-5, -5, 0, 1);
    gl_PointSize = 1;
}
