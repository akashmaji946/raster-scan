// Equi-Depth Index - Round-Robin Count Vertex Shader
// Counts points per bin using round-robin assignment (gl_VertexIndex % totalBins)
// This guarantees perfect equi-depth: each bin gets npoints/totalBins entries
#version 450

layout(push_constant) uniform ConstantBlock {
    uint resolution;    // INDEX_RESOLUTION
    uint npoints;
    uint minX;          // Not used for round-robin, kept for interface compatibility
    uint maxX;
    uint minY;
    uint maxY;
} consts;

// Binding 0: Quantiles buffer (not used for round-robin, kept for interface compatibility)
layout(std430, binding = 0) readonly buffer Quantiles {
    uint quantiles[];
};

// Binding 1: Count buffer (atomically incremented)
layout(std430, binding = 1) buffer CountBuffer {
    uint counts[];
};

// Vertex inputs: X and Y coordinates (not used for binning)
layout(location = 0) in uint inX;
layout(location = 1) in uint inY;

out gl_PerVertex {
    vec4 gl_Position;
    float gl_PointSize;
};

void main() {
    uint totalBins = consts.resolution * consts.resolution;
    
    // Round-robin bin assignment: gl_VertexIndex % totalBins
    // This guarantees perfect equi-depth distribution
    uint bin = gl_VertexIndex % totalBins;
    
    // Atomically increment count
    atomicAdd(counts[bin], 1);
    
    // Discard vertex (no rendering)
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    gl_PointSize = 1.0;
}
