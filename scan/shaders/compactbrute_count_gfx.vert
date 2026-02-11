// CompactBruteScan Count Pass - counts points per bin using x,y coordinates
#version 450

layout(push_constant) uniform ConstantBlock {
    uvec2 minVal;
    uvec2 binRange;
    uint res;
} consts;

layout (binding = 0) buffer countBuffer {
    int count[];
};

layout (location = 0) in uint valx;
layout (location = 1) in uint valy;

out gl_PerVertex {
    vec4 gl_Position;
    float gl_PointSize;
};

void main() {
    uvec2 val = uvec2(valx, valy);
    uvec2 binid = (val - consts.minVal) / consts.binRange;
    uint bin = binid.x + binid.y * consts.res;
    atomicAdd(count[bin], 1);

    gl_Position = vec4(-5, -5, 0, 1);
    gl_PointSize = 1;
}
