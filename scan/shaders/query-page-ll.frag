// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// Query Page Linked List - Fragment Shader
// Traverses the linked list of pages and checks data against query range

#version 450

#define PAGE_DATA_SIZE 16
#define PAGE_SIZE_UINTS (PAGE_DATA_SIZE * 4 + 2)
#define NULL_PAGE_PTR 0xFFFFFFFF

layout(push_constant) uniform ConstantBlock {
    uint res;
    uint ncols;
    uint pageDataSize;
} consts;

// Page buffer: stores all pages
layout (binding = 0) buffer pageBuffer {
    uint pages[];
};

// Result buffer: bitmap of matching row IDs
layout (binding = 1) buffer resBuffer {
    uint res[];
};

// Range buffer: query ranges
layout (binding = 2) buffer rangeBuffer {
    uint qrange[];
};

layout (location = 0) flat in uint pagePtr;

layout (location = 0) out vec4 fragColor;

// Get the offset into pageBuffer for a given page index
uint getPageOffset(uint pageIdx) {
    return pageIdx * PAGE_SIZE_UINTS;
}

// Get the count offset within a page
uint getCountOffset(uint pageOffset) {
    return pageOffset + consts.pageDataSize * 4;
}

// Get the nextPtr offset within a page
uint getNextPtrOffset(uint pageOffset) {
    return pageOffset + consts.pageDataSize * 4 + 1;
}

// Read data item from page
uvec4 readData(uint pageOffset, uint itemIdx) {
    uint dataOffset = pageOffset + itemIdx * 4;
    return uvec4(
        pages[dataOffset + 0],
        pages[dataOffset + 1],
        pages[dataOffset + 2],
        pages[dataOffset + 3]
    );
}

void main() {
    // Only process once per page (use fragment at (0,0) position within the quad)
    uvec2 coord = uvec2(gl_FragCoord.xy);
    
    // Traverse the linked list starting from pagePtr
    uint currentPage = pagePtr;
    
    while (currentPage != NULL_PAGE_PTR) {
        uint pageOffset = getPageOffset(currentPage);
        uint count = pages[getCountOffset(pageOffset)];
        
        // Clamp count to valid range
        count = min(count, consts.pageDataSize);
        
        // Check each item in the page
        for (uint i = 0; i < count; i++) {
            uvec4 data = readData(pageOffset, i);
            
            // Check if data matches query range
            // data = (x, y, z, rowId)
            bool flag = false;
            
            if (data.x >= qrange[0] && data.y >= qrange[1] &&
                data.x <= qrange[2] && data.y <= qrange[3]) {
                if (consts.ncols == 2) {
                    flag = true;
                } else if (data.z >= qrange[4] && data.z <= qrange[5]) {
                    flag = true;
                }
            }
            
            if (flag) {
                // Set bit in result bitmap
                uint rowId = data.w;
                uint ind = rowId >> 5;       // rowId / 32
                uint bit = 1 << (rowId & 0x1f);  // 1 << (rowId % 32)
                atomicOr(res[ind], bit);
            }
        }
        
        // Move to next page
        currentPage = pages[getNextPtrOffset(pageOffset)];
    }
    
    discard;
}
