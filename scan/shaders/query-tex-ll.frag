// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// Query Texture Linked List - Fragment Shader
// For each cell in the query range, traverses the linked list and checks data

#version 450

#define PAGE_DATA_SIZE 1
#define PAGE_SIZE_UINTS 8  // = 8
#define NULL_PAGE_PTR 0xFFFFFFFF
#define MAX_PAGES_PER_CELL 100000  // Safety limit to prevent infinite loops

// Valid bit is stored in bit 31 of the rowId field
#define VALID_BIT_MASK 0x80000000u
#define ROWID_MASK 0x7FFFFFFFu

layout(push_constant) uniform ConstantBlock {
    uvec2 minVal;
    uvec2 binRange;
    uint res;
} consts;

// Head pointer buffer: stores pointer to first page for each cell
layout (binding = 0) buffer headPtrBuffer {
    uint headPtr[];
};

// Page buffer: stores all pages
layout (binding = 1) buffer pageBuffer {
    uint pages[];
};

// Result buffer: bitmap of matching row IDs
layout (binding = 2) buffer resBuffer {
    uint res[];
};

layout (location = 0) flat in uint qind;
layout (location = 1) flat in uvec4 qrange;
layout (location = 2) flat in uvec2 zrange;

layout (location = 0) out vec4 fragColor;

// Get the offset into pageBuffer for a given page index
uint getPageOffset(uint pageIdx) {
    return pageIdx * PAGE_SIZE_UINTS;
}

// Get the count offset within a page
uint getCountOffset(uint pageOffset) {
    return pageOffset + PAGE_DATA_SIZE * 4;
}

// Get the nextPtr offset within a page
uint getNextPtrOffset(uint pageOffset) {
    return pageOffset + PAGE_DATA_SIZE * 4 + 1;
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
    uvec2 coord = uvec2(gl_FragCoord.xy);
    uint binid = coord.x + coord.y * consts.res;
    
    // Get head pointer for this cell
    uint currentPage = headPtr[binid];
    
    // Traverse the linked list with safety limit
    uint pageCount = 0;
    while (currentPage != NULL_PAGE_PTR && pageCount < MAX_PAGES_PER_CELL) {
        uint pageOffset = getPageOffset(currentPage);
        uint count = pages[getCountOffset(pageOffset)];
        
        // Clamp count to valid range
        count = min(count, PAGE_DATA_SIZE);
        
        // Check each item in the page
        for (uint i = 0; i < count; i++) {
            uvec4 data = readData(pageOffset, i);
            
            // Extract valid bit and rowId from data.w
            uint rowIdWithValid = data.w;
            bool isValid = (rowIdWithValid & VALID_BIT_MASK) != 0u;
            uint rowId = rowIdWithValid & ROWID_MASK;
            
            // Skip deleted entries (valid bit = 0)
            if (!isValid) {
                continue;
            }
            
            // Check if data matches query range
            // data = (x, y, z, rowId_with_valid)
            // qrange = (x1, y1, x2, y2), zrange = (z1, z2)
            bool flag = false;
            
            if (data.x >= qrange.x && data.y >= qrange.y &&
                data.x <= qrange.z && data.y <= qrange.w) {
                // Always check z range for 3D data
                if (data.z >= zrange.x && data.z <= zrange.y) {
                    flag = true;
                }
            }
            
            if (flag) {
                // Set bit in result bitmap
                uint ind = rowId >> 5;       // rowId / 32
                uint bit = 1 << (rowId & 0x1f);  // 1 << (rowId % 32)
                atomicOr(res[ind], bit);
            }
        }
        
        // Move to next page
        currentPage = pages[getNextPtrOffset(pageOffset)];
        pageCount++;
    }
    
    discard;
}
