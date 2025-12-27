// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// Range Delete shader - marks entries as invalid if they fall within a coordinate range
// Input: One vertex per grid cell, iterates through all pages in that cell's linked list
// 
// This shader deletes all points where:
//   x1 <= x <= x2 AND y1 <= y <= y2 AND z1 <= z <= z2

#version 450

// Page structure constants
#define PAGE_DATA_SIZE 1
#define PAGE_SIZE_UINTS (PAGE_DATA_SIZE * 4 + 2)  // data + count + nextPtr = 6
#define NULL_PAGE_PTR 0xFFFFFFFF
#define MAX_PAGES_PER_CELL 10000  // Safety limit per cell

// Valid bit is stored in bit 31 of the rowId field
#define VALID_BIT_MASK 0x80000000u
#define ROWID_MASK 0x7FFFFFFFu

// Push constants: delete range
layout(push_constant) uniform ConstantBlock {
    uint x1, x2;  // x range [x1, x2]
    uint y1, y2;  // y range [y1, y2]
    uint z1, z2;  // z range [z1, z2]
    uint res;     // grid resolution
} consts;

// Head pointer buffer: stores pointer to first page for each cell
layout (binding = 0) buffer headPtrBuffer {
    uint headPtr[];
};

// Page buffer: stores all pages
layout (binding = 1) buffer pageBuffer {
    uint pages[];
};

out gl_PerVertex {
    vec4 gl_Position;
    float gl_PointSize;
};

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

void main() {
    // Each vertex represents one grid cell
    uint cellIdx = gl_VertexIndex;
    uint totalCells = consts.res * consts.res;
    
    if (cellIdx >= totalCells) {
        gl_Position = vec4(-5, -5, 0, 1);
        gl_PointSize = 1;
        return;
    }
    
    // Traverse the linked list for this cell
    uint currentPage = headPtr[cellIdx];
    uint pageCount = 0;
    
    while (currentPage != NULL_PAGE_PTR && pageCount < MAX_PAGES_PER_CELL) {
        uint pageOffset = getPageOffset(currentPage);
        uint count = pages[getCountOffset(pageOffset)];
        count = min(count, PAGE_DATA_SIZE);
        
        // Check each item in the page
        for (uint i = 0; i < count; i++) {
            uint dataOffset = pageOffset + i * 4;
            uint x = pages[dataOffset];
            uint y = pages[dataOffset + 1];
            uint z = pages[dataOffset + 2];
            uint rowIdWithValid = pages[dataOffset + 3];
            
            bool isValid = (rowIdWithValid & VALID_BIT_MASK) != 0u;
            
            // Check if point is within delete range
            bool inRange = (x >= consts.x1 && x <= consts.x2) &&
                          (y >= consts.y1 && y <= consts.y2) &&
                          (z >= consts.z1 && z <= consts.z2);
            
            if (isValid && inRange) {
                // Clear the valid bit
                uint rowId = rowIdWithValid & ROWID_MASK;
                pages[dataOffset + 3] = rowId;  // Valid bit is now 0
            }
        }
        
        // Move to next page
        currentPage = pages[getNextPtrOffset(pageOffset)];
        pageCount++;
    }
    
    // Discard vertex (no actual rendering)
    gl_Position = vec4(-5, -5, 0, 1);
    gl_PointSize = 1;
}
