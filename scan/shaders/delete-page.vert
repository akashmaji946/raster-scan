// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// Delete shader - marks entries as invalid by clearing the valid bit
// Input: rowId to delete (passed as vertex attribute)
// 
// OPTIMIZATION: With 1-item-per-page, the page index equals the rowId!
// So we can directly access the page in O(1) instead of searching.

#version 450

// Page structure constants
#define PAGE_DATA_SIZE 1
#define PAGE_SIZE_UINTS (PAGE_DATA_SIZE * 4 + 2)  // data + count + nextPtr = 6

// Valid bit is stored in bit 31 of the rowId field
#define VALID_BIT_MASK 0x80000000u
#define ROWID_MASK 0x7FFFFFFFu

// Page buffer: stores all pages
layout (binding = 1) buffer pageBuffer {
    uint pages[];
};

// Input: rowId to delete
layout (location = 0) in uint deleteRowId;

out gl_PerVertex {
    vec4 gl_Position;
    float gl_PointSize;
};

// Get the offset into pageBuffer for a given page index
uint getPageOffset(uint pageIdx) {
    return pageIdx * PAGE_SIZE_UINTS;
}

void main() {
    uint targetRowId = deleteRowId;
    
    // With 1-item-per-page, page index == rowId
    // Directly access the page and clear the valid bit
    uint pageOffset = getPageOffset(targetRowId);
    
    // The rowId is stored at offset 3 within the first (and only) data item
    uint dataOffset = pageOffset + 3;
    uint rowIdWithValid = pages[dataOffset];
    
    // Clear the valid bit (set bit 31 to 0)
    // Keep the rowId value intact
    uint rowId = rowIdWithValid & ROWID_MASK;
    pages[dataOffset] = rowId;  // Valid bit is now 0
    
    // Discard vertex (no actual rendering)
    gl_Position = vec4(-5, -5, 0, 1);
    gl_PointSize = 1;
}
