// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// Mark-free shader: scans all pages and marks deleted ones as free in bitmap
// This shader processes one page per vertex invocation
// Input: gl_VertexIndex = page index to check

#version 450

// Page structure constants
#define PAGE_DATA_SIZE 1
#define PAGE_SIZE_UINTS 8  // data + count + nextPtr + padding = 8
#define NULL_PAGE_PTR 0xFFFFFFFF

// Valid bit is stored in bit 31 of the rowId field
#define VALID_BIT_MASK 0x80000000u
#define ROWID_MASK 0x7FFFFFFFu

layout(push_constant) uniform ConstantBlock {
    uint maxPages;      // Total number of pages to scan
    uint allocCounter;  // Current allocation counter (pages 0 to allocCounter-1 are potentially used)
} consts;

// Page buffer: stores all pages
layout (binding = 0) buffer pageBuffer {
    uint pages[];
};

// Free bitmap: 1 = free, 0 = allocated
layout (binding = 1) buffer freeBitmapBuffer {
    uint freeBitmap[];
};

// Free count: number of free pages
layout (binding = 2) buffer freeCountBuffer {
    uint freeCount;
};

out gl_PerVertex {
    vec4 gl_Position;
    float gl_PointSize;
};

// Get the offset into pageBuffer for a given page index
uint getPageOffset(uint pageIdx) {
    return pageIdx * PAGE_SIZE_UINTS;
}

void main() {
    uint pageIdx = uint(gl_VertexIndex);
    
    // Only process pages that are within bounds
    if (pageIdx >= consts.maxPages) {
        gl_Position = vec4(-5, -5, 0, 1);
        gl_PointSize = 1;
        return;
    }
    
    uint pageOffset = getPageOffset(pageIdx);
    
    // Check if the page's data is valid (valid bit set in rowId field)
    // The rowId with valid bit is stored at offset 3 within the page data
    uint rowIdWithValid = pages[pageOffset + 3];
    bool isValid = (rowIdWithValid & VALID_BIT_MASK) != 0;
    
    // Also check if count is 0 (empty page)
    uint count = pages[pageOffset + PAGE_DATA_SIZE * 4];
    
    if (!isValid || count == 0) {
        // This page is deleted/invalid - mark it as free in bitmap
        uint bitmapIdx = pageIdx / 32;
        uint bitPos = pageIdx % 32;
        uint mask = 1u << bitPos;
        
        // Atomically set the bit (mark as free)
        uint oldVal = atomicOr(freeBitmap[bitmapIdx], mask);
        
        // If bit was previously 0 (allocated), increment free count
        if ((oldVal & mask) == 0) {
            atomicAdd(freeCount, 1);
        }
    }
    
    gl_Position = vec4(-5, -5, 0, 1);
    gl_PointSize = 1;
}
