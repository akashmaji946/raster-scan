// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#version 450

// Page structure constants
#define PAGE_DATA_SIZE 1
#define PAGE_SIZE_UINTS (PAGE_DATA_SIZE * 4 + 2)  // data + count + nextPtr = 6
#define NULL_PAGE_PTR 0xFFFFFFFF

// Valid bit is stored in bit 31 of the rowId field
// rowId field = (valid << 31) | (rowId & 0x7FFFFFFF)
#define VALID_BIT_MASK 0x80000000u
#define ROWID_MASK 0x7FFFFFFFu

layout(push_constant) uniform ConstantBlock {
    uvec2 minVal;
    uvec2 binRange;
    uint res;
    uint pageDataSize;
    uint rowIdOffset;  // Offset to add to gl_VertexIndex for unique rowIds
} consts;

// Head pointer buffer: stores pointer to first page for each cell
layout (binding = 0) buffer headPtrBuffer {
    uint headPtr[];
};

// Page buffer: stores all pages
// Page layout: [data0, data1, ..., data15, count, nextPtr]
// Each data item is uvec4 (x, y, z, rowId_with_valid_bit)
layout (binding = 1) buffer pageBuffer {
    uint pages[];
};

// Allocation counter: next free page index
layout (binding = 2) buffer allocCounterBuffer {
    uint allocCounter;
};

layout (location = 0) in uint valx;
layout (location = 1) in uint valy;
layout (location = 2) in uint valz;

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

// Allocate a new page and return its index
uint allocatePage() {
    return atomicAdd(allocCounter, 1);
}

void main() {
    uvec2 val = uvec2(valx, valy);
    uvec2 binid = (val - consts.minVal) / consts.binRange;
    uint bin = binid.x + binid.y * consts.res;
    
    // Data to insert: (x, y, z, rowId_with_valid_bit)
    // Set valid bit (bit 31) to 1 to mark entry as valid
    // Add rowIdOffset to gl_VertexIndex to generate unique rowIds for inserted data
    uint rowId = (uint(gl_VertexIndex) + consts.rowIdOffset) & ROWID_MASK;
    uint rowIdWithValid = VALID_BIT_MASK | rowId;
    uvec4 data = uvec4(val, valz, rowIdWithValid);
    
    // Allocate new page
    uint newPageIdx = allocatePage();
    uint newPageOffset = getPageOffset(newPageIdx);
    
    // Initialize new page
    uint newCountOffset = getCountOffset(newPageOffset);
    uint newNextPtrOffset = getNextPtrOffset(newPageOffset);
    
    // Write data at position 0
    pages[newPageOffset + 0] = data.x;
    pages[newPageOffset + 1] = data.y;
    pages[newPageOffset + 2] = data.z;
    pages[newPageOffset + 3] = data.w;  // rowId with valid bit set
    
    // Set count to 1
    pages[newCountOffset] = 1;
    
    // Use compare-and-swap loop to safely insert at head
    // This ensures nextPtr is set BEFORE the page becomes visible
    uint expectedHead = headPtr[bin];
    while (true) {
        // Set our nextPtr to current head BEFORE trying to become the new head
        pages[newNextPtrOffset] = expectedHead;
        
        // Memory barrier to ensure nextPtr write is visible
        memoryBarrierBuffer();
        
        // Try to atomically set ourselves as the new head
        uint actualHead = atomicCompSwap(headPtr[bin], expectedHead, newPageIdx);
        
        if (actualHead == expectedHead) {
            // Success! We are now the head
            break;
        }
        
        // Failed - someone else changed the head. Update expectedHead and retry
        expectedHead = actualHead;
    }
    
    // Discard vertex (no actual rendering)
    gl_Position = vec4(-5, -5, 0, 1);
    gl_PointSize = 1;
}
