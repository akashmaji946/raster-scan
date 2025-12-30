// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// Delete-by-data shader: deletes rows by matching (x, y, z) coordinates
// Input: (x, y, z) data to delete (passed as vertex attributes)
// 
// This shader:
// 1. Computes the bin from (x, y) coordinates
// 2. Traverses the linked list in that bin
// 3. Finds the page with matching (x, y, z) data
// 4. Marks the page as invalid (clears valid bit in rowId)
// 5. Marks the page as free in the bitmap using the pageId stored in count field

#version 450

// Page structure constants
#define PAGE_DATA_SIZE 1
#define PAGE_SIZE_UINTS 8  // data + count + nextPtr + padding = 8
#define NULL_PAGE_PTR 0xFFFFFFFF

// Valid bit is stored in bit 31 of the rowId field
#define VALID_BIT_MASK 0x80000000u
#define ROWID_MASK 0x7FFFFFFFu

layout(push_constant) uniform ConstantBlock {
    uvec2 minVal;
    uvec2 binRange;
    uint res;
    uint maxPages;
} consts;

// Head pointer buffer: stores pointer to first page for each cell
layout (binding = 0) buffer headPtrBuffer {
    uint headPtr[];
};

// Page buffer: stores all pages
layout (binding = 1) buffer pageBuffer {
    uint pages[];
};

// Free bitmap: 1 = free, 0 = allocated
layout (binding = 2) buffer freeBitmapBuffer {
    uint freeBitmap[];
};

// Free count: number of free pages
layout (binding = 3) buffer freeCountBuffer {
    uint freeCount;
};

// Input: (x, y, z) data to delete
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

// Get the count offset within a page (stores pageId)
uint getCountOffset(uint pageOffset) {
    return pageOffset + PAGE_DATA_SIZE * 4;
}

// Get the nextPtr offset within a page
uint getNextPtrOffset(uint pageOffset) {
    return pageOffset + PAGE_DATA_SIZE * 4 + 1;
}

void main() {
    // Compute bin from (x, y)
    uvec2 val = uvec2(valx, valy);
    uvec2 binid = (val - consts.minVal) / consts.binRange;
    uint bin = binid.x + binid.y * consts.res;
    
    // Target data to find and delete
    uint targetX = valx;
    uint targetY = valy;
    uint targetZ = valz;
    
    // Traverse the linked list in this bin
    // SAFETY: Limit iterations to prevent infinite loops from corrupted linked lists
    uint currentPageIdx = headPtr[bin];
    uint iterCount = 0;
    const uint MAX_ITERATIONS = 1000000u;  // Safety limit
    
    while (currentPageIdx != NULL_PAGE_PTR && iterCount < MAX_ITERATIONS) {
        iterCount++;
        
        // Validate page index to prevent out-of-bounds access
        if (currentPageIdx >= consts.maxPages) {
            break;
        }
        
        uint pageOffset = getPageOffset(currentPageIdx);
        
        // Read page data
        uint pageX = pages[pageOffset + 0];
        uint pageY = pages[pageOffset + 1];
        uint pageZ = pages[pageOffset + 2];
        uint rowIdWithValid = pages[pageOffset + 3];
        
        // Check if this page matches our target data AND is valid
        bool isValid = (rowIdWithValid & VALID_BIT_MASK) != 0;
        bool matches = (pageX == targetX) && (pageY == targetY) && (pageZ == targetZ);
        
        if (matches && isValid) {
            // Found a valid page with matching (x, y, z)
            // Simply mark as invalid by clearing the valid bit
            // DO NOT modify bitmap or linked list structure
            // Compaction will clean up invalid pages later
            
            uint oldRowId = atomicAnd(pages[pageOffset + 3], ROWID_MASK);
            
            // If we successfully cleared the valid bit, we're done
            if ((oldRowId & VALID_BIT_MASK) != 0) {
                break;
            }
            // If we lost the race, continue searching for another match
        }
        
        // Move to next page in linked list
        uint nextPtrOffset = getNextPtrOffset(pageOffset);
        uint nextPageIdx = pages[nextPtrOffset];
        
        // SAFETY: Detect self-loop (page pointing to itself)
        if (nextPageIdx == currentPageIdx) {
            break;
        }
        
        currentPageIdx = nextPageIdx;
    }
    
    // Discard vertex (no actual rendering)
    gl_Position = vec4(-5, -5, 0, 1);
    gl_PointSize = 1;
}
