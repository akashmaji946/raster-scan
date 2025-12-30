// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// Compact shader: removes deleted pages from linked lists
// This shader processes one grid cell per vertex invocation
// It traverses the linked list and removes any deleted pages

#version 450

// Page structure constants
#define PAGE_DATA_SIZE 1
#define PAGE_SIZE_UINTS 8  // data + count + nextPtr + padding = 8
#define NULL_PAGE_PTR 0xFFFFFFFF

// Valid bit is stored in bit 31 of the rowId field
#define VALID_BIT_MASK 0x80000000u
#define ROWID_MASK 0x7FFFFFFFu

// Maximum iterations to prevent infinite loops
#define MAX_ITERATIONS 100000

layout(push_constant) uniform ConstantBlock {
    uint res;  // INDEX_RESOLUTION
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

// Free count
layout (binding = 3) buffer freeCountBuffer {
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

// Get the nextPtr offset within a page
uint getNextPtrOffset(uint pageOffset) {
    return pageOffset + PAGE_DATA_SIZE * 4 + 1;
}

// Check if a page is valid (not deleted)
bool isPageValid(uint pageIdx) {
    uint pageOffset = getPageOffset(pageIdx);
    uint rowIdWithValid = pages[pageOffset + 3];
    return (rowIdWithValid & VALID_BIT_MASK) != 0;
}

// Mark a page as free in the bitmap
void markPageFree(uint pageIdx) {
    uint bitmapIdx = pageIdx / 32;
    uint bitPos = pageIdx % 32;
    uint mask = 1u << bitPos;
    
    uint oldVal = atomicOr(freeBitmap[bitmapIdx], mask);
    if ((oldVal & mask) == 0) {
        atomicAdd(freeCount, 1);
    }
}

void main() {
    uint cellIdx = uint(gl_VertexIndex);
    uint totalCells = consts.res * consts.res;
    
    if (cellIdx >= totalCells) {
        gl_Position = vec4(-5, -5, 0, 1);
        gl_PointSize = 1;
        return;
    }
    
    // Find the first valid page to be the new head
    uint currentPtr = headPtr[cellIdx];
    uint newHead = NULL_PAGE_PTR;
    uint iterations = 0;
    
    // First pass: find new head (first valid page) and mark invalid pages as free
    while (currentPtr != NULL_PAGE_PTR && iterations < MAX_ITERATIONS) {
        iterations++;
        
        uint pageOffset = getPageOffset(currentPtr);
        uint nextPtr = pages[getNextPtrOffset(pageOffset)];
        
        // SAFETY: Detect self-loop
        if (nextPtr == currentPtr) {
            // Break self-loop
            pages[getNextPtrOffset(pageOffset)] = NULL_PAGE_PTR;
            nextPtr = NULL_PAGE_PTR;
        }
        
        if (isPageValid(currentPtr)) {
            newHead = currentPtr;
            break;
        } else {
            // Mark deleted page as free in bitmap
            markPageFree(currentPtr);
        }
        
        currentPtr = nextPtr;
    }
    
    // Update head pointer
    headPtr[cellIdx] = newHead;
    
    if (newHead == NULL_PAGE_PTR) {
        gl_Position = vec4(-5, -5, 0, 1);
        gl_PointSize = 1;
        return;
    }
    
    // Second pass: traverse from new head and remove deleted pages from the list
    // Reset iteration counter for second pass
    uint iterations2 = 0;
    uint prevValidPtr = newHead;
    uint prevPageOffset = getPageOffset(prevValidPtr);
    currentPtr = pages[getNextPtrOffset(prevPageOffset)];
    
    while (currentPtr != NULL_PAGE_PTR && iterations2 < MAX_ITERATIONS) {
        iterations2++;
        
        uint pageOffset = getPageOffset(currentPtr);
        uint nextPtr = pages[getNextPtrOffset(pageOffset)];
        
        // SAFETY: Detect self-loop
        if (nextPtr == currentPtr) {
            // Break self-loop by terminating list here
            nextPtr = NULL_PAGE_PTR;
        }
        
        if (isPageValid(currentPtr)) {
            // Valid page - keep it in the list, update prev pointer
            prevValidPtr = currentPtr;
            prevPageOffset = pageOffset;
        } else {
            // Deleted page - remove from list by updating prev's nextPtr
            pages[getNextPtrOffset(prevPageOffset)] = nextPtr;
            
            // Mark as free in bitmap
            markPageFree(currentPtr);
        }
        
        currentPtr = nextPtr;
    }
    
    // Ensure the last valid page's nextPtr is NULL (clean termination)
    // This handles the case where the last page(s) were deleted
    if (prevValidPtr != NULL_PAGE_PTR) {
        uint lastPageOffset = getPageOffset(prevValidPtr);
        uint lastNextPtr = pages[getNextPtrOffset(lastPageOffset)];
        // If the last valid page still points to something, verify it's valid or NULL
        // (This is already handled by the loop above, but adding for safety)
    }
    
    gl_Position = vec4(-5, -5, 0, 1);
    gl_PointSize = 1;
}
