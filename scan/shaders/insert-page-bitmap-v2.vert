// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// Insert shader with bitmap-based allocation (Version 2)
// Stores pageId in count field for efficient delete-by-data
// This allows O(1) bitmap update when deleting by data coordinates

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
    uint pageDataSize;
    uint rowIdOffset;
    uint maxPages;      // Maximum number of pages (128M)
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
// Each uint32 represents 32 pages
layout (binding = 2) buffer freeBitmapBuffer {
    uint freeBitmap[];
};

// Free count: number of free pages available
layout (binding = 3) buffer freeCountBuffer {
    uint freeCount;
};

// Next free hint: starting point for bitmap search
layout (binding = 4) buffer nextFreeHintBuffer {
    uint nextFreeHint;
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

// Find first set bit (returns 32 if no bit set)
uint findFirstSetBit(uint val) {
    if (val == 0) return 32;
    uint pos = 0;
    if ((val & 0x0000FFFFu) == 0) { pos += 16; val >>= 16; }
    if ((val & 0x000000FFu) == 0) { pos += 8;  val >>= 8; }
    if ((val & 0x0000000Fu) == 0) { pos += 4;  val >>= 4; }
    if ((val & 0x00000003u) == 0) { pos += 2;  val >>= 2; }
    if ((val & 0x00000001u) == 0) { pos += 1; }
    return pos;
}

// Allocate a page from the free bitmap
// Returns page index or NULL_PAGE_PTR if no free pages
uint allocatePageFromBitmap() {
    // Search the FULL bitmap (MAX_PAGES / 32 words)
    uint totalWords = consts.maxPages / 32;
    
    // Distribute threads across the entire bitmap using prime stride
    // This ensures better coverage when pages are fragmented
    uint threadId = uint(gl_VertexIndex);
    uint startWord = threadId % totalWords;
    
    // Use a prime stride to avoid clustering and improve coverage
    // 97 is prime, helps threads spread out across bitmap
    uint stride = 97u;
    
    // Search the ENTIRE bitmap if needed - critical for fragmented allocations
    // With 67M pages / 32 = ~2M words, we need full coverage
    uint maxSearch = totalWords;
    
    for (uint i = 0; i < maxSearch; i++) {
        // Use strided access pattern for better distribution
        uint bitmapIdx = (startWord + i * stride) % totalWords;
        uint bitmapWord = freeBitmap[bitmapIdx];
        
        // Quick check - skip if no free bits
        if (bitmapWord == 0) continue;
        
        // Find first set bit efficiently
        uint bit = findFirstSetBit(bitmapWord);
        if (bit < 32) {
            uint mask = 1u << bit;
            
            // Try to atomically clear this bit (mark as allocated)
            uint oldVal = atomicAnd(freeBitmap[bitmapIdx], ~mask);
            
            if ((oldVal & mask) != 0) {
                // Successfully allocated this page
                uint pageIdx = bitmapIdx * 32 + bit;
                
                // Validate page index
                if (pageIdx >= consts.maxPages) {
                    // Restore the bit we cleared
                    atomicOr(freeBitmap[bitmapIdx], mask);
                    continue;
                }
                
                // Decrement free count
                atomicAdd(freeCount, -1);
                
                return pageIdx;
            }
            // Someone else got it, continue searching
        }
    }
    
    // No free pages found
    return NULL_PAGE_PTR;
}

void main() {
    uvec2 val = uvec2(valx, valy);
    uvec2 binid = (val - consts.minVal) / consts.binRange;
    uint bin = binid.x + binid.y * consts.res;
    
    // Data to insert: (x, y, z, rowId_with_valid_bit)
    uint rowId = (uint(gl_VertexIndex) + consts.rowIdOffset) & ROWID_MASK;
    uint rowIdWithValid = VALID_BIT_MASK | rowId;
    uvec4 data = uvec4(val, valz, rowIdWithValid);
    
    // Allocate page from bitmap
    uint newPageIdx = allocatePageFromBitmap();
    
    if (newPageIdx == NULL_PAGE_PTR) {
        // No free pages available - discard this insert
        gl_Position = vec4(-5, -5, 0, 1);
        gl_PointSize = 1;
        return;
    }
    
    uint newPageOffset = getPageOffset(newPageIdx);
    
    // Initialize new page
    uint newCountOffset = getCountOffset(newPageOffset);
    uint newNextPtrOffset = getNextPtrOffset(newPageOffset);
    
    // Write data at position 0
    pages[newPageOffset + 0] = data.x;
    pages[newPageOffset + 1] = data.y;
    pages[newPageOffset + 2] = data.z;
    pages[newPageOffset + 3] = data.w;
    
    // IMPORTANT: Store pageId in count field (for delete-by-data to find bitmap entry)
    pages[newCountOffset] = newPageIdx;
    
    // Use compare-and-swap loop to safely insert at head
    // CRITICAL: First write nextPtr, then do CAS. If CAS fails, re-read and retry.
    // We must ensure we never create a self-loop (page pointing to itself).
    uint expectedHead = headPtr[bin];
    
    // Safety check: if somehow expectedHead equals our new page, something is wrong
    // This should never happen on first iteration, but guard against corruption
    if (expectedHead == newPageIdx) {
        expectedHead = NULL_PAGE_PTR;
    }
    
    pages[newNextPtrOffset] = expectedHead;
    memoryBarrierBuffer();
    
    while (true) {
        uint actualHead = atomicCompSwap(headPtr[bin], expectedHead, newPageIdx);
        
        if (actualHead == expectedHead) {
            // Success - we inserted at head
            break;
        }
        
        // CAS failed - another thread modified headPtr
        // Update expectedHead and retry
        expectedHead = actualHead;
        
        // CRITICAL: Prevent self-loop - if actualHead somehow equals our page, use NULL
        // This can happen in pathological race conditions
        if (expectedHead == newPageIdx) {
            expectedHead = NULL_PAGE_PTR;
        }
        
        // Update our nextPtr to point to the new head
        pages[newNextPtrOffset] = expectedHead;
        memoryBarrierBuffer();
    }
    
    gl_Position = vec4(-5, -5, 0, 1);
    gl_PointSize = 1;
}
