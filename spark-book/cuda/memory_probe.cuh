#pragma once
#include <cuda_runtime.h>
#include <cstdio>
// Query status belongs beside the value. An unsupported query is not false.
inline void print_memory_capabilities() {
    struct Item { const char* name; cudaDeviceAttr attribute; };
    const Item items[] = {
        {"integrated",cudaDevAttrIntegrated},
        {"managedMemory",cudaDevAttrManagedMemory},
        {"concurrentManagedAccess",cudaDevAttrConcurrentManagedAccess},
        {"pageableMemoryAccess",cudaDevAttrPageableMemoryAccess},
        {"pageableMemoryAccessUsesHostPageTables",cudaDevAttrPageableMemoryAccessUsesHostPageTables},
        {"hostNativeAtomicSupported",cudaDevAttrHostNativeAtomicSupported},
        {"canMapHostMemory",cudaDevAttrCanMapHostMemory}
    };
    std::printf("\"memory_capabilities\":{");
    bool first=true;
    for(const auto& item:items) {
        int value=0;
        cudaError_t status=cudaDeviceGetAttribute(&value,item.attribute,0);
        std::printf("%s\"%s\":{\"value\":",first?"":",",item.name);
        if(status==cudaSuccess) std::printf("%d",value);
        else std::printf("null");
        std::printf(",\"status\":\"%s\",\"cuda_error\":%d}",
                    status==cudaSuccess?"ok":"query_error",int(status));
        if(status!=cudaSuccess) (void)cudaGetLastError();
        first=false;
    }
    std::printf("}");
}
