set pagination off
set confirm off
break inspect_kernel
run
cuda kernel 0 block (0,0,0) thread (0,0,0)
info cuda kernels
info cuda warps
printf "OBSERVED thread_x=%u\n", threadIdx.x
x/4i $pc
info registers $R0 $R1
stepi
x/i $pc
continue
quit
