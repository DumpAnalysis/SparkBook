set pagination off
set confirm off
set print address off
break host_checkpoint
run snapshot runs/cuda/debug-snapshot
printf "OBSERVED count=%lu\n", count
printf "OBSERVED first=%u\n", *(@global unsigned int*)device_values
printf "OBSERVED second=%u\n", *((@global unsigned int*)device_values+1)
continue
quit
