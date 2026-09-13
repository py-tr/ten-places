# OpenVINO model cache: time to get each model ready on the CPU

Median of 3 fresh processes each; the model files are in the OS file cache beforehand.

| model | compiled from IR (cold) | first launch, writing the cache | from the cache | speed-up | cache size |
|---|---|---|---|---|---|
| drawer | 836 ms | 936 ms | 170 ms | 4.9× | 150.5 MB |
| fork | 850 ms | 934 ms | 167 ms | 5.1× | 150.5 MB |
| plate | 849 ms | 927 ms | 166 ms | 5.1× | 150.5 MB |
| spoon | 834 ms | 951 ms | 162 ms | 5.1× | 150.5 MB |
| cup | 860 ms | 928 ms | 165 ms | 5.2× | 150.5 MB |
| classifier | 58 ms | 83 ms | 34 ms | 1.7× | 21.4 MB |
| planner (Qwen3-VL-4B INT4) | 4724 ms | 11485 ms | 8759 ms | 0.5× | 3011.1 MB |
