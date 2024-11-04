Analyze your results, when does it make sense to use the various approaches?

From the graph, we can see that in terms of efficiency this is how the algorithm ranks from top to bottom: cuBLAS > Tiled GPU > GPU matrix > CPU matrix. The CPU matrix multiplication took about hour and a half to compute matrix multiplications of 10000 sizes. On the other hand, GPU matrix, tiled, and cuBLAS multiplications are significantly faster. This shows that for matrix multiplication, GPU and use of advanced programming has immense effect on computational speed. 

How did your speed compare with cuBLAS?

Compared to cuBLAS, my GPU codes are approximately close up until and including 1000. However, at 10000 input size, the difference is noticable. 

What went well with this assignment?

Matrix multiplication on GPUs and CPUs seemed easy and interpretable, mainly because of my background in mathematics. This assignment also included many of the same ideas from a3, so it didn't feel like I was working from scratch.

What was difficult?

Implementation of cuBLAS was the most difficult because I was having segmentation fault due to assigning matrices without proper malloc. I think it was a good reminder to look back on malloc and allocation of spaces. 

How would you approach differently?

I would probably pay extra attention in data structures and its corresponding allocation space on the GPU and CPU. 

Anything else you want me to know?
I have confirmed that cuBLAS works on non-squared matrices, and I have utilized malloc, cudaMalloc, and cudaMemcpy. 
