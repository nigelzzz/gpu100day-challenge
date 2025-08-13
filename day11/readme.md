# Objectives
- Understand the concept of flash attention and its significance in deep learning.
https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf
# reade above material and understand the formulas. 
- safesoftmax: 
![alt text](image.png)
    - prevent overflow and underflow in softmax computation.
    - find the maximum value in the input vector, subtract it from each element to prevent overflow. 
- online softmax:
    - 3 pass safesoftmax 
![alt text](image-1.png)    
    - 2 pass softmax. 
  ![alt text](image-2.png)
    - depends on above formula, dominated depend on past max, so we can rewrite it to 2 pass. 
![alt text](image-3.png)

- flash attention: reduce the 2 pass softmax to 1 pass. 
    - our target is not `attention score matrix A`, we need `O matrix which is equals A * V` 
![alt text](image-7.png)
-----

![alt text](image-5.png)

    - final: one pass 
  ![alt text](image-6.png)
> the `blue color` can reside in `sram`,
  
  ![alt text](image-8.png)