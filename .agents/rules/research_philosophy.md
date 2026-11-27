# Ocean Library - Agent Rules for Research & Pedagogical Philosophy

> [!IMPORTANT]
> **Research and Pedagogical Focus**:
> Unlike other libraries that may implement black-box logic to speed up production processes, this library favors the researcher, aiming to facilitate a deep understanding of the system rather than production deployment. The library is specifically designed to allow users to see and understand exactly what is going on behind the scenes.

When generating or refactoring code, agents must adhere to the following principles:

## 1. Prioritize Internal Legibility Over Production Black-Boxes
* **Explicit Implementations**: Avoid using complex, opaque library optimization tricks unless absolutely necessary. Code should be written "from scratch" to serve as a clear mathematical reference.
* **Readable Over Vectorized (if conflict arises)**: While vectorized code (like matrix multiplication) is encouraged for MATLAB/Python speed, it should never be written in a way that completely obscures the underlying mathematical formula. Include comments linking lines of code to the corresponding textbook formulas.
* **Preserve Parameter Context**: Retain parameter metadata (like sizes, standard deviations, boundary settings, or matrices) in the state of solver/transform objects rather than returning raw anonymous arrays. This allows the user to inspect the state and query configuration.

## 2. Reveal "Behind the Scenes" Math
* **Analytical Decompositions**: When implementing optimizations (such as separable convolution), show the analytical steps (like SVD or discrete binomial approximations) explicitly in the code.
* **Visual Clarity**: Design experiments to make the differences between mathematical methods as transparent and visually extreme as possible (e.g., using synthetic gradients or checkerboards to make padding transitions obvious).
