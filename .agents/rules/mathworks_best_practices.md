# STRICT RULE: MathWorks & Ocean Library Engineering Best Practices

Whenever generating or modifying MATLAB code in this codebase:

1. **Modern Input Validation**: Always use modern `arguments ... end` input validation blocks on public methods and functions.
2. **Vectorization & Pre-allocation**: Avoid growing arrays inside loops. Always pre-allocate output matrices using `zeros()` and use vectorized element-wise operations (`.*`, `./`, `.^`).
3. **Unit Test Coverage**: Every new class or strategy must include a corresponding unit test inheriting from `OceanTest.BaseTestCase` in `+OceanTest`.
4. **OceanViz Theme Compliance**: All visualizations must use `OceanViz.Core`, adhering to active dark/light mode themes, styled legends, and explicit axis styling (`viz.styleAxes(ax)`).
5. **Registration & Documentation**:
   - Register signatures in `resources/functionSignatures.json`.
   - Update package map in `.agents/architecture.md`.
   - Add copy-paste recipes in `docs/recipes/` and update `docs/index.md`.
