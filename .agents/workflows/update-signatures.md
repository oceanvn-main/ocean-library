---
description: Regenerate functionSignatures.json from source code
---

Regenerate the IDE autocomplete signatures file by scanning all `+Ocean*` packages.

## When to run
- After adding a new class or method to any package
- After changing an `arguments` block signature
- Before committing changes that include new public API

## Steps

1. Open MATLAB and set the working directory to the project root:
```matlab
cd('c:\Ocean Library\Matlab code practice')
```

// turbo
2. Run a dry-run first to preview what will be generated (no file written):
```matlab
OceanTools.generateSignatures('dry')
```

3. If the preview looks correct, run the full generation:
```matlab
OceanTools.generateSignatures
```

4. If you need to override or supplement any auto-generated entry, edit:
```
resources/signatures_overrides.json
```
Then re-run step 3. Override entries always win over auto-generated ones.

5. Verify the output file is valid JSON (MATLAB will report a parse error if not):
```matlab
jsondecode(fileread('resources/functionSignatures.json'));
disp('JSON is valid.')
```

// turbo
6. Reload the MATLAB path so the IDE picks up the new signatures:
```matlab
rehash toolboxcache
```
