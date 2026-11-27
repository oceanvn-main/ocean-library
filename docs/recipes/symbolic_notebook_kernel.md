# Symbolic notebook kernel

The notebook kernel exposes the existing static source extractor, typed registry,
and safe scalar workspace through a fixed command protocol. It does not evaluate
imported MATLAB or Python source.

```matlab
kernel = OceanTools.SymbolicNotebook.Kernel("local-session");
request = struct( ...
    'protocolVersion', "1.0.0", ...
    'sessionId', kernel.SessionId, ...
    'jobId', "scan-1", ...
    'notebookRevision', 1, ...
    'command', "scanText", ...
    'payload', struct( ...
        'sourceText', "function y = squareValue(x)" + newline + ...
            "y = x^2;" + newline + "end", ...
        'language', "matlab", ...
        'filePath', "squareValue.m"), ...
    'sourceRefs', struct([]));
response = kernel.execute(request);
```

Only record keys returned by `scanText` can be passed to `acceptRecord`. The
remaining fixed commands are `compose`, `transform`, `inspect`, `expand`, and
`cancelJob`. Every response carries a `valid`, `invalid`, `unknown`, or
`unsupported` verdict. `unknown` means the available source contracts cannot
prove or disprove the operation; it is not a successful proof.

`dispatchJson` is the transport-neutral JSON boundary used by a loopback bridge.
It caps request text at four million characters and delegates every operation to
the same guarded kernel. Cancellation is cooperative: queued work can be marked
cancelled, while an active MATLAB symbolic call runs to the limits enforced by
`OceanMath.Symbolic.Workspace`.

To serve the Information Extractor and start one persistent MATLAB kernel, run:

```text
node "App/Information Extractor/server/symbolic-bridge.cjs" --port 8767
```

Open the printed loopback URL. `MATLAB_BIN` may point to the MATLAB executable
when it is not on `PATH`. The service binds only to `127.0.0.1`, checks browser
origin and a per-process session secret, limits request bodies, and exposes no
general MATLAB evaluation route. `--no-matlab` intentionally starts review-only
mode for transport diagnostics.
