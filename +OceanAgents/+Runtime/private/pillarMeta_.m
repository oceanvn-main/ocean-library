function meta = pillarMeta_(obj, assumptions, reference)
% PILLARMETA_  getMetadata() struct for a runtime/container component (A0 section 8: IsSymbolic = false).
arguments
    obj
    assumptions cell
    reference (1,1) string
end
meta = struct();
meta.SymbolicLatex = strings(1, 0);
meta.Symbols = containers.Map();
meta.Assumptions = [{sprintf('%s: runtime/container component (no formula of its own)', class(obj))}, assumptions];
meta.IsSymbolic = false;
meta.Reference = reference;
end
