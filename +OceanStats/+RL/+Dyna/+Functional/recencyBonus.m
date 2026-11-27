function bonus = recencyBonus(kappa, elapsedRealSteps)
%RECENCYBONUS Dyna-Q+ exploration bonus kappa*sqrt(tau).
if ~(isnumeric(kappa) && isreal(kappa) && isscalar(kappa) && isfinite(kappa) && kappa >= 0)
    fail_("kappa must be a finite nonnegative scalar.");
end
if ~(isnumeric(elapsedRealSteps) && isreal(elapsedRealSteps) && all(isfinite(elapsedRealSteps), 'all') ...
        && all(elapsedRealSteps >= 0, 'all') && all(elapsedRealSteps == fix(elapsedRealSteps), 'all'))
    fail_("elapsedRealSteps must contain nonnegative integers.");
end
bonus = double(kappa) .* sqrt(double(elapsedRealSteps));
end
function fail_(message), error('OceanStats:RL:Dyna:Functional:InvalidInput', '%s', message); end
