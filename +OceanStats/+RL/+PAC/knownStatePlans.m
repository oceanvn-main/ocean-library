function plans = knownStatePlans(R, P, Pend, known, mask, gamma, horizon, opts)
% KNOWNSTATEPLANS  The two off-line computations of E³ on the known-state MDP (Kearns & Singh 2002,
% Definition 9 and Section 5.4).
%
%   plans = OceanStats.RL.PAC.knownStatePlans(R, P, Pend, known, mask, gamma, horizon, ...
%               'MinReward', 0, 'Tolerance', 1e-10, 'MaxIterations', 100000)
%     R, Pend   S x A empirical mean reward and episode-end probability     (TransitionCounts.estimates)
%     P         sparse (S*A) x S empirical next-state probabilities, row sub2ind([S A], s, a)
%     known     S x 1 logical: the known set S_K
%     mask      S x A logical admissible actions
%
% Induced MDP M̂_S: the known states plus one absorbing state s0; every transition from a known state to a
% state outside S_K is redirected to s0. A transition that ends the episode keeps no continuation value.
%
%   Exploitation (discounted value iteration on M̂_S to convergence -- a stationary policy; K&S use a T-step
%   non-stationary one -- s0 pays MinReward forever, K&S: 0 for rewards in [0, Rmax]):
%     Q(s,a) = R̂(s,a) + γ [ Σ_{j ∈ S_K} T̂(j|s,a) V(j) + T̂(out|s,a) MinReward/(1-γ) ],   V(s) = max_a Q(s,a)
%   Exploration (M̂'_S: payoff only for reaching s0; undiscounted, horizon T):
%     W_0 = 0,  W_t(s) = max_a [ T̂(out|s,a) + Σ_{j ∈ S_K} T̂(j|s,a) W_{t-1}(j) ]
%     W_T(s) = the largest probability of leaving S_K within T steps from s.
%   (K&S give s0 payoff Rmax per step on M̂'_S and then compute the policy's probability of reaching s0; here
%   the escape probability is maximised directly -- it is the quantity Lemma 8 bounds. Not in the paper verbatim.)
%
% plans: ExploitQ (S x A, NaN outside S_K or inadmissible), ExploitPolicy (S x 1, 0 outside S_K),
%        Escape (S x 1 = W_T), ExplorePolicy (S x T; column t = action with t steps left), Iterations.
arguments
    R double
    P
    Pend double
    known (:,1) logical
    mask logical
    gamma (1,1) double {mustBeGreaterThanOrEqual(gamma, 0), mustBeLessThan(gamma, 1)}
    horizon (1,1) double {mustBeInteger, mustBePositive}
    opts.MinReward (1,1) double {mustBeFinite} = 0
    opts.Tolerance (1,1) double {mustBePositive} = 1e-10
    opts.MaxIterations (1,1) double {mustBeInteger, mustBePositive} = 100000
end
[S, A] = size(R);
adm = mask & known;                                          % pairs that exist in M̂_S
PK = P(:, known);                                            % to known states
out = reshape(full(1 - Pend(:) - sum(PK, 2)), S, A);         % to s0: mass neither to S_K nor ending the episode
out(~adm) = 0; out = max(out, 0);
v0 = opts.MinReward / (1 - gamma);

% Exploitation: discounted value iteration.
V = zeros(S, 1); it = 0;
Q = nan(S, A);
while it < opts.MaxIterations
    it = it + 1;
    cont = reshape(full(PK * V(known)), S, A);
    Qn = R + gamma * (cont + out * v0);
    Qn(~adm) = -Inf;
    Vn = max(Qn, [], 2); Vn(~known | ~any(adm, 2)) = 0;
    delta = max(abs(Vn - V)); V = Vn; Q = Qn;
    if delta < opts.Tolerance, break; end
end
[~, pol] = max(Q, [], 2); pol(~known | ~any(adm, 2)) = 0;
Q(~adm) = NaN;

% Exploration: horizon-T maximal escape probability.
W = zeros(S, 1); explore = zeros(S, horizon);
for t = 1:horizon
    Qe = out + reshape(full(PK * W(known)), S, A);
    Qe(~adm) = -Inf;
    [Wn, ae] = max(Qe, [], 2);
    Wn(~known | ~any(adm, 2)) = 0; ae(~known | ~any(adm, 2)) = 0;
    W = Wn; explore(:, t) = ae;
end
plans = struct('ExploitQ', Q, 'ExploitPolicy', pol, 'Escape', W, 'ExplorePolicy', explore, 'Iterations', it);
end
