classdef (Abstract) IObserver < handle
    % IOBSERVER  Passive listener of what a backup does, so logging, recording and monitoring never live inside an update
    % loop. A backup calls the hooks; an observer never changes learning. Every hook has a no-op default: implement only
    % the ones you need (a new observer is ONE file, passed with the 'Observer' option of TDLearner).
    %
    %   startRun()                             a run begins (clear any state kept from a previous run)
    %   begin(s0, a0)                          an episode starts in state index s0 with first action a0
    %   step(r, s2, a2, outsideTable)          one transition: reward, successor index (a2 = next action, 0 if none)
    %   coin(c)                                a double-estimator coin flip (true = first table was updated)
    %   action(a)                              an action chosen after the previous transition was reported
    %   finish(terminated, truncated)          the episode ended
    %   finishArrays(S, A, R, terminated, truncated, outsideLast)   the episode ended, given as whole arrays
    %   ep = episodes()                        recorded episodes, if this observer keeps them ([] otherwise)
    methods
        function startRun(~), end
        function begin(~, ~, ~), end
        function step(~, ~, ~, ~, ~), end
        function coin(~, ~), end
        function action(~, ~), end
        function finish(~, ~, ~), end
        function finishArrays(~, ~, ~, ~, ~, ~, ~), end
        function ep = episodes(~), ep = []; end
    end
end
