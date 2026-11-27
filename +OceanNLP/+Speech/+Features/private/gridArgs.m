function args = gridArgs(g)
% GRIDARGS  Name-value arguments that rebuild a FrameGrid from its config() struct.
args = {'WindowLength', g.WindowLength, 'HopLength', g.HopLength, 'NFFT', g.NFFT, 'Window', g.Window, ...
    'Periodic', g.Periodic, 'Center', g.Center, 'PadMode', g.PadMode};
end
