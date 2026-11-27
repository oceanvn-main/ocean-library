%% Speech waveform, spectrum, and spectrogram
% Requires Audio Toolbox / Signal Processing Toolbox

clear;
close all;
clc;

%% 1. Load MATLAB's built-in speech sample
[x, fs] = audioread("SpeechDFT-16-8-mono-5secs.wav");

% Safety: convert stereo to mono if necessary
if size(x,2) > 1
    x = mean(x,2);
end

% Remove very small DC offset
x = x - mean(x);

Ntotal = length(x);
t = (0:Ntotal-1).' / fs;

fprintf("Sampling frequency fs = %.0f Hz\n", fs);
fprintf("Duration = %.3f s\n", Ntotal/fs);
fprintf("Number of samples = %d\n", Ntotal);

% Listen to it
soundsc(x,fs);


%% 2. Automatically choose a short high-energy speech frame
% Speech is approximately stationary only over short intervals,
% so use a 40 ms frame for the ordinary spectrum.

frameDuration = 0.040;             % 40 ms
hopDuration   = 0.010;             % 10 ms

frameLength = round(frameDuration * fs);
hopLength   = round(hopDuration   * fs);

starts = 1:hopLength:(Ntotal-frameLength+1);

frameRMS = zeros(size(starts));

for i = 1:length(starts)
    idx = starts(i):(starts(i)+frameLength-1);
    temp = x(idx);

    frameRMS(i) = sqrt(mean(temp.^2));
end

% Choose the highest-energy frame
[~,bestFrame] = max(frameRMS);

startIndex = starts(bestFrame);
stopIndex  = startIndex + frameLength - 1;

xFrame = x(startIndex:stopIndex);

frameStartTime = (startIndex-1)/fs;
frameEndTime   = (stopIndex-1)/fs;

fprintf("Selected frame: %.3f to %.3f s\n", ...
    frameStartTime, frameEndTime);


%% 3. Spectrum of the selected frame

% Window the frame to reduce spectral leakage
w = hann(frameLength,"periodic");
xWindowed = xFrame .* w;

% FFT length
NFFT = 2^nextpow2(frameLength);

X = fft(xWindowed,NFFT);

% Keep only positive frequencies
X = X(1:NFFT/2+1);

f = (0:NFFT/2).' * fs/NFFT;

% Magnitude spectrum in dB, normalized to strongest component
spectrumDB = 20*log10(abs(X) + eps);
spectrumDB = spectrumDB - max(spectrumDB);

fprintf("DFT bin spacing = %.3f Hz\n", fs/NFFT);


%% 4. Spectrogram parameters
% 25 ms analysis windows with 10 ms frame step are common in speech.

specWindowLength = round(0.025*fs);
specHopLength    = round(0.010*fs);
specOverlap      = specWindowLength - specHopLength;

specWindow = hann(specWindowLength,"periodic");

NFFTspec = 2^nextpow2(max(specWindowLength,512));

[S,F,T] = spectrogram( ...
    x, ...
    specWindow, ...
    specOverlap, ...
    NFFTspec, ...
    fs);

Sdb = 20*log10(abs(S) + eps);


%% 5. Plot all three representations

%% ============================================================
%  LIGHT / SCIENTIFIC-STYLE VISUALIZATION
% ============================================================

fontName = "Times New Roman";
fontSize = 11;
lineWidth = 1.2;

fMaxPlot = min(4000, fs/2);
specFloor = -80;
specDynamicRange = 70;

fig = figure( ...
    "Name","Speech Signal Analysis", ...
    "Color","w", ...
    "Units","centimeters", ...
    "Position",[2 2 18 22]);

tl = tiledlayout(fig,3,1, ...
    "TileSpacing","compact", ...
    "Padding","compact");

title(tl,"Speech Signal Analysis", ...
    "Interpreter","latex", ...
    "FontSize",14, ...
    "FontWeight","bold", ...
    "Color","k");


%% ------------------------------------------------------------
% A. TIME DOMAIN
% ------------------------------------------------------------
ax1 = nexttile(tl,1);

plot(ax1,t,x, ...
    "Color",[0 0.4470 0.7410], ...
    "LineWidth",0.9);
hold(ax1,"on");

yl = [min(x), max(x)];

p = patch(ax1, ...
    [frameStartTime frameEndTime frameEndTime frameStartTime], ...
    [yl(1) yl(1) yl(2) yl(2)], ...
    [0.88 0.93 1.00], ...
    "FaceAlpha",0.55, ...
    "EdgeColor","none");
uistack(p,"bottom");

xline(ax1,frameStartTime,"--", ...
    "Color",[0.25 0.25 0.25], ...
    "LineWidth",1.0);

xline(ax1,frameEndTime,"--", ...
    "Color",[0.25 0.25 0.25], ...
    "LineWidth",1.0);

xlabel(ax1,"$t$ (s)","Interpreter","latex");
ylabel(ax1,"$x(t)$","Interpreter","latex");
title(ax1,sprintf("(a) Time-domain waveform and selected %.0f ms frame",1000*frameDuration), ...
    "Interpreter","latex","Color","k");

grid(ax1,"on");
box(ax1,"on");
axis(ax1,"tight");


%% ------------------------------------------------------------
% B. FREQUENCY DOMAIN
% ------------------------------------------------------------
ax2 = nexttile(tl,2);

plot(ax2,f,spectrumDB, ...
    "Color",[0 0.4470 0.7410], ...
    "LineWidth",lineWidth);

xlabel(ax2,"$f$ (Hz)","Interpreter","latex");
ylabel(ax2,"$20\log_{10}|X[k]|$ (dB)","Interpreter","latex");
title(ax2,sprintf("(b) One-sided magnitude spectrum, $N_{\\mathrm{FFT}}=%d$",NFFT), ...
    "Interpreter","latex","Color","k");

grid(ax2,"on");
box(ax2,"on");
xlim(ax2,[0 fMaxPlot]);
ylim(ax2,[specFloor 5]);

yline(ax2,-20,":", ...
    "Color",[0.5 0.5 0.5], ...
    "LineWidth",0.9, ...
    "Label","$-20$ dB", ...
    "Interpreter","latex", ...
    "LabelHorizontalAlignment","left");

yline(ax2,-40,":", ...
    "Color",[0.5 0.5 0.5], ...
    "LineWidth",0.9, ...
    "Label","$-40$ dB", ...
    "Interpreter","latex", ...
    "LabelHorizontalAlignment","left");


%% ------------------------------------------------------------
% C. TIME-FREQUENCY DOMAIN
% ------------------------------------------------------------
ax3 = nexttile(tl,3);

imagesc(ax3,T,F,Sdb);
axis(ax3,"xy");

xlabel(ax3,"$t$ (s)","Interpreter","latex");
ylabel(ax3,"$f$ (Hz)","Interpreter","latex");
title(ax3,"(c) Short-time Fourier transform spectrogram", ...
    "Interpreter","latex","Color","k");

ylim(ax3,[0 fMaxPlot]);

maxDB = max(Sdb(:));
clim(ax3,[maxDB-specDynamicRange, maxDB]);

colormap(ax3,turbo(256));

cb = colorbar(ax3);
cb.Label.String = "Magnitude (dB)";
cb.Label.Interpreter = "latex";
cb.Color = "k";
cb.FontName = fontName;
cb.FontSize = fontSize;


%% ------------------------------------------------------------
% Force LIGHT appearance on all axes
% ------------------------------------------------------------
set([ax1 ax2 ax3], ...
    "Color","w", ...
    "XColor","k", ...
    "YColor","k", ...
    "GridColor",[0.82 0.82 0.82], ...
    "MinorGridColor",[0.90 0.90 0.90], ...
    "GridAlpha",0.35, ...
    "MinorGridAlpha",0.2, ...
    "FontName",fontName, ...
    "FontSize",fontSize, ...
    "LineWidth",0.8, ...
    "TickDir","out", ...
    "Layer","top", ...
    "Box","on", ...
    "TickLabelInterpreter","latex");

exportgraphics(fig,"speech_analysis_light.png","Resolution",300);   