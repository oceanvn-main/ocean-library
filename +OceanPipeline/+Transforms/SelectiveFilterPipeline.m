classdef SelectiveFilterPipeline < handle
    % SELECTIVEFILTERPIPELINE Transparent long-cycle spectral filtering.
    % Removes deterministic trend, annual and half-year frequencies, then
    % applies causal short-period smoothing. Every layer exposes its signal,
    % raw periodogram, coefficients, and cumulative transfer gain.

    properties
        SamplesPerYear (1,1) double {mustBePositive} = 365.2425/7
        SmoothWindow (1,1) double {mustBeInteger,mustBePositive} = 13
        SearchPeriodYears (1,2) double {mustBePositive} = [2 6]
        TargetPeriodYears (1,1) double {mustBePositive} = 4
    end

    methods
        function obj = SelectiveFilterPipeline(options)
            arguments
                options.SamplesPerYear (1,1) double = 365.2425/7
                options.SmoothWindow (1,1) double = 13
                options.SearchPeriodYears (1,2) double = [2 6]
                options.TargetPeriodYears (1,1) double = 4
            end
            obj.SamplesPerYear=options.SamplesPerYear;
            obj.SmoothWindow=options.SmoothWindow;
            obj.SearchPeriodYears=sort(options.SearchPeriodYears);
            obj.TargetPeriodYears=options.TargetPeriodYears;
        end

        function result = run(obj,time,signal)
            arguments
                obj
                time (:,1) datetime
                signal (:,1) double {mustBeFinite}
            end
            if numel(time)~=numel(signal) || numel(signal)<4*obj.SamplesPerYear
                error('OceanPipeline:SelectiveFilter:InvalidInput', ...
                    'Time and signal must match and contain at least four years.');
            end
            names=["Linear trend removed","Annual frequency removed", ...
                "Half-year frequency removed","Sub-quarter variation suppressed"];
            coefficients={1,localNotch(obj.SamplesPerYear), ...
                localNotch(obj.SamplesPerYear/2),ones(1,obj.SmoothWindow)/obj.SmoothWindow};

            values=cell(4,1); times=cell(4,1);
            values{1}=detrend(signal,'linear'); times{1}=time;
            for k=2:4
                transient=numel(coefficients{k})-1;
                filtered=filter(coefficients{k},1,values{k-1});
                values{k}=filtered(transient+1:end);
                times{k}=times{k-1}(transient+1:end);
            end

            layers=repmat(struct('Name',"",'Time',[],'Signal',[], ...
                'Coefficients',[],'Frequency',[],'Periodogram',[], ...
                'CumulativeGain',[]),4,1);
            cumulative={1,coefficients{2},conv(coefficients{2},coefficients{3}), ...
                conv(conv(coefficients{2},coefficients{3}),coefficients{4})};
            for k=1:4
                [frequency,periodogram]=localPeriodogram(values{k},obj.SamplesPerYear);
                omega=2*pi*frequency/obj.SamplesPerYear;
                gain=abs(localResponse(cumulative{k},omega)).^2;
                layers(k)=struct('Name',names(k),'Time',times{k},'Signal',values{k}, ...
                    'Coefficients',coefficients{k},'Frequency',frequency, ...
                    'Periodogram',periodogram,'CumulativeGain',gain);
            end

            f=layers(4).Frequency; p=layers(4).Periodogram;
            band=f>=1/obj.SearchPeriodYears(2) & f<=1/obj.SearchPeriodYears(1);
            [peakPower,index]=max(p(band)); bandFrequency=f(band);
            peakFrequency=bandFrequency(index);
            audit=struct('EstimatedPeriodYears',1/peakFrequency, ...
                'PeakFrequencyCyclesPerYear',peakFrequency, ...
                'TargetFrequencyCyclesPerYear',1/obj.TargetPeriodYears, ...
                'PeakToBandMedian',peakPower/median(p(band)), ...
                'FrequencyResolution',obj.SamplesPerYear/numel(values{4}), ...
                'ObservedTargetCycles',days(time(end)-time(1))/365.2425/obj.TargetPeriodYears);
            result=struct('Layers',layers,'Audit',audit);
        end

        function metadata=getMetadata(obj)
            metadata=struct('Name','SelectiveFilterPipeline', ...
                'SymbolicLatex','H_{notch}(L)=1-2\cos(\omega_0)L+L^2', ...
                'Symbols',[struct('Symbol','L','Description','Lag operator'), ...
                struct('Symbol','omega_0','Description','Frequency removed by the notch')], ...
                'SamplesPerYear',obj.SamplesPerYear,'SmoothWindow',obj.SmoothWindow);
        end
    end
end

function b=localNotch(periodSamples)
b=[1 -2*cos(2*pi/periodSamples) 1];
end

function response=localResponse(b,omega)
k=0:numel(b)-1;
response=exp(-1i*omega(:)*k)*b(:);
end

function [frequency,periodogram]=localPeriodogram(x,samplesPerYear)
x=x(:)-mean(x);
n=numel(x); transform=fft(x); last=floor(n/2)+1;
frequency=((0:last-1)'/n)*samplesPerYear;
periodogram=abs(transform(1:last)).^2/(2*pi*n);
if rem(n,2)==0
    periodogram(2:end-1)=2*periodogram(2:end-1);
else
    periodogram(2:end)=2*periodogram(2:end);
end
end
