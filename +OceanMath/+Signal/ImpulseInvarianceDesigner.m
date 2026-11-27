classdef ImpulseInvarianceDesigner
    % IMPULSEINVARIANCEDESIGNER  Design analog prototype filter; convert to digital
    %                             via Impulse Invariance method.
    %
    %   PEDAGOGICAL PURPOSE:
    %     One approach to digital filter design is: start from an analog prototype
    %     H_c(s) with well-understood frequency characteristics (Butterworth,
    %     Chebyshev), then convert to a digital filter H(z) that preserves the
    %     impulse response shape.
    %
    %     IMPULSE INVARIANCE RULE (Oppenheim §4.4.2):
    %       h[n] = Ts * h_c(n*Ts)
    %       i.e., the digital impulse response samples the analog impulse response.
    %
    %     CRITICAL LIMITATION (what makes this interesting to study):
    %       In the frequency domain, this mapping is NOT exact:
    %         H(e^{j*omega}) = (1/Ts) * sum_{k=-inf}^{inf} H_c(j*(omega - 2*pi*k)/Ts)
    %       Infinite periodic copies of H_c are summed together.
    %       Since real analog filters are NOT strictly bandlimited, these copies
    %       overlap in the digital stopband, causing ALIASING distortion.
    %
    %     The aliasing error shrinks as Ts -> 0 (fs -> inf). Exp 04 shows
    %     this explicitly by repeating the comparison at increasing fs values.
    %
    %   USAGE:
    %     d = OceanMath.Signal.ImpulseInvarianceDesigner('Type','Butterworth', ...
    %             'Order', 6, 'CutoffHz', 300, 'AnalogRate', 100000);
    %     [b_a, a_a]    = d.getAnalogCoeffs();
    %     [b_d, a_d]    = d.toDigital(fs);
    %     H_a           = d.evalAnalog(Omega_vec);    % |H_c(j*Omega)|
    %     H_d           = d.evalDigital(omega_vec, fs); % |H(e^{j*omega})|
    %
    %   DEPENDENCIES:
    %     Requires MATLAB Signal Processing Toolbox (for butter, impinvar).
    %     butter()   — Butterworth IIR analog prototype
    %     impinvar() — Impulse invariance analog-to-digital conversion
    %
    %   REFERENCE: Oppenheim & Schafer 3rd Ed.
    %              §4.4.2 — Impulse Invariance, Eq. 4.47–4.54
    %              §4.4.1 — LTI Discrete-Time Systems
    %
    %   Author: Ocean Library — Chapter 4 Signal Processing Series
    %   Date:   2026-08-26

    properties (SetAccess = private)
        FilterType  (1,:) char     % 'Butterworth' (only type currently supported)
        Order       (1,1) double   % Filter order N
        CutoffHz    (1,1) double   % Analog cutoff frequency (Hz)
        AnalogRate  (1,1) double   % F_ANALOG reference rate (Hz) — for eval grids
        % Analog prototype coefficients (continuous-time)
        B_analog    (1,:) double   % numerator polynomial b(s)
        A_analog    (1,:) double   % denominator polynomial a(s)
        OmegaC      (1,1) double   % analog cutoff in rad/s = 2*pi*CutoffHz
    end

    properties (Constant, Hidden)
        ClassDescription = ['Designs analog IIR filters (Butterworth) and converts '...
            'them to digital via impulse invariance (h[n]=Ts*h_c(n*Ts)). '...
            'Exposes evalAnalog() and evalDigital() for direct frequency response '...
            'comparison, making the stopband aliasing distortion visible. '...
            'See Oppenheim & Schafer §4.4.2 Eq. 4.47-4.54.'];
        ConceptNodeId = 'ch4-impulse-invariance';
        TextbookRef   = 'Oppenheim & Schafer 3rd Ed., §4.4.2 Eq. 4.47–4.54';
        PackagePath   = '+OceanMath/+Signal/ImpulseInvarianceDesigner.m';
    end

    methods
        % ── Constructor ──────────────────────────────────────────────────────────
        function obj = ImpulseInvarianceDesigner(options)
            % IMPULSEINVARIANCEDESIGNER  Construct and design the analog prototype.
            %
            %   d = ImpulseInvarianceDesigner('Type','Butterworth','Order',6, ...
            %           'CutoffHz', 300, 'AnalogRate', 100000)
            %
            %   Parameters:
            %     Type       — Filter family: 'Butterworth' (default)
            %     Order      — Filter order N (default 6)
            %     CutoffHz   — Analog -3dB cutoff in Hz (default 300)
            %     AnalogRate — Reference analog rate in Hz (for validation)
            arguments
                options.Type       (1,:) char   = 'Butterworth'
                options.Order      (1,1) double {mustBePositive, mustBeInteger} = 6
                options.CutoffHz   (1,1) double {mustBePositive} = 300
                options.AnalogRate (1,1) double {mustBePositive} = 100000
            end
            assert(strcmpi(options.Type, 'Butterworth'), ...
                'ImpulseInvarianceDesigner:UnsupportedType', ...
                'Only ''Butterworth'' is currently supported.');
            obj.FilterType = options.Type;
            obj.Order      = options.Order;
            obj.CutoffHz   = options.CutoffHz;
            obj.AnalogRate = options.AnalogRate;
            obj.OmegaC     = 2 * pi * options.CutoffHz;  % rad/s

            % Design analog Butterworth prototype: H_c(s)
            % butter() with 's' flag returns analog TF in rad/s
            [obj.B_analog, obj.A_analog] = butter(obj.Order, obj.OmegaC, 's');
        end

        % ── Core Conversion ──────────────────────────────────────────────────────

        function [b_d, a_d] = toDigital(obj, fs)
            % TODIGITAL  Convert analog prototype to digital via impulse invariance.
            %
            %   [b_d, a_d] = d.toDigital(fs)
            %
            %   Math (Oppenheim §4.4.2, Eq. 4.47):
            %     h[n] = Ts * h_c(n*Ts)   =>   H(z) = impinvar(B_a, A_a, fs)
            %
            %   Parameters:
            %     fs — Target digital sample rate in Hz
            %
            %   Returns:
            %     b_d, a_d — digital filter TF polynomials (for use with filter())
            %
            %   Note on aliasing: impinvar implements the partial-fraction
            %   sum exactly for the given poles. The aliasing distortion in
            %   the stopband comes from the infinite sum of spectral copies
            %   that are folded into [0, pi] (see Eq. 4.54).
            arguments
                obj
                fs (1,1) double {mustBePositive}
            end
            % impinvar: analog-to-digital conversion by impulse invariance
            % Internally performs partial-fraction expansion of H_c(s),
            % then maps each pole s_k -> exp(s_k / fs) in the z-domain.
            [b_d, a_d] = impinvar(obj.B_analog, obj.A_analog, fs);
        end

        % ── Frequency Response Evaluation ────────────────────────────────────────

        function H_mag = evalAnalog(obj, Omega)
            % EVALANALOG  Evaluate |H_c(j*Omega)| at given angular frequencies.
            %
            %   H_mag = d.evalAnalog(Omega)
            %
            %   Math: H_c(j*Omega) = B_a(j*Omega) / A_a(j*Omega)
            %         evaluated by polynomial substitution s = j*Omega.
            %
            %   Parameters:
            %     Omega — angular frequency vector (rad/s)
            %
            %   Returns:
            %     H_mag — |H_c(j*Omega)|, the analog magnitude response
            arguments
                obj
                Omega (1,:) double
            end
            % Evaluate B_a(j*Omega) and A_a(j*Omega) as polynomials in s=j*Omega
            s    = 1j * Omega;
            H_num = polyval(obj.B_analog, s);
            H_den = polyval(obj.A_analog, s);
            H_mag = abs(H_num ./ H_den);
        end

        function H_mag = evalDigital(obj, omega, fs)
            % EVALDIGITAL  Evaluate |H(e^{j*omega})| for digital filter at given fs.
            %
            %   H_mag = d.evalDigital(omega, fs)
            %
            %   Math: H(e^{j*omega}) evaluated at normalized frequencies omega.
            %   omega is in [0, pi] (rad/sample).
            %
            %   Parameters:
            %     omega — normalized angular frequency vector (rad/sample), [0, pi]
            %     fs    — digital sample rate (Hz)
            arguments
                obj
                omega (1,:) double
                fs    (1,1) double {mustBePositive}
            end
            [b_d, a_d] = obj.toDigital(fs);
            % freqz evaluates H(z) = B(z)/A(z) on the unit circle
            % at the specified normalized frequencies
            H_complex = freqz(b_d, a_d, omega);
            H_mag     = abs(H_complex);
        end

        function [b_a, a_a] = getAnalogCoeffs(obj)
            % GETANALOGCOEFFS  Return analog prototype polynomial coefficients.
            b_a = obj.B_analog;
            a_a = obj.A_analog;
        end

        % ── Metadata ─────────────────────────────────────────────────────────────

        function s = getMetadata(obj)
            % GETMETADATA  Return description struct for agents and OceanLearn.
            s.ClassName       = 'ImpulseInvarianceDesigner';
            s.ConceptNodeId   = obj.ConceptNodeId;
            s.TextbookRef     = obj.TextbookRef;
            s.Description     = obj.ClassDescription;
            s.FilterType      = obj.FilterType;
            s.Order           = obj.Order;
            s.CutoffHz        = obj.CutoffHz;
            s.OmegaC_rad_s    = obj.OmegaC;
        end

        function printSummary(obj)
            % PRINTSUMMARY  Print design summary to console.
            fprintf('=== ImpulseInvarianceDesigner ===\n');
            fprintf('  Filter type    : %s\n',  obj.FilterType);
            fprintf('  Order          : %d\n',  obj.Order);
            fprintf('  Cutoff         : %g Hz  (Omega_c = %.4f rad/s)\n', ...
                obj.CutoffHz, obj.OmegaC);
            fprintf('  Concept node   : %s\n',  obj.ConceptNodeId);
            fprintf('  Note: Use toDigital(fs) to convert at any sample rate.\n');
            fprintf('  Note: Higher fs -> less aliasing distortion in stopband.\n');
        end
    end
end
