classdef Validator
    % VALIDATOR Accessibility and Quality Assurance engine for visualizations.
    %
    % This class implements the Web Content Accessibility Guidelines (WCAG 2.0)
    % to ensure that scientific plots are legible for all users, including those
    % with visual impairments or color blindness.
    %
    % == KEY CONCEPTS ==
    %   1. Relative Luminance: How bright a color appears to the human eye.
    %      Green is perceived as much brighter than Blue, even at equal intensity.
    %
    %   2. Contrast Ratio: The difference in luminance between foreground (Text)
    %      and background.
    %      - Ratio 1:1  = Invisible (Same color)
    %      - Ratio 21:1 = Maximum (Black text on White background)
    %      - Ratio 4.5:1 = Minimum required for standard text (Level AA).
    %
    % Author: Ocean Library System
    % Date: 2026-01-30

    methods (Static)
        function checkContrast(bgColor, textColor)
            % CHECKCONTRAST Verifies if text is readable against a background.
            %
            % Inputs:
            %   bgColor   - [R, G, B] vector (0-1). The background color.
            %   textColor - [R, G, B] vector (0-1). The text/line color.
            %
            % Output:
            %   Prints a pass/fail message to the console. Generates a warning
            %   if the contrast is below the industry standard (4.5:1).
            
            % 1. Calculate Perceived Brightness (Luminance)
            %    We cannot just add R+G+B because the eye sees them differently.
            L1 = OceanViz.Utils.Validator.getLuminance(bgColor);
            L2 = OceanViz.Utils.Validator.getLuminance(textColor);
            
            % 2. Calculate Contrast Ratio (WCAG Formula)
            %    (L1 + 0.05) / (L2 + 0.05)
            %    The 0.05 is a "flare correction" for ambient light on screens.
            light = max(L1, L2);
            dark  = min(L1, L2);
            ratio = (light + 0.05) / (dark + 0.05);
            
            % 3. Check against Standards
            %    4.5 is the cutoff for WCAG Level AA (Normal Text).
            if ratio < 4.5
                warning('OceanViz:LowContrast', ...
                    'Accessibility Fail: Contrast Ratio is %.1f:1. Minimum required is 4.5:1.', ratio);
            else
                fprintf('Accessibility Pass: Contrast Ratio %.1f:1 (Excellent)\n', ratio);
            end
        end
        
        function L = getLuminance(rgb)
            % GETLUMINANCE Converts sRGB color to Relative Luminance.
            %
            % The math here converts "Screen Code" (sRGB) into "Linear Light"
            % (physics) to accurately model how photons hit the eye.
            
            % Ensure input is double precision for calculation
            rgb = double(rgb);
            
            % == STEP A: Gamma Correction (Linearization) ==
            % Screens apply a "Gamma Curve" to compress colors. We must
            % reverse this to get linear light values.
            % Threshold 0.03928 is where the sRGB curve switches from linear
            % to exponential.
            idx = rgb <= 0.03928;
            
            % Low values are linear (Slope 1/12.92)
            rgb(idx) = rgb(idx) / 12.92;
            
            % High values follow a power law (Gamma ~2.4)
            rgb(~idx) = ((rgb(~idx) + 0.055) / 1.055) .^ 2.4;
            
            % == STEP B: Weighted Sum (Spectral Sensitivity) ==
            % The human eye is most sensitive to Green (71%), then Red (21%),
            % and least sensitive to Blue (7%).
            % These coefficients are standard ITU-R BT.709.
            L = 0.2126 * rgb(1) + ... % Red Contribution
                0.7152 * rgb(2) + ... % Green Contribution
                0.0722 * rgb(3);      % Blue Contribution
        end
    end
end