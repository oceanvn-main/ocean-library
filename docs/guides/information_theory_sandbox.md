# Information Theory Sandbox — User & Developer Guide

## Overview

The **Information Theory Sandbox** is a programmatic, highly interactive GUI application (`OceanViz.App.InformationSandbox`) designed to explore the mathematical dualities of **Gambling & Log-Optimal Portfolios (Kelly Criterion)** and **Data Compression (Source Coding)** in a unified Dark Mode workspace.

The application decouples statistical simulation algorithms from premium vector-rendered graphics, utilizing high-precision timers, event loops, and interactive elements.

---

## Architecture & Layout

The app is built entirely programmatically as a single window divided into two primary workspaces via a modern tab interface:

1.  **Gambling & Kelly (Tab 1)**:
    *   **Left Controls**: Horse counts, normalized probability sliders, fair/unfair odds configurations, strategy presets (Kelly, Uniform, Favorite, Custom), starting wealth, and playback controls.
    *   **Center Panel**: Horizontal horse race lanes (`RaceTrackRenderer`) and real-time log-wealth trajectory comparisons (`log2(S_t)` vs. expected Kelly growth slope `W * t`).
    *   **Right Panel**: Live metrics: active race counter, current wealth, expected theoretical growth, empirical growth rate, source entropy, and betting divergence loss.
2.  **Data Compression (Tab 2)**:
    *   **Left Controls**: Alphabet size, normalized probability sliders, coder selection (Huffman vs. Shannon-Fano), and symbol path exploration.
    *   **Center Panel**: Interactive prefix-free code trees (`CodeTreeRenderer`) and a tabular dashboard displaying probabilities, codewords, lengths, ideal lengths, and expected bits.
    *   **Right Panel**: Live coding metrics: source entropy, expected average length, coding efficiency, redundancy, Kraft inequality sum, and Shannon limit bounds.

---

## Core Systems & Logic

### 1. Multiplicative Normalized Slider Callback
To enforce the axiomatic probability constraint (sum = 1.0), the sliders implement a relative ratio preservation re-normalization scheme:
If slider $i$ changes to $p_i^{new}$:
$$p_j^{new} = p_j^{old} \times \frac{1 - p_i^{new}}{\sum_{k \neq i} p_k^{old}} \quad \forall j \neq i$$
If the other probabilities were zero, the delta is distributed equally:
$$p_j^{new} = \frac{1 - p_i^{new}}{K - 1} \quad \forall j \neq i$$
This ensures smooth, natural transitions and exact normalization.

### 2. Guaranteed-Distribution Race Simulation
To animate the races smoothly while guaranteeing the winners obey the exact probability distribution:
1.  Prior to starting the race, the backend `RaceSimulator` draws the winning index $W$.
2.  The maximum distance is set to `1.0` for horse $W$, and random bounds in `[0.55, 0.93]` for all other horses.
3.  Each timer tick, positive random steps are added. When horse $W$ crosses the finish line, the race finishes.
4.  Empirical wealth $S_t$ updates by:
    $$S_t = S_{t-1} \times (b_W \times o_W + 1 - \sum_{k=1}^K b_k)$$

---

## Usage

Launch the sandbox app directly from the MATLAB command window:

```matlab
% Instantiate and display the application
app = OceanViz.App.InformationSandbox();
app.show();
```

---

## Examples

### 1. Proving Kelly Superiority
1.  Configure 4 horses with unequal probabilities: `p = [0.4, 0.3, 0.2, 0.1]`.
2.  Use **Fair Odds** (`Auto-calculate` enabled), setting odds multipliers to `[2.5, 3.33, 5.0, 10.0]`.
3.  Set the Betting Strategy to `Kelly Proportional`. Note that expected growth rate $W$ is positive.
4.  Hit `Play` (at 10 FPS) and let it run for 100 races. Note that the empirical log-wealth grows steadily along the theoretical Kelly slope.
5.  Pause the simulation and change the strategy to `Uniform`. Note that expected growth $W$ drops to `0` (since bets match fair odds but uniform doesn't match the source distribution, or becomes negative if non-Kelly).
6.  Change the strategy to `All-In Favorite`. Notice the extreme volatility: empirical wealth will grow fast, but eventually hit absolute bankruptcy (wealth = $0) and stop the app due to extreme risk.

### 2. Huffman vs. Shannon-Fano Coding
1.  Switch to the **Data Compression** tab.
2.  Set Alphabet Size to `5` and probabilities to: `p = [0.45, 0.2, 0.15, 0.1, 0.1]`.
3.  Compare the results:
    *   **Huffman**: Renders an optimal tree, average length `L = 2.10 bits`.
    *   **Shannon-Fano**: Renders a sub-optimal split, average length `L = 2.20 bits`.
4.  Under **Path Exploration**, select `S3`. The corresponding path branches inside the visual tree highlight in vibrant cyan, showing you exactly how prefix-free codewords correspond to left/right tree branches.

---

## See Also

*   `OceanMath.Information.Processes.RaceSimulator`
*   `OceanMath.Information.Processes.KellyOptimizer`
*   `OceanMath.Information.Coding.HuffmanCoder`
*   `OceanMath.Information.Coding.ShannonFanoCoder`
*   `OceanViz.Strategies.RaceTrackRenderer`
*   `OceanViz.Strategies.CodeTreeRenderer`
