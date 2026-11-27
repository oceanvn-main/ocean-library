# Ocean Library — Architecture Reference

## OceanStorage Rust engine (initial milestone, 2026-10-16)

`+OceanStorage/` is a separate, from-scratch storage-engine study package.
`rust/` contains the `ocean_storage` library and `ocean-storage` executable:
fixed 4096-byte pages, B+ tree indexing with linked leaves and tombstone deletion,
bounded clean-page cache, full-page redo WAL, checksums, recovery and exclusive
OS directory ownership. The first milestone exposes bounded byte keys/values,
single-record transactions and a versioned stdio protocol. Python and Java
clients live under `clients/`; `OceanStorage.Client` is the MATLAB uint8 adapter.
It does not migrate or replace existing OceanData stores. LSM, columnar tables,
catalog integration, SQL and distributed operation remain later milestones.
Contract: `+OceanStorage/CONTRACT.md`; execution evidence and review gate:
`+OceanStorage/BUILD_LOG.md`; recipes: `docs/recipes/ocean_storage.md`.

## Social Operations local application (2026-10-11)

`App/Social Operations/` is an isolated .NET 10 Blazor application for the
social operations plan in `Planning/System/Social App/`. Its initial slice has
a SQLite-backed prepared-post store, manual reminder UI, Telegram bot text
publishing and polled inbox/replies, and account capability labels. Meta token
connections, bounded comment polling, and manual comment replies are an initial
developer test path; reviewed OAuth and Meta publishing remain pending. A local
dashboard reads operational counts from SQLite and labels missing remote metrics.
The product plan now includes multi-platform automations and analytics as later
delivery tracks. This application does not depend on
the MATLAB Ocean packages; connector and delivery boundaries are described in
its plan and README.

## Python RL distribution (2026-09-15)

`Experiments/Scripts/RL/Python` is an installable `ocean-rl` project with
pyproject.toml and optional image/plot/Gym dependencies. `ocean_rl.cli.main`
dispatches `--system racetrack`; legacy launchers remain compatible. Default
maps and MC metadata ship as package data. Results are caller-relative, outside
the installed package. Algorithms, environments, evaluators and experiments
retain separate responsibilities. GitHub Actions checks Windows/Linux Python
3.11/3.12, regression tests, critical lint, builds and isolated wheel assets.
See the project CONTRIBUTING.md for extension and publication instructions.
> **For AI agents and future developers. Read this before writing any code.**  
> Root: `c:\Ocean Library\Matlab code practice\` | Updated: 2026-09-29 (Hamilton Ch.6 Spectral Analysis Replication)

---

## 1. Philosophy — Six Principles Baked Into Every Class

### NLP Process Explorer (initial vertical slice, 2026-10-05)

`+OceanNLP/+Pipeline/ComposablePipeline.m` holds validated `OceanNLP.Pipeline.Stage` objects (built with `Stage.fromFunction`/`fromMethod`; struct stages are deprecated). Stages are checked for one-input callability on `add`/`replace`, may declare `Requires`/`Provides` state fields, and `run` calls `validate` (spaCy `analyze_pipes`-style contract walk) before executing; uncontracted stages are reported but never rejected. Edits are silent by default; `AutoValidate = true` warns (`ChainBroken`) after any edit that breaks the chain, using the pipeline's `InitialFields`. `Stage.apply` is the single invocation path shared by `run` and `PipelineAdapter`. Plan: `Planning/NLP/Pipeline_Validation_Upgrade_Plan.md`.

`+OceanNLP/+Inspect/MemoryReader.m` is the first inspection read model for real, bounded process runs. `PipelineAdapter` captures enabled `ComposablePipeline` stages and their actual inputs and outputs; `capture` retains failed-step metadata, while `run` rethrows failures. `+OceanViz/+App/+NLP/ProcessExplorer.m` is a standalone text-first `uifigure` app over the reader contract (`listSteps`, `getStep`, `getArtifact`). It is separate from `OceanReplay`/Replay Studio. `+OceanAlgo/+Search/+Inspect/HNSWInspector.m` collects optional decision events from the domain-neutral hierarchical `HNSWIndex` and exposes bounded native MATLAB graph neighborhoods. This is an in-memory initial slice; paging, persistent artifacts, richer presenters, and HNSW-to-reader integration remain subsequent phases. Design: `Planning/NLP/NLP_Process_Explorer_Implementation_Plan.md`; recipe: `docs/recipes/nlp_process_explorer.md`.

### Information Extractor symbolic notebook (Phase 10, 2026-10-12)

`App/Information Extractor/src/symbolic/` provides the browser-side, versioned symbolic notebook state, guarded kernel client, and three-pane notebook view. Review and editing remain available from `file://`; computation is enabled only from the loopback companion in `App/Information Extractor/server/`. `+OceanTools/+SymbolicNotebook/Kernel.m` is the fixed-command MATLAB boundary over `OceanMath.Symbolic.SourceExtractor`, `Registry`, and `Workspace`; `dispatchJson` and `worker` expose newline-delimited JSON only to the local HTTP bridge. Notebook aliases, assumptions, source fingerprints, kernel versions, immutable history, cancellation state, and explicit candidate pagination are persisted or disclosed by contract. The frozen protocol and acceptance boundary are in `Planning/Information_Extractor_Phase_10_Execution_Contract.md`; the implementation plan is `Planning/Information_Extractor_Phase_10_Symbolic_Notebook_Implementation_Plan.md`.

| Principle                      | How It's Applied                                                                                |
| ------------------------------ | ----------------------------------------------------------------------------------------------- |
| **Open/Closed (OCP)**          | Add algorithms by creating new classes, never modifying existing ones                           |
| **Dependency Inversion (DIP)** | High-level code depends on interfaces (`IModelStrategy`, `IDistribution`), not concrete classes |
| **Strategy Pattern**           | Every swappable algorithm is a pluggable object injected at runtime                             |
| **Observer Pattern**           | Simulators fire events; visualizers listen — no direct coupling                                 |
| **Adapter Pattern**            | `ISpatialMapper` bridges abstract state indices to physical coordinates                         |
| **Template Method**            | Base classes orchestrate; abstract hooks implement domain-specific steps                        |

### 1.1 Research and Pedagogical Focus (Legibility Over Production)

Unlike enterprise production frameworks that prioritize raw speed and black-box optimization at the expense of transparency, the Ocean Library is built primarily for researchers and students. The core mission is to enable deep understanding of systems and algorithms by showing what goes on behind the scenes. Optimization (e.g. separable convolutions) must always remain transparent, parameterized, and readable, rather than hidden inside un-inspectable black boxes. For details, see the behavioral rule [`research_philosophy.md`](file:///.agents/rules/research_philosophy.md).

---

## 2. Package Map

```
+OceanAgents/     ← NEW 2026-09-28: game-theory agent system (A1 host/mind/runners + A2 beliefs/trust done). NOT statistics,
                     so not in OceanStats. Plan/contract: Planning/System/Game Theory/Implementation_Build_Plan_A0_A1.md,
                     A0_Contract_Freeze.md. Guide docs/guides/ocean_agents.md; recipes docs/recipes/agent_recipes.md.
                     +Core/      ← value records (ActorView, Decision, Feedback, Transition, DecisionFrame, ActionBatch,
                                   EnvSpec, CapabilitySet), interfaces (IPolicy, ILearner, IDecisionProcess,
                                   IJointEnvironment, ISingleAgentEnvironment), assertValueData (value-data-only rule)
                     +Env/       ← NativeSingleAgentBridge (any OceanStats.RL.Env.IEnvironment), +Reference/RepeatedMatrixGameEnv
                     +Policies/  ← RandomLegalPolicy, TablePolicy, EpsilonGreedyQPolicy;  +Learners/ ← OneStepQLearner
                     +Minds/     ← PolicyMind (reactive profile);  +Artifacts/ ← ActionCodec, DiscreteStateCodec, PolicyArtifact
                     +Runtime/   ← AgentHost, SingleAgentRunner, JointRunner, StreamRegistry, TransitionBuilder,
                                   RunRecord, IsolationChecker
                     +Beliefs/   ← (A2) BeliefRecord, IBeliefStore, FixedBeliefStore, BetaTrustStore; evidence via
                                   Core.EvidenceRecord in ActorView.Evidence, each ID consumed once
                     +Minds/TrustMind + +Minds/+TrustRules/ (ExpectedUtilityRule, ThompsonRule); +Env/+Reference/DelegationEnv
                     Contract: Planning/System/Game Theory/A2_Beliefs_Contract.md
                     +World/     ← (W1) generic hierarchical world kernel: Timeline (clock only), Registry, Store
                                   (plain arrays; value/linear/clock components), Calendar (predictions, scheduled
                                   commands, skip-safe monitor checks), World loop (phases, sync points, fail-stop);
                                   records RuleSpec, DoorSpec, Event, Command. +Guard/: Dimension, ComponentType,
                                   AxiomSpec, Certificate, GuardEngine (admission by types-with-dimensions + axioms,
                                   never names; V1..V13). Design World_Kernel_Design.md; contract W1_Kernel_Contract.md
                                   W2 (2026-09-30): AgentWorld (world as IJointEnvironment); +Doors/: Agency, GridMotion,
                                   RewardTermination, GridWorlds (native grid envs rebuilt from doors, bit-identical to
                                   env_golden.mat), RLDoor (module axioms RL-1..9). +Env/WorldAsEnvironment = reverse bridge
                                   (door world as native IEnvironment). Contract W2_Grid_Doors_Contract.md
                     AB1 (2026-10-01): +Minds/+Signals (SignalType, Signal, Wave, Port, Router), +Components (IComponent verify-
                                   then-act, IConverter, +Converters/ThresholdConverter, Sampler, Motor, Scheduler), +Memory (IMemory
                                   one writer, TableMemory, EpisodeMemory), +Body (BodySignature, IBody, EmptyBody, GridRobotBody,
                                   IBehavior, RepeatMove), Minds/SignalMind + QSignalMind (learners inside the mind; AgentHost keeps the
                                   gate), +Embodied/Dualism. Design Agent_Body_Signal_Design.md; contract AB1_Signal_Body_Contract.md
                     AB2 (2026-10-01): +Body/+Sensors (IObservation sensors: DigitObservation, MovesObservation = what the
                                   Laws permit, BehaviorObservation, VisionObservation + IRenderer/GridRenderer), +Minds/+Perception
                                   (IPerception: Digit, Moves, Behavior), PolicySignalMind; the mind decides AND learns only from
                                   observations; Env/+Reference/TicTacToeEnv (turn-based). Terms: Laws = world rules, Policy = RL.
                                   Contract AB2_Observation_Contract.md
                     DV1 (2026-10-01): +Devices (IDevice, Terrain, Reward, DeviceWorld = world rules as devices, native
                                   IEnvironment facade; cliff walking bit-identical to env_golden.mat); Components/GridPosition (body-owned
                                   cell, moved only by the terrain). Contract DV1_Device_World_Contract.md
                     PG1 (2026-10-01): Player (mind + body + host; observers via +Observe/IPlayerObserver, PlayerTrace),
                                   Core/ActionFactors, Policies/FactoredPolicy + FictitiousPlayPolicy, Learners/FactorLearner +
                                   FictitiousPlayLearner, Memory/BeliefMemory, Env/+Reference/ParityGuessEnv. Fictitious play itself
                                   lives in OceanStats.GameTheory.Learning (EmpiricalBelief, FictitiousPlay) and
                                   OceanStats.GameTheory.Solvers.FictitiousPlaySolver. Contract PG1_Parity_Guess_Contract.md
                     Invariants: labels not columns; records hold no handles; one rollout owner (runner); learning only
                     after a committed step; per-purpose mrg32k3a streams, never the global RNG; fail-stop RunRecord.
                     Extend: new policy -> implement IPolicy; new learner -> ILearner; new mind -> IDecisionProcess;
                     new shared world -> IJointEnvironment; new belief model -> IBeliefStore; new trust rule -> ITrustRule;
                     new world behaviour -> a DoorSpec (rules + owned components + axioms), never a kernel edit.
                     Tests: OceanTest.OceanAgents.* (541).
+OceanAlgo/       ← Generic algorithm primitives (IOptimizationStrategy, GreedySearch),
                     domain-neutral vector search (IVectorIndex, HNSWIndex, IVFPQIndex,
                     VectorSearchAdapter), and compression (ProductQuantizer,
                     ResidualProductQuantizer)
                     +StateSearch/        ← NEW 2026-10-11: domain-neutral state-space search on a planning model (never a live env).
                       Contracts: ISearchModel (actions/key/isGoal), IDistributionModel (outcomes), ISampleModel (sample), SearchResult (Status ok|budget_exhausted|no_path|no_legal_action|at_goal).
                       Engines: AStarSearch (A*, weighted A*), IDAStarSearch, BeamSearch (+ diverse beam), DepthLimitedSearch (S&B 8.9), RolloutSearch (8.10), MonteCarloTreeSearch (8.11, UCB, chance nodes, subtree reuse).
                       Parts: MinHeap, StateTable, UCB1Selection, IRolloutPolicy/UniformRandomRolloutPolicy, rolloutReturn, DistributionSampleModel, checkOutcomes.
                       Plan: Planning/System/Search/State_Search_Build_Plan.md; recipe docs/recipes/state_search_recipes.md; tests OceanTest.OceanAlgo.StateSearch.*.
                     +Optimization/       ← NEW 2026-09-03: Continuous Unconstrained Numerical Optimization
                       MatlabOptimizationWrapper.m ← Wrapper around MATLAB's fminunc solver providing live telemetry logging via OutputFcn callback (steepdesc, dfp, bfgs, newton).
                     +Spatial/            ← NEW 2026-05-12
                       KDTree.m           ← Generic N-d KD-Tree (Bentley 1975). Max-spread split, AABB pruning. findKNearest/rangeQuery. Metrics: euclidean/manhattan/chebyshev/custom.
+OceanDL/         ← Deep learning (top-level since 2026-09-23): Tensor autograd, nn, optim, Layout, Recurrent,
                     +Generative/+Diffusion (DiffusionModel, MLPDenoiser, INoiseSchedule/LinearVP/CosineVP/EDM/RectifiedFlow, ISampler, Targets, TimeSampler);
                     legacy NN/CNN stack. See section 5b.
+OceanCS/         ← Scheduling utilities (ISchedulingStrategy, Scheduler, Heuristic/Serial strategies)
+OceanEngine/     ← System infrastructure layer (memory, histograms, pipeline orchestration)
                     +Accumulators/      ← NEW 2026-09-10: Domain-Agnostic Online Statistics Accumulators
                       IAccumulator.m              ← Abstract strategy interface: update(x,w,idx), getValue(), getWeight(), reset(), getMetadata(), getDiagramSpec()
                       AbstractAccumulator.m       ← Abstract base class: template update loop, table allocation, subscript indexing, reset logic
                       WeightedMeanAccumulator.m   ← Incremental weighted mean & importance sampling (S&B Eq 5.8) with subscript indexing
                       ExponentialMovingAverage.m  ← Constant step-size alpha tracking (S&B Sec 2.5)
                       WelfordVarianceAccumulator.m← Welford (1962) online mean, variance, and standard deviation
                       AccumulatorFactory.m        ← Static factory & catalog: creates accumulators by name ('Weighted', 'EMA', 'Welford') + static 3-Pillar introspection
+OceanMath/+Interpolation/ ← NEW 2026-08-18: Generic N-D Scattered Data Interpolation (Ex 4.1 §4.1)
                       IScatteredInterpolant.m     ← Abstract strategy: fit(X,y)/predict(Xq)/isFitted()/getDim()
                       ScatteredInterpolant.m      ← UNIFIED MASTER CLASS: wraps RBF, Kernel Regression, KNN, Natural, Spline into one interface
                       ScatteredRBF.m              ← N-D RBF strategy: (K + lambda*I)*w = y. Pluggable IRBFKernel (eq 4.3-4.11)
                       ScatteredKernelRegression.m ← N-D Nadaraya-Watson regression strategy: distance-weighted average (eq 4.12-4.14)
                       ScatteredNatural.m          ← Adapter: MATLAB scatteredInterpolant (natural/linear/nearest)
                       ScatteredKNN.m              ← Adapter: KNNClassifier in regression mode
                       ScatteredSpline.m           ← Adapter: SmoothingSpline (1D only)
                       InterpolationFactory.m      ← Static factory: returns configured ScatteredInterpolant instance
                       +meta/*.meta.json           ← Pedagogical metadata sidecars for GUI learner app (getMetadata)
+OceanMath/       
                    +Symbolic/          ← Scalar Symbolic Math Toolbox workspace plus static MATLAB/Python SourceExtractor and typed Registry for transparent/opaque operators
                       Workspace.m       ← Safe expression parser, named blocks, domain checks, hygienic rename, portable JSON import/export. No source execution or AI dependency.
                    +Information/       ← Information Theory & Statistical Information
                       ScoreFunction.m           ← NEW 2026-09-08: Standalone Score vector V_i(theta) = d/d theta ln f(x_i; theta), regularity condition check E[V] = 0 (Cover & Thomas Eq. 11.268-11.269), and getMetadata() self-describing reflection.
                       FisherInformationMatrix.m ← NEW 2026-09-08: Score vector V, Expected Fisher Info J(theta)=E_theta[(d/d theta ln f)^2] (Cover & Thomas Eq. 11.270), empirical Observed Fisher Info I_n(theta_hat)=-Hessian or OPG, and getMetadata() self-describing reflection.
                    +Signal/            <- NEW 2026-05-24: Signal Processing Package
                       IdealLowpass.m     <- Ideal lowpass filter, truncated Fourier approximation, Gibbs overshoot grid search, Parseval verification
                       ContinuousSignalGenerator.m <- NEW 2026-08-26: Ch.4 — Simulates analog signals as dense discrete arrays (sinusoid, dualTone, chirp, multiTone). Ground-truth for all Ch.4 experiments. getMetadata() pullable by OceanLearn.
                       SpectralAnalyzer.m <- NEW 2026-08-26: Ch.4 — Static spectral analysis utilities (spectrumTwoSided, spectrumOneSided, stft, findPeaks, aliasFormula, sqnr, powerSpectralDensity). Single source of truth for FFT axis conventions.
                       SamplingEngine.m   <- NEW 2026-08-26: Ch.4 — 1D periodic downsampling. Two modes: sample() direct (shows aliasing), sampleWithFilter() anti-aliased (prefilter + decimate). predictAlias() computes folding formula.
                       ImpulseInvarianceDesigner.m <- NEW 2026-08-26: Ch.4 — Butterworth analog prototype + impulse invariance A/D conversion. evalAnalog(Omega) vs evalDigital(omega,fs) reveals stopband aliasing. Requires Signal Processing Toolbox.
                       QuantizationEngine.m <- NEW 2026-08-26: Ch.4 — Uniform B-bit midrise quantizer. quantize(), error(), theoreticalSQNR() (6.02*B+1.76 dB), empiricalSQNR(). Used by Exp 09 and as 1-bit comparator in SigmaDeltaModulator.
                       PolyphaseEngine.m  <- NEW 2026-08-26: Ch.4 — Arbitrary-M polyphase FIR decomposition. filterDirect() vs filterPolyphase() (Noble Identity). verifyEquivalence() asserts bit-identical output. Speedup = M.
                       SigmaDeltaModulator.m <- NEW 2026-08-26: Ch.4 — First/second-order sigma-delta ADC. modulate() runs feedback loop. reconstruct() applies LPF+decimate. theoreticalSQNR() = 10*log10(9*OSR^3/pi^2). NTF(z)=1-z^{-1}.
                       +SimulinkGuide/    <- NEW 2026-09-04: Step-by-step manual build guide for Simulink App GUI (docs/guides/ch4_simulink_workflow_guide.md and recipes/ch4_simulink_workflow_recipes.md)
                       +Kernels/          <- NEW 2026-06-15: Decoupled filter kernel strategies
                         IFilterKernel.m    <- Abstract interface for discrete 1D/2D kernels
                         BoxKernel.m        <- Discrete N-D/average box blur kernel
                         TriangleKernel.m   <- Bilinear/triangle kernel
                         GaussianKernel.m   <- Gaussian binomial kernel approximation
                         SincKernel.m       <- Windowed Sinc ideal low-pass filter
                         SobelKernel.m      <- Sobel derivative kernel
                         SecondDerivativeKernel.m <- 2D cross-derivative (corner) kernel
                         CustomKernel.m     <- Adapter wrapping raw numeric arrays
                         KernelFactory.m    <- Static factory and resolver for kernels
                       +Lifting/          <- NEW 2026-06-17: Algebraic lifting transforms
                         ILiftingTransform.m <- Abstract interface for algebraic lifting (DWT)
                         LiftingScheme.m    <- In-place wavelet transforms (Haar and CDF 5/3)
                       +meta/             <- NEW 2026-08-26: Pedagogical metadata sidecars for OceanLearn
                         ch4_signal_classes.meta.json <- ConceptNode JSON for all 7 Ch.4 classes (id, domain, methods, cross_bridges, textbook_ref)
                    +Regularization/     <- PROMOTED 2026-08-05: Domain-Agnostic Robust Penalty Functions (1D, 2D, Graphs)
                      IRobustPenalty.m   <- Abstract strategy interface: evaluate, weight, derivative, getName, isConvex
                      QuadraticPenalty.m <- L2 quadratic penalty rho(d)=d^2
                      L1Penalty.m        <- L1 Total Variation / trend filtering penalty rho(d)=|d|
                      PNormPenalty.m     <- p-norm penalty rho(d)=|d|^p (p in (0,2])
                      HuberPenalty.m     <- Piecewise Huber penalty
                      GNCPenalty.m       <- Blake-Zisserman non-convex penalty min(d^2, V)
                      PenaltyFactory.m   <- Static factory for penalty creation
                    +Operators/          <- NEW 2026-08-05: Linear Difference Operators (1D, 2D, Graphs)
                      IDifferenceOperator.m <- Abstract interface for linear difference operator D
                      DifferenceOperator1D.m <- 1D 1st (TV) and 2nd (Hodrick-Prescott/L1 trend) difference operator
                      DifferenceOperator2D.m <- 2D image spatial difference operator
                      DifferenceOperatorGraph.m <- Graph incidence difference operator (E x V)
                      DifferenceOperatorAffinity.m <- Luminance-guided bilateral difference operator (L = I - W)
                    +Variational/        <- NEW 2026-08-05: Unified Domain-Agnostic Variational Energy & Solvers
                      VariationalEnergy.m <- Master orchestrator: totalEnergy, buildDifferenceMatrix, buildWeightMatrix
                      +Solvers/
                        ISolver.m          <- Abstract: solve(energy, g)→f, getHistory()→struct(Energy,Residual,NumIter)
                        IRLSSolver.m       <- IRLS solver: works on 1D signals, 2D images, and N-D graphs
                        PrimalDualSolver.m <- Chambolle-Pock solver for 1D, 2D, and N-D graphs
                        GNCSolver.m        <- Blake-Zisserman GNC continuation solver
                    +Geometry/          <- NEW 2026-06-23: Spatial Mathematics and Coordinate Systems
                      CameraProjection.m   <- Models 3D to 2D projections (Perspective, Orthographic, WeakPerspective)
                      LensDistortion.m     <- Models lens aberrations (radial and tangential models)
                      Transform2D.m        <- Homogeneous transformations for 2D points & lines
                      Transform3D.m        <- Homogeneous transformations for 3D points & planes
                      Rotation3D.m         <- Procedural conversions between Quat, Euler, Axis-Angle, Exp-Map, Matrix
                      +Rotations/          <- Object-oriented strategy pattern for 3D rotation representations
                        RotationRepresentation3D.m <- Abstract strategy interface for rotations
                        QuaternionRotation3D.m    <- Concrete strategy for quaternions
                        EulerRotation3D.m         <- Concrete strategy for Euler angles
                        AxisAngleRotation3D.m     <- Concrete strategy for Axis-Angle representation
                        ExpMapRotation3D.m        <- Concrete strategy for Exponential Map / Rodrigues' formula
                        MatrixRotation3D.m        <- Concrete strategy for raw SO(3) matrices
                    +Graphics/          <- NEW 2026-06-23: 3D Rendering and Shading
                      +Scene/              <- Virtual world structures (Meshes, Objects, Cameras)
                      +Lighting/           <- Virtual light sources (Point, Directional, Spot)
                      +Shading/            <- Material light interaction models (Lambertian, Phong)
                      +Renderer/           <- Ray tracing and mesh rasterization engines
                    +Image/             <- NEW 2026-06-23: Dense grid pixel operations and Computer Vision
                      +Filters/            ← Spatial neighborhood filters
                        INeighborhoodFilter.m ← Abstract Template Method base class for spatial filters
                        BoxFilter.m        ← O(1) Box filter via Integral Images
                        MedianFilter.m     ← Non-linear Median filter
                        BilateralFilter.m  ← Edge-preserving Bilateral filter (direct & grid)
                        GaussianFilter.m   ← Linear Gaussian blur filter convolving from scratch
                        NonLocalMeansFilter.m ← Advanced patch-based Non-Local Means denoising filter (direct & grid)
                        GuidedFilter.m     ← NEW 2026-08-19: Guided Image Filter (He et al. 2013, TPAMI). O(N) box-filter integral image. Self-guided: apply(I). Cross-guided: applyGuided(p, guidance_I). Ex 4.3 §4.3.
                        JointBilateralFilter.m ← NEW 2026-08-19: Joint Bilateral Filter (Kopf et al. 2007). Range weights from guidance image I, decoupled from filtered signal p. applyGuided(p, guidance_I). Ex 4.3 §4.3.
                      +Compression/        ← NEW 2026-08-19: JPEG-style image compression/decompression (Ex 4.3 §4.3)
                        JpegSyntheticCompressor.m ← Software DCT encoder/decoder with custom Q tables. compress(img)→{img_out,q_all,Q_table}. JPEG Annex K quality scaling formula.
                        ChromaSubsampler.m ← YCbCr 4:2:0/4:2:2/4:4:4 chroma subsampling. subsample(YCbCr)→{Y,Cb,Cr}. upsample() via GuidedFilter (default), JBF, or nearest.
                        PocsDeblockingEngine.m ← POCS alternating projection loop. Per-block DCT clamp to [Q*(q-0.5), Q*(q+0.5)]. Pluggable INeighborhoodFilter smoother (default BilateralFilter). deblock(img,q_all,Q_table)→{img_out,history}.
                      +Lifting/            <- Dimensional lifting engines
                        BilateralGrid.m    <- Splat-Blur-Slice 3D grid computational engine
                      +Resampling/         <- Image warping and texture sampling strategies
                        ImageResampler.m   <- High-quality separable image resampling (Nearest, Bilinear, Bicubic, Lanczos)
                        EWASampler.m       <- Elliptical Weighted Average anisotropic filtering
                        PiecewiseAffineWarp.m <- Piecewise affine image warping over a triangle mesh (C0 continuous)
                        BilinearQuadWarp.m <- Quadrilateral grid image warping using triangulation
                        ThinPlateSpline.m  <- Thin Plate Spline (TPS) coordinate interpolation and warping (smooth C-inf)
                        DisplacementFieldWarp.m <- Forward displacement warping with hole filling
                        BeierNeelyWarp.m   <- Beier-Neely line-based image morphing
                        BSplineWarp.m      <- Cubic B-Spline Free-Form Deformation (FFD) — C2-continuous control-grid warping (Ex 3.23 Part 3)
                        ImageMorpher.m     <- Feature-based cross-dissolve image morphing engine (Ex 3.25)
                        ContentAdaptedMesh.m <- Feature-aware adaptive tri-mesh generator
                        DemosaicBayer.m    <- Bayer Color Filter Array (CFA) demosaicing (Bilinear & MHC)
                        BlueScreenMatting.m <- Blue/green screen matting (difference matting & despilling)
                        BackgroundSubtractor.m <- Statistical background subtraction (difference keying, MedianMAD robust estimation, RGBA alpha matting, morphology)
                        IRBFKernel.m       <- Abstract strategy interface for RBF kernels
                        ThinPlateKernel.m  <- RBF thin plate spline kernel
                        GaussianRBFKernel.m <- RBF Gaussian kernel
                        MultiquadricKernel.m <- RBF Multiquadric kernel
                        InverseDistanceKernel.m <- RBF Inverse Distance Weighting kernel
                      +Regularization/     <- NEW 2026-08-05: Robust penalty functions (Szeliski §4.2)
                        IRobustPenalty.m   <- Abstract strategy interface: evaluate, weight, derivative, getName, isConvex
                        QuadraticPenalty.m <- rho(x)=x^2, w=1 (constant IRLS weights, 1-step direct solve)
                        L1Penalty.m        <- rho(x)=|x| Total Variation (Rudin-Osher-Fatemi 1992), IRLS w=1/|x|
                        PNormPenalty.m     <- rho(x)=|x|^p  p∈(0,2]: Quadratic→TV→Hyper-Laplacian; convex iff p>=1
                        HuberPenalty.m     <- Classic piecewise Huber (NOT pseudo-Huber): L2 for |d|<=sigma, L1 beyond
                        GNCPenalty.m       <- Blake-Zisserman (1987) clamped penalty: min(d^2,V); non-convex
                        PenaltyFactory.m   <- Static factory: create('l1'), create('pnorm','P',0.8), etc.
                      +Variational/        <- NEW 2026-08-05: Image variational energy & solvers (Szeliski §4.2 - §4.2.4)
                        ImageRegularizationEnergy.m <- Orchestrator: totalEnergy, buildDifferenceMatrix(rows,cols), buildWeightMatrix(f)
                        LevinColorizer.m   <- NEW 2026-08-05: Optimization-based colorization (Levin et al. 2004, §4.2.4). YUV chrominance WLS solve.
                        +Solvers/
                          ISolver.m          <- Abstract: solve(energy, g)→f, getHistory()→struct(Energy,Residual,NumIter)
                          IRLSSolver.m       <- IRLS: iterates (I+lambda*D'W_kD)f=g. MaxIter/Tol/LinearSolve('backslash'|'pcg')
                          PrimalDualSolver.m <- Chambolle-Pock (2011): dual ascent on p + primal proximal step. Tau/Sigma auto from normest(D).
                          GNCSolver.m        <- GNC continuation: anneals V from VMax→Vtarget (log-spaced). NumScales/VMax/InnerMaxIter.
                      +MultiResolution/    <- Multi-scale/multi-resolution engines
                        IPyramid.m         <- Abstract interface for multi-resolution pyramids
                        PyramidOperator.m  <- Decimation and interpolation utilities (BypassFilter for aliasing demo)
                        GaussianPyramid.m  <- Low-pass Gaussian scale representation (BypassFilter for aliasing demo)
                        LaplacianPyramid.m  <- Band-pass Laplacian detail representation. report() method audits overcompleteness (pixel count, per-level entropy contribution).
                        PyramidBlender.m   <- NEW 2026-07-13: Laplacian pyramid blending (Burt & Adelson 1983, Szeliski Ex 3.18). N-image blend via Gaussian mask pyramid. Supports BypassFilter for aliasing comparison.
                        LocalLaplacianFilter.m <- NEW 2026-07-15: Local Laplacian Filter (Paris et al. 2011, Szeliski Ex 3.20). Edge-preserving contrast manipulation via fast bin-interpolated remapping in Laplacian pyramid space. Stores InputGaussianPyramid, InputLaplacianPyramid, OutputLaplacianPyramid as properties. plotPyramids() visualizes all three pyramids side-by-side using OceanViz Dark theme. Operates in CIE L*a*b* space on color images.
                        MipMap.m           <- NEW 2026-07-15: Texture mapping mip-map implementation. Builds Gaussian scales and computes continuous levels d = log2(max_singular_value(J)) from inverse transform Jacobians. Supports Nearest, Floor, Ceil, and Trilinear filtering lookups.
                      +Pixels/             <- Point and neighborhood pixel transformation strategies
                        IPixelTransform.m    <- Abstract interface for all pixel operations
                        TransformPipeline.m  <- Chains multiple transformations sequentially
                        LinearOperator.m     <- Adjusts contrast scale and brightness bias
                        GammaCorrection.m    <- Applies power-law correction
                        ColorBalance.m       <- Applies channel-wise gains for color balancing (linear or sRGB)
                        HistogramEqualization.m <- Applies full or partial histogram equalization
                        AdaptiveHistogramEqualization.m <- Contrast-Limited Adaptive Histogram Equalization (CLAHE) (with bilinear accumulation and spatial CDF filtering options)
                        MedianFilter.m       <- Non-linear rank-based median noise filter
                        BilateralFilter.m    <- Edge-preserving non-linear bilateral filter
                        SteerableFilter.m    <- Steerable filter responses for edge/feature extraction (Freeman & Adelson)
                        LookupTableTransform.m <- Performs O(1) intensity mappings
                        ColorSpaceConverter.m <- Transforms color spaces (RGB, Gray, HSV, YCbCr)
                        BoundaryPadder.m     <- Unified boundary padding (border effects) strategy
                        SpatialFilter.m      <- Applies neighborhood convolutions
                        CrossCorrelation.m   <- Applies 2D neighborhood cross-correlation
                        TrueConvolution.m    <- Applies 2D mathematical convolution
                        MinMaxStretch.m      <- Dynamically stretches dynamic range
                        MeanMatchStretch.m   <- Centering dynamically around luma mean
                        ZeroStuffing.m       <- Upsamples by inserting zeros between original samples
                        SynthesisFilter.m    <- Applies wavelet synthesis filtering
                        Decimation.m         <- Downsamples an image with optional anti-aliasing
                    +Transit/
                      TransitLine.m      <- Modified 2026-04-04: Upgraded public APIs with modern arguments blocks for autocomplete
                      TransitNetwork.m   <- Modified 2026-04-04: Upgraded public APIs with modern arguments blocks for autocomplete
                      TransitAgent.m     <- Modified 2026-04-04: Upgraded public APIs with modern arguments blocks for autocomplete
                      TransitSimulation.m <- Modified 2026-03-21: Multi-agent orchestrator, passes AccelRate/BrakeRate through addAgent()
                      +AgentStates/       <- NEW 2026-03-21: State Pattern for TransitAgent
                        IAgentState.m      Abstract interface: step, getPosition, getTangent, getProgress, isArrived, getName
                        RunningState.m     Moving along curve segment, detects segment end
                        DwellingState.m    Dwell countdown at station
                        ArrivedState.m     Terminal no-op state
                    +Simulation/
                      CurveSimulator.m   <- Physics engine: SUVAT kinematics, AccelProfile mode for custom a(t) functions
                      KinematicState.m   <- Immutable snapshot (Time, Parameter, Position, Velocity, Acceleration, Tangent)
                    +Information/        <- NEW 2026-05-21: Information Theory Sandbox
                      +Processes/
                        RaceSimulator.m    <- Monte Carlo K-horse race simulator using InverseTransformSampler
                        KellyOptimizer.m   <- Growth rate optimization, proportional betting fractions, entropy/divergence metrics
                        GamblingSimulator.m <- Sequential betting campaign, empirical growth vs analytical theory
                      +Coding/
                        ShannonFanoCoder.m <- Top-down binary prefix source coding strategy
+OceanStats/      ← Statistical learning (the core ML engine)
                    +Optim/              ← UPDATED 2026-09-16: Domain-Agnostic Tensor Optimizer Engine
                      IOptimizer.m         ← Abstract legacy interface for per-parameter solvers
                      IOptimizerStrategy.m ← NEW 2026-09-16: Abstract 3-Pillar Optimizer Strategy interface (getMetadata, getPipelineSpec, state_dict)
                      ITensorTarget.m      ← NEW 2026-09-16: Abstract domain adapter contract (getParameters, setParameters, computeGradients)
                      ParameterGroup.m     ← NEW 2026-09-16: PyTorch-style layer-wise hyperparameter group container
                      VectorizedOptimizerEngine.m ← NEW 2026-09-16: High-performance 1D contiguous vector optimization engine (3-Pillar O=(phi,f,D))
                    +Image/              ← NEW 2026-08-19: Image quality metrics (Ex 4.3 §4.3)
                      ImageQualityMetrics.m ← Static PSNR and SSIM metrics. psnr(img,ref)→dB. ssim(img,ref)→[0,1]. Wang et al. (2004) 11x11 Gaussian window SSIM. printTable(results) formatted table.
                    +Base/               ← NEW 2026-04-24: IAffinityStrategy (Affinity graph construction interface)
                    +Supervised/+Discriminant/
                      MDA2.m               ← NEW 2026-03-24: Shared-center MDA (ESL Ex 12.11), wraps MDA2Model + Recipe_MDA2_EM + GenericOptimizer
                    +Supervised/+Neighbors/
                      KNNClassifier.m      ← Modified 2026-05-12: KDTree-backed O(log N) predict. Adds Metric/Weights/Task/regression. Falls back to vectorized brute-force for custom DistanceFcn.
                      DANNClassifier.m     ← NEW 2026-04-06: Discriminant Adaptive NN (ESL 13.4.1), local metric via W/B covariance, getLocalMetric() for ellipse viz
                    +Inference/
                      +HypothesisTest/      ← NEW 2026-09-08: Statistical Hypothesis Testing Framework
                        +Base/
                          IStatisticalTest.m ← Abstract strategy interface for hypothesis tests (test, getName, getNullHypothesis)
                          TestResult.m       ← Immutable container for test statistic, p-value, df, rejection flag, and metadata
                        +Likelihood/
                          LikelihoodRatioTest.m ← Likelihood Ratio Test strategy computing lambda = 2*(logL_unrestricted - logL_restricted) and Chi-Square upper-tail p-value
                      +Models/
                        MDA2Model.m          ← NEW 2026-03-24: Pure state for MDA2 (Mu, Sigma, Pi, ClassProfile)
                        GMM.m                ← NEW 2026-03-25: Robust Gaussian Mixture Model (ESL Ex 14.3.2) bypassing singular covariance matrices
                      +Recipes/
                        Recipe_MDA2_EM.m     ← NEW 2026-03-24: IMMRecipe E/M steps for MDA2
                        Recipe_FixedSigmaGMM_EM.m ← NEW 2026-04-06: Fixed-sigma GMM EM (ESL Ex 13.1), freezes Sigma=sigma*I, centroid trajectory tracking
                    +Unsupervised/
                      SpectralClustering.m ← NEW 2026-04-24: Spectral Clustering using pluggable IAffinityStrategy
                      NMF.m                ← NEW 2026-04-29: Non-negative Matrix Factorization (Lee & Seung, 1999). ESL §14.6. Multiplicative updates (Frobenius norm). fit/transform/reconstruct/getBasisImages.
                      ArchetypalAnalysis.m ← NEW 2026-04-29: Archetypal Analysis (Cutler & Breiman, 1994). ESL §14.6.1. Alternating NNLS + simplex projection. W rows & B rows sum to 1. Archetypes on convex hull.
                      FactorAnalysis.m     ← NEW 2026-04-29: Classical Factor Analysis (ML). X=A*S+eps, Sigma=A*A^T+D_eps. ESL §14.7.1. Wraps factoran. fit/transform/getLoadings/getCovModel/printSummary.
                      ICA.m                ← NEW 2026-04-30: ICA orchestrator (whitening + strategy dispatch). ESL §14.7. Pluggable IICASolver strategies. fit/transform/getComponents/getMixingMatrix/getWhitenedData.
                      +ICASolvers/         ← NEW 2026-04-30: Pluggable ICA solver strategies (implements IICASolver interface)
                        IICASolver.m         ← Abstract interface: solve(X_w, q, verbose) -> [W, converged, iters]. Protected deflate() helper.
                        FastICAStrategy.m    ← Fixed contrast (tanh/cubic). ESL §14.7.3. Configurable Alpha, MaxIter, Tol.
                        ProDenICAStrategy.m  ← Adaptive KDE contrast. ESL §14.7.4. Gaussian KDE, Silverman bandwidth, O(N^2) per step.
                        KernelICAStrategy.m  ← STUB. Bach & Jordan (2002). HSIC dependence minimization on Stiefel manifold. Not yet implemented.
                        amariDist.m          ← NEW 2026-04-30: Amari distance d(A0, A_hat). ESL Eq. 14.97. Measures ICA mixing matrix recovery quality (0=perfect, invariant to permutation/sign).
                      +Affinity/           ← NEW 2026-04-24: Affinity graph generators
                        MutualKNNStrategy.m
                        FullyConnectedRBFStrategy.m
                    +RL/                  ← Reinforcement Learning (Sutton & Barto)
                      +Env/               ← NEW 2026-09: ALGORITHM-FREE environments (no learner package is referenced; enforced by EnvCoreTest). Gymnasium-style contract.
                        IEnvironment.m      ← Abstract: [obs,info]=reset('Seed',s); [obs,r,terminated,truncated,info]=step(a); setSeed; ObservationSpace/ActionSpace; unwrapped(); 3 pillars. info.action_mask = admissible actions at the NEW observation (all false at a terminal).
                        IEnumerableEnvironment.m ← Capability: finite listable state space (get_all_states, state_to_index [0 = terminal outside the table], action_to_index, get_valid_actions, is_terminal).
                        IModelEnvironment.m ← Capability (extends Enumerable): exact dynamics get_transitions(s,a) -> next, prob, mean reward (the env.P idea). Sample-only environments do not implement it.
                        +External/          ← NEW 2026-09-30: GymEnv / GymTabularEnv (+ gym_bridge.py): a Gymnasium (Python) environment behind IEnvironment / IModelEnvironment. Optional dependency: Python with gymnasium visible to MATLAB (pyenv); tests skip without it. Actions 1..A, Discrete obs s+1; Strict refusals satisfy RL-5.
                        TabularEnvironment.m ← Concrete base built from ONE transition table (sampled + exact model cannot disagree); validates the model; exactPolicyValues, uniformPolicy.
                        EnvWrapper.m        ← Wrapper base (delegates; unwrapped chain).
                        +Wrappers/TimeLimit.m ← Truncation after MaxSteps (terminal on the last step stays a termination).
                        +Spaces/            ← ISpace, Discrete(n,'Start'), Box(low,high), MultiDiscrete(nvec): contains(x), sample([stream]).
                        +Classic/           ← RandomWalkEnv, CliffWalkingEnv (Gymnasium CliffWalking-v1 dynamics), WindyGridworldEnv, MaximizationBiasEnv, GridTabularEnv (abstract helper), BlackjackEnv (sample-only), RacetrackEnv (sample-only), ModifiedCarRentalEnv (model-only).
                                              Randomness: own RandStream after setSeed/reset('Seed'); until then MATLAB's global stream (rng(k) still reproduces an unseeded run).
                      +Compat/            ← NEW 2026-09: the ONLY meeting point of +Env and the learner packages (deprecated, removed with the learners' migration)
                        LegacyEnvironment.m ← Presents an IEnvironment through the older MonteCarlo.IEpisodeEnvironment interface (three-output step, was_truncated); adapt(env, class) for sampling solvers, adaptModel(env) for exact solvers.
                      +TD/                ← Temporal-difference learning (Ch.6-7, 12) — DECOMPOSED 2026-09 (Phase 2 of the RL architecture refactor): a thin TDLearner wires independent strategies; a new algorithm/component = ONE new file
                        TDLearner.m         ← Thin orchestrator: options -> parts, run() restarts from Seed; Task control|prediction; Mode = Backup name; Double_learning; accepts any Env.IEnvironment (adapted); hooks 'ValueFunction' and 'Policy'
                        Sampler.m           ← The ONLY code that touches the environment: reset/step, indices, admissible-action table, MaxSteps truncation, terminal outside the table (placeholder), seeding
                        Context.m           ← Everything a backup needs for one run (sampler, values, policy, target, stepSize, trace, observer, stream, gamma, lambda, n, progress)
                        ITDComponent.m, Factory.m ← 3-pillar base + convention-based lookup (target|stepsize|exploration|trace|backup)
                        +Backups/           ← the update scheme ("Mode"): IBackup (capabilities supportsValueKind/allowsOffPolicyTarget/supportsOffline/supportsDoubleValues), OneStepBackup, NStepBackup, TraceBackup, LambdaReturnBackup
                        +Values/            ← what is learned: IValueFunction, TabularActionValues (base), TabularQ, DoubleTabularQ (Double Q-learning), TabularV
                        +Policies/          ← behaviour: IBehaviourPolicy, ExplorationPolicy (control), FixedPolicy (prediction, validated)
                        +Observers/         ← IObserver, Recorder (Record=true; replay format), NullObserver
                        +Targets/ (TD0, Sarsa, ExpectedSarsa, QLearning; ITarget capabilities valueKind/supportsDouble/bootstrap), +StepSize/ (Constant, Harmonic), +Exploration/ (EpsilonGreedy, DecayingEpsilonGreedy, Softmax), +Traces/ (Accumulating, Replacing; visitWith for Dutch-style), +Functional/ (pure return/probability functions)
                      +DP/                ← Dynamic Programming (Ch.4)
                        TabularEnvironment.m ← DEPRECATED legacy abstract base (get_all_states, get_valid_actions, get_transitions); new code implements Env.IModelEnvironment. Solvers accept either.
                        ModifiedCarRentalEnv.m ← DEPRECATED name: empty subclass of Env.Classic.ModifiedCarRentalEnv
                        PolicyIterationSolver.m ← Decoupled DP Policy Iteration solver (compute_q_value, evaluate_policy, improve_policy, run); accepts any IModelEnvironment
                        ValueIterationSolver.m  ← Decoupled DP Value Iteration solver (compute_q_value, evaluate_step, extract_policy, run); accepts any IModelEnvironment
                        BellmanOptimality.m ← Pure expected optimality backup over an exact Dyna.Models.IWorldModel distribution; validates successor and terminal semantics.
                        RTDPSolver.m        ← Real-time DP over indexed exact models: bounded greedy planning trials from a start state, owned repeatable sampling stream, inspectable backups and trajectory, explored-state policy. See docs/recipes/rtdp_recipes.md and Planning/RL/Planning and Learning with Tabular method/RTDP_Build_Contract.md. LRTDP solved labeling is a separate future phase.
                      +PAC/               ← NEW 2026-10-12: PAC model-based exploration. TransitionCounts (first-m-samples empirical model, sparse T̂), RMaxAgent (Strehl, Li & Littman 2009 Alg. 1: optimistic U = Rmax/(1-γ), VI on known pairs), E3Agent (Kearns & Singh 2002: balanced wandering, known-state MDP, exploration-biased or α(t) schedule), knownStatePlans (exploit/escape plans), IExplorationAgent (run loop). Env.Classic.ChainEnv (Strens 2000). Recipe docs/recipes/pac_exploration_recipes.md; experiment Experiments/Scripts/RL/exp_pac_exploration_chain.m.
                      +DecisionTime/      ← NEW 2026-10-11: RL application of OceanAlgo.StateSearch (S&B 8.8): WorldModelSearchModel (IWorldModel / IModelEnvironment -> planning model; read-only models cached), TableRolloutPolicy, DecisionTimeAgent (plan, act once, replan; MCTS tree reuse). Experiment: Experiments/Scripts/RL/exp_ch8_decision_time_planning.m.
                      +Dyna/              ← Planning and learning with tabular models (Ch.8): DynaLearner owns real rollout; pluggable deterministic/empirical/exact models, uniform/prioritized planning, recency bonus, causal observers; reuses TD values/targets/step sizes/exploration; owned RNG streams
                      +MonteCarlo/        ← Monte Carlo Methods (Ch.5 & Sec 5.8)
                        IEpisodeEnvironment.m ← LEGACY abstract base for episodic sample-based environments (three-output step, was_truncated). Provides generate_episode() with pre-allocated buffers. New environments implement Env.IEnvironment and are wrapped by Compat.LegacyEnvironment.
                        BlackjackEnv.m        ← DEPRECATED name: thin Compat.LegacyEnvironment subclass of Env.Classic.BlackjackEnv (read-only forwarding of the old public properties)
                        RacetrackEnv.m        ← DEPRECATED name: thin Compat.LegacyEnvironment subclass of Env.Classic.RacetrackEnv (+ static getMetadata/getPipelineSpec kept)
                        MonteCarloSolver.m    ← Generic MC solver Context class delegating execution to +Strategies/ algorithms and +Weighting/ importance sampling schemes. Full 3-Pillar contract: Pillar 1 (getMetadata LaTeX/symbols), Pillar 2 (run numerical execution), Pillar 3 (getPipelineSpec diagram port signature).
                        +Weighting/           ← Decoupled Importance Sampling Weighting Schemes (S&B Sec 5.5, 5.6, 5.8)
                          IImportanceWeighting.m               ← Abstract Strategy Base Class for IS weighting schemes (3-Pillar contract)
                          OrdinaryISWeighting.m                ← S&B Eq 5.5: Unweighted cumulative empirical mean
                          WeightedISWeighting.m                ← S&B Eq 5.6: Bounded-variance normalized weighted average
                          DiscountingAwareOrdinaryISWeighting.m← S&B Eq 5.9: Flat partial return unweighted estimator (gamma < 1)
                          DiscountingAwareWeightedISWeighting.m← S&B Eq 5.10: Flat partial return normalized weighted estimator (gamma < 1)
                        +Strategies/          ← Decoupled MC Algorithm Strategies (S&B Ch.5 & Sec 5.8)
                          IMonteCarloStrategy.m                ← Abstract Strategy Base Class for MC algorithms (3-Pillar contract)
                          FirstVisitPredictionStrategy.m        ← S&B p.92: First-visit MC prediction V_pi(s)
                          EveryVisitPredictionStrategy.m        ← S&B Sec 5.1: Every-visit MC prediction V_pi(s)
                          ExploringStartsStrategy.m            ← S&B p.99: MC Exploring Starts control Q*(s,a) & pi*(s)
                          OnPolicyControlStrategy.m            ← S&B p.101: On-policy epsilon-greedy control
                          OffPolicyPredictionStrategy.m        ← S&B Sec 5.5-5.6: Off-policy prediction via IS
                          OffPolicyControlStrategy.m           ← S&B p.111: Off-policy control via weighted IS
                          DiscountingAwarePredictionStrategy.m <-- S&B Sec 5.8: Discounting-aware off-policy prediction (Eq 5.9 & 5.10)
                          DiscountingAwareControlStrategy.m    <-- S&B Ch.5 Ex 5.14: Discounting-aware off-policy control for Q(s,a)


                    +GameTheory/          ← NEW 2026-09-23: Strategic-Form Games & Dominated Strategies (Bonanno Ch.1 §1.1)
                      +Core/
                        IPayoffStore.m           ← Abstract payoff-storage contract: numPlayers/numStrategies/getPayoff/getPlayerTensor (OCP seam for future sparse/symmetric backends)
                        DensePayoffTensor.m      ← Default IPayoffStore: one N-D array per player, O(N*M^N) (mirrors OpenSpiel matrix_game/tensor_game)
                        NormalFormGame.m         ← Strategic-form game G=(N,{S_i},{u_i}) facade over an IPayoffStore. Bimatrix (PayoffA/PayoffB) or N-player (PayoffTensors) construction.
                        RoundRobinTournament.m   ← Many-player (20-30+) scoring via bounded k-player session decomposition (Axelrod 1980/1984 tournaments; session-sharding lesson from GTA Online) instead of one P-player tensor
                        IAuctionRule.m           ← NEW 2026-09-23: Abstract single-item auction rule as ONE pure function: resolveAgainstMax(ownBid,maxOtherBid,ownValue)->payoff (aggregative-game reduction; swap point for first-price/all-pay/etc.)
                        SecondPriceRule.m        ← NEW 2026-09-23: Concrete IAuctionRule -- Vickrey (1961) second-price rule
                        IBidStrategy.m           ← NEW 2026-09-23: Minimal contract for any bidder: proposeBid(value,bidGrid)->bid
                        ILearnableBidStrategy.m  ← NEW 2026-09-23: Optional capability extending IBidStrategy: observeOutcome(bid,payoff)->obj (mirrors IModelEnvironment's opt-in-capability pattern)
                        TruthfulBidStrategy.m    ← NEW 2026-09-23: Deterministic IBidStrategy -- always bids own value (the Vickrey-optimal baseline)
                        SingleItemAuctionGame.m  ← NEW 2026-09-23: Generator -- random values + IAuctionRule -> one reduced 2-player NormalFormGame per bidder, for analytic IteratedDominanceSolver checks (cost independent of NumBidders)
                        RepeatedAuctionSimulator.m ← NEW 2026-09-23: The auction "environment" -- Rule + BidGrid + Values + {IBidStrategy} -> runRounds() history; rule and strategies fully decoupled
                      +Solvers/
                        ISolver.m                    ← Abstract: solve(game)->result, getHistory() (mirrors OceanMath.Image.Variational.Solvers.ISolver)
                        IteratedDominanceSolver.m    ← Strict/weak dominance checks + vectorized Iterated Elimination of Strictly Dominated Strategies (IESD), with round-by-round elimination history
                        MonteCarloBidStrategy.m      ← NEW 2026-09-23: Concrete ILearnableBidStrategy -- epsilon-greedy K-armed-bandit bid learner (reuses OceanStats.RL.TD.Functional.epsilonGreedyProbs)
                    +Time/                ← NEW 2026-09-03: Stochastic Time Series & Likelihood Modeling
                      ARMAProcess.m        ← Synthetic ARMA(1,1) time series realization generator y(t) = phi*y(t-1) + e(t) + theta*e(t-1).
                      ARMALikelihood.m     ← Box-Jenkins conditional Gaussian log-likelihood calculator & 2D surface grid sweeper.
                    +Data/               ← Data processing and transformations
                      +Weighting/          ← Term weighting schemes (TF-IDF)
                        TfidfWeighting.m
                        +Idf/                ← Strategy pattern for IDF calculation
                          IIdfScheme.m
                          SmoothIdf.m
                          ClassicIdf.m
                          ProbabilisticIdf.m
                          IdfSchemeFactory.m
                      +Norm/               ← Strategy pattern for row-wise vector normalization
                        INormScheme.m        ← Abstract strategy for normalization
                        L1Norm.m             ← L1 normalization (sum of absolute values)
                        L2Norm.m             ← L2 normalization (Euclidean length)
                        NoneNorm.m           ← No-op normalization
                        NormSchemeFactory.m  ← Factory for instantiation
+OceanNLP/        ← Natural Language Processing & RAG Pipeline components
                    +Chunking/           ← Text segmentation strategy pattern
                      IChunker.m                ← Abstract chunker interface
                      ChunkerFactory.m          ← Static factory ("recursive", "sliding", "sentence-window")
                      FixedTokenChunker.m       ← Fixed word-count chunking strategy
                      SlidingWindowChunker.m     ← Overlapping sliding-window chunking strategy
                      SentenceWindowChunker.m   ← Sentence boundary window with context radius
                      RecursiveCharChunker.m    ← Line-by-line / character length chunking strategy
                    +Index/              ← Corpus indexing strategies
                      IIndexingStrategy.m       ← Abstract indexing strategy interface
                      IndexStrategyFactory.m    ← Static factory ("inverted_index", "matrix")
                      InvertedIndex.m           ← Inverted postings list strategy (decoupled from BM25 math)
                      MatrixIndexStrategy.m     ← Sparse N x V weight matrix strategy
                    +Retrieval/          ← Information retrieval & reranking strategies
                      IRanker.m                 ← Abstract retriever interface
                      IReranker.m               ← Abstract 2nd-stage candidate reranker interface
                      BM25Ranker.m              ← BM25 ranker delegating to IIndexingStrategy
                      BiEncoderRanker.m         :- Dense semantic bi-encoder retriever (SBERT/DPR)
                      DenseRanker.m             ← Dense full-scan retriever
                      ANNRanker.m               ← Approximate Nearest Neighbor vector search (HNSW)
                      CrossEncoderReranker.m    ← Two-stage retrieve-and-rerank pipeline
                      FunctionReranker.m        ← Function handle adapter for custom reranking handles
                      HybridRanker.m            ← Multi-retriever rank fusion (RRF)
                      MultiQueryRanker.m        ← Query expansion multi-query rank fusion
                    +Store/              ← Corpus storage layers for vector search & retrieval
                      ICorpusStore.m            ← Abstract corpus storage interface
                      InMemoryCorpusStore.m     ← Flat in-memory text chunk and source document ID store
                      ParentChildCorpusStore.m   ← Hierarchical Parent-Child retrieval store (child search -> parent expansion)
+OceanData/       ← Data lifecycle: storage layers (DA-1 to DA-3)  ← NEW 2026-03

                    ExperimentResult.m  ← NEW 2026-08-05: Self-describing experiment output container & reflection protocol
                    +Generators/
                      ESLBinaryGenerator.m ← NEW 2026-04-07: ESL 13.3.1 easy/difficult binary classification (uniform features, deterministic labels, Bayes oracle)
                      SmoothFunctionGenerator.m ← NEW 2026-08-18: Ex 4.1 §4.1. Random superpositions of sinusoids+Gaussians in 1D/2D. Modes: sinusoid/gaussian/mixed/piecewise. Stores f_true handle in MetaData.
                      SyntheticImageGenerator.m ← NEW 2026-08-19: Ex 4.3 §4.3. Procedural test image (gradient/checkerboard/mixed). Mixed mode: bilinear gradient + geometric shapes + sinusoidal texture + step edge. generate()→(H x W x 3) double [0,1].
+OceanPipeline/   ← Data movement & ingestion (DA-4 to DA-6)       ← NEW 2026-03-11
                    +Ingestion/          ← Data readers & API interfaces
                      BinanceDataReader.m           ← NEW 2026-09-28: Binance REST API reader (Spot/Futures Klines, Orderbook Depth, Recent Trades, 24hr Ticker). Implements IDataReader.
                      BinanceFetcher.m              ← NEW 2026-10-02: Windowed Binance OHLCV ingestion ('1d','3d','1w','1m') with auto-pagination across REST pages. Wraps BinanceDataReader; computes startTime/endTime; merges pages; deduplicates.
                      RobustBinanceHistoryReader.m ← NEW 2026-10-02: Fail-stop historical range reader with bounded retry/backoff, closed-candle selection, incremental MAT cache, coverage provenance, and market-data validation.
                      MarketDataValidator.m        ← NEW 2026-10-02: Audits OHLC ordering, positivity, finite values, duplicates, missing intervals, and irregular timestamp spacing.
                    +Transforms/         ← Pipeline operator transforms & dual representation blocks
                      BinanceTSAProcessor.m         ← NEW 2026-09-28: Time Series Analysis (TSA) preprocessor & EDA engine. Causal retime, log returns, realized vol, VWAP, ADF stationarity, ACF/PACF, spectral density, anti-leakage temporal split, Parquet lake export.
                      DifferenceSpectrumFilter.m    ← NEW 2026-10-02: Nth-order backshift-difference (1-B)^d spectrum filter. Computes before/after Welch PSD, theoretical |H(f)|^d gain overlay, Parseval verification (< 1% threshold), and variance-reduction summary.
                      SelectiveFilterPipeline.m     ← NEW 2026-10-02: Transparent long-cycle analysis pipeline exposing detrended/notched/smoothed time layers, raw periodograms, cumulative transfer gains, and a predeclared-band cycle audit.
                      LikelihoodRatioTestOperator.m ← NEW 2026-09-08: Decoupled pipeline operator for Likelihood Ratio Test (Hamilton TSA Ex 5.8.5). Dual representation layer for statistical test.
+OceanViz/        ← Visualization (themes, strategies, simulation rendering, apps)
                    +Charts/             ← NEW 2026-09-29: Standalone modular chart renderers
                      CandlestickChart.m    ← NEW 2026-09-29: Performance-optimized financial OHLCV candlestick & volume chart renderer (vectorized O(1) handles wicks, bodies, MA7/25/99).
                    +Adapters/
                      GraphStyle.m     <- NEW 2026-03-20: Mutable visual attributes (node color/size/icon, edge color/width, path highlights, annotations)
                      LegendManager.m  <- NEW 2026-03-20: Collects legend entries (lines, icons, symbols) and renders on axes
                    +Strategies/
                      ACFStrategy.m           ← NEW 2026-09-29: Autocorrelation (ACF) & PACF visualization strategy with 95% confidence ribbon, zero baseline, TeX rendering, and significant lag highlights.
                      CandlestickStrategy.m   ← UPDATED 2026-09-29: Binance-style OHLCV strategy delegating to OceanViz.Charts.CandlestickChart.
                      ImageVizStrategy.m      ← NEW 2026-08-05: 2D image, heatmap, and RGB tensor renderer
                      TabularRLStrategy.m     ← NEW 2026-08-19: Ex 4.2 & Ex 4.7 Tabular RL renderer (3D state value surface, 2D policy heatmap, dark theme)
                      MonteCarloRLStrategy.m  ← NEW 2026-08-27: Ch.5 MC RL renderer (S&B Figs 5.1-5.3). Modes: ValueSurface3D (V(s) 3D surface by ace flag), PolicyHeatmap (hit/stick 2D boundary, teal/rose cmap), ConvergenceCurve (avg return + MA trend vs episodes), Combined (4-panel dashboard). All legends themed to active OceanViz Dark palette.
                      LikelihoodOptimizationStrategy.m ← NEW 2026-09-03: Likelihood surface optimization 4-panel dashboard renderer (Contour terrain + trajectories, score convergence, gradient norm decay, step size trajectory).
                      DeblockingStrategy.m    ← NEW 2026-08-19: Ex 4.3 §4.3. 6-panel OceanViz Dark-theme dashboard: Ground Truth / JPEG Compressed / POCS Deblocked / Error Heatmap x2 / POCS Convergence curve with PSNR/SSIM table.
                      ScatterInterpStrategy.m ← NEW 2026-08-18: Ex 4.1 dashboard. 3-4 panels: ground truth+samples, fitted surface, error heatmap/ribbon, optional CV curve. Supports Dim=1 and Dim=2.
                      TransitGraphStrategy.m  ← NEW 2026-03-20: Spline-based transit/graph renderer (curved routes, interchange icons, path glow)
                      PrototypeClassifierStrategy.m ← NEW 2026-03-24: ESL-style prototype classifier plot (stippled regions, boundaries, prototypes)
                      FunctionalDataStrategy.m ← NEW 2026-03-24: Generic overlaid curves grouped by class (spectra, time series, waveforms)
                      RaceTrackRenderer.m     ← NEW 2026-05-21: High-fidelity horse race track simulation visualization (cyberpunk glow lanes, trailing sparks, winner flash/reset)
                      CodeTreeRenderer.m      ← NEW 2026-05-21: Reingold-Tilford prefix code tree visualizer with interactive path highlights
                      TimeSeriesVizStrategy.m ← NEW 2026-06-09: 3-panel time series dashboard (Realization, empirical/theoretical ACF, Durbin-Levinson PACF, confidence bounds)
                    +Designer/  <- NEW 2026-03-20 (Phase 4: Interactive Spline GUI)
                      TrackDesigner.m    <- Modified 2026-04-04: Refactored base parameter passing to modern arguments blocks
                      TrackRenderer.m    <- Modified 2026-04-04: Converted constructor to modern arguments block
                      DraggablePoint.m   Lightweight draggable marker with OnMove/OnRelease callbacks
                      +Commands/         <- NEW 2026-03-20: UndoManager rewrite
                        TimelineController.m <- Tracks history (MaxHistory=40), isolates state mutability from UI, manages redo branches
                    +Simulator/
                      TransitSimulatorApp.m <- Modified 2026-04-04: Python GUI wrapped with public loadMap/launch API; decoupled ax_map/ax_ui for blitting performance.
                    +App/
                      ExperimentStudio.m      ← NEW 2026-08-05: Master generic interactive studio UI & drag-drop experiment inspector
                      InteractiveImageCanvas.m ← NEW 2026-08-05: Interactive mouse brush tool for Szeliski §4.2.4 colorization scribbles
                      InformationSandbox.m    ← NEW 2026-05-21: Interactive Information Theory Sandbox containing Kelly race simulator & source coding tree explorer.
                      Computer Vision/
                         ComputerVisionSandbox.m <- UPDATED 2026-07-13: Tab 7 Image Blending (Ex 3.18) added. Programmatic interactive Computer Vision Sandbox app (warps, rotations, projections, lens distortions, image pyramids, resampling, pyramid blending). All tabs include Show Theory popup and Bypass Filter checkbox.
                        LightSimulationSandbox.m  ← NEW 2026-06-06: Programmatic interactive 3D Room Light Simulation Sandbox app (camera, lights, shading strategies, vector/raytrace)
                        TransformEditor2D.m       ← NEW 2026-06-10: Programmatic interactive 2D Transform Editor app (Ex 2.2) with homography/Procrustes solvers and JSON persistence.
                        Viewer3D.m                ← UPDATED 2026-07-16: Programmatic interactive 3D Primitives Viewer app (Ex 2.3) now supporting perspective-correct texture-mapped polygon rendering (Ex 3.27) with depth buffering.
                         ImagePreprocessor.m       ← NEW 2026-06-11: Programmatic interactive Image Preprocessing Sandbox app (Phases 1-5).
                         RainbowEnhancer.m         ← NEW 2026-07-16: Programmatic interactive Rainbow Enhancer app (Ex 3.29) with polar unrolling, radial background fitting, and signal enhancement.
                         NNStudio.m                ← UPDATED 2026-09-16: Programmatic interactive Neural Network Studio app; updated NNCircularLayout to accept options struct and map circular positions.

+OceanTest/       ← Unit test framework (BaseTestCase, TestRunner, test suites)
                    +OceanNLP/
                      +System/
                        +Validation/  ← Python cross-validation reference data and MATLAB test comparisons
+OceanTools/      ← Dev utilities (ParametricGUI, crossvalIndices, snippets)
                    +SymbolicWorkbench/ ← Native UIFigure symbolic explorer; scans .m/.py text without execution, reviews source-linked formulas and opaque interfaces, edits contracts, composes accepted operators, and uses MATLAB Symbolic Math Toolbox only for supported scalar expansions. Optional aiDraft/validateAIDraft keep model proposals separate from verified algebra.
                    cachedFit          ← NEW 2026-03-17: train-once, load-on-rerun model caching
                    ExperimentContext  ← NEW 2026-03-17: reusable Steps 2-4 ML pipeline handle
                    (DataLake + PostgresWarehouse + Redis + run_id + saveResult + cachedFit + queryHistory)
Experiments/      ← Research entry-point scripts (exp_*.m) — NOT part of any package
                    Scripts/Unsupervised/exp_kernel_topology_ablation.m  ← NEW 2026-04-28:
                      ESL §14.5 Kernel Topology Ablation Study. 4-stage cyclical design:
                      (1) Dense RBF kernel K [diag=1, off-diag>0]
                      (2) KPCA branch: double-centering K_tilde=HKH vs Spectral branch: sparse W=mask.*K, L=G-W
                      (3) Eigengap analysis, indicator function comparison (step-fn vs wave)
                      (4) Cross-injection ablation: W->KPCA, K->Laplacian — proves matrix topology governs success
                    Scripts/Unsupervised/exp_spectral_clustering.m    ← ESL Fig 14.29: mutual KNN affinity (k=10), 3 concentric circles
                    Scripts/Unsupervised/exp_spectral_clustering2.m   ← NEW 2026-04-29: RBF affinity contrast — sigma auto via median-distance heuristic, 4-panel visualization
                    Scripts/Unsupervised/exp_pca_digits.m             ← NEW 2026-04-29: ESL §14.5.1 PCA on USPS digit 3. Economy SVD, scree plot, PC directions as images, PC1 vs PC2 scatter, reconstruction quality panels
                    Scripts/Unsupervised/exp_spca_digits.m            ← NEW 2026-04-29: ESL §14.5.5 Sparse PCA on same digit. Uses SparsePCA library class. Lambda1 sweep, sparsity vs variance tradeoff, loading image comparison (standard vs sparse rows), reconstruction comparison
                    Scripts/Unsupervised/exp_nmf_digits.m             ← NEW 2026-04-29: ESL §14.6 NMF vs PCA vs VQ on USPS digit 3. NMF uses multiplicative updates (r=25). Basis image grids, sample reconstruction, MSE comparison bar chart, convergence curve.
                    Scripts/Unsupervised/exp_archetypal_digits.m      ← NEW 2026-04-29: ESL §14.6.1 Archetypal Analysis. Part A: 2D synthetic AA vs K-Means (Fig 14.35). Part B: digit 3 archetypes r=2,3,4 (Fig 14.36).
                    Scripts/Unsupervised/exp_factor_analysis.m         ← NEW 2026-04-29: ESL Ex 14.15. Factor Analysis vs PCA on synthetic 3-variable model. Shows PCA dominated by high-variance X3; FA recovers correlated X1+X2 (latent Z1). 2-figure output.
                    Scripts/Unsupervised/exp_ica_digits.m               ← NEW 2026-04-30: ESL §14.7, Figs 14.39 & 14.40. FastICA vs PCA on USPS digit 3. 5x5 scatter matrix (ICA above / PCA below diagonal) + extreme digit images per IC.
                    Scripts/Unsupervised/exp_ica_simulation.m            ← NEW 2026-04-30: ESL §14.7.4, Fig 14.42. FastICA vs ProDenICA on 18 source distributions. 30 MC reps x 5 starts. Amari distance log-scale comparison.
                    Scripts/Unsupervised/exp_tree_clustering_demographics.m ← NEW 2026-05-04: ESL §14.3.12 / Ex 14.4. Classification tree-based clustering on Bay Area demographics. Null reference via column-wise permutation. CART tree identifies dense regions (P(Class=1) leaves). K-Means comparison. 3-panel heatmap + leaf scatter.
                    Scripts/Unsupervised/exp_som_sphere.m                ← NEW 2026-05-04: ESL Ex 14.5. SOM [1x3] vs K-Means (K=3) on 3-class spherical cluster data (sigma=0.6, N=30/class). Permutation-matched accuracy. 3-panel 3D scatter + SOM convergence curve.
                    Scripts/Unsupervised/exp_nci_som_kmeans.m            ← NEW 2026-05-04: ESL Ex 14.6. K-Means (K=2,5,10,20) and 2D-grid SOM on NCI60 microarray data (64 cell lines x 6830 genes, raw). Neighbourhood convergence experiment: FinalRadius sweep [2.0→0.005] shows SOM→K-Means as radius→0. 3-figure output: WCSS/MQE bars, cluster heatmap (K=5), convergence curve.
                    Scripts/Unsupervised/exp_procrustes_signatures.m     ← NEW 2026-05-05: ESL Ex 14.9. Generalized Procrustes Analysis (GPA) on L=3 synthetic "S" landmark configurations (N=100, p=2). Closed-form solution: eigen-decompose averaged projection matrices H_bar to recover template M (ESL Eq. 14.60). Per-shape orthogonal Procrustes (SVD) recovers R_l and beta_l. 4-panel figure: preshape overlay + M mapped to each raw shape. Idempotency verification.
                    Scripts/Unsupervised/exp_pagerank.m                  ← NEW 2026-05-06: ESL Ex 14.22. PageRank on 6-node network (Fig 14.47). Uses power iteration. Visualized with OceanViz-styled directed graph and score-proportional node sizes.
                    Scripts/Unsupervised/exp_nmf_digits.m                ← NEW 2026-05-07: ESL Ex 14.25. NMF vs PCA on '2's from digits database. Demonstrates parts-based (non-negative, sparse) representation of NMF against holistic representation of PCA.
                    Scripts/Unsupervised/exp_nmf_ocean.m                 ← NEW 2026-05-07: NMF vs PCA dictionary learning on single high-res ocean image via 32x32 patch extraction.
                    Scripts/Unsupervised/exp_nmf_ocean_texture.m         ← NEW 2026-05-07: True texture extraction using local mean subtraction (PCA) and edge maps (NMF).
                    Scripts/Prototype Methods and Nearest-Neighbors/exp_tangent_distance_manifold.m  ← NEW 2026-06-01:
                      ESL §13.8 Tangent Distance & Transformation Manifolds. (1) Loads digit 3 pairs from ZipGenerator, (2) Computes horizontal, vertical, and rotational gradients/tangents (Lie Algebra generators), (3) Sweeps rotation group (SO(2) Lie Group) to trace 1D curved manifolds, (4) Performs 3D SVD projection of the high-dimensional space, (5) Solves invariant least-squares Tangent Distance showing dramatic reduction over Euclidean, (6) Renders 3D manifold projection and divergence curves.
                    Scripts/Prototype Methods and Nearest-Neighbors/exp_covector_normal_transform.m  ← NEW 2026-06-01:
                      Co-vectors vs. Vectors (The Inverse Transpose). Demonstrates why co-vectors (2D homogeneous line equations and 3D surface mesh normals) must be transformed via the transposed inverse H^-T of the point transformation matrix. (1) Algebraic & physical verification in 2D using intersecting points and lines, (2) Verification in 3D using a sheared paraboloid mesh with vertex normal dot products, (3) Renders visual alignments under non-orthogonal scale and shear.
                    Scripts/Prototype Methods and Nearest-Neighbors/exp_euler_gimbal_lock.m  ← NEW 2026-06-02:
                      3D Euler Rotations and Gimbal Lock. (1) Performs algebraic verification of lost DOF at Pitch = 90 degrees, (2) Visualizes world coordinates and alignment of active gimbal axes in 3D under Lock vs. Normal states, (3) Compares linear interpolation of Euler angles vs. Quaternion SLERP through the lock zone, proving smooth, geodesic shortest-arc coordinate trace.
                     Scripts/Computer Vision/exp_light_simulation.m       ← NEW 2026-06-06: 3D room light simulation comparing shading and ray-tracing models.
                     Scripts/Computer Vision/exp_color_jpeg_spaces.m      ← NEW 2026-06-10: Replicates Figure 2.33 (color space transformations) and Figure 2.34 (JPEG quality settings) on custom ocean image.
                     Scripts/Computer Vision/exp_transform_editor_2d.m    ← NEW 2026-06-10: Launches interactive MATLAB 2D Transform Editor (Ex 2.2).
                     Scripts/Computer Vision/exp_viewer_3d.m              ← NEW 2026-06-10: Launches interactive MATLAB 3D Primitives Viewer (Ex 2.3).
                     Scripts/Computer Vision/exp_texture_viewer.m          ← NEW 2026-07-16: Launches interactive MATLAB 3D Texture-Mapped Viewer (Ex 3.27).
                     Scripts/Computer Vision/exp_histogram_equalization.m ← NEW 2026-06-11: Replicates Figure 3.7 (histogram analysis, CDFs, transfer functions, full & partial equalization) on custom ocean image.
                     Scripts/Computer Vision/exp_image_preprocessor.m     ← NEW 2026-06-11: Launches interactive MATLAB Image Preprocessing Sandbox GUI (Phases 1-5).
                     Scripts/Computer Vision/exp_separable_filters.m      ← NEW 2026-06-15: Replicates Szeliski Figure 3.14 (Separable Linear Filters) on custom ocean image.
                     Scripts/Computer Vision/exp_separable_filters.py     ← NEW 2026-06-15: Replicates Szeliski Figure 3.14 (Separable Linear Filters) in Python/PyTorch.
                     Scripts/Computer Vision/exp_image_pyramids.m         ← NEW 2026-06-19: Constructs 5-level Gaussian/Laplacian pyramids on elon_musk.jpg and verifies perfect reconstruction.
                     Scripts/Computer Vision/exp_laplacian_pyramid_blending.m ← NEW 2026-06-24: Replicates Figure 3.41 Laplacian Pyramid Blending (Orapple) using multiresolution masks.
                     Scripts/Computer Vision/Python/exp_pytorch_pyramid_blending.py ← NEW 2026-07-13: Solves Exercise 3.19. PyTorch vs SciPy Laplacian pyramid blending comparison and verification.
                     Scripts/Computer Vision/exp_local_laplacian.m        ← NEW 2026-07-15: Solves Exercise 3.20. Local Laplacian contrast manipulation, edge-preserving filtering, and HDR tone mapping on custom images.
                     Scripts/Computer Vision/exp_forward_vs_inverse_warp.m ← NEW 2026-06-24: Demonstrates holes and overlaps in Forward (Scatter) mapping vs clean interpolation in Inverse (Gather) mapping.
                     Scripts/Computer Vision/exp_domain_scaling_direction.m ← NEW 2026-06-25: Verifies the textbook Domain Scaling Law (3.76) under non-uniform scale transform (stretching X and squashing Y), demonstrating directional frequency compression vs expansion and aliasing.
                     Scripts/Computer Vision/exp_non_separable_prefilter.m ← NEW 2026-06-25: Verifies the non-separability of 2D rotated anisotropic Gaussian prefilters using SVD analysis, full 2D vs. separable approximation convolution, 2D FFT spectral comparison, and computational benchmarks.
                     Scripts/Computer Vision/exp_mesh_warping.m            ← NEW 2026-06-30: Demonstrates and compares Piecewise Affine Warping vs. Thin Plate Spline (TPS) Warping on a checkered grid.
                     Scripts/Computer Vision/exp_local_warping.m           ← UPDATED 2026-07-15: Solves Exercise 3.23 (all parts). Benchmarks Piecewise Affine, Thin Plate Spline, Cubic B-Spline FFD, Beier-Neely Line Morphing, and Paintbrush Swirl flow-field warping on a synthetic grid.
                     Scripts/Computer Vision/exp_bilateral_effect.py      ← NEW 2026-06-17: Evaluates Gaussian Blur vs. Bilateral Filter on synthetic step-edge in Python/OpenCV.
                     Scripts/Computer Vision/exp_image_denoising.m        ← NEW 2026-07-16: Benchmarks Box, Median, Gaussian, Bilateral, and NLM filters under parameter sweeps on AWGN and S&P noise.
                     Scripts/Computer Vision/exp_rainbow_enhancer.m       ← NEW 2026-07-16: Replicates Exercise 3.29 rainbow separation, polar unrolling, and radial profile polynomial subtraction.
                     Scripts/Time Series Analysis/exp_ar_fitting_forecast.m ← NEW 2026-06-09: AR(2) simulation, Yule-Walker parameter recovery, order selection, and multi-step forecasting with 95% confidence bands.
                     Scripts/Time Series Analysis/exp_time_series_demo.m   ← NEW 2026-06-09: ARMA simulations, empirical vs theoretical ACF/PACF comparisons, and MA(1) duality analysis.
                     Scripts/Time Series Analysis/exp_hidden_cycles_spectral.m ← NEW 2026-06-16: Spectral vs Time-Domain analysis comparing noisy autocovariances to power spectrum peak concentration on built-in sunspot data.
                     Scripts/Simulation/exp_lorenz_butterfly.m               ← NEW 2026-06-25: Edward Lorenz Chaotic Butterfly Attractor (3D). Phase 1: RK4 integration from scratch (dt=0.005, T=50s). Phase 2: 3D phase portrait color-coded by time (parula map). Phase 3: Butterfly effect — two trajectories separated by epsilon=1e-8 diverge exponentially; estimates Lyapunov exponent via linear regression on log-divergence.
Data/             ← Raw datasets

Results/          ← Output figures and tables
Python/            ← Python tools, scripts, and interactive CV applications
                     transform_editor_2d.py  ← NEW 2026-06-10: Interactive 2D Transform Editor (Ex 2.2) with homography/affine solvers and JSON persistence.
                     viewer_3d.py            ← NEW 2026-06-10: Interactive 3D Primitives Viewer (Ex 2.3) with camera/object transforms and Painter's algorithm.
```

---

## 3. Dependency Direction (NEVER violate this)

### Python racetrack experiment support (2026-09-14)

Organization update (2026-09-15): reusable Python code now lives in
`Experiments/Scripts/RL/Python/ocean_rl/`. The package contains `contracts.py`
(agent/environment interfaces), `registry.py` (algorithm registration and
creation), `algorithms/monte_carlo/off_policy.py` (updates/checkpoints),
`environments/racetrack.py` (dynamics), `maps/io.py` and `maps/tiled.py`
(imports), `evaluation/racetrack.py` (policy-callable evaluation),
`visualization/racetrack.py` (optional plots), `experiments/racetrack.py`
(CLI/reporting), and `tools/import_dot_map.py` (diagram preprocessing).
The existing top-level Python files are compatibility facades. User maps,
configs, results and tests retain stable locations outside the implementation
package. `python -m ocean_rl --algorithm off-policy-mc` is the package launcher;
`--agent-options` loads algorithm parameters. Only MC is currently implemented.
See `Experiments/Scripts/RL/README.md` for the MATLAB/Python workspace index and
`Experiments/Scripts/RL/Python/README.md` for extension contracts and verification.

`tiled_maps.py` adds semantic Tiled JSON (.tmj/.json) import for finite
orthogonal maps with a top-level `Racetrack` tile layer. Tiles declare integer
`cell_type` properties; inline and external JSON tilesets are supported.
`maps/tiled/` contains editable Harbor Bend and uploaded-image starter maps,
a shared .tsj tileset and artwork. `tests/test_tiled.py` covers import contracts;
`docs/recipes/tiled_racetrack.md` documents the paint-save-train workflow.

`import_dot_map.py` (2026-09-15) adds calibrated black-dot diagram extraction:
regular border lattice estimation, cell occupancy classification, explicit pixel
endpoints, exact-palette PNG/config export and image-import round-trip validation.
It is specialized to regular diagrams, not arbitrary scene understanding.

`Experiments/Scripts/RL/Python/` separates reusable Python map, environment and
learning modules from the CLI experiment. `racetrack_maps.py` provides immutable
`MapSpec`, strict ASCII/NumPy/labeled-image/JSON loading and directional
reachability validation. `racetrack_env.py` provides textbook and bidirectional
inertial dynamics, exact ordered supercover collision, seeded reset/clone and
explicit truncation. `offpolicy_mc_racetrack.py` provides sparse tabular MC with
recorded behavior probabilities, log cumulative weights and isolated evaluation.
`exp_multi_map_racetrack.py` loads `maps/` assets or user paths, runs map/seed
experiments and writes actual metrics, grids and sparse policy tables. Plotting
is optional and isolated from training. Each map needs a separate learned policy.
Regression tests live in `Experiments/Scripts/RL/Python/tests/`; usage is in
`docs/guides/racetrack_maps.md` and `docs/recipes/racetrack_maps.md`.

```
OceanEngine  →  OceanStats (uses trained models), OceanMath (pure math)  ← sits ABOVE model layer
OceanViz     →  OceanMath.Simulation  (event-based only, no class coupling)
OceanStats   →  OceanMath.Approximation, OceanMath.LinearAlgebra
OceanStats   →  OceanAlgo.Core, OceanAlgo.StateSearch (RL.DecisionTime only)
OceanMath    →  (nothing — fully self-contained)
OceanDL      →  OceanStats.Optim, OceanStats.Loss, OceanMath   (never the reverse)
OceanNLP     →  OceanDL
OceanViz     →  OceanDL  (NN apps and strategies)
OceanAlgo, OceanCS → (standalone)
OceanAgents  →  OceanStats.RL (Env, TD values/targets/step sizes), OceanStats.GameTheory.Core   (never the reverse)
```

**Rule**: Lower layers NEVER import upper layers.

---

## 4. OceanMath — Pure Math Engine

```
+Approximation/       IApproximationMethod, ScoreTestApproximation, SecondOrderApproximation
+Calculus/            analyzeLineIntegral, analyzeSurface
+Information/
  +Distributions/     IDistribution (abstract), DiscreteDistribution, ContinuousDistribution,
                      MultivariateNormalDistribution  <- NEW 2026-05-14
  +Processes/         IStochasticProcess (abstract), MarkovChain, RandomWalkGraph
  +Strategies/        IMetricStrategy (abstract), ShannonEntropyStrategy, MarkovEntropyRateStrategy
                      ExpectedCodeLengthStrategy ← NEW 2026-05-16: L_avg = sum(p_k * l_k)
                      RedundancyStrategy         ← NEW 2026-05-16: R = L_avg - H(X)
                      CodingEfficiencyStrategy   ← NEW 2026-05-16: eta = H(X)/L_avg
                      KraftInequalityStrategy    ← NEW 2026-05-16: K = sum(Base^(-l_k))
                      CodingGapStrategy          ← NEW 2026-05-16: L_avg - H(X) in bits
  +Coding/            ← NEW 2026-05-16: Source Coding Package
    CodeTable.m         Immutable value class: symbols, probs, codewords, lengths.
                        Methods: expectedLength, redundancy, efficiency, kraftSum,
                                 isComplete, codingGap, getOptimalLengths, printSummary, toStruct.
    HuffmanTree.m       Handle class backed by DigraphCus. Greedy bottom-up merge.
                        Methods: build(symbols,probs), getCodeTable, toDigraph, getRootId,
                                 depth, numLeaves, numInternal.
    ISourceCoder.m      Abstract interface: encode(symbols,probs)→CodeTable, getName, isOptimal.
                        Protected helper: validateInputs_.
    HuffmanCoder.m      Huffman (1952) optimal prefix-free coder. Implements ISourceCoder.
                        Options: Base (k-ary), k-ary padding auto-applied.
                        Methods: encode, buildTree, getName, isOptimal.
    huffmanEncode.m     One-liner convenience: huffmanEncode(symbols, probs) → CodeTable.
    optimalCodeLength.m Theoretical -log_Base(p_k) for each symbol.
    compareCoding.m     Full comparison struct vs Shannon bounds. Verbose option.
  Correlation.m, Entropy.m, MutualInfo.m, InformationEngine.m (orchestrator)
  InformationEngine     UPDATED 2026-05-16: +5 coding aliases (ExpectedLength, Redundancy,
                        CodingEfficiency, KraftSum, CodingGap); encodeSummary() method.
  DiscreteDistribution  UPDATED 2026-05-16: +huffmanEncode() bridge method.
+LinearAlgebra/
  +Optimization/       generateSpatialPenalty, generateRoughnessPenalty1D,
                       generateRoughnessPenalty2D  ← NEW 2026-03-16 (2D Kronecker-sum image penalty)
  +Orthogonalization/  IOrthogonalizer, HouseholderReflection, ModifiedGramSchmidt
  +Solvers/            ILinearSolver, JacobiSolver, GaussSeidelSolver
  IncrementalQR.m, IncrementalWLS.m
+Random/               <- NEW 2026-05-14 (Generic Probability Generation)
  +Samplers/           ISampler (abstract strategy for random variate generation)
                       InverseTransformSampler.m  <- Exact for PMF, fzero fallback for PDF
                       RejectionSampler.m         <- Auto-bounds finite support with Uniform proposal
                       CholeskySampler.m          <- O(p^3) generic multivariate normal sampler
                       LatentVariableSampler.m    <- O(N*p) fast block-correlated normal sampler
+Quat/                 9 pure quaternion functions (create_from_euler, quat_mul, squad_point, ...)
+Geometry/             ← NEW 2026-03-19 (Phase 1: Parametric Curves)
  IParametricCurve.m     Abstract interface: evaluate(t), tangent(t), secondDerivative(t),
                           arcLength(t0,t1), totalLength(), parameterize(N), parameterizeByArcLength(N)
  CatmullRomSpline.m     Centripetal Catmull-Rom (alpha=0.5). Barry-Goldman recursive evaluation.
                           Analytical 1st/2nd derivatives. Mutation: setControlPoint, addControlPoint,
                           removeControlPoint. Phantom endpoints for boundary segments.
  FrenetFrame.m          Frenet-Serret TNB frame on any IParametricCurve. curvature(t), quaternion(t).
                           Manual cross product, Shepperd rotation-matrix-to-quaternion conversion.
  BernsteinBasis.m       <- NEW (Phase A) Static: evaluate, derivative, secondDerivative, matrix.
                           Bernstein polynomial B_{i,n}(t) = C(n,i)*t^i*(1-t)^{n-i}
  BezierCurve.m          <- NEW (Phase A) Implements IParametricCurve via de Casteljau.
                           Single segment, degree = K-1, endpoint interpolation.
  KnotVector.m           <- NEW (Phase B) Static: uniform, clamped, open, fromInteriorKnots,
                           validate, findSpans. Knot sequence generation for B-splines.
  BSplineBasis.m         <- NEW (Phase B) Static: iterative Cox-de Boor evaluate, derivative,
                           basisMatrix. O(n*p) per point. Replaces recursive bspline.m.
  BSplineCurve.m         <- NEW (Phase C) Implements IParametricCurve via BSplineBasis + KnotVector.
                           Arbitrary degree, local control, clamped endpoints, add/remove points.
  NURBSCurve.m           <- NEW (Phase D) Rational B-spline (+ weights). Exact conics.
                           C(t) = sum(w_i*N_i*P_i) / sum(w_i*N_i). setWeight for sculpting.
  QuadTree.m             <- NEW 2026-04-01 2D spatial index. Bulk-build recursive quadrant
                           subdivision (MAX_LEAF_SIZE=8). findNearest(pt,r) O(log N),
                           rangeQuery(center,r), queryBox(AABB). Used by TrackDesigner for
                           fast station lookup and curve collision pre-filtering.
+Simulation/           ISimulator (abstract, Observer), MarkovSimulator, SimulationHistory,
                       StateChangedEventArgs
                       ← NEW 2026-03-20 (Phase 2: Kinematic Physics)
                       KinematicState (immutable value: Position, Velocity, Acceleration,
                           ArcDistance, Parameter, Tangent, Orientation quaternion)
                       CurveSimulator (SUVAT physics on IParametricCurve, 3 accel modes:
                           constant / uniform / profile(t), fires CurveStateChangedEventArgs)
                       CurveStateChangedEventArgs (event payload: old + new KinematicState)
+Graph/                ← NEW 2026-03-20 (Phase 3: Generic Graph Architecture)
  Node.m                 Immutable value: Id, Name, Position, Metadata (struct)
  Edge.m                 Immutable value: Source, Target, Weight, Label
  DigraphCus.m           GENERIC digraph-style: Node/Edge cell arrays + (V x V) weight matrix.
                           addNode, addEdge, removeNode, removeEdge, neighbors, hasEdge,
                           getAdjacencyMatrix, getNodePositions, toStruct
+Algorithms/            <- NEW 2026-03-20 (Algorithms separated from data structures)
  +Graph/                Graph algorithms (operate on DigraphCus or raw weight matrices)
    Dijkstra.m             Static solver: [path, dist] = solve(W, src, tgt). Pure math O(V^2).
    NetworkAnalytics.m     Static utilities: totalLength, averageEdgeLength, degreeDistribution,
                             density, isConnected (BFS), connectedComponents, numComponents
+Transit/               <- NEW 2026-03-20 (Domain wrapper for transit systems)
  TransitNetwork.m       Wraps DigraphCus + manages splines. addStation, addRoute(spline),
                           getSpline, updateRouteSpline, totalNetworkLength, toStruct
  TransitAgent.m         <- NEW 2026-03-21: Multi-segment train agent. Chains CurveSimulator
                           across splines with dwell times. step(), getPosition(), getProgress(), isArrived().
  TransitSimulation.m    <- NEW 2026-03-21: Multi-train orchestrator. Ticks agents in lockstep.
                           addAgent(lineCode), step(), getPositions(), isAllArrived().
  KinematicAgent.m       <- NEW 2026-03-27: 6-DOF 3D Newtonian physics standalone agent. Injects exact
                           analytical Z-axis gravity acceleration profile directly into CurveSimulator.
+Signal/               <- NEW 2026-05-24: Signal Processing Package
  IdealLowpass.m       Ideal lowpass filter modeling class. Computes analytic impulse response (h_lp[n]),
                       frequency response approximation (H_M(e^(j*w))), numerical/empirical Gibbs overshoot
                       location and percentage amplitude, and validates Parseval's relation.
+Vision/               <- NEW 2026-06-01: Computer Vision Package
  Transform2D.m        Generates 2D homogeneous coordinate transformation matrices (translation, rotation, scale, shear, rigid, similarity, stretchSquash, affine, homography)
  Transform3D.m        Generates 3D homogeneous coordinate transformation matrices (translation, rotation, scale, shear, rigid, similarity, affine, projective)
  Rotation3D.m         Converts between Roll/Pitch/Yaw, quaternions, axis-angle, exponential maps, and SLERP interpolation
  +Rotations/          <- NEW 2026-06-02: Object-Oriented 3D Rotation Representations (Strategy Pattern)
    RotationRepresentation3D.m  Abstract interface contract for all 3D rotation strategies
    EulerRotation3D.m            ZYX Euler angle 3D rotation strategy (wraps Rotation3D.eulerToMatrix)
    QuaternionRotation3D.m       Unit quaternion 3D rotation strategy (wraps Quat.*)
    AxisAngleRotation3D.m        Axis-angle 3D rotation strategy (wraps Rotation3D.axisAngleToMatrix)
    ExpMapRotation3D.m           Exponential map 3D rotation strategy (wraps Rodrigues' formula/expMapToMatrix)
    MatrixRotation3D.m           3x3 orthonormal SO(3) matrix strategy
  CameraProjection.m   Models camera extrinsics/intrinsics and projects 3D space to Perspective, Orthographic, Weak-Perspective viewports
  LensDistortion.m     Models Brown-Conrady radial (k1, k2, k3) and tangential (p1, p2) decentering lens distortion aberrations
  +Geometry/           <- NEW 2026-06-06: 3D Geometries with ray intersection capabilities
    IGeometry.m        Abstract interface for 3D geometries (intersect, toMesh, getNormal)
    PlaneGeometry.m    Infinite or finite bounded flat plane geometry
    SphereGeometry.m   3D sphere geometry with ray intersection quadratic solver
    BoxGeometry.m      Oriented box: slab-method ray casting (vectorised intersectRays), inside test, uniform surface sampling  <- NEW 2026-09-23
    BoxAssembly.m      Union of boxes (CSG): exact inside, nearest-hit rays, union-boundary sampling (normal-offset rule), EXACT volume by coordinate compression
  +Scene/              <- NEW 2026-06-06: 3D Scene Representation
    Material.m         Immutable value class representing optical surface properties
    SceneObject.m      Instantiated 3D geometry and material under a 4x4 coordinate transform pose
    Scene.m            Aggregates objects and lights, and resolves closest-intersection ray queries
  +Lighting/           <- NEW 2026-06-06: 3D Light Sources
    ILight.m           Abstract interface for all light sources
    AmbientLight.m     Constant omnidirectional ambient light source
    PointLight.m       Positional light source radiating with quadratic distance attenuation
    DirectionalLight.m Distant light source emitting parallel rays (constant direction)
    SpotLight.m        Positional light source constrained to a cone angle with falloff
  +Shading/            <- NEW 2026-06-06: 3D Shading Reflection Models
    ILightingStrategy.m Abstract strategy interface for custom illumination equations
    LambertianStrategy.m Diffuse-only reflection model with shadow mapping
    PhongStrategy.m    Classical Phong specular reflection model
    BlinnPhongStrategy.m Blinn-Phong specular model using the half-angle vector
  +Renderer/           <- NEW 2026-06-06: 3D Room Light Renderers
    IRenderer.m        Abstract base interface for 3D room light renderers
    VectorMeshRenderer.m Mesh-based renderer utilizing MATLAB patch graphics (2D/3D, Flat/Gouraud)
    RayTraceRenderer.m Pixel-by-pixel recursive ray tracing renderer with reflections and shadows
+Geometry/+Mesh/      <- NEW 2026-09-23: Triangle-mesh toolkit for mesh-prediction pipelines (Mesh R-CNN / Pixel2Mesh)
  TriMesh.m          (V x 3, F x 3) mesh: edges, sparse adjacency, normals, areas, volume, Euler/genus, watertight/orientation checks
  Cubify.m           Voxel grid (D x H x W, MaxPool3D layout) -> watertight mesh (Mesh R-CNN Sec. 3.1); functional core compute()
  SurfaceSampler.m   Area-weighted uniform surface sampling with backward() (dL/dV through barycentric weights)
  Voxelizer.m        Solid -> occupancy: implicit inside-fn or closed mesh (generic-direction ray parity), supersampled MinFraction rule
  IcoSphere.m        Geodesic sphere (V = 10*4^L + 2): the genus-0 template of sphere-deforming methods
  MeshBatch.m        Packed batch of meshes (PyTorch3D-style): VertMeshIdx, offsets, block-diagonal edges
+Slerp/               interpolate, raw_interpolate
+TimeSeries/          <- NEW 2026-06-09: Time Series Analysis Package
  +Models/           (Univariate and multivariate model definitions)
    ITimeSeriesModel.m Abstract interface: simulate, theoreticalACF, theoreticalPACF, isStationary, isInvertible, getEquation
    ARMA.m             Concrete ARMA(p, q) process model with roots-based stationarity/invertibility checks and MA(inf) ACF solver
    AR.m               Concrete AR(p) process model supporting simulation, Yule-Walker fitting, lag order selection (AIC/BIC), and multi-step forecasting
  +Diagnostics/      (Statistical estimation and diagnostic testing)
    autocorr.m         Computes empirical autocorrelation function (ACF) up to lag K
    parcorr.m          Computes empirical partial autocorrelation function (PACF) using Durbin-Levinson recursion
SimpsonEstimate.m  (bspline.m DELETED — replaced by +Geometry/BSplineBasis.m)
```

**Information flow**:  
`IDistribution/IStochasticProcess` → `IMetricStrategy.calculate(context)` → `InformationEngine`

---

## 5. OceanStats — Statistical Learning Engine (largest package)

### 5.1 +Base — All Interfaces Live Here
| Interface                        | Contract                                                                                 |
| -------------------------------- | ---------------------------------------------------------------------------------------- |
| `IModelStrategy`                 | `train(X,Y,W)` / `predict(model,X)` / `evaluate(Ytrue,Ypred)` / `getComplexity(model)`   |
| `IUnivariateSmoother`            | `fit(X,Y,W)` / `predict(model,Xnew)`                                                     |
| `IObjectiveApproximation`        | Template Method: `computeGain()` → `extractComponents()` (abstract) → `Method.compute()` |
| `ICrossValidator`                | CV split contract                                                                        |
| `ILinkFunction`                  | GLM link function contract                                                               |
| `IPruningCriterion`              | Pruning cost contract                                                                    |
| `IForwardPass` / `IBackwardPass` | Basis expansion & pruning pass contracts                                                 |
| `IBasisFunction`                 | Basis function evaluation                                                                |
| `IFeatureSelector`               | Feature selection contract                                                               |
| `IDataEncoder`                   | Data encoding contract                                                                   |
| `IValidator`                     | General validation contract                                                              |
| `ProbabilisticModel`             | Abstract base with likelihood interface                                                  |

### 5.2 +Data
`DataFactory` (named dataset loader), `BootstrapSampler`, `IndicatorEncoder`, `RawEncoder`, `polyFeatures`  
`+Generators/` `WaveformGenerator` (ESL Eq 12.64), `ESLMixtureGenerator` (K-class mixture, NumClasses param, Bayes oracle), `CorrelatedGenerator` (ESL 16.2.3 block-correlated regression data)

### 5.2a +Regularization & Linear Modeling ← NEW (2026-05-14)
Unified architecture for penalized regression paths (Lasso, Ridge, Elastic Net, Stagewise).
`RegularizedLinearModel.m` — Context/Facade class replacing standalone specific models.
`+Solvers/`
  `IRegularizationSolver.m` — Abstract strategy for coefficient path generation.
  `CoordinateDescentSolver.m` — Fast path generation via soft-thresholding warm starts (wraps MATLAB lasso).
  `ForwardStagewiseSolver.m` — Infinitesimal forward stagewise path (ESL Algorithm 16.1).
### 5.3 +Ensemble
`Bagging`, `BMA`, `WeightedEnsemble`, `ScoreWeightedEnsemble`, `StackedModel`, `RandomForest`, `ExtraTreesForest`, `ProximityMatrix`, `ImportanceCalculator`, `OOBTracker`, `ISLEForest` (Unified ISLE Orchestrator)

`+HME/`: Full Hierarchical Mixture of Experts tree — `BaseNode`, `InternalNode`, `LeafNode`, `GenericExpert`, `CompetitiveExpert`, `EMTrainer`, `HierarchicalMixture`, `+Initialization/KMeansInitializer`

**Random Forest** ← NEW 2026-05-12 (ESL §15, Breiman 2001)  
`RandomForest` — Orchestrator. B bootstrap trees with `RandomFeatureCandidateGenerator`. Parallel `parfor` when toolbox present AND `NumTrees >= MinTreesForParallel`. fit/predict/predictProba/oobError/featureImportance/printSummary.  
`OOBTracker` — Accumulates OOB soft votes (classification) or sums (regression) per sample. oobError/oobPredictions.  
`ImportanceCalculator` — Static MDA (permutation, OOB-based) and MDI (weighted impurity reduction) importance scores.

**Extra-Trees** ← NEW 2026-05-12 (Geurts, Ernst & Wehenkel 2006)  
`ExtraTreesForest` — Extremely Randomized Trees. Uses `ExtraTreesCandidateGenerator` (random thresholds). UseBootstrap=false by default (full dataset per tree). ImportanceMethod defaults to 'MDI'. Same parfor threshold logic as RandomForest. fit/predict/predictProba/oobError/featureImportance/printSummary.

**Proximity Matrix** ← NEW 2026-05-12 (Breiman 2001, §2.3)  
`ProximityMatrix` — Static utility. compute(forest, X) → N×N symmetric co-leaf proximity matrix. outlierScores(P) → low-proximity outlier detection. mdsCoords(P) → 2D/3D MDS embedding. visualize(P, ax, Y) → sorted heatmap.

**Experiments (RF/ExtraTrees/ISLE):**  
`Experiments/Scripts/Supervised/exp_random_forest_esl.m`   — ESL §15 error convergence & importance on synthetic mixture.  
`Experiments/Scripts/Supervised/exp_extra_trees_comparison.m` — RF vs ExtraTrees: error convergence, importance, proximity MDS.  
`Experiments/Scripts/Supervised/exp_rf_spam_mtry.m`        — ESL Ex 15.6: RF mtry sweep on Spambase, OOB vs test error dual-curve.  
`Experiments/Scripts/Supervised/exp_isle_gbm_knobs.m`      — NEW 2026-05-14: ISLE Grand Unification (Friedman & Hall 2007). 3-part: (A) (eta x nu) heatmap showing ensemble continuum from Bagging→GBM→ISLE, (B) error convergence curves for 4 representative configs, (C) training loss descent comparison.

### 5.14 +Boosting  ← NEW (2026-03-02)
Sequential ensemble methods. All algorithms share `IBoostingStrategy` interface; `BoostingEnsemble` (the Context/Facade) exposes the result as a standard `IModelStrategy`.  
`+Base/IBoostingStrategy` — abstract contract: `boostingStep()`, `aggregate()`  
`+Algorithms/AdaBoostM1` — Freund & Schapire (1997), discrete AdaBoost  
`+Algorithms/AdaBoostM2` — SAMME algorithm (Zhu et al. 2005), multi-class extension  
`+Algorithms/LogitBoost` — Friedman et al. (2000), additive logistic regression  
`BoostingEnsemble` — Context: drives the loop, implements `IModelStrategy`  

**GradientBoostingMachine** ← UPDATED 2026-05-13 (Phase 1 ISLE)  
Friedman (2001) Gradient Boosted Trees + Stochastic GBM extension (Friedman 2002, Friedman & Hall 2007).  
- `LearningRate` (nu) — shrinkage / memory knob. Default: 0.1  
- `SubsampleFraction` (eta) ← **NEW** — row subsampling WITHOUT replacement per tree. Default: 1.0 (standard GBM, backward-compatible). Set eta<=0.5 for ISLE optimal recipe.  
- `OutputMode` — `'auto'|'regression'|'binary'|'multiclass'` (MART, K trees/round)  
- Internal `subsampleRows_()` helper draws `floor(eta*N)` unique row indices via `randperm`. Full-N F tally updated after each tree.  
- ISLE optimal recipe: `LearningRate=0.1, SubsampleFraction=0.5`  

Planned: `DARTBoosting`


### 5.15 +Loss
Swappable, generic loss function strategies. Any gradient-based algorithm (Gradient Boosting, NN Trainer, etc.) injects an `ILoss` to decouple optimization math from algorithm structure.  
`ILoss` — abstract interface: `forward()`, `gradient()`, `hessian()`, `name()`  
`+Regression/MSELoss`       — Mean Squared Error  
`+Regression/MAELoss`       — Mean Absolute Error, L1  
`+Regression/HuberLoss`     — Pseudo-Huber, smooth L1/L2 blend  
`+Classification/CrossEntropyLoss`  — Binary log-loss. Y_pred = raw logit  
`+Classification/ExponentialLoss`   — AdaBoost loss. Y_true ∈ {-1,+1}  
`+Classification/HingeLoss`         — SVM hinge. Y_true ∈ {-1,+1}  
`+Classification/BinaryBCELoss`     — Binary cross-entropy for NN binary heads. Logit input.  ← NEW NN-2  
`+Classification/MulticlassDevianceLoss` — Softmax cross-entropy for MART. N×K logit scores.  
`+Classification/MarginSquaredLoss` — Margin-based squared error. Y_true ∈ {-1,+1}  
`+Classification/HuberisedSquareHingeLoss` — Huberised square hinge loss. Y_true ∈ {-1,+1}  

### 5.17 +Preprocessing
```
IPreprocessor.m        Abstract: encode(X_raw)→X_enc, outputDim(), name()   ← NEW NN-5
FourierFeatureMap.m    Implements IPreprocessor. gamma(X)=[cos(2πBX);sin(2πBX)].  B frozen.
                       encode(X)          — fit + transform (training data)
                       encodeNew(X)       — apply frozen B (val/test data)
                       outputDim()        — returns 2*M
                       recommendSigma(X)  — Nyquist-safe sigma (static)
                       sweepSigma(X,Y)    — empirical U-curve tuning (static)
PolynomialMapper.m     Implements IPreprocessor. X → [X, X^2, ..., X^p].  ← NEW NN-5
                       Stacks d_in*Degree features column-major.
```

> **Why in +OceanStats not +OceanPipeline?**  
> `IPreprocessor` is a *statistical learn* concept (it transforms data for an ML model).  
> `+OceanPipeline` is for orchestration/ingestion. Dependency direction: `OceanPipeline` → `OceanStats`, not vice versa.


### 5.4 +GAM
`AdditiveModel` (facade) → `Backfitting` (iterates `IUnivariateSmoother` per predictor) → `LocalScoring` (GLM extension)

### 5.5 +Inference

#### +Recipes/
Specialized inference recipe implementations (existing).

#### +HypothesisTest/  ← NEW (2026-03-10)
Statistical hypothesis tests. Strategy pattern: every test implements `IStatisticalTest`.

```
+HypothesisTest/
  +Base/
    IStatisticalTest.m   ← Abstract: test(data)→TestResult, name(), assumptions()
                            describe() default: prints name + assumption list
    TestResult.m         ← Immutable value class: Statistic, PValue, Df, Alpha,
                            Reject (auto-derived from PValue<Alpha), CI, TestName, Notes
                            summary() prints a formatted result block
  +Paired/
    McNemarTest.m        ← DONE. Paired binary classifier comparison.
                            chi-sq statistic = (n01-n10)^2 / (n01+n10)
                            Auto-switches to exact binomial when n01+n10 < 25.
                            Edwards' continuity correction (default on).
                            Input: {Yhat1, Yhat2, Ytrue} OR {n01, n10}
                            Tests: Tests/test_mcnemar.m (10/10 PASS)
  +ModelComparison/      ← PLANNED
    DietterichTest.m     ← corrected McNemar for k-fold CV results
    FriedmanTest.m       ← multi-algorithm, multi-dataset ranking test
```

**Usage:**
```matlab
t      = OceanStats.Inference.HypothesisTest.Paired.McNemarTest();
result = t.test(struct('Yhat1', Yhat1, 'Yhat2', Yhat2, 'Ytrue', Ytrue));
result.summary();     % prints formatted block
result.Reject         % logical
result.PValue         % scalar
```

**Known MATLAB gotcha:** Package namespaces cannot be aliased inside class methods.
Always use the full path: `OceanStats.Inference.HypothesisTest.Base.TestResult(...)`.

### 5.6 +Link
`IdentityLink`, `SigmoidLink`, `SoftmaxLink` — all implement `ILinkFunction`

### 5.7 +Metrics
`ROC`, `Calibration`

### 5.8 +NonParametric
`+Strategies/`: `ISplineStrategy` (extends `IUnivariateSmoother`), `NaturalCubicStrategy`, `PSplineStrategy`, `RegressionSplineStrategy`  
Utilities: `calcLoocv`, `calculateKDE`, `generateBSplineBasis`, `getNSBasis`, `getSmootherMatrix`  
Legacy: `SmoothingSpline` (being retired)

### 5.9 +Optimization
`IOptimizationRecipe` (abstract), `IMMRecipe`, `GLMRecipe`, `GenericOptimizer` (driver loop)  
Pattern: `GenericOptimizer` + any `IOptimizationRecipe` → decouples "when to stop" from "how to step"

`IPrimalSolver` (abstract) → solvers for unconstrained models (finds $w$) [Planned: SGD, Proximal Gradient]
`CompositeObjective` Wrapper combining `ILoss` and `IRegularizer` for primal solvers.

> **DO NOT CONFUSE with `+Optim` (§ 5.10c).** These are two different abstraction levels:
>
> | Package | Abstraction | Consumer | Key method |
> |---|---|---|---|
> | `+Optimization` | Full iterative algorithm with loop, convergence check, history | EM, GLM fitting, GenericOptimizer, IPrimalSolver | `recipe.step(data)` — takes full data, moves whole model |
> | `+Optim` | Pure gradient step function on a single parameter tensor | BackpropEngine, Trainer | `opt.step(param, grad)` — takes one array, returns one array |

### 5.9b +Regularization  ← NEW (2026-03-13)
Abstract penalties $r(w)$ independent of loss functions. Used by `CompositeObjective` to create $f(w) + \lambda r(w)$.
`IRegularizer` — abstract interface: `evaluate()`, `gradient()`, `proximity()`
`L2Regularizer` — Ridge penalty.
`L1Regularizer` — Lasso penalty (soft thresholding).

### 5.10 +Pipeline
`Pipeline` (sequential fit-transform chain), `PCA`, `PolynomialExpander`, `processTies`

### 5.10b +Preprocessing
Input feature transformers applied before NN training.
```
FourierFeatureMap   gamma(X) = [cos(2pi*B*X); sin(2pi*B*X)],  B ~ N(0, sigma^2*I) frozen
                    Overcomes MLP spectral bias (NTK theory).
                    Static: recommendSigma(X) — analytical Nyquist bound.
                    Static: sweepSigma(X, Y) — empirical U-Curve tuning.
                    Ref: Tancik et al., NeurIPS 2020 (arXiv:2006.10739)
```

### 5.10c +Optim
Per-parameter gradient-step optimizers for NN training loops.  
Every class implements `IOptimizer`: `step(param, grad)→param`, `reset()`, `describe()`.

> **Lives in `+OceanStats` (not `+OceanMath`)** because these classes depend on the concept of
> "model parameters and gradients" — a statistical learning idea. `+OceanMath` is pure mathematics
> with no knowledge of training.

```
IOptimizer            Abstract interface: step(param, grad), reset(), describe()
AdamOptimizer         Per-parameter Adam (Kingma & Ba, 2015).  ← existed pre-NN2
                      Lazy-inits on first step(). Static: forParams(N,'LR',lr).
SGDOptimizer          W = W - lr*dW. Stateless.
SGDMomentumOptimizer  EMA velocity: v = beta*v + (1-beta)*g; W = W - lr*v
RMSPropOptimizer      s = rho*s + (1-rho)*g^2; W = W - lr*g/sqrt(s+eps)
```

### 5.11 +Selection — Model Selection (most complex sub-package)
```
+Criteria/          aic, bic, mallows, GCVCriterion
+CrossValidation/   CrossValidator (orchestrator), KFold, HoldOut, LeaveOneOut, BootstrapSplitter
+FeatureSelection/  BestSubsetSelector, ForwardStepwiseSelector, ForwardBasisSelector,
                    GCVBackwardSelector, PolyMARSForwardSelector
+HyperParameter/    ComplexitySweeper
+Strategies/        ModelFactory + thin IModelStrategy wrappers:
                    LinearReg, LassoReg, RidgeReg, RidgeReg, PLSReg, PolyReg, NNLS,
                    DecisionTree, RegressionTree, SplineSmoother, LocalLinearSmoother
+Validation/        Bootstrap632, NestedCV
Auditors:           BiasVarianceAuditor, ComplexityAudit, ComplexityAudit_v2,
                    FeatureImportanceAuditor, FeatureSelectionAudit
```

### 5.12 +Supervised
```
+Discriminant/   IDiscriminantModel (abstract contract), LocalLDA, QDA, LDA, FDA, MDA
+MARS/           MARS + +Basis/(ConstantBasis, HingeBasis, ProductBasis, CategoricalBasis) + +Strategy/
+Neighbors/      KNNModel, LVQ1 (Learning Vector Quantization, Kohonen 1989, K-means init + competitive learning)
+PRIM/           PRIM + Box + +Criterion/(IObjectiveFunction, MeanResponse, TargetClass)
                 + +Selection/(IBoxSelector, CVSelector, HeuristicSelector)
+Regression/     LinearModel, LogisticModel, LogisticRegressionManual,
                 PolynomialLogisticModel, PolynomialModel, PenalizedRegression
+Tree/           DecisionTree (core) + ExhaustiveCandidateGenerator + QuantileCandidateGenerator
                 + ICandidateGenerator + TreeSplitProblem
                 + RandomFeatureCandidateGenerator  ← NEW 2026-05-12: mtry random feature subset per node (RF split randomization)
                 + ExtraTreesCandidateGenerator     ← NEW 2026-05-12: random threshold from feature range (Geurts 2006 K=1 split)
                 +Criterion/ (INodeCriterion, ClassificationCriterion, RegressionCriterion, GainRatioEvaluator)
                 +Impurity/  (IImpurityMeasure, GiniIndex, CrossEntropy, MisclassificationError, ExpectedLoss)
                 +Strategy/  (ISplitStrategy, CartSplitter, C50Splitter, SplitResult)
                 +Surrogate/ (SurrogateSplitStrategy)
+KernelMethods/  KernelMachine (unified core), IKernelSolver, SMOSolver, BuiltInKernelSolver
+SVM/            SVMClassifier (legacy wrapper) + +Kernels/(ISVMKernel, LinearKernel, RBFKernel, PolynomialKernel, SigmoidKernel)
```

### 5.13 +Unsupervised
`GMM` (EM algorithm), `KMeans`, `SOM` (Self-Organizing Map, Kohonen 1982: online Gaussian-neighborhood update, linear decay, U-matrix, topographic error), `SpectralClustering` (Unnormalized graph Laplacian, mutual k-NN affinity)
`SparsePCA` ← NEW 2026-04-29: Sparse PCA (Zou et al. 2006, ESL §14.5.5). Alternating minimization: SVD Procrustes step for Theta, elastic-net (L2+L1) step for V via L1Regularizer.proximity(). Deflation-based multi-component. API: fit(X,K,...), transform(X_new), sparsityRatio(), printSummary().
`rigid_procrustes_average`  ← NEW 2026-05-05: Iterative Generalized Procrustes Analysis (rigid+scale). ESL Ex. 14.9, Eq. 14.115. ALS: SVD Procrustes R=U*V', scale beta=trace(S)/||X||_F^2, mean renormalization. Input: [N,p,L] tensor. Outputs: M, R, beta, X_aligned.
`affine_invariant_average`  ← NEW 2026-05-05: Closed-form affine-invariant Procrustes mean. ESL Ex. 14.10, Eq. 14.60. Economy QR → H_l=Q*Q' → H_avg=mean(H_l) → eigs(H_avg,p). Input: [N,p,L] tensor. Output: M (N×p, cols orthonormal), X_aligned (optional projection).
`PrincipalCurve.m` - Hastie & Stuetzle (1989) Principal Curves.
`PageRank.m`       ← NEW 2026-05-06: PageRank algorithm (power method). ESL §14.5.3. formulation (14.110). fit(L, 'DampingFactor', 0.85).

#### +FrequentItemsets/  ← NEW (2026-04-15, Market Basket Analysis Pipeline)
```
TransactionEncoder.m   ← DONE Phase 2: fit/apply pattern, COO sparse construction, decode + validate round-trip
IMinerStrategy.m       ← DONE Phase 3: Abstract interface: mine(X_sparse, minSupport, opts)
AprioriMiner.m         ← DONE Phase 3: Level-wise search, matrix-multiply k=2, anti-monotone pruning
FPGrowthMiner.m        ← PLANNED Phase 3: FP-Tree prefix compression, no candidate generation
```

#### +Rules/  ← DONE (2026-04-16)
```
RuleGenerator.m        ← DONE Phase 4: Itemsets → 2^k-2 directed rules, cached Confidence/Lift, vocabulary translation
RuleFilter.m           ← DONE Phase 5: Threshold, redundancy pruning, ranking, top-N
```

---

### 5.17 +DataArch — (moved to +OceanData)
See section 6 below.

### 5.18 +DataPipeline — (moved to +OceanData)
See section 6 below.

---

## 5b. OceanDL — Deep Learning (top-level package, 2026-09-23)

Deep learning is NOT statistics: everything neural-network lives in `+OceanDL`, shared across domains
(CV, NLP, time series). Moved from `OceanStats.NeuralNet` on 2026-09-23 (package names, error ids
`OceanDL:...`, tests `OceanTest.OceanDL.*`; 195/195 tests identical before/after). Historical planning
documents under `Planning/` and `Architecture/` still use the old `OceanStats.NeuralNet` names on purpose.

```
+OceanDL/
  +Autograd/        Tensor, Function, Engine, Broadcast, GradMode, GradCheck, F, +Functions/   (modern)
                    Tape                                                                         (legacy, CNNEngine/Interpretability)
  +nn/              Module (to(DataType, Device), dataType(), device(), visitModules, applyBuffers hook), Parameter,
                    Linear, Conv2d, Embedding, Dropout, Sequential, Activation (+silu), GroupNorm, Upsample,
                    RMSNorm, TimestepEmbedding, Legacy, clipGradNorm
  +optim/           ModuleOptimizer (setLR; forModule = AdamW groups without decay on Bias/Norm/PosEmbed/ClsToken),
                    WarmupCosine schedule. Step rules (Adam, AdamW, SGD...) stay in OceanStats.Optim.
  +Layout/          DimFormat -- the mandatory import contract (features x observations)
  +Recurrent/       RNN cells/layers, CRNN, CTC decoder (Goodfellow Ch. 10)
  +Transformer/     MultiHeadAttention (batched pages, key/attn masks, stored maps), FeedForward, TransformerBlock
                    (Pre-LN default | Post-LN), TransformerEncoder (DropPath schedule, final norm), PositionalEncoding
                    (sincos 1-D/2-D), LearnedPositionalEmbedding. == PyTorch nn.MultiheadAttention /
                    nn.TransformerEncoderLayer (1e-12) and reproduces the legacy OceanNLP BERT attention/block.
  +Vision/          PatchEmbedding (stride-p conv == patchify+linear, column-major token order), ViT (positions
                    learned|sincos|none, pooling cls|mean, any resolution with sincos/none), AttentionRollout,
                    ImageAugment.
                    Swin (Liu 2021): WindowOps (partition/reverse/cyclic shift/3x3 region shiftMask/relativeIndex),
                    SwinBlock (W-MSA / SW-MSA, relative position bias, rule: grid <= window -> window = grid, no
                    shift), PatchMerging, Swin (stages; image side must be divisible by PatchSize*WindowSize*2^(S-1)).
                    MaskedAutoencoder (He 2022): wraps a ViT built with Pooling "mean" + Positions "sincos"; encoder
                    sees visible tokens only; decoder with [MASK] token; MSE on masked patches; after pre-training
                    fine-tune the wrapped ViT directly (stateDict/loadStateDict to clone the backbone).
                    NormPixLoss suits natural images; on flat-background data (USPS: 25% flat 4x4 patches) it
                    turns noise into unit-variance targets -- use NormPixLoss=false there.
                    Conv autoencoder blocks (AutoencoderKL layout): ConvResBlock (GN-SiLU-conv x2, 1x1 skip),
                    SpatialSelfAttention (reuses MultiHeadAttention over H*W positions), ConvEncoder / ConvDecoder
                    (Widths, ResPerLevel, MidAttention, Groups options; factor 2^(L-1)).
                    Diffusion denoisers (forward(x, t, y)): ConvResBlock 'EmbedDim' (ADM scale-shift FiLM), UNet2D
                    (skip concat, zero-init out), DiTBlock / DiT (adaLN-Zero), MMDiTBlock / MMDiT (two streams).
  +Generative/      Latent (SSCB|CB layout helpers), DiagonalGaussian (rsample, KL, klTo, logProb), VAE (gaussian|bernoulli,
                    fixed or learned sigma, Beta, FreeBits, activeUnits), VQVAE, CodePrior (causal Transformer over
                    code grids, sample with Temperature/TopK), +Quantizers/ IQuantizer, VectorQuantizer, FSQ -- see 5b.5
                    +Diffusion/ schedules, targets, samplers, DiffusionModel, LatentDiffusion -- see 5b.6
                    HierarchicalVAE + TopDownBlock (configurable levels / latents per level) -- see 5b.7
  Device            cast(X, DataType, Device), name, dataType, gpuAvailable -- hardware-neutral precision/placement
  +Activation/ +Norm/ +Pooling/ +Conv/ +Graph/ +Simulation/ +Commands/ +Viz/ +Interpretability/   (legacy stack)
  ComputeBackend, CPUBackend, GPUBackend, ResidualBlock
```
Dependencies: `OceanDL -> OceanStats.Optim (per-parameter step rules), OceanStats.Loss (legacy Trainer), OceanMath`.
Consumers: `OceanNLP` (BERT uses Norm/Activation; will move onto OceanDL.Transformer), `OceanViz` (NNStudio,
CNNInterpretabilityStudio, NN/CNN strategies). `OceanStats` must never import `OceanDL`.


**New ops (2026-09-23):** BatchMatMul (pagemtimes, all transpose flags, broadcast pages), GELU (exact erf | tanh),
fused LayerNorm (PyTorch eps-inside-sqrt; the legacy LayerNormLayer uses std+eps), F.dropPath, crossEntropy 'LabelSmoothing'.

**AffectNet copy (Data/Datasets/AffectNet) is NOT a clean benchmark -- see OceanData.Adapters.AffectNetReader:**
1,688 test files are byte copies of train files (ALL 1,584 test "sad" images), labels.csv disagrees with folders
for 37% of rows, and happy/neutral are ~all FFHQ PNGs (other classes AffectNet JPEGs: source confound).
The reader lists jpg+png, dedups, and exposes 'Labels' and 'Sources' options with a per-class x source report.

### 5b.1 Modern autograd (Tensor graph, PyTorch-style) ← NEW 2026-09-23
The primary way to build and train networks. The older `Autograd.Tape` + `CNNEngine` stay (legacy API,
still used by `+Interpretability`); `LegacyEquivalenceTest` proves both engines give identical gradients.
```
+Autograd/
  Tensor.m       Array + RequiresGrad/GradFn/Grad; MATLAB operators build the graph (RedefinesParen indexing,
                 cat, * / .* / + ...). No in-place assignment. Grad accumulates until zeroGrad(). Precision follows Data.
  Function.m     Base of every op AND its graph node: forward(raw...) saves on obj.Saved, backward(g) = VJP.
                 Non-differentiated config goes to the constructor. Extend by subclassing (see class doc).
  Engine.m       backward(root) / grad(out, inputs): iterative DFS + decreasing creation Id (= reverse topo order),
                 multi-use gradient accumulation (BPTT), Broadcast.reduceTo on every input grad, releases buffers
                 unless RetainGraph. First-order only (VJPs act on raw arrays).
  Broadcast.m    reduceTo(g, size): sum over broadcast dims.     GradMode.m  disable()/noGrad() (torch.no_grad).
  GradCheck.m    central finite differences vs autograd with random output projection.
  F.m            functional API: relu/sigmoid/tanh/softmax/logSoftmax, linear, conv2d, embedding, dropout,
                 crossEntropy (Mask for padded sequences), mseLoss, bceWithLogits -- losses are COMPOSED ops.
  +Functions/    Add Sub Mul Div Neg Pow MatMul Transpose Reshape Permute Sum Mean Max Exp Log Sqrt Tanh Sigmoid
                 ReLU Abs Index (scatter-add VJP) Cat Softmax LogSoftmax Conv2d (shift-and-GEMM) LegacyLayer
+nn/             Module (parameters()/namedParameters() by property walk incl. cells, train/eval, zeroGrad,
                 stateDict/loadStateDict), Parameter, Linear, Sequential, Activation, Embedding, Conv2d, Dropout,
                 Legacy (wrap parameter-free legacy layers; stale-cache guard), clipGradNorm (Goodfellow 10.11.1)
+Optim/ModuleOptimizer.m   zeroGrad()/step() over Parameters with one existing IOptimizer per parameter.
```
Layout convention: FEATURES x BATCH [x TIME] (samples are columns); images H x W x C x N.
Training loop: `opt.zeroGrad(); L = F.crossEntropy(net.forward(X), y); L.backward(); opt.step();`

### 5b.2 Layout contract (decision B, 2026-09-23) -- `+OceanDL/+Layout/DimFormat.m`
Canonical layouts are FEATURES x OBSERVATIONS, column-major: "CB" (D x N), "CBT" (D x N x T), "SSCB" (H x W x C x N);
one observation / one time step is always a contiguous block. ANY imported data goes through an explicit step:
`fromRowMajor(buffer, shape)` (memory order: C-order bytes -> same index meaning), `toCanonical(X, "BTC"|"TBC"|"BCSS"|"BSSC"|"BC")`
(axis meaning), `fromTable(T)` (columnar table -> CB), then `validate(X, fmt, 'FeatureDim', D, 'Mask', m)` (errors name the likely
wrong layout). Readers return data UNCONVERTED plus a format description: `OceanData.Adapters.NpyReader` (.npy v1-3, C/F order),
`OceanData.Adapters.UspsDigitsReader` (zip.train/test: rows, row-major pixels, fmt.Format = "BSSC").

### 5b.3 +Recurrent (Goodfellow Ch. 10, built on the Tensor autograd) ← NEW 2026-09-23
```
IRecurrentCell   step(x_t (D x N), state) -> [h, state]; initialState(N). Cells share Parameters over time; BPTT = Engine.
  ElmanCell      Eq. 10.8, tanh|relu                       LSTMCell   PyTorch gate order i,f,g,o; ForgetBias=1; RecordGates
  GRUCell        Variant "pytorch" (default, reset after W h) | "original" (Cho 2014 / Eq. 10.45)
  LeakyCell      10.9.2 leaky integrators, log-uniform alphas, optional learnable (sigmoid(rho))
  EchoStateCell  10.8 fixed sparse reservoir scaled to SpectralRadius; no Parameters; fitRidgeReadout (closed form)
RecurrentLayer   unroll over "CBT"; Mask (right padding: state carried, output 0); Reverse; InitialState + detachState (TBPTT);
                 TrackStates -> StepStates for Diagnostics.gradientNormsOverTime (10.7)
Bidirectional (10.3, concat|sum)   StackedRecurrent (10.5; variational dropout, residual)   SequencePool (last|mean|max, masked)
ImageSequence.toSequence(SSCB, "rows"|"columns"|"pixels") -> CBT      ReNet2D (patches + vertical + horizontal BiRNN sweeps)
SequenceClassifier (encoder -> pool -> Linear; ImageMode)            ReNetClassifier (ReNet2D -> global mean -> Linear)
CRNN (CNN -> columns as time -> (Bi)GRU -> per-step logits; Shi et al. 2017)   CTCDecoder (greedy best path, editDistance, CER / string acc)
Autograd/+Functions/CTCLoss + F.ctcLoss: Graves 2006 forward-backward in log space; Blank (default 1), InputLengths,
  Reduction "mean" (PyTorch: mean_n L_n/U_n) | "sum", ZeroInfinity. Verified vs PyTorch nn.CTCLoss (loss+grad, 1e-10) and
  brute-force enumeration of all alignments.
```
Verified: Elman/LSTM/GRU equal PyTorch nn.RNN/LSTM/GRU to 1e-12 (fixtures in +OceanTest/.../+Recurrent/fixtures, regenerate with
gen_torch_reference.py); every cell's BPTT vs finite differences.
### 5b.5 Precision/device + generative models (VAE, VQ-VAE) <- NEW 2026-09-24

**Precision / device (PyTorch `model.to`).** `net.to('DataType', "single", 'Device', "gpu")` casts every
Parameter and every module-owned buffer (subclasses with non-Parameter state override `applyBuffers(fn)`);
optimizer state (Adam/AdamW/RMSProp/SGDMomentum) is created `'like'` the gradient; `loadStateDict` casts to
the module's type. Never branch on a GPU model: FP32 is the right training default on consumer GPUs (FP64 at
1/32-1/64 rate) and halves CPU memory traffic; keep double for GradCheck. `OceanDL.Device.cast` never casts
integer/logical arrays (labels, masks). Test gotcha: `verifyEqual` IGNORES AbsTol/RelTol when both values
are single -- compare `double(a), double(b)`.

**Generative (Szeliski Ch. 5, Goodfellow Ch. 14/19/20).** Encoder/decoder agnostic: any Module pair.
- VAE: loss = negative ELBO per sample (nats) with 1 reparameterised sample + closed-form KL; `Beta` public
  (warm-up), `FreeBits` = sum_c max(KL_c, lambda) (posterior-collapse guard), `LearnSigma` (sigma-VAE calibrated
  decoder), `LastStats` (Recon, KL, KLPerChannel, NegELBO, BitsPerDim), `activeUnits` (Burda 2016).
  Validated exactly: linear-Gaussian VAE with the true posterior gives NegELBO == -log p(x) (PPCA marginal) and
  a shifted encoder loses exactly KL(q || posterior).
- Quantizers (IQuantizer strategy): VectorQuantizer (straight-through; Update "grad" (codebook + Beta
  commitment) | "ema" (online k-means, Laplace smoothing; Decay 0 == one Lloyd step); dead-code restart;
  Init "data"; CodeDim projection + Normalize (ViT-VQGAN)) and FSQ (Mentzer 2023; no codebook/aux loss).
  VQ (grad) and FSQ == PyTorch/paper reference to 1e-12 (values, indices, gradients). The decoder gradient
  never reaches the codebook (straight-through goes to z_e only).
- VQVAE (encodeIndices/decodeIndices/reconstruct), CodePrior (GPT-style, BOS = K+1, raster column-major
  order, causal AttnMask, no KV cache).
- Fixtures: `+OceanTest/+OceanDL/+Generative/fixtures/gen_torch_generative_reference.py` (GroupNorm, Upsample,
  ResBlock + stride-2 conv, VQ, FSQ). Out of scope so far: perceptual (LPIPS) / adversarial losses (need a
  pretrained network), hierarchical VAEs, experiments (P5 deferred by the user).

### 5b.6 Diffusion family (foundation + common variants) <- NEW 2026-09-24

One formulation, `x_t = alpha(t) x0 + sigma(t) eps`, t in [0,1] (t = 1 noise). Strategy objects:
- `+Diffusion/+Schedules/` INoiseSchedule (alpha, sigma, dalpha, dsigma; grid(N), priorScale, netTime):
  LinearVP (DDPM; t = i/L hits the discrete chain exactly), CosineVP (clipped logSNR +-15, 'Shift'),
  NOTATION = Lai et al., "The Principles of Diffusion Models" Sec. 2.2: beta_i is the step noise STD,
  alpha_i = sqrt(1 - beta_i^2), alphaBar_i = prod alpha_k; LinearVP('L', 'BetaSqStart', 'BetaSqEnd') takes
  Ho's variances as beta_i^2; accessors betaSq(i), beta(i), alphaStep(i), alphaBar(i). DDPMAncestral / DDIM
  use the book's variable names (alphaBar_i, betaSq_i, tildeBetaSq_i, x_phi) with Eq. (2.2.4)/(2.2.5)/(2.2.15).
  EDM (sigma = Karras rho-spacing of t, SigmaData, tFromSigma, netTime = log(sigma)/4), RectifiedFlow
  (alpha = 1 - tau, sigma = tau, SD3 'Shift').
- `Targets` exact conversions eps / x0 / v / velocity (+ x0ErrorFactor for min-SNR); `TimeSampler`
  uniform | logitnormal (SD3) | lognormal (EDM); `Analytic` exact Bayes denoisers for Gaussian mixtures
  + closed-form Gaussian flow map (reference for tests / optimal-loss measurement).
- `+Samplers/` ISampler.sample(predict, schedule, shape) with predict(x,t) -> [x0hat, epshat]:
  DDPMAncestral (general posterior formula), DDIM (Eta; Eta=1 == ancestral), EulerODE, HeunODE (EDM Alg. 1).
  Measured: Heun 2nd order, DDIM/Euler 1st order; Karras' 18 Heun steps are too few for narrow toy modes.
- `DiffusionModel(net, schedule)`: 'Target' eps|x0|v|velocity|edm (EDM preconditioning), 'Weighting'
  simple|minsnr, 'CondDropout' + NULL label K+1 (classifier-free guidance), predictor/generate with 'Guidance'
  (one batched cond+uncond pass). Guards: edm needs EDM schedule; eps is singular on RectifiedFlow.
- `LatentDiffusion(VAE|VQVAE, DiffusionModel)`: calibrate() -> ScaleFactor = 1/std(latents); autoencoder in a
  PRIVATE property (frozen, invisible to parameters()). `MLPDenoiser` for vector data. `optim.ModelEMA`.
- MultiHeadAttention refactor: splitHeads -> headAttention -> mergeHeads (static, reused by JointAttention);
  options 'QKNorm' rms|layer and forward 'Rope' (Transformer.RoPE: 1-D and 2-D axial, half convention,
  column-major tokens). FeedForward 'Activation' "swiglu". Defaults unchanged (270 Transformer/Vision/NLP
  regression tests green after the refactor).
- Validation: `DiffusionCoreTest` (exact, no training), `DiffusionTrainingTest` (trained MLP within 10% of
  the analytic Bayes-optimal loss; PyTorch fixtures for time embedding + FiLM block), `DiffusionTransformerTest`
  (DiT block and MMDiT block with RMSNorm/SwiGLU/QK-norm/RoPE/joint attention == PyTorch 1e-10),
  `LatentDiffusionTest`. Experiments (F6) deferred by the user.
- Gotchas: `X = pkg.ClassName` constructs an instance -- fine for Targets/Analytic (no-arg), an ERROR for
  classes with required constructor args (use fully qualified static calls); packages cannot be aliased
  (use `import`); `Embedding.forward(ids)` returns D x size(ids) -> pass labels as N x 1.

### 5b.7 Hierarchical VAE (configurable) <- NEW 2026-09-25

`OceanDL.Generative.HierarchicalVAE` (Ladder VAE / NVAE / VDVAE family). The hierarchy is pure configuration,
one entry per resolution level (level 1 = full size, each next level halves H, W):
`'Widths'`, `'LatentsPerLevel'` (any non-negative integers, total >= 1), `'LatentChannels'` (scalar or per
level), `'BlocksPerLevel'`, `'BlockType'` "res" (ConvResBlock) | "bottleneck" (Vision.BottleneckBlock, VDVAE),
`'Posterior'` "residual" (NVAE) | "direct", `'TopPrior'` "standard" N(0,I) | "learned", plus the VAE options
(Likelihood, Sigma/LearnSigma, Beta, FreeBits -- here PER LATENT LAYER).
- Bottom-up: ConvIn, blocks per level, stride-2 down; one feature per level. Top-down: learned constant H0 at the
  lowest level -> per level: TopDownBlocks (top first), extra blocks, upsample + Lift conv -> GN/SiLU/ConvOut.
- `TopDownBlock`: prior p(z|h) (zero-init head = starts N(0,I); N(0,I) when Standard), posterior q(z|h,e) from
  [h; e], KL = `DiagonalGaussian.klTo`, h = Block(h + ZProj(z)); modes "sample" | "mode" | "given" | "prior".
- API: forward (neg. ELBO; LastStats.KLPerLayer top first), encode (posteriors), reconstruct, decodeLatents,
  sample(n, imageSize, 'Temperature'), activeUnits (per layer), costEstimate(imageSize, 'Batch', 'GFLOPS')
  (conv FLOPs per level, hand-count verified). `VAE.nll` is the shared likelihood.
- Validated: klTo vs MC; BottleneckBlock and TopDownBlock == PyTorch 1e-10; a one-latent hierarchy gives EXACTLY
  the VAE loss (test adapters HvaeEncoderAdapter / HvaeDecoderAdapter run OceanDL.Generative.VAE on the same
  network); five configurations incl. empty levels, per-level channels, bottleneck, learned prior; per-layer
  free bits; gradient check; 2-layer USPS training with both layers active. H4 experiments deferred.
- `Module.children()` now recurses into NESTED cell arrays (train/eval/to/visitModules reach modules stored as
  cells of cells, e.g. HierarchicalVAE.TD{level}{j}).
- Review fixes 2026-09-26: levels above the top latent layer are NOT built (were dead: 23% of the USPS VAE's
  parameters); `'FreeBitsPer'` "layer" (default, old behaviour) | "channel" | "unit" -- only "unit" stops single
  units collapsing (a per-layer floor of 0.25 nats never binds); `'LogVarClamp'` c (default 10 in
  HierarchicalVAE, 0 in a bare TopDownBlock so the PyTorch-equivalence test stays exact) soft-clamps prior and
  posterior log-variances to c tanh(lv / c). Bernoulli likelihood needs 0/1 pixels: USPS is 31% grey, which gave
  the loss a 33-nat floor -> gen_compare_vae_hvae `cfg.Binarize` (dynamic binarisation of training batches,
  fixed-seed binarised test set) makes the negative ELBO a true bound. That function also trains on the GPU
  (`cfg.Device`) and saves ModelCards (Results/generative/cards/<dataset>_<model>.mat).

### 5b.8 Inspection, ModelCard, Model Lab GUI <- NEW 2026-09-26

- `OceanDL.Generative.Inspect.ForwardProcess` (static): `at(x0, schedule, t, eps)` -> [x_t, info] with the book's
  quantities (AlphaBar, Sigma, SNR; LinearVP also Step i, BetaSq, Beta, AlphaStep); `trajectory`, `scheduleTable`.
- `Inspect.ReverseProcess` (handle): steppable sampler run -- `advance(n)` in chunks (GUI progress/cancel),
  `Frames` (X, X0, Eps, Mu, Var per recorded step, 'Stride'); final sample == `sampler.sample` exactly.
- `ISampler.sample(..., 'ClipX0', [lo hi])` (Ho et al. clipping; eps recomputed from the clipped x0).
  `TimeSampler('Discrete', L)`: t = randi(L)/L (textbook integer steps; uniform only).
- `OceanDL.Training.ModelCard`: weights + a nested constructor RECIPE + DataType + Meta (Dataset, ImageSize,
  ClassNames, DataRange). `save` / `load` (rebuilt, cast, eval mode) / `build`. Anything that should open in a
  GUI or another script is saved as a card, not a bare checkpoint.
- `OceanViz.App.ModelLab` (replaces the monolithic CNNInterpretabilityStudio; that class is now a launcher):
  `lab = OceanViz.App.ModelLab.launch(modelOrCardOrBuiltin, 'Visible', tf)`.
  - `Shell` (window, top bar, tabs rebuilt on ModelChanged), `Context` (Model, Image in [0,1], Dataset, Theme;
    events ModelChanged / ImageChanged), `Datasets` (shapes16/shapes32/usps, cached). Shared widgets
    `JobRunner` (chunked long jobs, progress + Cancel), `Sidebar` and `ImageView` moved to
    `OceanViz.App.Common` on 2026-09-27, because Replay Studio uses them too.
  - `+Models/IModelHandle` adapts a model to the GUI: Capabilities ("diffusion", "sample", "conditional",
    "vae", "hierarchical", "cnn-legacy", ...), ImageSize, DataRange, ClassNames, toModel/toDisplay.
    Handles: DiffusionHandle (trained or ideal EmpiricalDenoiser), VAEHandle, LegacyCNNModel. `ModelFactory`
    (builtins, fromCard, idealDDPM, dummyCNN, legacyCNN).
  - `IPanel` (title, requires, build, reference, onImageChanged, supports). A tab appears only when the model
    has the panel's required capability. `PanelRegistry.all()` lists panels in tab order; `register(ctor)`
    adds one at run time. Panels: Forward process, Reverse process, Generate, Training step (Eq. 2.2.10),
    VAE explorer, and the 8 CNN tabs (`CnnPanelBase` + Cnn*Panel; same modules/strategies as before).
  - ADD A FEATURE = one IPanel subclass (+ a capability on the handle if new) + one line in PanelRegistry.
    Every panel exposes its main action as a public method (run / update / generate / traverse) so
    `OceanTest.OceanViz.ModelLabTest` drives it headless (hidden uifigure).
- Cards produced so far: `Results/ddpm_textbook/ddpm_shapes16.mat` (exp_ddpm_textbook_shapes.m, class-cond.
  16x16 shapes DDPM), `Results/ddpm_affectnet/ddpm_affectnet16.mat` (exp_ddpm_affectnet16.m). The old USPS
  VAE/HVAE cards were deleted 2026-09-26 (Bernoulli-on-grey likelihood issue); retraining pending.

### 5b.9 Interpret -- feature visualisation for modern-autograd models <- NEW 2026-09-26

`+OceanDL/+Interpret/` (maths: Szeliski 2nd ed. Sec. 5.4, Fig. 5.47-5.49; VAE parts Kingma & Welling Ch. 2/4).
Not the legacy `OceanDL.Interpretability` (CNNEngine only).
- `ITraceable` interface: `[acts, order] = trace(X, 'Parts', "all"|"encoder")` -> containers.Map name -> Tensor
  (H x W x C x N; `Tensor.raw` for arrays), `traceNames(inputSize)` table Name/Size/Section. Naming
  `enc.L<l>.in|block<b>`, `z<i>.mu|kl` (top first), `dec.L<l>.start|td<j>|res<b>|lift`, `dec.out.act`, `dec.out`.
  Implemented by `HierarchicalVAE` (so both VAE and HVAE) via an optional recorder in bottomUp/topDown.
  `topDown` "given" with an EMPTY zsGiven{i} = that layer at its prior mean (layer contributions).
- `Probe` (activation, unitGradient under a backward rule), `ReceptiveField` (EFFECTIVE field: GroupNorm makes the
  theoretical field global; Luo et al. 2016), `TopActivations` (top-k per channel, Fig. 5.47 right), `FeatureVis`
  (activation maximisation, Fig. 5.49; Lucent-style DCT decorrelation + jitter + L2; 'Decorrelate' false = pixels),
  `Occlusion` (Fig. 5.48), `LatentStats` (KL per latent unit, mean codes), `DecoderAtoms` (d E[x|z]/d z_j by central
  differences -- NOT in the texts -- and HVAE layer contributions).
- `OceanDL.Autograd.BackpropRule` "standard"|"deconvnet"|"guided" (GradMode-style switch); `F.silu` then uses
  `Functions.RuledSiLU`. The published rules are for ReLU; the SiLU form is our adaptation (announced in the UI).
- Model Lab tab `InsideVAEPanel` ("Inside the VAE", capability "traceable"): Encoder / Latent / Decoder sections;
  every non-textbook view is labelled. Tests: `OceanTest.OceanDL.Interpret.InterpretTest` (closed-form answers on
  `TinyTraceable`: one known 3x3 conv + SiLU) and `ModelLabTest.testInsideVaePanelRunsEveryView`.
- ALSO NOT IN THE TEXTS (labelled): `LatentStats.codes` + a t-SNE map of the codes (panel, via
  `OceanViz.Adapters.EmbeddingMapper2D`; OceanDL never calls OceanViz), and `SparseDictionary` -- a sparse
  autoencoder on a layer's position vectors (Bricken et al. 2023: ReLU code, unit-norm atoms, L1, data scaled to
  E||a||^2 = C; test: recovers 24/24 known atoms). `TopActivations.scan(..., 'Transform', f)` ranks any feature maps.
  Panel views: Latent '2-D map of all codes (t-SNE)'; Encoder/Decoder 'Sparse features: overview' / 'Sparse feature #'.
  A layer that does not depend on x (dec.L*.start = learned constant) gets zero gradients / whole-image field.
- GUI refinements 2026-09-26 (all texts say HOW IT WAS MADE / HOW THEY WERE PICKED; non-textbook items marked):
  descriptive layer names (ItemsData = internal names); Channel field shows the valid range per layer/view, frames the
  chosen channel in all-channel views, disabled when unused; Top-9 = 9 strongest DISTINCT crops from one candidate
  per test image (ties on 0/1 data), first layer adds its 3x3 kernel tile; activation maps show the current x;
  activation maximisation reports RELIABILITY vs the best real digit; 'accentuate best real digit' = FeatureVis
  'Init' from the top-1 digit at its winning unit (Hamblin et al. 2024; Decay 0, Jitter 0, LR 0.2 -- decay/jitter
  made it fail); Hinton diagram of the conv producing a layer (`HierarchicalVAE.weightsFor`; dec.out.act / td map to
  the conv before them). ReverseProcessPanel round trip with the textbook DDPM = exactly round(t0 L) integer steps.
  Generate 'Temperature' labelled as not in Kingma & Welling Ch. 2.
- USPS HVAE numbers: 10-NN digit agreement of codes z1 84% vs z2 43% (chance 10%); enc.L2 dictionary (256
  features, EV 76%, L0 5) = stroke pieces (bars, diagonals, curves, corners, junctions).
- Findings on the USPS HVAE: top latent z1 = digit identity (a channel's top-9 are 5/3/S shapes); decoding the top
  layer alone already gives the digit, z2 adds stroke detail; z2 atoms are local stroke pieces; activation
  maximisation is weak here (GroupNorm + 16x16) -- top-9 / projections are more informative.

### 5b.4 Legacy dense / CNN stack (built 2026-03 as OceanStats.NeuralNet, moved 2026-09-23)
Interactive-first neural network module. Graph phase done; NN-1 adds vectorized engine + activations; NN-2 closes the training loop.

**Design principles:**
- **Topology ≠ State**: `NNTopologySpec` (value class, immutable) is separate from `NNGraph` (mutable handle). Inspired by JAX/Flax NNX `nnx.split()`.
- **Composite** for layer hierarchy; **Observer** for GUI coupling; **Command** for undo/redo.
- `NNGraph` implements `events` (TopologyChanged, WeightsChanged, ActivationsUpdated) — renderers subscribe.

```
+Graph/
  NNTopologySpec.m   ← VALUE class: shape blueprint. Valid activations: linear, relu,
                        sigmoid, tanh, softmax, gelu, swish, sine   ← updated NN-1
  NNGraph.m          ← Observable handle: fromSpec(), forwardPass(), disableNode(), swapNodes()
                        getBiasVector(L) → (d_L x 1)               ← added NN-1
  NNLayer.m          ← Composite: owns NNNode[]
  NNNode.m           ← Leaf: bias param, state machine (Active/Disabled/Highlighted)
  NNEdge.m           ← Directed weight
  ParameterStore.m   ← flatten(graph)→double[] / setFromFlat(graph,v)

+Activation/                                                         ← NEW NN-1
  IActivation.m         Abstract: forward(Z), grad(Z), name()  [Z is (d x N) column-major]
  LinearActivation.m    A = Z
  ReLUActivation.m      A = max(0, Z)
  SigmoidActivation.m   A = 1/(1+exp(-Z))
  TanhActivation.m      A = tanh(Z)
  SoftmaxActivation.m   Numerically stable, column-wise
  GELUActivation.m      A = Z .* normcdf(Z)   (Hendrycks & Gimpel, 2016)
  SwishActivation.m     A = Z .* sigmoid(Z)   (Ramachandran et al., 2017)
  SineActivation.m      A = sin(omega_0*Z)    SIREN (added NN-5)
  ActivationFactory.m   create('relu') → IActivation object

+Simulation/
  ForwardPassEngine.m   ← legacy diagnostic, keep
  VectorizedEngine.m    Batch forward pass.  [Y,cache] = eng.forward(graph, X, mode, Backend)
                          cache{L}.Z (post-norm), cache{L}.Z_raw (pre-norm), cache{L}.A, cache{L}.A_skip
  BackpropEngine.m      Chain-rule backprop. grads = back.backward(graph,cache,Y,Ytrue,loss, Backend)
                          Residual skip grad via dA_acc accumulator. Softmax+CE fused.
  Trainer.m             Minibatch loop. history = trainer.fit(X,Y,'Epochs',500,'BatchSize',32)

+Norm/                                                               ← NEW NN-3
  INormalization.m      Abstract: normalize(Z,mode), parameters(), name()
  LayerNormLayer.m      Per-column mean=0, std=1 + learnable gamma/beta. Mode-agnostic.
  RMSNormLayer.m        Per-column RMS=1 + learnable gamma. No mean subtraction. (LLaMA-style)
  BatchNormLayer.m      Per-feature across N. train: batch stats+EMA update. infer: frozen stats.
  > Applied AFTER Z=W*A+b, BEFORE activation. Controlled by NNTopologySpec.NormalizationMode.

+Pooling/                                                            ← NEW 2026-09-22, 1D/3D added same day
  Three parallel rank-specific families (IPoolingLayer[1D|(2D=default)|3D] each own
  PoolSize/Stride/Padding arity -- kept separate per rank rather than one generalized
  N-D interface, since the window loop nesting genuinely differs; mirrors PyTorch's own
  independent nn.MaxPool{1,2,3}d classes) -- NOT wired into VectorizedEngine/BackpropEngine
  (those only understand dense (d x N) chain-rule). See docs/guides/cnn_pooling_unpooling.md.

  IPoolingLayer1D.m / IPoolingLayer.m (2-D) / IPoolingLayer3D.m
                          Abstract: forward(X), backward(dY), getMetadata(), getPipelineSpec(), name()
                          X/Y tensor rank per family: (L x C x N) 1-D, (H x W x C x N) 2-D,
                          (D x H x W x C x N) 3-D -- matches dlarray SS..CB, NOT the (d x N)
                          convention of the rest of +OceanDL.
  MaxPool1D.m / MaxPool2D.m / MaxPool3D.m
                          Sliding-window max. Padding = -Inf (never wins). Caches argmax
                          "switches" (linear idx into the un-padded spatial plane) on self --
                          getSwitches()/getInputSize() for pairing with the matching
                          MaxUnpoolND. backward() accumulates (accumarray) when
                          Stride < PoolSize causes overlapping windows.
  AvgPool1D.m / AvgPool2D.m / AvgPool3D.m
                          Sliding-window mean. Padding = 0. CountIncludePad (true, PyTorch
                          default) divides by full pool area/volume; false divides by
                          real-cell count only.
  MaxUnpool1D.m / MaxUnpool2D.m / MaxUnpool3D.m
                          SegNet-style inverse (Badrinarayanan et al. 2017): scatters a pooled
                          tensor back to switches' original locations (functional -- switches
                          passed explicitly to forward(), not read from a paired object's hidden
                          state). No learnable weights (contrast: transposed conv).
  > No Conv1D/2D/3D or spatial engine yet -- scoped to pooling/unpooling only, standalone.
  > GOTCHA: MATLAB drops TRAILING singleton array dims (size(zeros(2,2,1,1)) == [2 2]).
  > Any shape-equality check across these classes destructures size() into named outputs
  > per axis and compares element-wise -- never isequal(size(A), [h w c n]) directly.

ResidualBlock.m         Static helpers for skip connections.                ← NEW NN-4
  applySkip(A_next, A_skip)          identity shortcut (same dims)
  applySkip(A_next, A_skip, W_skip)  projection shortcut (dim change)
  makeProjection(d_in, d_out)        Glorot-init projection matrix
  > BlockType='residual' in NNTopologySpec creates 2-layer blocks; L=3..nL-1.

ComputeBackend.m        Abstract: send(X), recv(X), isGPU(), name()        ← NEW NN-6
CPUBackend.m            Identity passthrough. Zero overhead. Bit-identical.
GPUBackend.m            gpuArray/gather. Silent CPU fallback if no Parallel Computing Toolbox.
  > Pass as optional arg to VectorizedEngine.forward() and BackpropEngine.backward().
  > Backward-compatible: omitting Backend defaults to CPUBackend.
```

**Weight initialization (statistically motivated):**
- ReLU layers → He Normal: `W ~ N(0, sqrt(2/d_in))`
- All others → Glorot Uniform: `W ~ U(-sqrt(6/(d_in+d_out)), ...)`

**Column-major convention inside engines:** X enters as `(N x d_in)` row-major; engines transpose immediately to `(d_in x N)` for `W * A`. All `cache{L}.Z` and `cache{L}.A` are `(d_L x N)`. Final output transposed back to `(N x d_out)`.

**NNTopologySpec key-value options (all optional):**
| Option              | Values                                       | Effect                            |
| ------------------- | -------------------------------------------- | --------------------------------- |
| `NormalizationMode` | `'none'`(def)\|`'batch'`\|`'layer'`\|`'rms'` | Inserts norm into hidden layers   |
| `BlockType`         | `'plain'`(def)\|`'residual'`                 | Enables He-style skip connections |

## 5c. OceanReplay — record once, replay in MATLAB  ← NEW 2026-09-27 (G0+G1 done)

**Purpose:** understand a model without re-running it. Any computation is recorded once and replayed
in MATLAB with no model, GPU or retraining. It can be recorded locally in MATLAB, or in Python on the
cloud later (G3). The recorded computation can be a diffusion sampler, a training run, k-means, a
Transformer, a filter, or an RL episode.
- **Plan:** `Planning/System/Replay/Replay_Master_Plan.md`
- **Normative file format** (HDF5 + JSON header; the same for MATLAB and Python):
  `Planning/System/Replay/Replay_Format_Spec.md`

```text
+OceanReplay/            domain-neutral; depends on NO other Ocean* package (test-guarded)
  +Core/   Kinds (scalar vector matrix image sequence points tokens), Axis, ChannelInfo, ViewInfo,
           Sampling (all | every | everyIndex | list | window | union, on axis VALUES),
           Spec (groups x sampling + views; JSON), Recording (read-only; get = NEAREST STORED point,
           never blended; series for scalars)
  +IO/     Hdf5Writer (streaming, status partial->complete), Hdf5Reader (hyperslab reads, attribute
           normalisation), Schema.validate, extract (archive -> smaller file, same format)
  +Record/ IRecordable (THE domain contract), Recorder (pause -> record -> continue), NullRecorder,
           RngGuard, verifyInvariance
```

**Dependency rule:**
- `OceanReplay` → none;
- domain adapters (`OceanDL`, `OceanStats`, …) → `OceanReplay`;
- `OceanViz` → `OceanReplay`.

**New replay domain.** Implement `OceanReplay.Record.IRecordable`, with thin adapters living in their
domain:
- `axesInfo()` returns `Axis` objects. An axis with empty `Values` is open and is filled as the run
  visits it.
- `catalog()` returns the `ChannelInfo`s and `ViewInfo`s (explanation methods with `Compute`,
  `HowMade`, `Reference`, `InTexts`).
- `run(recorder)` does the real work and calls `recorder.point(coords, state)` at each point. `state`
  is a struct of **lazy** providers (function handles), so channels that aren't requested cost
  nothing.
- Prove it with `OceanReplay.Record.verifyInvariance`: the output is identical with and without
  recording.

**Gotchas:**
- Positional `name`/`kind` arguments must stay required, otherwise `"label"` parses as the
  `LabelsChannel` option.
- MATLAB has no logical HDF5 attributes, so `inTexts` is stored as uint8.
- String attributes are `string` or `char`, and shape (1,) or scalar, depending on the writer. Always
  read through `Hdf5Reader`.
- There are no zero-width datasets, so channels without axes have no `points`.
- `matlab -batch` garbles UTF-8 in the console only; the data is correct.

Tests: `+OceanTest/+OceanReplay` (28 G1 tests; `ToyRecordable` is the reference domain), plus the G2 tests
below.

**G2 (2026-09-27): Replay Studio + three domains.**
- **Domain adapters (thin, in their domains):**
  - `OceanDL.Generative.Inspect.DiffusionRecordable`: axis `step` is the book index i; images x_i,
    eps_phi, x_phi, mu; the Eq. 2.2.4 scalars; `result`; an `occlusion` view (forward passes only, NOT
    in Lai et al.). It uses `ReverseProcess` with the new optional `'OnStep'` callback.
  - `OceanStats.Unsupervised.KMeansRecordable`: an open `iteration` axis; data and centroids as
    `points`; `assignment` categorical; `sse`; `changed`. It uses `KMeans.fit`'s new optional
    `'OnIteration'` observer.
  - `OceanViz.Visualizers.NNCheckpointRecordable`: converts a legacy NNReplayEngine checkpoint; the
    prediction maps come from `DecisionBoundaryRenderer.predictMap`, the renderer's own code, which
    was extracted from `render`.
- **GUI** `OceanViz.App.Replay`:
  - `ReplayStudio(file, 'Visible', 'Layout', 'Theme')`, with `setCoord(axis, value)` and
    `setLayout(name)`;
  - `+Renderers` (one per kind; the `Registry` is the only kind → drawing map);
  - `+Layouts`: `DefaultLayout` (any recording), `DiffusionLayout` (Eq. 2.2.4–2.2.5 via the shared
    `ReverseProcess.equationText`), `ClusteringLayout` (ESL Algorithm 14.1). A layout can name its
    timeline scalar. `meta.layout` picks the layout.
- **Shared widgets:** `ImageView`, `Sidebar` and `JobRunner` now live in `OceanViz.App.Common`.
- **Launchers:** `Experiments/Scripts/Replay/exp_replay_record_{diffusion,kmeans}.m` and
  `exp_replay_convert_mandelbrot.m`, writing to `Experiments/Scripts/Replay/Results/`.

**G3–G5 (2026-09-27).**
- **Python twin `Python/ocean_replay`** (same format; tests use `unittest` only):
  - `adapters/diffusers_ddpm.py`, with the same channel names as `DiffusionRecordable`;
  - `cloud/record_ddpm.py` and `cloud/cloudshell_record.sh`, which record the trained AffectNet 64×64
    DDPM in Cloud Shell from its checkpoint, with no training.
- **MATLAB ↔ Python parity:** `OceanTest.OceanReplay.CrossLanguageTest` and `ocean_replay/tests/test_cross_language.py`
  (fixtures in `+OceanTest/+OceanReplay/fixtures`, regenerated by
  `OceanTest.OceanReplay.makeCrossLanguageFixtures` and `python -m ocean_replay.tests.make_fixtures`).
  Empty string-list attributes must be written as an HDF5 null attribute (`h5py.Empty`); MATLAB cannot read
  zero-length string arrays.
- **Optional `symbolLatex` channel attribute:** Replay Studio renders notation with the LaTeX
  interpreter (`OceanViz.App.Replay.Notation`, with a fallback table for older files).
- **GUI design (user preferences):**
  - long explanations only behind ⓘ (details drawer);
  - notation lines, colour bars and legends always visible;
  - no "not in the textbook" badges or notes on the cards; that is stated in ⓘ;
  - the timeline shows one series with no legend box, a labelled cursor, stored points on a separate
    strip, and x in play order.
- **Model Lab:** the Reverse process tab has **Save recording…** (`saveRecording(file)`), which records
  exactly the run the tab shows (shared `setup()`), via `DiffusionRecordable`'s `'Stream'` option.
- **Signatures:** `OceanTools.generateSignatures('write', 'Packages', {...}, 'OutFile', f)` generates a
  subset, which is then merged. A full regeneration would drop packages outside its list, such as
  OceanDL. Recipes: `docs/recipes/replay_recipes.md`.

## 5d. OceanNLP.Speech — speech features and ASR  ← NEW (2026-10-07)

Native implementation of J&M Ch. 14–15. Plan `Planning/NLP/Speech_and_ASR_System_Implementation_Plan.md`
(living log §9), frozen contracts `Planning/NLP/Speech_ASR_Phase0_Contracts.md`, recipe `docs/recipes/speech_asr.md`.
```
+OceanMath/+Signal/   FrameSignal (frame/window/numFrames), ResampleSignal (rational polyphase, Kaiser sinc),
                      MelFilterBank (htk|slaney, librosa parity), CepstralTransform (DCT-II, lifter),
                      SpectralAnalyzer.framePowerSpectrum
+OceanNLP/+Speech/
  +Core/        Waveform (S x C, fs, History), FrameGrid (time convention: start (t-1)H-P, centre (s+(L-1)/2)/fs),
                FeatureSequence (F x T + times/Hz/units/config), Transcript (no invented timestamps), IFeatureTransform
  +Features/    LogMel, MFCC, DeltaFeatures (HTK regression), EnergyContour, PitchContour (YIN), FeaturePipeline
  +Phonetics/   PhonemeInventory (ARPAbet<->IPA, CMUdict digit lexicon), ArticulatoryFeatures
  +Data/        SpeechManifest (fromFSDD, canonicalFSDDSplit: test speaker theo), UtteranceComposer, TranscriptNormalizer,
                SpeechVocabulary (1 blank 2 pad 3 bos 4 eos 5 unk; "|" word boundary), AudioDataset, SpeechBatcher
                (CBT + Mask + Lengths), AudioAugment
  +Models/      ISpeechRecognizer (forward/loss/transcribe/config), RecognizerFactory (fromConfig; builtin tables +
                register()), CTCRecognizer, EncoderDecoderRecognizer;
                +Encoders/ ConvEncoder, RecurrentEncoder, ConformerEncoder; +Pretrain/ UnitCodebook, HuBERTModel,
                MaskedUnitTrainer
  +Decoding/    CTCPrefixBeamDecoder (+TokenNGramLM shallow fusion), AttentionBeamDecoder
  +Training/    SpeechTrainer, SpeechCheckpoint (bitwise exact resume)
  +Evaluation/  EditAlignment, WordErrorRate, CharacterErrorRate (== jiwer 4), CorpusEvaluator
  +Inspect/     SpeechTrace, SpeechProcessAdapter (-> OceanNLP.Inspect.MemoryReader)
+OceanDL/       nn.Conv1d (+Functions.Conv1d, groups/depthwise), Layout.SequenceMask, Recurrent.CTCBeamSearch,
                Transformer.ConformerBlock, Transformer.DecoderBlock
+OceanViz/      Strategies.Speech.* registered in Core.plot: SpeechWaveform, Spectrogram, MelFilterBank, SpeechFeatures,
                PitchContour, CTCPosterior, BeamCandidates, ErrorAlignment;  App.NLP.SpeechExplorer
```
Extension: new feature = implement `IFeatureTransform` (+ add its config case in `RecognizerFactory`); new encoder =
`[H,lens]=forward(X,lens)`, `outputLength`, `InputDim/OutputDim`, `config`, static `fromConfig` + one line in
`RecognizerFactory.builtinEncoders`; new recogniser = subclass `ISpeechRecognizer` + `builtinRecognizers` line.
Corpus: MathWorks-hosted FSDD in `Data/Datasets/Speech/FSDD` (CC BY-SA 4.0). Reference libraries (librosa, jiwer,
torch, Audio Toolbox) are used only in tests/validation scripts.

---

## 6. OceanData — Data Lifecycle Layer  ← NEW (2026-03-10)

Top-level package. Covers the **entire data journey** from raw source ingestion to ML-ready features and BI outputs. Deliberately separate from `+OceanStats` (which only contains statistical algorithms).

```
+OceanData/
  +Generators/     ← Synthetic datasets. IDataGenerator (Template Method).
                      Contract: generate() → [X, Y]; generateCore() (abstract).
                      Bayes Oracle (optional): hasBayesOracle(), bayesPredict(X, y), bayesDensity(X).
                      Synthetic generators (ESLMixtureGenerator) override oracle methods.
                      File-based generators (ZipGenerator) inherit default (error).
    +Shapes3D/FurnitureShapeFactory.m  ← NEW 2026-09-23: seeded procedural chairs/tables/stools as BoxAssembly (y-up, normalized); style controls topology (frame back = genus 1)
    +Sequences/AddingProblem.m  ← NEW 2026-09-23: Hochreiter-Schmidhuber adding task, canonical 2 x N x T (baseline MSE 1/6)
    +Sequences/DigitStrings.m   ← NEW 2026-09-23: compose multi-digit strings from canonical digit images (CRNN/CTC data)
  +DataArch/       ← Storage layers (DA-1 to DA-3)
+OceanPipeline/
  +Ingestion/      ← DA-4: Readers (CSV, Mat, Lake, Synthetic, Streaming stub)
  +Transforms/     ← DA-5: Quality transforms [planned]
  +Orchestration/  ← DA-6: DataPipeline + MLDataAdapter [planned]
```

> **Note:** `IPreprocessor`, `FourierFeatureMap`, and `PolynomialMapper` live in `+OceanStats/+Preprocessing/`,  
> NOT in `+OceanPipeline/+Transforms/`. They are stat concepts, not pipeline-orchestration concerns.

### 6.1 +DataArch — Storage Layer

**Design principle:** Every store implements `IDataStore` but enforces different contracts on write.

| Class            | Schema           | Purpose                                                             | Status |
| ---------------- | ---------------- | ------------------------------------------------------------------- | ------ |
| `IDataStore`     | —                | Abstract base: put/get/list/exists/delete                           | ✅ DA-1 |
| `DataLake`       | On READ          | Raw, partitioned .mat storage. No validation on write.              | ✅ DA-1 |
| `DWSchema`       | —                | Column spec value object.                                           | ✅ DA-2 |
| `SchemaRegistry` | —                | Column type + constraint registry.                                  | ✅ DA-2 |
| `DataWarehouse`  | On WRITE         | Enforces DWSchema on insert. Filtered reads. Bloom-filter dedup.    | ✅ DA-2 |
| `BloomFilter`    | —                | Probabilistic O(1) membership test. SHA-256 multi-hash. Auto-sized. | ✅ DA-2 |
| `MDMStore`       | Canonical entity | Microscopic golden record store. merge/resolve per entity ID.       | ✅ DA-2 |
| `FeatureStore`   | Feature vectors  | Versioned, named, disk-backed feature matrices. getOrCompute.       | ✅ DA-3 |
| `DataSplitter`   | —                | holdout/kfold/stratified index splitting. Zero-overlap asserted.    | ✅ DA-3 |

**Adapters** (Database backends, same API as above):

| Class               | Backed By      | Status |
| ------------------- | -------------- | ------ |
| `DbConfig`          | —              | ✅ DA-2 |
| `IDbAdapter`        | —              | ✅ DA-2 |
| `PostgresAdapter`   | PostgreSQL     | ✅ DA-2 |
| `PostgresWarehouse` | PostgreSQL     | ✅ DA-2 |
| `PostgresMDMStore`  | PostgreSQL     | ✅ DA-2 |
| `RedisConfig`       | —              | ✅ DA-3 |
| `RedisAdapter`      | Jedis jar      | ✅ DA-3 |
| `RedisCacheStore`   | Redis          | ✅ DA-3 |
| `FeatureCache`      | Redis          | ✅ DA-3 |
| `ModelCache`        | Redis          | ✅ DA-3 |
| `CachingWarehouse`  | Redis+Postgres | ✅ DA-3 |

**IDataStore contract:**
```matlab
put(key, data, meta)   % write (overwrite silently)
get(key)               % → struct{data, meta, timestamp, key}
list()                 % → (1×N) cell of key strings
exists(key)            % → logical
delete(key)            % idempotent
describe()             % default: prints class + key listing
```
Error ID convention: `'OceanData:DataArch:<Class>:KeyNotFound'`

**DataLake specifics:**
```matlab
lake = OceanData.DataArch.DataLake('rootDir', './lake/');
lake.put('raw/demo', X, struct('source','csv'));   % no validation
entry = lake.get('raw/demo');                       % schema inferred by caller
lake.list()    % scans **/*.mat, returns forward-slash keys
```
- Keys use `/` separator → mapped to OS subdirectories on disk
- Partition format: `<rootDir>/<key>.mat` containing `struct(data, meta, timestamp, key)`
- Tests: `Tests/da1_fulltest.m` (10/10 PASS)

**Storage flow:**
```
Source → ODS → [DataLake (raw)] → re-process → [DataWarehouse (BI)]
                                 → feature-eng → [FeatureStore (ML)]
                                 → entity-res  → [MDMStore (C360)]
```

### 6.2 +OceanPipeline/+Ingestion — DA-4 ✅ DONE

Unifies all data sources under a single `IDataReader` contract producing identical `{data, meta}` structs.

| Class                 | Source                       | Key Feature                                         |
| --------------------- | ---------------------------- | --------------------------------------------------- |
| `IDataReader`         | —                            | Abstract: read(), getSource(), buildMeta            |
| `CsvFileReader`       | CSV file                     | Auto type inference, NaN handling                   |
| `MatFileReader`       | .mat file                    | Named var or auto-detect numeric                    |
| `DataLakeReader`      | DataLake.get(key)            | Preserves lake provenance metadata                  |
| `SyntheticReader`     | DataFactory / IDataGenerator | Wraps OceanStats generators                         |
| `StreamingReader`     | —                            | STUB — throws NotImplemented                        |
| `TransactionalReader` | Relational CSV / flat basket | REFACTORED 2026-04-15: Generic column-based config (PrimaryFile, TransactionCol, ItemCol, EnrichFile, LookupFile). Works with any transactional dataset, not limited to Instacart. Chunked streaming for large files. |

**Output contract (all readers):**
```matlab
result.data              % (N × p) double
result.meta.nSamples     % N
result.meta.nFeatures    % p
result.meta.source       % URI string (e.g. 'datalake://raw/spam/2026')
result.meta.header       % (1×p) cell of column names
result.meta.timestamp    % datetime
```
- Tests: `Tests/test_da4.m`

### 6.3 +OceanPipeline/+Transforms — DA-5 ✅ DONE

Data quality: all transforms follow `fit(X_train)→state` / `apply(X, state)→X'` anti-leakage discipline.

| Class                 | Operation     | Key Behaviour                                   |
| :-------------------- | :------------ | :---------------------------------------------- |
| `IDataTransform`      | Abstract      | `fit/apply` contract + `fitApply()` convenience |
| `MissingValueImputer` | Fill NaN      | `mean`/`median`/`zero`/`drop` strategies        |
| `OutlierClipper`      | Clip extremes | `sigma` (N-std) or `iqr` mode                   |
| `DataValidator`       | Audit schema  | Pass-through + `ValidationReport` struct        |
| `DataMasker`          | Privacy       | Column drop + Gaussian noise injection          |

**Anti-leakage rule:** `fit()` only ever sees training data. `apply()` is a pure `(data, state)` function.
- Tests: `Tests/test_da5.m`

### 6.4 +OceanStats/+Data/PredictiveImputer — DA-5.5 ✅ DONE

Model-based stochastic imputer (MICE + Predictive Mean Matching). Lives in `+OceanStats` so it remains a pure algorithm with **no pipeline dependency**.

| Component           | Location                      | Role                                          |
| :------------------ | :---------------------------- | :-------------------------------------------- |
| `PredictiveImputer` | `+OceanStats/+Data/`          | Full MICE + PMM algorithm (OLS or CART)       |
| `ModelImputer`      | `+OceanPipeline/+Transforms/` | Thin adapter — delegates to PredictiveImputer |

**Dependency arrow:** `+OceanPipeline` → `+OceanStats` (one way only, never circular).
- Tests: `Tests/test_da55.m`

### 6.5 +OceanPipeline/+Orchestration — DA-6 ✅ DONE

Wires all DA layers into a single reusable, configurable pipeline.

| Class           | Role                                                                                     |
| :-------------- | :--------------------------------------------------------------------------------------- |
| `DataPipeline`  | Chain reader + N transforms. `fit(train)` / `run(data)`                                  |
| `MLDataAdapter` | Produces `{X_tr, X_te, Y_tr, Y_te}` for holdout or k-fold. Optional `FeatureStore` cache |
| `BIAdapter`     | STUB — `export/push/schedule` reserved for DA-7                                          |

**Full end-to-end usage:**
```matlab
pipeline = OceanPipeline.Orchestration.DataPipeline( ...
    OceanPipeline.Ingestion.CsvFileReader('spam.csv'), ...
    OceanPipeline.Transforms.MissingValueImputer('Strategy','mean'), ...
    OceanPipeline.Transforms.OutlierClipper('Mode','iqr'));
pipeline.fit(X_raw);

adapter = OceanPipeline.Orchestration.MLDataAdapter(pipeline, ...
    OceanData.DataArch.DataSplitter('Strategy','kfold','K',5));
pkg = adapter.prepare(X_raw, Y);
% → pkg.folds{k}.X_train / X_test / Y_train / Y_test
```
- Tests: `Tests/test_da6.m`

```
Core.m              ← Central hub: theme management + plot() factory dispatcher
VizFactory.m        ← Factory helper
Themes.json         ← Dark/Scientific theme color palettes

+Interfaces/        IVizStrategy (abstract: plot(ax,data,options)), IProjection
+Strategies/        BaseStrategy + 11 concrete: Regression, Distribution, Surface,
                    SurfaceManifoldStrategy, DualMetrics, BarGroup, MultiComparison,
                    Calibration, ErrorRibbonStrategy, FeatureAudit,
                    PartialDependence2D (pre-computed 2D surface),
                    GeographicScatter (generic geo scatter — DIRECT or GRID input mode,
                      any polygon overlay; alias 'GeographicPD' kept for compat)
                    IPredictionRenderer (abstract) ← renders model global output into one ax
                    NetworkGraphRenderer     ← NNGraph + activations (classification)
                    DecisionBoundaryRenderer ← 2D probability heatmap (binary classifiers)
                    (future: RegressionSurfaceRenderer)
                    FrenetVizStrategy    <- NEW 2026-03-21: TNB arrows, curvature comb, osculating circle on IParametricCurve
                    PrototypeClassifierStrategy <- NEW 2026-03-24: decision regions, prototype markers, Bayes boundary overlay
                    SOMGridStrategy         <- NEW 2026-04-22: 2D SOM grid with jittered data-point assignments (ESL Fig 14.16 style)
+Adapters/          ISpatialMapper (abstract), GridMapper2D, EmbeddingMapper2D, GraphMapper
                    ISampleRenderer (abstract) ← bridge data sample → visual panel
                    ImageSampleRenderer      ← pixel grid (MNIST, shapes)
                    ComplexPointRenderer     ← 2D dot on complex plane (Mandelbrot, Julia)
+Projections/       Planar, Manifold
+Visualizers/       IVisualizer (abstract, Observer), ParticleVisualizer
                    IReplayLayout (abstract) ← dashboard panel arrangement
                    NNGraphLayout            ← 3-left-col + NNGraph right (MNIST/shapes)
                    NNReplayEngine           ← generic training-history animation orchestrator
+App/               InformationStudio.m (full App Designer app for interactive Markov simulation)
+Tools/             createRevolutionMovie, exploreCurves, plotParametricSurface,
                    plotParametricSurfaceWithGrid, solveAndAnimate
+Utils/             Exporter, Validator
```

**How to use OceanViz (ALWAYS use this pattern)**:
```matlab
viz = OceanViz.Core('Theme', 'Dark');   % or 'Scientific'
viz.plot(ax, 'Regression', data, 'Title', 'My Model', 'XLabel', 'x');
```

**Full simulation rendering pipeline**:
```
MarkovChain → MarkovSimulator.step() → fires StateChanged event
    → IVisualizer.onSimulatorStep_() → updateToStep(t)
    → queries SimulationHistory + ISpatialMapper → renders on UIAxes
```

---

### `+OceanStats/+NonParametric/`
*Kernel and Spline Methods*
- `KernelSmoother.m` *(IUnivariateSmoother)* - Nadaraya-Watson local kernel regression (Gaussian, Epanechnikov, Tricube).
- `SmoothingSpline.m` *(ProbabilisticModel)* - B-Spline regularised with Ridge penalty.
- `calcLoocv.m` - Efficient kernel leave-one-out cross-validation.

### `+OceanStats/+Interpolation/`  ← NEW 2026-08-18 (Ex 4.1 §4.1 Scattered Data Interpolation)
*Statistical orchestration layer for fitting, evaluation, and CV tuning of scattered interpolants.*
- `ScatterFitter.m` — Orchestrator: manages multiple `IScatteredInterpolant` models, `fitAll/evaluateAll/printSummary`.
- `InterpolationError.m` — Value object: RMSE, MAE, MaxErr, R^2. Static factories `fromFunc` (grid/random) and `fromPoints`.
- `InterpolationCV.m` — K-fold CV tuner: sweeps `rbf-epsilon/rbf-lambda/knn-k/spline-lambda`, selects best, refits on full data. `plotCurve()` for diagnostic.

## 7. OceanTest — Testing Framework

```
BaseTestCase.m          ← Abstract base (assert helpers, setup/teardown)
TestRunner.m            ← Test discovery + execution engine
+Templates/             ← Templates for new test cases
+OceanMath/+LinearAlgebra/  ← Linear algebra tests
+OceanStats/            ← LocalLDATest, demo_test_loocv, test_local_lda
```

## 8. Extension Points — How to Add New Things

### New ML Model
1. Create in `+OceanStats.+Supervised.<Category>/MyModel.m`
2. Implement `IModelStrategy`: `train`, `predict`, `evaluate`, `getComplexity`
3. Add thin wrapper in `+Selection.+Strategies/` + register in `ModelFactory`

### New Information-Theoretic Metric  
1. Create in `+OceanMath.+Information.+Strategies/MyMetric.m`
2. Implement `IMetricStrategy`: `calculate(context)`, `isCompatible(context)`, `getName()`

### New Spline / 1-D Smoother
1. Create in `+OceanStats.+NonParametric.+Strategies/MySmoother.m`
2. Implement `ISplineStrategy` (extends `IUnivariateSmoother`): `fit`, `predict`, `getDF()`
3. Drop-in compatible with `GAM.Backfitting`

### New Loss Function
1. Create in `+OceanStats.+Loss.+Regression/` or `+OceanStats.+Loss.+Classification/`
2. Implement `ILoss`: `forward(Y_true, Y_pred)`, `gradient(Y_true, Y_pred)`, `hessian(Y_true, Y_pred)`, `name()`
3. Call `obj.checkDims(Y_true, Y_pred)` at the top of each method
4. Inject into any gradient-based algorithm via its constructor

### New System Infrastructure Component
1. Create in `+OceanEngine.+<SubPackage>/MyComponent.m`
2. Components here are NOT ML models — they are execution-layer infrastructure (memory layout, histograms, pipeline orchestration)
3. `OceanEngine` may depend on `OceanStats` and `OceanMath`, but never the reverse

### +OceanEngine — System Infrastructure (NEW 2026-03-03)
```
+Interfaces/
  IQuantizer.m          ← abstract: quantize(X) → DataMatrix
  IHistogramBuilder.m   ← abstract: build(dm,g,h,idx)→H, subtract(Hp,Hl)→Hr
  ISplitFinder.m        ← abstract: findBestSplit(H, dm) → SplitResult

+Data/
  DataMatrix.m          ← Pure data container: uint8 BinnedX + BinEdges.
                           Created by IQuantizer, consumed by IHistogramBuilder.

+Quantizers/
  UniformBinQuantizer.m ← IQuantizer impl: equal-width bins [min,max]/NumBins

+Histogram/
  HistogramBuilder.m    ← IHistogramBuilder impl: O(N*p) accumulation + subtraction trick

+SplitFinders/
  HistogramSplitFinder.m ← ISplitFinder impl: O(bins*p) cumulative scan, XGBoost gain
```

### New Plot Type
1. Create in `+OceanViz.+Strategies/MyStrategy.m`, implement `IVizStrategy`
2. Add `case 'MyStrategy'` in `OceanViz.Core.plot()` switch block

### New Spatial Layout (for simulations)
1. Create in `+OceanViz.+Adapters/MyMapper.m`, implement `ISpatialMapper`
2. Inject into any `IVisualizer` constructor

### New Simulation
1. Extend `ISimulator` in `+OceanMath.+Simulation/`, implement `step()` and `reset()`
2. Call `notifyStateChange_()` inside `step()` — all existing visualizers auto-connect

### New Parametric Curve  ← NEW (2026-03-19)
1. Create in `+OceanMath.+Geometry/MyCurve.m`, extend `IParametricCurve`
2. Implement: `evaluate(t)`, `tangent(t)`, `secondDerivative(t)` — all pure math, no built-ins
3. Arc length, parameterization, and FrenetFrame support are inherited automatically
4. Existing curves: `CatmullRomSpline` (centripetal, alpha=0.5)

### New Replay Experiment (new dataset type)
1. **Sample Renderer**: Create `+OceanViz.+Adapters/MySampleRenderer.m`, implement `ISampleRenderer`
   - Implement `setup(ax, config, colors)`, `render(ax, frame, ep)`, `getTitle(frame, ep)`
   - `frame` struct contains `.X` (input), `.Y` (label), `.A1` (hidden activations), `.A2` (output)
2. **Prediction Renderer** (optional, if you need custom output viz): Create `+OceanViz.+Strategies/MyPredRenderer.m`, implement `IPredictionRenderer`
3. **Layout** (optional): Create `+OceanViz.+Visualizers/MyLayout.m`, implement `IReplayLayout`
4. **Training script**: Save checkpoint with standard fields: `weightHistory`, `costHistory`, `accHistory`, `layerSizes`, `activationFns`, `demoX`, `demoY`, `demoA1`, `demoA2`, `numEpochs`, `nSamples`
5. **Experiment launcher**: Create `exp_<name>_replay.m` — just configure and call `NNReplayEngine.run()`

**NNTrainingCheckpoint standard fields:**
```matlab
ck.weightHistory    % {numEpochs x 1} flat weight vectors per epoch
ck.costHistory      % [1 x numEpochs]
ck.accHistory       % [1 x numEpochs]
ck.layerSizes       % [d1, d2, ..., dL]  — required by NNReplayEngine
ck.activationFns    % {'linear','tanh','softmax'} — required by NNReplayEngine
ck.demoX            % [INPUT_DIM x numEpochs] — pre-computed demo sample pixels
ck.demoY            % [1 x numEpochs] — true labels (0-indexed)
ck.demoA1           % [HID_DIM x numEpochs] — hidden activations
ck.demoA2           % [OUT_DIM x numEpochs] — output activations
ck.numEpochs        % scalar
ck.nSamples         % scalar
```

---

## 9. Naming Conventions

| Convention         | Rule                                                         |
| ------------------ | ------------------------------------------------------------ |
| Interface prefix   | All abstract classes start with `I` (e.g., `IModelStrategy`) |
| Private properties | Trailing `_` (e.g., `Process_`, `Mapper_`)                   |
| Protected methods  | Trailing `_` (e.g., `notifyStateChange_()`)                  |
| Experiment scripts | `exp_` prefix (e.g., `exp_hme_classification_3d.m`)          |
| Tests              | `test_` prefix                                               |
| Event args         | `EventArgs` suffix (e.g., `StateChangedEventArgs`)           |
| Error IDs          | `'Package:SubPackage:Class:ErrorType'`                       |

---

## 10. Critical Rules for AI Agents

1. **`handle` everywhere** — All classes inherit `handle`. All mutations are in-place; objects are reference-typed.
2. **`arguments` blocks** — Methods use MATLAB `arguments` for validation. Named options use `'Name', value` syntax.
3. **`+Selection.+Strategies` are thin wrappers** — `LinearReg.m`, `DecisionTree.m` etc. there are NOT the full implementations. They adapt `+Supervised.*` to `IModelStrategy` for the selection/audit infrastructure.
4. **`InformationEngine` is the orchestrator** for information theory — don't call strategies directly in experiments.
5. **`OceanViz.Core` is the single viz entry point** — never instantiate `+Strategies` directly in experiments.
6. **`DataFactory` is the single dataset entry point** — `OceanStats.Data.DataFactory.load('name')`.
7. **Experiment scripts are ephemeral** — all reusable logic must live in packages.
8. **`+OceanStats.+Boosting`** — sequential boosting algorithms. `IBoostingStrategy` is the extension point; plug in any `IModelStrategy` as the weak learner. Planned next: `AdaBoostM2`, `GradientBoosting`, `LogitBoost`.
9. **3-Pillar Metadata & Symbol Interpretation Dictionary (`Symbols`)** — All algorithm and component classes implementing `getMetadata()` MUST provide both `SymbolicLatex` (closed-form LaTeX formula) and `Symbols` (an array of struct objects mapping each LaTeX symbol token to its plain-language interpretation, e.g. `[struct('Symbol', 'V_n', 'Description', '...'), ...]`) to assist UI visualizers and static component extractors (`component_extractor.js`).

---

## 11. Computer Vision Experiments

| Script | Exercise | Topic | Key Classes Used |
| ------ | -------- | ----- | ---------------- |
| `Experiments/Scripts/Computer Vision/exp_local_laplacian.m` | Ex 3.20 | Local Laplacian contrast manipulation (detail enhancement, denoising, tone mapping) | `LocalLaplacianFilter`, `OceanViz.Core` |
| `Experiments/Scripts/Computer Vision/exp_wavelet_compression.m` | Ex 3.21 | Wavelet construction & compression entropy benchmark. Rate-distortion sweep (9 q-steps) comparing Laplacian Pyramid (overcomplete), Haar DWT, and CDF 5/3 DWT. Per-band entropy audit at q=1/4 gray level. CDF 5/3 operates in integer [0,255] space to satisfy lifting contract. | `LaplacianPyramid`, `LiftingScheme`, `OceanMath.Information.Entropy`, `OceanViz.Core` |
| `Experiments/Scripts/Computer Vision/exp_parametric_warping.m` | Ex 3.22 | Parametric image warping, MIP-mapping, and anisotropic filtering. Benchmarks round-trip PSNR of Nearest vs Bilinear vs Bicubic interpolation under affine warp. Performs Gaussian MIP-map level selection analysis (scale 0.35x), showcasing aliasing (finer level) vs blurring (coarser level) vs trilinear blending. Performs tilted perspective floor mapping comparing Nearest vs Bilinear vs Trilinear MIP vs EWA Anisotropic filtering. | `MipMap`, `EWASampler`, `Transform2D`, `OceanViz.Core` |
| `Experiments/Scripts/Computer Vision/exp_local_warping.m` | Ex 3.23 | Local image warping and mesh deformation. Shows Piecewise Affine mesh warping (C0 barycentric mapping over Delaunay triangles) and swappable RBF / Thin Plate Spline interpolations (TPS, Gaussian, MQ, IDW) under control point drag operations. | `PiecewiseAffineWarp`, `ThinPlateSpline`, `OceanViz.Core` |
| `Experiments/Scripts/Computer Vision/exp_collage_editor.m` | Ex 3.26 | 2D image collage editor. Programmatically composites landscape, similarity, and perspective image layers onto a single background canvas via homographies. | `PiecewiseAffineWarp`, `OceanViz.Core` |
| `Experiments/Scripts/Computer Vision/exp_texture_viewer.m` | Ex 3.27 | 3D texture-mapped viewer. Projects a 3D unit cube and rasterizes image textures onto its faces under perspective camera rotation using bilinear homography mapping. | `Viewer3D`, `OceanViz.Core` |
| `Experiments/Scripts/Computer Vision/exp_image_denoising.m` | Ex 3.28 | Image denoising benchmark. Evaluates Box, Median, Gaussian, Bilateral, and NLM filters under sweeps on AWGN and Salt & Pepper noise, generating PSNR/SSIM sensitivity curves. | `MedianFilter`, `BilateralFilter`, `NonLocalMeansFilter`, `OceanViz.Core` |
| `Experiments/Scripts/Computer Vision/exp_rainbow_enhancer.m` | Ex 3.29 | Rainbow separation and enhancement. Performs polar unrolling, radial background polynomial fitting, and signal reconstruction. Launches interactive GUI for manual arc delineation and real-time radial profile enhancement. | `RainbowEnhancer`, `OceanViz.Core` |
| `Experiments/Scripts/Computer Vision/exp_mesh_rcnn_geometry.m` | Fig 5.54 (Phase 1) | Mesh R-CNN geometry blocks. Procedural furniture (exact box unions) -> voxels -> Cubify; sweeps G = 8..48 measuring volume error vs EXACT union volume and genus vs true genus for majority vs any-subsample voxelisation (3^3 probes). | `FurnitureShapeFactory`, `BoxAssembly`, `Voxelizer`, `Cubify`, `TriMesh`, `IcoSphere`, `OceanViz.Core` (`TriMesh` strategy) |
| `Experiments/Scripts/Computer Vision/exp_rnn_sequential_digits.m` | Goodfellow Ch. 10 (CV) | USPS digits read row-by-row: Elman / Leaky / GRU / LSTM / BiGRU / Echo-state (ridge readout) vs ReNet2D and a small CNN; per-epoch test accuracy and ||dL/dh_t|| over t. Shows the explicit import contract (UspsDigitsReader -> DimFormat.fromRowMajor -> toCanonical -> validate). | `+Recurrent/*`, `DimFormat`, `UspsDigitsReader`, `nn.*`, `ModuleOptimizer` |
| `Experiments/Scripts/Computer Vision/exp_rnn_long_dependencies.m` | Goodfellow 10.7-10.10 | Adding problem T=10..120 (MSE vs 1/6 baseline, ESN ridge reference), gradient reaching step 1 at init, pixel-by-pixel USPS (T=256). | `+Recurrent/*`, `AddingProblem` |
| `Experiments/Scripts/Computer Vision/exp_crnn_digit_strings.m` | CV sequence labelling | CRNN + CTC reads 3-5 digit USPS strings; CER / exact-string accuracy, BiGRU vs GRU, learned CTC alignment. | `CRNN`, `F.ctcLoss`, `CTCDecoder`, `DigitStrings` |
| `Experiments/Scripts/Computer Vision/exp_esn_reservoir_sweep.m` | Goodfellow 10.8 | Echo-state grid over spectral radius / input scale / leak / readout (final, mean, all states) / ridge lambda, validation-selected, test once. | `EchoStateCell` |



## Selectable Python RL objectives

`ocean_rl.objectives` registers stateless reward strategies. Minimum-time remains
backward compatible; geometric-distance charges physical travel to collision or
finish boundary, excluding reset teleportation, with gamma=1. Environment clones,
JSON/Tiled settings, CLI and reports preserve the objective. Evaluation reports
physical distance alongside steps and crashes. See the Python README for extension
contracts and successful-episode limitations.
