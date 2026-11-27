# Ocean Library — Documentation

> Master table of contents for all guides, concepts, and recipes.

## Getting Started

- [OceanStorage](../+OceanStorage/README.md): from-scratch Rust storage engine, durable B+ tree, and language clients.
- [OceanStorage recipes](recipes/ocean_storage.md): build and access the same byte store from Rust, Python, Java and MATLAB.

- [RL workspace organization](../Experiments/Scripts/RL/README.md): experiment index and implementation ownership.
- [Python RL extension guide](../Experiments/Scripts/RL/Python/README.md): algorithms, environments, registry and run commands.

- [Draw racetracks with Tiled](recipes/tiled_racetrack.md): editable starter maps and direct .tmj import.

- [Imported Python racetracks](guides/racetrack_maps.md): map validation, configurable dynamics and Monte Carlo training.
- [Racetrack import recipes](recipes/racetrack_maps.md): map files, batch CLI and regression tests.

| Doc                                        | Description                              |
| ------------------------------------------ | ---------------------------------------- |
| [README](../README.md)                     | Project overview and quick start         |
| [Architecture](../.agents/architecture.md) | Full package map and interface contracts |
| [MATLAB Symbolic Workbench](recipes/symbolic_workbench.md) | Native app, named mathematical blocks, domains, imports, optional AI drafts |
| [Information Extractor symbolic notebook plan](../Planning/Information_Extractor_Phase_10_Symbolic_Notebook_Implementation_Plan.md) | Browser review notebook, guarded loopback MATLAB kernel, persistence, staleness, and acceptance scenarios |
| [Symbolic notebook execution contract](../Planning/Information_Extractor_Phase_10_Execution_Contract.md) | Frozen notebook and kernel protocol, ownership boundaries, and end-to-end proof requirements |

## Guides — "How do I do X?"

Task-oriented, step-by-step instructions for each subsystem.

| Guide                                            | Covers                                                                                 |
| ------------------------------------------------ | -------------------------------------------------------------------------------------- |
| [OceanAgents](guides/ocean_agents.md) | Agent host, swappable minds, single/joint runners, private state, reproducible streams, fail-stop runs; fixed and Beta–Bernoulli trust from evidence; world kernel with doors, time functions and an axiom guard; grid worlds built from doors, identical to the native environments; agents as mind, body and memory talking through signals; sensors and observations (the mind sees only through its body); worlds made of devices; players, factored actions and fictitious play |
| [Parametric Curves](guides/parametric-curves.md) | BezierCurve, CatmullRomSpline, BSplineCurve, NURBSCurve — choosing, creating, swapping |
| [Transit Network](guides/transit-network.md)     | Building networks, adding stations/routes/lines, querying                              |
| [Simulation](guides/simulation.md)               | CurveSimulator, TransitAgent, TransitSimulation                                        |
| [Visualization](guides/visualization.md)         | OceanViz themes, strategies, legends, FrenetVizStrategy                                |
| [Prototype Methods](guides/prototype-methods.md) | ESLMixtureGenerator K-class, LVQ1, PrototypeClassifierStrategy                         |
| [TrackDesigner Architecture](guides/TrackDesigner_Architecture.md) | Complete 4-layer MVC architecture, 6 design patterns, 3 raycasting systems, event flows |
| [QuadTree Spatial Index](guides/quadtree_guide.md) | 2D spatial index: bulk-build, findNearest, rangeQuery, queryBox                        |
| [Market Basket Analysis](guides/market_basket_analysis.md) | Full MBA pipeline: ingestion, encoding, Apriori mining, rule generation, filtering     |
| [Self-Organizing Map](guides/som.md) | SOM algorithm, training options, evaluation metrics, SOMGridStrategy visualization      |
| [Kernel Topology Ablation](guides/kernel_topology_ablation.md) | KPCA vs Spectral Clustering: matrix topology as control variable, eigengap theorem, cross-injection failure modes |
| [Sparse PCA](guides/sparse_pca.md) | SparsePCA class, alternating minimization, lambda1 sparsity control, comparison with standard PCA |
| [Procrustes Analysis](guides/procrustes.md) | Generalized Procrustes Analysis: preshaping, projection matrices, eigen-decomposed template M, per-shape SVD Procrustes alignment |
| [PageRank](guides/pagerank.md) | Network ranking algorithm, formulation using the power method and link matrix, dampening factor |
| [Random Forest](guides/random_forest.md) | RandomForest ensemble: OOB error, MDA/MDI importance, parfor parallelism, IModelStrategy adapter |
| [KD-Tree & KNN](guides/kdtree_knn.md) | Generic N-d KDTree (OceanAlgo.Spatial), KNNClassifier, distance metrics, bias-variance K sweep |
| [Source Coding](guides/source_coding.md) | Huffman coding, Shannon's theorem, expected code length, efficiency, redundancy, Kraft inequality, tree visualization |
| [Kelly Criterion & Racetrack](guides/kelly_criterion_racetrack.md) | Kelly optimal capital allocation, mutual odds, and RaceTrackRenderer visualization |
| [Information Theory Sandbox](guides/information_theory_sandbox.md) | Integrated programmatic dashboard for horse races, Kelly strategy Presets, and prefix tree explorer |
| [Computer Vision Primitives](guides/computer_vision_primitives.md) | Homogeneous 2D/3D transformations, Euler/Quat/Axis-Angle rotations, Slerp, pinhole projections, and lens distortion aberration models |
| [Pixel Transformations](guides/pixel_transforms.md) | Modular pixel transformation strategies, gamma correction, lookup tables, and histogram equalization |
| [Spatial Filtering & Lifting](guides/filtering_and_lifting.md) | Advanced linear and non-linear filtering, O(1) Integral Box filter, Bilateral Grid dimensional lifting, and Wavelet Lifting scheme |
| [3D Room Light Simulation](guides/light_simulation.md) | Simulated 3D room environment: modular shapes, ambient/point/spot lights, reflection models, and Gouraud/Flat/Ray-Trace renderers |
| [Time Series Analysis](guides/time_series.md) | ARMA processes: simulation, empirical autocorrelation estimation, Durbin-Levinson PACF, and multi-panel visualization |
| [Image Pyramids](guides/pyramids.md) | Multi-resolution representation: decimation (reduce), interpolation (expand), Gaussian and Laplacian Pyramids with perfect reconstruction |
| [Local Laplacian Filter](guides/local_laplacian.md) | Edge-preserving contrast manipulation (Ex 3.20): detail enhancement, denoising, tone mapping; fast bin-interpolated Laplacian remapping |
| [Wavelet Compression](guides/wavelet_compression.md) | Entropy-based compression benchmark (Ex 3.21): Laplacian Pyramid vs Haar vs CDF 5/3 DWT; rate-distortion sweep, per-band entropy audit, integer lifting contract |
| [Parametric Warping & Mipmapping](guides/parametric_warping.md) | Image warps (affine and perspective), interpolant comparison (PSNR), MIP-map level selection analysis (aliasing vs blurring), and EWA Anisotropic texture filtering (Ex 3.22) |
| [Directional Scaling](guides/exp_domain_scaling_direction_guide.md) | Verification of the Directional Domain Scaling Law (textbook Section 3.5.2) under non-uniform scaling |
| [Non-Separable Prefilter](guides/exp_non_separable_prefilter_guide.md) | Verification of the non-separability of 2D rotated anisotropic Gaussian prefilters using SVD analysis and benchmarks |
| [Mesh-Based Warping](guides/mesh_based_warping_guide.md) | Mesh-based and spline-based warping: PiecewiseAffineWarp, BilinearQuadWarp, and ThinPlateSpline |
| [Local Image Warping](guides/local_image_warping.md) | Ex 3.23 (all parts): Piecewise Affine, TPS, Beier-Neely Lines, B-Spline FFD, and Paintbrush Flow warping; GUI Tab 8 integration |
| [Feature-Based Morphing](guides/feature_morphing.md) | Ex 3.25: Feature-based morphing, 4-step cross-dissolve pipeline (Delaunay triangulation, intermediate skeleton, double inverse mapping, blend), and rotation chord-vs-arc artifact analysis; GUI Tab 9 integration |
| [2D Image Collage Editor](guides/collage_editor.md) | Ex 3.26: 2D image collage editor, layer rendering and Z-order control, interactive crop ROIs, and translation, rigid, similarity, affine, and perspective homography transformation systems; GUI Tab 10 integration |
| [3D Texture-Mapped Viewer](guides/texture_viewer.md) | Ex 3.27: 3D texture-mapped polygon projection and rasterization, perspective-correct homography mapping, and bilinear sampling; GUI integration |
| [Image Denoising Benchmark](guides/image_denoising.md) | Ex 3.28: Benchmark comparing Box, Median, Gaussian, Bilateral, and Non-Local Means (NLM) spatial filtering strategies under noise sweeps |
| [Rainbow Enhancer](guides/rainbow_enhancer.md) | Ex 3.29: Rainbow separation and enhancement: polar unrolling, radial background fitting, and signal reconstruction; interactive GUI |
| [Scattered Data Interpolation](guides/scattered_interpolation.md) | Ex 4.1: Scattered data interpolation via RBF (Gaussian, Multiquadric, Inverse Multiquadric, TPS), Nadaraya-Watson Kernel Regression, Delaunay Natural Neighbour, KNN, and 1D Smoothing Spline |
| [Tabular Reinforcement Learning](guides/tabular_rl_guide.md) | Ex 4.2 & Ex 4.7: Tabular Reinforcement Learning and Dynamic Programming Policy Iteration for Jack's Car Rental |
| [Gambler's Problem](guides/gamblers_problem_guide.md) | Ex 4.9: Dynamic Programming Value Iteration for the Gambler's Problem (ph = 0.25, 0.40, 0.55) |
| [Chapter 4 Simulink Workflow](guides/ch4_simulink_workflow_guide.md) | Step-by-step tutorial for manually constructing Chapter 4 continuous-discrete sampling, filter, and reconstruction workflow in the Simulink App GUI |
| [Likelihood Ratio Test](guides/likelihood_ratio_test_guide.md) | Likelihood Ratio Test (LRT) hypothesis testing strategy and dual-representation pipeline operator (Hamilton TSA Ex 5.8.5) |
| [Fisher Information Matrix](guides/fisher_information_guide.md) | Score function V, theoretical expected Fisher Info J(theta), and empirical observed Fisher Info I_n(theta_hat) |
| [Online Accumulators](guides/online_accumulators_guide.md) | Streaming statistics accumulators: Weighted IS, EMA, and Welford variance (+OceanEngine/+Accumulators) |
| [OceanNLP Text Processing](guides/ocean_nlp_text_processing.md) | Text analysis pipeline, ICharFilter, ITokenizer, ITokenFilter, IStemmer, and Analyzer integration |
| [OceanNLP RAG Pipeline](guides/rag_pipeline.md) | Editable chunking, retrieval, ANN/PQ vector search, prompt construction, generation, and evaluation |
| [NLP Process Explorer plan](../Planning/NLP/NLP_Process_Explorer_Implementation_Plan.md) | Architecture and text-first interface for inspecting NLP process steps and representations |
| [CNN Pooling & Unpooling](guides/cnn_pooling_unpooling.md) | MaxPool/AvgPool/MaxUnpool for 1D, 2D, and 3D — sequence/image/volumetric tensors, switches-based SegNet unpooling, buffer-on-self forward/backward |
| [Binance TSA Pipeline](guides/binance_tsa_pipeline.md) | End-to-end Binance REST API ingestion (Spot/Futures/OrderBook/Trades), causal regularisation, feature engineering, EDA (ADF/ACF/PACF/Spectral), anti-leakage train/val/test splits, and Parquet lake export |
| [Hamilton Spectral Analysis](guides/hamilton_spectral_analysis_binance.md) | Replication of Hamilton Chapter 6 (Figures 6.3-6.6) using real Binance BTC daily prices: non-stationary level periodograms, high-pass 1-day differencing, 30-day filter null ripples, and Parseval identity |
| [Candlestick Chart Renderer](guides/candlestick_chart.md) | Standalone performance-optimized OHLCV candlestick & volume chart renderer (vectorized O(1) handles wicks, bodies, MA7/25/99) |
| [BinanceFetcher & DifferenceSpectrumFilter](guides/binance_fetcher_diff_spectrum.md) | Windowed Binance OHLCV ingestion ('1d','3d','1w','1m') with auto-pagination; nth-order backshift-difference (1-B)^d spectrum filter with before/after Welch PSD and Parseval verification |
| [Binance Long-Cycle Analysis](guides/binance_long_cycle_analysis.md) | Robust cached weekly range ingestion, OHLCV quality and coverage audit, transparent seasonal filtering, transfer gain, and cautious four-year-cycle interpretation |
| [Hamilton Ch 6 Selective Seasonal Filtering](guides/hamilton_ch6_selective_filter.md) | Replication of Hamilton Figures 6.3 & 6.4: trend (w~0), 12m, 6m, and 3m seasonal cycles; targeted filter zero placement removing 0, 3, and 12m to isolate 6m cycle |

## Concepts — "Why is it designed this way?"

Architecture explanations for contributors and advanced users.

| Concept                                                      | Covers                                                     |
| ------------------------------------------------------------ | ---------------------------------------------------------- |
| [Transit Layer Architecture](concepts/layer-architecture.md) | The 4-layer design: Graph → Geometry → Domain → Simulation |

## Recipes — Copy-paste code snippets

| Recipe                                            | What you get                                              |
| ------------------------------------------------- | --------------------------------------------------------- |
| [Curves](recipes/curves.md)                       | Create, evaluate, compare, animate curves                 |
| [Prototype Methods](recipes/prototype-methods.md) | Mixture data, LVQ training, Bayes boundary, prototype viz |
| [Association Rules](recipes/association_rules.md) | MBA pipeline quick start, filtering, visualization, tuning |
| [SOM Training](recipes/som_training.md) | SOM training, grid visualization, U-matrix, before/after comparison |
| [Replay (record once, replay in MATLAB)](recipes/replay_recipes.md) | Replay Studio, recording any domain, Model Lab "Save recording", Google Cloud / Python recorder, extract |
| [Speech & ASR](recipes/speech_asr.md) | System diagram, workflow, FSDD corpus, features, CTC / encoder–decoder / HuBERT training, decoding, WER, inspection and Speech Explorer |
| [KPCA vs Spectral Clustering](recipes/kpca_vs_spectral.md) | RBF kernel construction, double-centering, sparse Laplacian, eigengap test, cross-injection |
| [Sparse PCA](recipes/sparse_pca.md) | Fit SparsePCA, sweep lambda1, compare loadings to standard PCA, transform new data |
| [Random Forest](recipes/random_forest_recipes.md) | Quick-start, OOB validation, importance bar chart, mtry CV sweep, ROC soft votes |
| [Huffman Coding](recipes/huffman_coding_recipes.md) | Encode distributions, inspect CodeTable metrics, InformationEngine aliases, tree + bar chart viz |
| [Kelly Criterion](recipes/kelly_criterion_recipes.md) | Compute log-optimal betting allocations and run smooth animated horse race visualizations |
| [Information Theory Sandbox](recipes/information_theory_sandbox.md) | Launch app, interact with programmatic controls, stop background timers, custom standalone layouts |
| [Computer Vision Primitives](recipes/computer_vision_recipes.md) | Homogeneous point warping, quaternion SLERP interpolation steps, projective camera frustum setups, radial/tangential distortions, and 3D room light simulation |
| [Pixel Transformations](recipes/pixel_transforms.md) | Adjusting scale/bias, gamma correction, dynamic contrast stretching, and histogram equalization |
| [Spatial Filtering & Lifting](recipes/filtering_recipes.md) | Copy-pasteable Box, Median, Bilateral, and Wavelet Lifting scheme code snippets |
| [Time Series Analysis](recipes/time_series.md) | Simulating AR(1), MA(1), and ARMA(p,q) processes, computing empirical ACF/PACF, and verifying duality |
| [Image Pyramids](recipes/pyramids_recipes.md) | Copy-pasteable Gaussian and Laplacian pyramid build, slicing, and reconstruction code snippets |
| [Wavelet Compression](recipes/wavelet_compression_recipes.md) | Rate-distortion sweep, per-band entropy audit, Haar vs CDF 5/3 quantize/reconstruct loop |
| [Parametric Warping & Mipmapping](recipes/parametric_warping_recipes.md) | Affine and perspective inverse mapping loops, MipMap level query and trilinear sampling, and EWA anisotropic rendering with Jacobian calculation |
| [Directional Scaling](recipes/exp_domain_scaling_direction_recipe.md) | Simulating directional domain scaling, manual gather warping, 1D FFT spectral analysis, and peak verification |
| [Non-Separable Prefilter](recipes/exp_non_separable_prefilter_recipe.md) | Copy-pasteable rotated Gaussian kernel construction, SVD factorization, and 2D vs. 1D benchmarking |
| [Mesh-Based Warping](recipes/mesh_based_warping_recipe.md) | Copy-pasteable PiecewiseAffineWarp, BilinearQuadWarp, and ThinPlateSpline code snippets |
| [Local Image Warping](recipes/local_image_warping_recipes.md) | Copy-pasteable PiecewiseAffineWarp, TPS, BSplineWarp, BeierNeelyWarp, and Paintbrush flow field code snippets (Ex 3.23) |
| [Feature-Based Morphing](recipes/feature_morphing_recipes.md) | Copy-pasteable PiecewiseAffineWarp, TPS, BSplineWarp, and rotation-simulation morphing code snippets (Ex 3.25) |
| [2D Image Collage Editor](recipes/collage_editor_recipes.md) | Copy-pasteable programmatic collage compositor loop (Ex 3.26) |
| [3D Texture-Mapped Viewer](recipes/texture_viewer_recipes.md) | Copy-pasteable programmatic 3D texture projection render loop (Ex 3.27) |
| [Image Denoising Benchmark](recipes/image_denoising_recipes.md) | Copy-pasteable Box, Median, Gaussian, Bilateral, and NLM filters for image denoising (Ex 3.28) |
| [Rainbow Enhancer](recipes/rainbow_enhancer_recipes.md) | Copy-pasteable circle fitting, polar unrolling, and baseline subtraction loops (Ex 3.29) |
| [Scattered Data Interpolation](recipes/scattered_interpolation_recipes.md) | Copy-pasteable RBF, Nadaraya-Watson, Delaunay, KNN, and CV tuning recipes (Ex 4.1) |
| [OceanAgents](recipes/agent_recipes.md) | Q-learning on CliffWalking through an agent, evaluation-only runs, acting from a saved table, matching pennies, isolation check, recorded failure, Beta trust store, imported scores, fixed vs adaptive trust, frozen-belief evaluation, battery world that jumps time, guard refusing by axiom, door-built grid world equal to the native one, RL door admission by probing, an agent as mind/body/memory, a mind moved to another body, an exact wave crossing, a moves-only tic-tac-toe agent, acting only on observations, cliff walking made of devices, a player that learns from the other exploits it |
| [Tabular Reinforcement Learning](recipes/tabular_rl_recipes.md) | Copy-pasteable Tabular RL Policy Iteration and OceanViz dark theme dashboard recipes |
| [Tabular Dyna](recipes/dyna_recipes.md) | Dyna-Q, inspectable real/planning updates, and prioritized sweeping composition |
| [Real-Time Dynamic Programming](recipes/rtdp_recipes.md) | Exact-model RTDP trials, bounded solving, and inspection of backups and trajectories |
| [State-Space Search & Decision-Time Planning](recipes/state_search_recipes.md) | A*, weighted A*, IDA*, beam / diverse beam, S&B 8.9 depth-limited search, 8.10 rollout, 8.11 MCTS on RL models; decision-time agent |
| [PAC Exploration: R-max & E³](recipes/pac_exploration_recipes.md) | Optimistic R-max and explicit explore-or-exploit E³ on the Chain problem; known-state planning |
| [Gambler's Problem](recipes/gamblers_problem_recipes.md) | Copy-pasteable Gambler's Problem Value Iteration sweeps and dark theme figure recipes |
| [Likelihood Surface Optimization](recipes/likelihood_optimization_recipes.md) | Copy-pasteable Steepest Ascent, Newton-Raphson, DFP, and BFGS optimization recipes |
| [Chapter 4 Simulink Recipes](recipes/ch4_simulink_workflow_recipes.md) | Workspace setup script and block parameter reference table for Simulink Chapter 4 workflow |
| [Likelihood Ratio Test](recipes/likelihood_ratio_test_recipe.m) | Copy-pasteable Likelihood Ratio Test (LRT) and dual-representation pipeline operator recipes |
| [Fisher Information Matrix](recipes/fisher_information_recipe.m) | Copy-pasteable Score V, Expected Fisher Info, and Observed Fisher Info recipes |
| [Online Accumulators](recipes/using_online_accumulators.m) | Copy-pasteable Weighted IS, Welford variance, and EMA accumulator recipes |
| [OceanNLP Text Processing](recipes/ocean_nlp_recipes.md) | Copy-pasteable recipes for text analysis, CountVectorizer, TF-IDF, and BM25 search ranking |
| [NLP Process Explorer](recipes/nlp_process_explorer.md) | Inspect actual pipeline stages in the separate text-first MATLAB app |
| [CNN Pooling & Unpooling](recipes/cnn_pooling_unpooling_recipes.md) | Copy-pasteable 1D/2D/3D MaxPool/AvgPool forward-backward, overlapping-window gradient accumulation, and SegNet-style pool/unpool sketch |
| [Binance TSA Pipeline](recipes/binance_tsa_recipes.md) | Copy-pasteable recipes for Binance REST API ingestion, feature engineering, EDA, anti-leakage splitting, and OceanViz dark theme dashboard |
| [Hamilton Spectral Analysis](recipes/hamilton_spectral_recipes.md) | Copy-pasteable recipes for FFT sample periodograms, Parseval energy conservation, theoretical AR(p) population spectra, and transfer function gains |
| [Candlestick Chart Renderer](recipes/candlestick_recipes.md) | Copy-pasteable recipes for rendering OHLCV candlestick and volume charts from timetables or structs |
| [BinanceFetcher & DifferenceSpectrumFilter](recipes/binance_fetcher_diff_spectrum_recipes.md) | Copy-pasteable recipes for windowed Binance fetch (1d/3d/1w/1m), auto-pagination, and nth-order difference spectrum filtering with Parseval verification |
| [Binance Long-Cycle Analysis](recipes/binance_long_cycle_recipes.md) | Read cached weekly Binance history and run the selective-filter pipeline with auditable frequency response |


## API Reference

Each class has inline `help` documentation via MATLAB's doc system:

```matlab
help OceanMath.Geometry.BezierCurve
help OceanMath.Transit.TransitNetwork
help OceanMath.Simulation.CurveSimulator
```

Autocomplete is available in MATLAB via `resources/functionSignatures.json`.
- [TrackRenderer Architecture & Mechanics Guide](guides/TrackRenderer_DeepDive.md)
