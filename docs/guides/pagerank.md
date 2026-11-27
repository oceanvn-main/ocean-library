# PageRank

**PageRank** is an unsupervised algorithm originally designed by Larry Page and Sergey Brin to rank web pages. It models the behavior of a "random surfer" navigating a network. In the Ocean Library, it is implemented as a standard statistical algorithm in `+OceanStats/+Unsupervised/PageRank.m`.

This implementation specifically follows the formulation from *The Elements of Statistical Learning* (ESL) Section 14.5.3.

## Overview

Given a network with $N$ nodes, let $\mathbf{L}$ be the link matrix where $L_{ij} = 1$ if node $j$ points to node $i$.
Let $c_j = \sum_i L_{ij}$ be the number of out-links from node $j$.
The PageRank vector $\mathbf{p}$ (where $\sum_i p_i = N$) is found by finding the stationary distribution of a modified Markov chain:

$$ \mathbf{p}_k \leftarrow \left[ \frac{1-d}{N} \mathbf{e}\mathbf{e}^T + d \mathbf{L} \mathbf{D}_c^{-1} \right] \mathbf{p}_{k-1} $$

The parameter $d$ is the **damping factor** (usually $0.85$), representing the probability that the surfer follows a link rather than jumping to a random page.

## Usage

```matlab
% 1. Create the link matrix L (N x N)
% L_{ij} = 1 if j -> i
L = zeros(4);
L(2, 1) = 1; % 1 -> 2
L(3, 1) = 1; % 1 -> 3
L(1, 2) = 1; % 2 -> 1
L(4, 3) = 1; % 3 -> 4
L(1, 4) = 1; % 4 -> 1

% 2. Instantiate and fit
pr = OceanStats.Unsupervised.PageRank();
p = pr.fit(L, 'DampingFactor', 0.85, 'MaxIter', 1000);

% 3. View scores
disp(p)
```

## Examples

For a detailed visual experiment demonstrating PageRank on the ESL Figure 14.47 network, run:
```matlab
run('Experiments/Scripts/Unsupervised/exp_pagerank.m')
```

## See Also
*   `OceanMath.Information.Processes.MarkovChain`
*   `OceanStats.Unsupervised.SpectralClustering`
