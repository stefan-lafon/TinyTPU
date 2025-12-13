# TinyTPU-Compiler

TinyTPU-Compiler simulates a systolic array architecture, taking high-level JAX math and executing it on a cycle-accurate hardware model.

The goal is to visualize how linear algebra operations (like matrix multiplication) are actually executed on hardware similar to a TPU, specifically focusing on data flow, latency, and quantization effects.

## Architecture

* **Frontend**: JAX (for extracting XLA/HLO graphs).
* **Compiler**: Custom logic to tile and pad matrices for a fixed hardware grid.
* **Backend**: A cycle-accurate simulator of a weight-stationary systolic array.

Work in progress...