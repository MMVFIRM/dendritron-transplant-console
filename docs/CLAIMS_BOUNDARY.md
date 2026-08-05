# Claims boundary

## Supported by this package

- The official Qwen2.5-0.5B checkpoint was used offline to produce the retained
  numerical phrase and definition assets.
- The donor checkpoint is not included and is not loaded by the demo.
- The packaged recipient runs locally on CPU.
- Exact phrase addresses retrieve frozen early and late representation values.
- The recipient exposes sparse expert, branch, recurrent, and output routing.
- Fixed-model memory substitutions can change target probability, rank, hidden
  state, and output candidates.
- The selected VM32 VIVERE/MACSL representation is smaller than full 32D rows
  while retaining similar measured fixture quality.

## Not supported

Do not describe this package as:

- “Qwen compressed into 532K parameters”;
- “Qwen running on CPU through Dendritrons”;
- “Qwen-equivalent performance”;
- “a production chatbot”;
- “a proven Transformer replacement”;
- “a native ARM runtime”;
- “a demonstrated latency improvement from VIVERE.”

The correct description is:

> A sparse recurrent CPU recipient using selected frozen representations
> extracted from a larger offline donor, with mechanistic routing and causal
> memory-binding controls.

## Benchmark interpretation

The included language measurements come from a controlled CPython-documentation
fixture and a 2,048-token recipient vocabulary. They are useful for matched
ablation and falsification, not for comparison with standard large-language
model perplexity reports.

The Gate 2C-VM32 result operates after a 896D-to-32D joint-transfer projection.
Raw-896D MACSL transfer remains unestablished in this package.
