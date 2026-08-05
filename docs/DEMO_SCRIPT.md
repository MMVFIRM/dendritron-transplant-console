# Ninety-second demonstration script

## 0–15 seconds — establish the boundary

Point to the header:

> “Qwen2.5-0.5B was the offline donor. It is not loaded now. The live process is
> the small CPU recipient plus frozen memory assets.”

Show the donor provenance and recipient parameter cards.

## 15–35 seconds — run a diagnostic phrase

Choose **Boolean condition**:

```text
true if → the
```

Select **Correct transplanted memory** and click **Run trace**.

Point out:

- the exact phrase address;
- the retrieved value row;
- the MACSL card;
- the expected token at rank 1;
- the two selected experts;
- the two selected branches inside each expert;
- 3.04% active conditional capacity;
- donor status `OFF`.

## 35–65 seconds — causal intervention

Click **Compare all controls**.

Explain:

> “The model weights and phrase address are frozen. We are replacing only the
> representation bound to that address.”

Compare correct memory against shuffled, semantic-opposite, random, zero, and
hash-only conditions. Emphasize target probability, NLL, rank, and the retrieved
value row.

## 65–80 seconds — storage and sparsity

Scroll to the benchmark panel:

- 2,000 phrase rows;
- 5,000 definition rows;
- 2.60× smaller phrase payload than full 32D rows;
- top-2 of 12 experts;
- top-2 of 6 branches per active expert;
- 12.5% active vocabulary.

## 80–90 seconds — close with the exact claim

> “This is not a tiny Qwen chatbot. It is a mechanistic proof that a large model
> can act as an offline representation factory, while a separate sparse CPU
> architecture retrieves and uses selected transplanted state—and fails in a
> measurable way when the wrong state is transplanted.”
