These held-out Android test images were copied from
`datasets/processed/combined_kaggle_v2_5class/test` after training.

Source datasets:

- Kaggle `lukex9442/indian-bovine-breeds`, CC0: Public Domain
- Kaggle `atharvadarpude/indian-buffalo-dataset`, CC0: Public Domain
- Kaggle `algsoch/breed-cattle-buffalo`, MIT

The selected samples were not part of the training/validation split. They are
used only by `BundledModelSanityTest` to verify that the production model
bundled in main assets can classify real held-out photos with the true breed in
the top-3 predictions.
