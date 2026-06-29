"""
Evaluation of the DINO embeddings against annotated data (to be built out).

Planned modules:
  - segmentation: plug a lightweight segmentation head on top of the (frozen) DINO
    patch embeddings and measure performance (mIoU / accuracy) on annotated tiles.
  - separability: cluster the patch/CLS embeddings and score the clustering against
    the annotations (purity / NMI / ARI) to observe how separable the classes are
    without any supervision.

Nothing here yet — this package is a placeholder created during the src/ reorg.
"""
