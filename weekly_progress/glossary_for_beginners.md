# Glossary for Beginners

Plain-English definitions, each with an everyday analogy.

### Zero-shot learning
Doing a task **without any training examples of it**. We never show the model a defective
carpet; we just describe "damaged carpet" in words.
*Analogy:* recognising a zebra the first time you see one because someone told you it's
"a horse with black and white stripes".

### CLIP
A model from OpenAI (we use the open-source **OpenCLIP** version) that turns images **and**
text into lists of numbers that can be compared directly.

### ViT (Vision Transformer)
The image half of CLIP. It cuts the image into small square **patches** (16×16 pixels for us),
so a 240×240 image becomes a 15×15 grid, and reasons about how the patches relate.
*Analogy:* reading a picture like a page of words, one tile at a time.

### Embedding
A list of numbers (a vector, e.g. 640 numbers) that captures the *meaning* of an image or a
sentence. Similar meanings give similar vectors.
*Analogy:* GPS coordinates for ideas. "Cat" and "kitten" sit close together; "cat" and
"truck" sit far apart.

### Cosine similarity
A score for how closely two embeddings point in the same direction: 1 = same, 0 = unrelated.
*Analogy:* two arrows; the smaller the angle between them, the more alike they are.

### Prompt ensembling
Instead of one sentence, we use many ("flawless carpet", "perfect carpet", "carpet without
defect"…) and average their embeddings for a more reliable "normal" vs "anomalous" reference.
*Analogy:* asking several people for directions and going with the consensus.

### Softmax
Turns raw scores into probabilities that add up to 1, e.g. normal 0.8 / anomalous 0.2.

### Sliding windows (`unfold`)
Cutting the image into many overlapping small crops so tiny defects aren't drowned out by
the rest of the image. `F.unfold` does this in PyTorch, and `F.fold` stitches results back.
*Analogy:* inspecting a carpet with a magnifying glass, square by square.

### Harmonic mean
A way of averaging that is pulled strongly toward **small** values:
`n / (1/a + 1/b + …)`. When a pixel is covered by several windows, we combine their scores this
way, so a pixel only scores high if *all* its windows agree it looks abnormal.
*Analogy:* a convoy only moves as fast as its slowest truck.

### AUROC (area under the ROC curve)
One number from 0 to 1 for how well scores separate normal from defective:
1.0 = perfect, 0.5 = coin flip. It doesn't depend on choosing a threshold.
- **Image AUROC:** is the *image* defective?
- **Pixel AUROC:** is *this pixel* part of a defect?

*Analogy:* pick one bad and one good item at random. AUROC is the chance the model gives the
bad one the higher score.

### PRO (per-region overlap)
Pixel AUROC is dominated by big defects. PRO gives **every defect region equal weight**, so a
tiny scratch counts as much as a big stain. We measure it only where false alarms stay below
30% (`fpr_limit = 0.3`).
*Analogy:* grading a student on every question equally, not by how long each answer is.

### Gradient / backpropagation
A gradient tells you: *"if I nudge this input a little, how much does the output change?"*
Backpropagation computes it for every input automatically.

### Input gradients
Gradients with respect to the **image pixels**. They show which pixels pushed the anomaly score
up. This is the basis of our Week 4 explanations (Integrated Gradients, ABIGX). That's why the
detector must be pure PyTorch with no `torch.no_grad()` inside it.
*Analogy:* a heat-sensitive camera showing which parts of the picture "made the model nervous".

### Integrated Gradients (IG) / ABIGX
Week 4 explanation methods. IG averages gradients along a path from a blank image to the real
one. ABIGX searches for the smallest change that would "repair" the defect, which shows what
is wrong.

### Seed / reproducibility
Computers use pseudo-random numbers. Fixing the **seed** makes the "random" choices identical
every run, so results can be repeated.
*Analogy:* shuffling a deck in exactly the same way every time.

### Config (YAML)
A plain text file (`configs/zeroshot.yaml`) holding every setting, so you change experiments
without editing code.

### CPU vs CUDA vs MPS
Where the maths runs:
- **CPU:** the normal processor. Works everywhere, but slow.
- **CUDA:** NVIDIA graphics cards (Windows/Linux). Very fast.
- **MPS:** Apple Silicon GPU (M1/M2/M3 Macs). Fast on Mac.

Our code picks the best available automatically (`cuda → mps → cpu`).
*Analogy:* a delivery by bicycle (CPU) vs a truck (GPU).

### Tensor
PyTorch's name for a multi-dimensional array of numbers. An image batch is a tensor of shape
`[batch, 3 colours, height, width]`, e.g. `[1, 3, 240, 240]`.

### Checkpoint (`laion400m_e31` / `e32`)
Saved model weights. `e31` and `e32` are the same CLIP model after training epochs 31 and 32
on the LAION-400M dataset. Step 10 compares them.
