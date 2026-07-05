# assets/ — test images

Put the images you want to run through the model here, e.g.

```
assets/
├── demo.jpg
├── scene1.png
└── receipt.png
```

Then point the CLI at one of them:

```bash
./build/penguinvl --model ./penguin-vl-2b-ncnn --image ./assets/demo.jpg \
  --prompt "Describe this image in detail." --threads 8
```

Supported formats (decoded by the vendored `stb_image.h`): JPG, PNG, BMP, TGA,
PSD, GIF, HDR, PIC — no OpenCV required. Any path works; `assets/` is just the
conventional place so commands stay short.

> Model weight files (`*.ncnn.param/bin`) are git-ignored; small test images may
> be committed if you want to share a repro.
