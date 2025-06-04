Folder to generate example color normalisation pictures and their histograms.


From the `FePh-sample-cropped-image.png`, you can generate the histogram with:

    python PlotColorHistogram.py -i FePh-sample-cropped-image.png -o FePh-sample-cropped-image-histogram.pdf

Which is showing the histogram of an image from FePh after cropping the face, with original colors.

From the `fePh_original.png` picture you can also generate a version with only color normalization (no rotations nor scalings).
Put the image in directory `conv_in` and run.

    cd ../Scripts
    PYTHONPATH=.. python NormalizeImages.py  -i ../ColorNormalizationExample/conv_in -o ../ColorNormalizationExample/conv_out -cn hist_eq -s -bbs 1.1

Then again create the histogram with

    python PlotColorHistogram.py -i fePh_HistNorm.png -o fePh_HistNorm-histogram.pdf
