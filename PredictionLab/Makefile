# Root directory with the trained models
#MODELDIR=/Users/fanu01-admin/Nextcloud/Documents/Projects/EASIER/Models/ResNet50-AffNet-nopreproc-210921

ifndef DATADIR
$(error Environment variable DATADIR is not set. It must be the path to the data dir containing the "images" dir and the "labels.csv" file)
endif

ifndef MODELDIR
$(error Environment variable MODELDIR is not set. It must be the path to the model dir containing the "model.pth" file and the "config.json" file)
endif

ifndef IMG_PREPROC
$(error Environment variable IMG_PREPROC is not set. It must be one among: NoProc, CropRot, CropRotBL)
endif


# ground truth labels
LABELS=$(DATADIR)/labels.csv
NORMALIZED_LABELS=$(DATADIR)/labels-normalized.csv
DATA_LABEL_NORMALIZATION_MAP=$(DATADIR)/labels-normalization-map.json
MODEL_LABEL_NORMALIZATION_MAP=$(MODELDIR)/labels-normalization-map.json


# Source images directory
IMAGES=$(DATADIR)/images
# This directory will contain the images after pre-processing
NORMALIZED_IMAGES=$(DATADIR)/images-$(IMG_PREPROC)
STATS_FILE=$(DATADIR)/stats.txt

# Compose prediction name via model+datadir names
MODEL_NAME = $(basename $(notdir $(MODELDIR)))
DATA_NAME = $(notdir $(DATADIR))

OUT_DIR=results/$(MODEL_NAME)-$(DATA_NAME)-$(IMG_PREPROC)


help:
	@echo "Choose between:"
	@echo "make normalize - from the images folder, generates an images-normalized folder"
	@echo "make predict - generates a CSV file with predicitons for each normalized input image"
	@echo "make test - performs a comparison between the predictions and the ground truth labels (if present)"
	@echo "make video - concatenate all the normalized images into a video with the predicted label in overlay"
	@echo "make stats - computes some statistics about the dataset"
	@echo ""
	@echo "make all - shortcut to make everything"
	@echo ""
	@echo "DATADIR set to $(DATADIR)"
	@echo "MODELDIR set to $(MODELDIR)"
	@echo "IMG_PREPROC set to $(IMG_PREPROC)"
	@echo "Output will be written to dir $(OUT_DIR)"


all: normalize predict test video stats

#
# Normalizing
normalize: $(NORMALIZED_IMAGES) $(NORMALIZED_LABELS)

ifeq ($(IMG_PREPROC), CropRot)
NORMALIZATION_PARAMS=--square --rotate
endif

ifeq ($(IMG_PREPROC), Crop11Rot)
NORMALIZATION_PARAMS=--square --rotate --bbox-scale 1.1
endif

ifeq ($(IMG_PREPROC), Crop12Rot)
NORMALIZATION_PARAMS=--square --rotate --bbox-scale 1.2
endif

ifeq ($(IMG_PREPROC), CropRotBL)
NORMALIZATION_PARAMS=--square --rotate --rot_filter_bilinear
endif

ifeq ($(IMG_PREPROC), CNormCrop11Rot)
NORMALIZATION_PARAMS=--normalize-color --square --rotate --bbox-scale 1.1
endif

ifeq ($(IMG_PREPROC), CNormCrop12Rot)
NORMALIZATION_PARAMS=--normalize-color --square --rotate --bbox-scale 1.2
endif



# Normalize the images of the tested data
$(NORMALIZED_IMAGES): $(IMAGES)
	python ../Scripts/NormalizeImages.py \
		-i "$<" \
		-o "$@" \
		--tolerant \
		$(NORMALIZATION_PARAMS)
	touch "$@"

# Make sure the symbolic link to the original images is always there
# Thsi has priority over the previous rule to generate normalized images
$(IMAGES)-NoProc: $(IMAGES)
	ln -s $< $@
	touch "$@"

# Normalize the labels of the tested data
$(NORMALIZED_LABELS): $(LABELS) $(DATA_LABEL_NORMALIZATION_MAP)
	PYTHONPATH=.. python ../Scripts/NormalizeLabels.py -i "$<" -m "$(DATA_LABEL_NORMALIZATION_MAP)" -o "$@"

#
# Dataset statistics
stats: $(STATS_FILE)

$(STATS_FILE): $(NORMALIZED_LABELS)
	PYTHONPATH=.. python ../Scripts/AnalyseDataset.py -l "$<" -o "$@"


#
# Predictions
PREDICTIONS=$(OUT_DIR)/predictions.csv
PREDICTIONS_NORMALIZED=$(OUT_DIR)/predictions-normalized.csv

PREDICTION_SCRIPT=../predict.py
ifdef MODEL_IS_SAVCHENKO
PREDICTION_SCRIPT=../predict_savchenko_enet.py
endif

predict: $(PREDICTIONS) $(PREDICTIONS_NORMALIZED)

$(PREDICTIONS): $(NORMALIZED_IMAGES) $(MODEL) $(CONFIG) | $(OUT_DIR)
	python $(PREDICTION_SCRIPT) -p -m "$(MODELDIR)" -i "$<" -o "$@"

$(PREDICTIONS_NORMALIZED): $(PREDICTIONS) $(MODEL_LABEL_NORMALIZATION_MAP) | $(OUT_DIR)
	PYTHONPATH=.. python ../Scripts/NormalizeLabels.py -i "$<" -m "$(MODEL_LABEL_NORMALIZATION_MAP)" -o "$@"

$(OUT_DIR):
	@echo "Creating dir '$@'"
	mkdir -p $@


#
# The demo videos
VIDEO_FILE=$(OUT_DIR)/composite_video.mp4
video: $(VIDEO_FILE)

ifneq ("$(wildcard $(NORMALIZED_LABELS))","")
    VIDEO_EXTRA_OPT = -l $(NORMALIZED_LABELS)
else
    VIDEO_EXTRA_OPT =
endif

$(VIDEO_FILE): $(PREDICTIONS_NORMALIZED) $(NORMALIZED_IMAGES)
	python ../Scripts/ComposeLabeledVideo.py -p $< --dir $(NORMALIZED_IMAGES) $(VIDEO_EXTRA_OPT) -o $@


#
# Test
TEST_RESULTS=$(OUT_DIR)/test_results.csv
test: $(TEST_RESULTS)

$(TEST_RESULTS): $(PREDICTIONS_NORMALIZED) $(NORMALIZED_LABELS)
	python ../test_csv.py -p $(PREDICTIONS_NORMALIZED) -t $(NORMALIZED_LABELS) -o $@
