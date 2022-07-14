# Root directory with the trained models
#MODELDIR=/Users/fanu01-admin/Nextcloud/Documents/Projects/EASIER/Models/ResNet50-AffNet-nopreproc-210921

ifndef DATADIR
$(error Environment variable DATADIR is not set. It must be the path to the data dir containing the "images" dir and the "labels.csv" file)
endif

ifndef MODELDIR
$(error Environment variable MODELDIR is not set. It must be the path to the model dir containing the "model.pth" file and the "config.json" file)
endif


# Source images directory
IMAGES=$(DATADIR)/images
# groud truth labels
LABELS=$(DATADIR)/labels.csv
NORMALIZED_LABELS=$(DATADIR)/labels-normalized.csv
DATA_LABEL_NORMALIZATION_MAP=$(DATADIR)/labels-normalization-map.json
# This directory will contain the images after pre-processing
NORMALIZED_IMAGES=$(DATADIR)/images-normalized
STATS_FILE=$(DATADIR)/stats.txt

# Compose prediction name via model+datadir names
MODEL_NAME = $(basename $(notdir $(MODELDIR)))
DATA_NAME = $(notdir $(DATADIR))
OUT_DIR=$(MODEL_NAME)-$(DATA_NAME)

MODEL_LABEL_NORMALIZATION_MAP=$(MODELDIR)/labels-normalization-map.json


all:
	@echo "Choose between:"
	@echo "make normalize - from the images folder, generates an images-normalized folder"
	@echo "make predict - generates a CSV file with predicitons for each normalized input image"
	@echo "make test - performs a comparison between the predictions and the ground truth labels (if present)"
	@echo "make video - concatenate all the normalized images into a video with the predicted label in overlay"
	@echo "make stats - computes some statistics about the dataset"
	@echo ""
	@echo "make everything - safe shortcut to make all"
	@echo ""
	@echo "DATADIR set to $(DATADIR)"
	@echo "MODELDIR set to $(MODELDIR)"
	@echo "Output will be written to dir $(OUT_DIR)"


everything: normalize predict test video stats

#
# Normalizing
normalize: $(NORMALIZED_IMAGES) $(NORMALIZED_LABELS)

# Normalize the images of the tested data
$(NORMALIZED_IMAGES): $(IMAGES)
	python ../Scripts/NormalizeImages.py \
		-i "$<" \
		-o "$@" \
		--square --rotate --tolerant
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

predict: $(PREDICTIONS) $(PREDICTIONS_NORMALIZED)

$(PREDICTIONS): $(NORMALIZED_IMAGES) $(MODEL) $(CONFIG) | $(OUT_DIR)
	python ../predict.py -p -m "$(MODELDIR)" -i "$<" -o "$@"

$(PREDICTIONS_NORMALIZED): $(PREDICTIONS) $(MODEL_LABEL_NORMALIZATION_MAP) | $(OUT_DIR)
	PYTHONPATH=.. python ../Scripts/NormalizeLabels.py -i "$<" -m "$(MODEL_LABEL_NORMALIZATION_MAP)" -o "$@"

$(OUT_DIR):
	@echo "Creating dir '$@'"
	mkdir -p $@


#
# The demo videos
VIDEO_FILE=$(OUT_DIR)/composite_video.mp4
video: $(VIDEO_FILE)

$(VIDEO_FILE): $(PREDICTIONS) $(NORMALIZED_IMAGES)
	python ../Scripts/ComposeLabeledVideo.py -p $< --dir $(NORMALIZED_IMAGES) -o $@


#
# Test
TEST_RESULTS=$(OUT_DIR)/test_results.csv
test: $(TEST_RESULTS)

$(TEST_RESULTS): $(PREDICTIONS_NORMALIZED) $(NORMALIZED_LABELS)
	python ../test_csv.py -p $(PREDICTIONS_NORMALIZED) -t $(NORMALIZED_LABELS) -o $@
