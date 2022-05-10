# Root directory with the trained models
MODELS_DIR=/Users/fanu01-admin/Nextcloud/Documents/Projects/EASIER/Models

MODEL=$(MODELS_DIR)/ResNet50-AffNet-nopreproc-210921/ResNet50_affectnet.pth
CONFIG=$(MODELS_DIR)/ResNet50-AffNet-nopreproc-210921/config-AffectNet_chirag.json

ifndef DATADIR
$(error Environment variable DATADIR is not set. It must be the path to the data dir containing the "images" dir and the "labels.csv" file)
endif

# Source images directory
IMAGES=$(DATADIR)/images
#
LABELS=$(DATADIR)/labels.csv

# This directory will contain the images after pre-processing
NORMALIZED_IMAGES=$(DATADIR)/images-normalized

all:
	@echo "Choose between:"
	@echo "make normalize - from the images folder, generates an images-normalized folder"
	@echo "make predict - generates a CSV file with predicitons for each normalized input image"
	@echo "make test - TODO performs a comparison between the predictions and the groud truth labels (if present)"
	@echo "make video - concatenate all the normalized images into a video with the predicted label in overlay"

#
# Normalizing
normalize: $(NORMALIZED_IMAGES)

$(NORMALIZED_IMAGES): $(IMAGES)
	python ../Scripts/NormalizeImages.py \
		-i "$<" \
		-o "$@" \
		--square --rotate --tolerant
	touch "$@"

#
# Predictions
# TODO -- Compose prediction name via model+datadir names
MODEL_NAME = $(basename $(notdir $(MODEL)))
DATA_NAME = $(notdir $(DATADIR))

OUT_DIR=$(MODEL_NAME)-$(DATA_NAME)
PREDICTIONS=$(OUT_DIR)/predictions.csv

predict: $(PREDICTIONS)

$(PREDICTIONS): $(NORMALIZED_IMAGES) $(MODEL) $(CONFIG) | $(OUT_DIR)
	python ../predict.py -p -c "$(CONFIG)" -m "$(MODEL)" -i "$<" -o "$@"

$(OUT_DIR):
	@echo "Creating dir '$@'"
	mkdir -p $@


#
# The demo videos
VIDEO_FILE=$(OUT_DIR)/composite_video.mp4
video: $(VIDEO_FILE)

$(VIDEO_FILE): $(PREDICTIONS) $(NORMALIZED_IMAGES)
	python ../Scripts/ComposeLabeledVideo.py -p $< --dir $(NORMALIZED_IMAGES) -o $@
