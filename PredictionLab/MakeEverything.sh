#!/usr/bin/env bash

# Use the two following statements on the terminal to train on the Savchenko model, or our own.
#export MODEL_IS_SAVCHENKO=1
#unset MODEL_IS_SAVCHENKO

if test $MODEL_IS_SAVCHENKO
then
  echo "Variable MODEL_IS_SAVCHENKO is set. Training only Savchenko models."
  MODELS=/Users/fanu01-admin/Nextcloud/Documents/Projects/EASIER/EkmanClassifierSharedData/models/ASavchenko-ENet-b0_7
else
  MODELS="/Users/fanu01-admin/Nextcloud/Documents/Projects/EASIER/EkmanClassifierSharedData/models/ResNet50-AffNet-nopreproc-210921 \
   /Users/fanu01-admin/Nextcloud/Documents/Projects/EASIER/EkmanClassifierSharedData/models/MobileNet-Affnet-CropRot-220623 \
   /Users/fanu01-admin/Nextcloud/Documents/Projects/EASIER/EkmanClassifierSharedData/models/ResNet50-AfNet-CropRot-220701"
fi

make_count=0

for IMG_PREPROC in NoProc CropRot Crop11Rot Crop12Rot CNormCrop11Rot CNormCrop12Rot
do
  echo 11111 $IMG_PREPROC
  for MODELDIR in $MODELS
  do
    echo 2222 Model $MODELDIR
    for DATADIR in /Users/fanu01-admin/Nextcloud/Documents/Projects/EASIER/EkmanClassifierSharedData/data/EkmanReferences \
      /Users/fanu01-admin/Nextcloud/Documents/Projects/EASIER/EkmanClassifierSharedData/data/DaFEx \
      /Users/fanu01-admin/Nextcloud/Documents/Projects/EASIER/EkmanClassifierSharedData/data/FePh \
      /Users/fanu01-admin/Nextcloud/Documents/Projects/EASIER/EkmanClassifierSharedData/data/AffectNet-Val
    do
      echo 333 Data $DATADIR
      ((make_count+=1))
      echo Make number $make_count
      make test
    done
  done
done

