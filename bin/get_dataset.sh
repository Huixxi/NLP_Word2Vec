#!/bin/bash

DATASET_DIR="dataset"
mkdir -p $DATASET_DIR

cd $DATASET_DIR

# Get TED Talks DataSet
FILENAME="ted_en-20160408.zip"
if hash wget 2>/dev/null; then
wget -O $FILENAME https://wit3.fbk.eu/get.php?path=XML_releases/xml/ted_en-20160408.zip&filename=ted_en-20160408.zip
else
curl -L https://wit3.fbk.eu/get.php?path=XML_releases/xml/ted_en-20160408.zip&filename=ted_en-20160408.zip
fi
wait
unzip $FILENAME
rm $FILENAME