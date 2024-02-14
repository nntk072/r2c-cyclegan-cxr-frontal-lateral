mkdir datasets
FILE=$1

if [[ $FILE != "CheXpert"]]; then
    echo "Available dataset is: CheXpert"
    exit 1
fi

URL=https://www.kaggle.com/datasets/mimsadiislam/chexpert/download?datasetVersionNumber=1
ZIP_FILE=./datasets/$FILE.zip
TARGET_DIR=./datasets/$FILE/
wget -N $URL -O $ZIP_FILE
mkdir $TARGET_DIR
unzip $ZIP_FILE -d ./datasets/
rm $ZIP_FILE
