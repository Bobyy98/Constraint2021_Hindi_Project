1. Create a virtual environment

2. Preprocess the data:
     convert xlsx files into csv
     Maps "fake" → 1, everything else → 0
   Use preprocess.py
   (preprocessing is already done so no need to do it again), csv files are present in the github repo

4. pip install tensorflow scikit-learn matplotlib pandas numpy openpyxl

5. Download fasttext hindi embeddings file: open https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.hi.300.vec.gz
6. Extract cc.hi.300.vec from cc.hi.300.vec.gz (size is approx 4.6GB)

7. Place cc.hi.300.vec in project directory

8. Run the project using:
   
   python train_cnn_lstm_hindi_fixed.py \
     --train_csv train_clean.csv \
     --val_csv val_clean.csv \
     --test_csv test_clean.csv \
     --fasttext_vec cc.hi.300.vec \
     --outdir outputs \
     --upsample

   local usage: (only for reference)
python "E:\constraint2021_implement\main3.py" `
>>   --train_csv "E:\constraint2021_implement\train_clean.csv" `
>>   --val_csv   "E:\constraint2021_implement\val_clean.csv" `
>>   --test_csv  "E:\constraint2021_implement\test_clean.csv" `
>>   --fasttext_vec "E:\constraint2021_implement\cc.hi.300.vec" `
>>   --outdir "E:\constraint2021_implement\outputs"

Project runs, yay!
