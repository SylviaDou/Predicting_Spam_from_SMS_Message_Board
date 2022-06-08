To run these files, first you need to download the dataset that used for this project.
You can obtain the dataset from https://www.kaggle.com/uciml/sms-spam-collection-dataset

After you download the dataset, you should put the data file and these files in the same folder.
Then you could run the analysis.py. It should take around 30 mins to run the whole file. If you
want to speed up the process, you can comment out two max_depth functions in the file because the 
functions for finding the max_depth takes a long time.

special package requirement (IMPORTANT):
In the preprocess.py we from nltk.corpus import stopwords. Since we first develop it on Ed, we
did not encounter any problems. However, when we switch to local developing, we found out that
it needs us to import nltk first and use nltk.download('stopwords'). If you run our analysis.py
and have the same problem, we suggest you could uncomment the two lines commented in the import 
part of preprocess.py. Once you have nltk.download('stopwords'), you should be good to run our file
and you may comment them out again if you want.

For testing our preprocess.py, you should run the preprocess_test.py. It will use the sample_data.csv
we provide along the files and also cse163_utils.py for using the assert_equals() function 
(credit to cse163 faculties). You should see no output after you run the test file.
