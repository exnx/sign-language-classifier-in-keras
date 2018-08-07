#!/bin/bin

# save subdirectors for each dataset type
cd sign-language-digits-dataset
mkdir train
mkdir valid
mkdir test

# move all class directories with images into train/
mv 0/ 1/ 2/ 3/ 4/ 5/ 6/ 7/ 8/ 9/ train/

# make class directories for valid and test data sets
cd valid
mkdir 0/ 1/ 2/ 3/ 4/ 5/ 6/ 7/ 8/ 9/

cd ../test
mkdir 0/ 1/ 2/ 3/ 4/ 5/ 6/ 7/ 8/ 9/

# loop through each class in /train and randomly remove 30 images from each class and 
# move them into the corresponding dir in /valid, and do the same with 5 images into /test
cd ../train
for ((i=0; i<=9; i++)); do
    a=$(find $i/ -type f | gshuf -n 30)
    mv $a ../valid/$i/
    b=$(find $i/ -type f | gshuf -n 5)
    mv $b ../test/$i/
done
