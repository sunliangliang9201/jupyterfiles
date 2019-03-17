rm -rf data/*.bmp
rm -rf data/thumbnail/*
echo $1
cp ../mnist_digits_images/$1/$2/* data/
cp ../mnist_digits_images/$1/$2/* data/thumbnail/
