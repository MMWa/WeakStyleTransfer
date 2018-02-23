# Weak Style Transfer using MobileNet
Transfer colour style from any image.

## Details
MobileNet is used because it runs much faster than VGG16, however it preforms badly at style transfer tasks.
This implementation works by heavily applying a style image onto a content image, then the resultant image is applied onto the content image again.
This allows the main features of the content image to be preserved.

This proves very useful when dealing with large images, ie 1200x1200 image would run for 80 iterations in 90 seconds, on a GeForce 940MX graphics card.  

## Dependencies
* Tensorflow
* NumPy
* SciPy - for applying gaussian smoothing on output
* TQDM - for the nice progress bar
