# Comprobo Machine Vision Project
*ENGR 3590: A Computational Introduction to Robotics*

*Christopher Nie*

This project is related to my PIE project, which performs the following actions:

1. Scan an image using a camera
2. Reduce Image to a line drawing
3. Draw this line drawing using a mounted paintball mechanism

The Computer vision aspect of this project is **Step 2: Reduce Image to a line drawing**. The major parts of this task can be broken down as follows: 

- [Identify the "frame of the picture"](#identify-the-frame)
- [Mask out all extraneous elements](#masking)
- [Rotate remaining elemnets](#rotation)
- [Reduce the picture to a line drawing](#picture-to-line-drawing)
- [Putting it all together](#putting-it-all-together)

## Identify the frame <a name="identify-the-frmae"></a>

### Description
Given an image that contains a canvas and a background, identify the canvas. Since the canvas can be any color, and so can the background, image thresholding is not able to accurately identify the canvas. Thus, I opted to find contours using Canny Edge Detection. 

### Methodology

The original methodology used hough line transforms. However, due to many false positive lines, I ended up settling for the Canny Edge Detection method. 

The `detect_paper` module follows the following steps: 

- Find edges using Canny Edge Detection
- Find all contours
- Identify "rectangle-like" contours
- Identify the biggest "rectangle-like" contour

#### Canny Edge Detection  

To prepare for Canny Edge Detection, we simplify the image to `grayscale`. Then, we apply `Gaussian blur` to reduce noise. 

<img src="intermediate_images/edges.png" alt="Canny Edge Detection" width="440"/>

#### Find all Contours

Next, we find the contours using `openCV`'s `findContour` method. This returns a `list` of all `contours` identified in the image. The syntax can be seen as follows: 
```Python
contours, _ = cv2.findContours(img, Mode, Method)
```
I chose to use `cv2.RETR_EXTERNAL` for the `Mode` and `cv2.CHAIN_APPROX_SIMPLE` as the `Method`. `cv2.RETR_EXTERNAL` means that the function will only return the outermost contour in any case of nesting. `cv2.CHAIN_APPROX_SIMPLE` means that the function will only return "keypoints" instead of entire lines. An illustration is shown below, with `cv2.CHAIN_APPROX_NONE` on the left and `cv2.CHAIN_APPROX_SIMPLE` on the right. 

<img src="intermediate_images/chain_approx.jpg" alt="CHAIN_APPROX" width="440"/>

> Although `cv2.CHAIN_APPROX_SIMPLE` is able to approximate here, real world images are not sharp enough to immediately approximate to a rectangle. This method is only taken to reduce data size

#### Rectangle-like Contours <a name="rectangle-like-contours"></a>

To filter out rectangle-like contours, we iterate through every `contour in contours` and approximate the shape using `cv2.approxPolyDP()`. The implimented code looks like this
```Python
arc_length = cv2.arcLength(contour, True)
approx = cv2.approxPolyDP(contour, 0.02 * arc_length, True)
```
`arc_length` is used to find the precision to which the approximated edges must look like a straight line. I found that `0.02` worked best with the test set I was using. 

`approx` is a contour that contains only the edges of the polygon that was approximated. For example, a rectangle would have 4 points, a pentagon would have 5, etc. For our case, we can simply store all contours that have 4 points into a `list` and throw out any contours that do not. 

#### Biggest Rectangle-like Contour

We iterate through our `list` of all "rectangle-like" objects and identify the biggest one through `cv2.contourArea()`. 

<img src="test_pics/Figure_4.png" alt="Biggest Rectangle-like contour" width="440"/>

## Masking <a name="masking"></a>
### Methodology

First we create white image called `mask`.

With the frame having been identified, we can create the mask using `cv2.drawContours` which takes the syntax
```Python
cv2.drawContours(img, contours, contourIdx, color, thickness)
```
We use the following values: 
- `mask` for `img`
- `max_contour` for `contours`, 
- `-1` for `contourIdx` (which draws all contours in `max_contour`), 
- `(255,255,255)` as `color` (for black)
- `-1` for `thickness` (which fills in the contour)

This draws a black filled in contour onto an otherwise white `mask`. Thus, locations to be excluded are 0, and locations that are to be included are 255. This allows us to perform a `cv2.bitwise_and` operation on the original image and the `mask`. 


### Results
Below are a couple of the images used to test the masking part of this project. Left images are the raw images, while the right images are the masked images. 

<p float="left">
  <img src="detect_paper_test_image/z13MW.jpg" width="220" />
  <img src="test_pics/Figure_11.png" width="220" /> 
</p>
<p float="left">
  <img src="detect_paper_test_image/lone_paper_slanted.png" width="220" />
  <img src="test_pics/Figure_31.png" width="220" /> 
</p>
<p float="left">
  <img src="detect_paper_test_image/envelope.jpg" width="220" />
  <img src="test_pics/Figure_41.png" width="220" /> 
</p>


## Rotation <a name="rotation"></a>
> I was able to get this part working after the in-class demo
### Description

Centering and orienting the image is an important part of the ultimate goal of painting a picture off of an image scan. 

### Methodology

We use the keypoints from `approx`, which was mentioned earlier in [Rectangle-like contours](#rectangle-like-contours). These keypoints come in the form 
$$\begin{bmatrix}
x_1 & y_1\\
x_2 & y_2\\
x_3 & y_3\\
x_4 & y_4\\
\end{bmatrix}$$

We find the slope of the keypoints by doing
$$\begin{bmatrix}
\frac{y_1-y_2}{x_1-x_2}\\
\frac{y_2-y_3}{x_2-x_3}\\
\frac{y_3-y_4}{x_3-x_4}
\end{bmatrix}$$

Then, we can take the `arctan` of each row to obtain the angles needed to transform. Since we are dealing with rectangles, we essentially get two angles that are $90\degree$ apart. We take the one with a smaller absolute value (to minimize the total rotation needed), and create a rotation matrix. 

However, before we rotate the image, we need to ensure that the image is properly centered. We can find the center of the contour by using 

```Python
M = cv2.moments(contour)
```
Again, we use the contour of the mask. `M` is a dictionary containing the moments of all contours. We use `M` to create a translation matrix
$$\begin{bmatrix}
1 & 0 & w/2 - x_1\\
0 & 1 & h/2 - y_1\\
\end{bmatrix}$$

Where $w$ and $h$ are the width and height of the image and $x_1$ and $y_1$ are the center coordinates of the mask. 

Finally, we can apply all these transformations through `cv2.warpAffine(img, Matrix, dSize)`, first translating to center the image then rotating to vertical or horizontal. 

### Results
Below are a couple of the images used to test the rotating part of this project. The left is the masked image and the right is the centered and rotated images. 
<p float="left">
  <img src="test_pics/Figure_11.png" width="220" /> 
  <img src="test_pics/Figure_111.png" width="220" /> 
</p>
<p float="left">
  <img src="test_pics/Figure_31.png" width="220" /> 
  <img src="test_pics/Figure_311.png" width="220" /> 
</p>
<p float="left">
  <img src="test_pics/Figure_41.png" width="220" /> 
  <img src="test_pics/Figure_411.png" width="220" /> 
</p>

## Picture to Line Drawing <a name="picture-to-line-drawing"></a>

- changing K
- No resizing
- No Gaussian blur

Putting it all together: IMG_3901.jpg (FINAL_FIGURE1, 2)


## Putting it All Together <a name='putting-it-all-together'></a>

### Running 
Run the file on a given image under name `filename`. 
```Bash
python3 main.py filename
```