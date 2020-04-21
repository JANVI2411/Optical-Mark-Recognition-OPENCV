import cv2
import numpy as np
from opencv.OMR import utils


heightImg = 700
widthImg  = 700
questions=5
choices=5
ans= [1,2,0,2,4]


img= cv2.imread("1.jpg")
img=cv2.resize(img, (widthImg, heightImg))
imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur=cv2.GaussianBlur(imgGray,(5, 5), 1)
imgCanny=cv2.Canny(imgBlur,10,70)
imgBlank=np.zeros((widthImg,heightImg,3),np.uint8)

#Find Contours

imgContour=img.copy()
contours,hierarchy=cv2.findContours(imgCanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
cv2.drawContours(imgContour,contours,-1,(0,255,0),3)

#Find mcqbox and grade contours points (4 corner points)
rectContour=utils.get_rectangle_contour(contours)
McqBoxContourPoint=utils.get_contour_points(rectContour[0])
gradeContourPoint=utils.get_contour_points(rectContour[1])

imgBiggestRect=img.copy()
cv2.drawContours(imgBiggestRect,McqBoxContourPoint,-1,(0,255,0),15)
cv2.drawContours(imgBiggestRect,gradeContourPoint,-1,(255,0,0),15)

#crop the image ( using 4 corner points of both mcq and grid ,we will crop those portions from the image)

pt1=np.float32(McqBoxContourPoint)
pt2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
matrix=cv2.getPerspectiveTransform(pt1,pt2)
# imgWarpColoured=cv2.warpPerspective(img,matrix,(widthImg,heightImg))
imgMcqBoxDisplay=cv2.warpPerspective(img,matrix,(widthImg,heightImg))

ptG1=np.float32(gradeContourPoint)
ptG2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
matrix=cv2.getPerspectiveTransform(ptG1,ptG2)
imgGradeDisplay=cv2.warpPerspective(img,matrix,(widthImg,heightImg))


#distinguish the filled circle(paper's ticked answer)

imgWarpGray=cv2.cvtColor(imgMcqBoxDisplay,cv2.COLOR_BGR2GRAY)
imgThresh=cv2.threshold(imgWarpGray,175,255,cv2.THRESH_BINARY_INV)[1]
boxes = utils.split_boxes(imgThresh,questions,choices)


#Calculate the grade
box_pixels=np.zeros((questions,choices))
r,c=0,0
for box in boxes:
    box_pixels[r][c]=cv2.countNonZero(box)
    c+=1
    if c==choices:
        r+=1
        c=0

my_ans=[]
for box_pixel in box_pixels:
    my_ans.append(np.argmax(box_pixel))


marks=[]
i=0
for a in my_ans:
    if(a==ans[i]):
        marks.append(1)
    else:
        marks.append(0)
    i+=1

score = (sum(marks)/questions)*100


#show the correct and wrong answers on mcqbox image AND show final score on grade image

utils.show_answers(imgMcqBoxDisplay,my_ans,marks,ans,choices,questions)
# cv2.putText(imgGradeDisplay,str(int(score))+"%",(200,400),cv2.FONT_HERSHEY_COMPLEX,5,(0,0,255),3) # AD


#project the above answers on the original image

#MCQ
imgFinal=img.copy()
imgMcqBoxDisplayCopy = np.zeros_like(imgMcqBoxDisplay) # NEW BLANK IMAGE WITH WARP IMAGE SIZE
utils.show_answers(imgMcqBoxDisplayCopy,my_ans,marks, ans,choices,questions) # DRAW ON NEW IMAGE
invMatrix = cv2.getPerspectiveTransform(pt2, pt1) # INVERSE TRANSFORMATION MATRIX
imgInvWarp = cv2.warpPerspective(imgMcqBoxDisplayCopy, invMatrix, (widthImg, heightImg)) # INV IMAGE WARP
imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp, 1,0)
#Grade
imgRawGrade = np.zeros_like(imgGradeDisplay,np.uint8) # NEW BLANK IMAGE WITH GRADE AREA SIZE
cv2.putText(imgRawGrade,str(int(score))+"%",(70,100),cv2.FONT_HERSHEY_COMPLEX,3,(0,0,255),3) # ADD THE GRADE TO NEW IMAGE
invMatrixG = cv2.getPerspectiveTransform(ptG2, ptG1) # INVERSE TRANSFORMATION MATRIX
imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, invMatrixG, (widthImg, heightImg)) # INV IMAGE WARP
imgFinal = cv2.addWeighted(imgFinal, 1, imgInvGradeDisplay, 1,0)

#/////////////////////////////////////////

imageArray = ([img,imgGray,imgCanny,imgContour],
              [imgBiggestRect,imgMcqBoxDisplay,imgInvWarp,imgFinal],
              [imgBlank,imgGradeDisplay,imgInvGradeDisplay,imgGradeDisplay])
stackimage=utils.stackImages(imageArray,0.3)
cv2.imshow(" ",stackimage)
cv2.waitKey(0)

#---------------------------------------------------------------------------------

