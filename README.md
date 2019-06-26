## Curved-Lane-Detector

[YOUTUBE LINK](https://www.youtube.com/watch?v=-fqk28GNcOU)

<img src = "https://user-images.githubusercontent.com/47768726/60156668-09151080-9828-11e9-9d9d-bbaf43e1fc5b.JPG" width = "90%"></img>

## 프로젝트 목적
  * OpenCv를 이용하여 곡선 차선인식을 구현합니다.
  * 주변환경 그림자, 햇빛 등으로 열악해도 차선을 추측할 수 있도록합니다.
  * 인식된 영상을 사용자에게 실시간으로 보여줍니다.
  
## 프로젝트 구성요소 및 제한요소
  * OpenCv 함수와 numpy로 이미지 처리
  * Filtering은 Hsv공간 및 Rgb공간에서의 차선 검출 및 이진화
  * 관심영역 (Resion of Interesting)을 통해 차선이 존재하는 도로 이외의 환경은 제외
  * 동영상 파일을 통한 테스팅
  * BirdEyedView로 차선을 일자로 정렬후 numpy 처리
  * Sliding Windows를 기반으로한 곡선 인식
  * 2차원 함수를 통한 차선 추측
  
  * 하나의 영상에 기준을 맞춰 제작해, 다른 영상으로 테스트 할 시에 변수조절등이 없다면 인식이 제한됨
  
  
## 구현  사양
  * PyCharm Python 3.6
  * OpenCv 3.4.6
  * numpy
  

## Sequence Diagram

<img src = "https://user-images.githubusercontent.com/47768726/60181733-d259ed00-985d-11e9-9d96-f05e6a3360a7.jpg" widtd= "90%"></img>.

* Pipeline
  * Video의 한 프레임을 받아옵니다
  * 해당 프레임을 Gaussian Blur처리를 수행합니다.
  * 4점(좌상,좌하,우상,우하)의 기하학적 변환을 통해 관심영역(ROI)를 BirdEyedView로 변환합니다.
  * HSV공간에서 노란색, BGR공간에서 흰색 차선을 검출합니다.
  * 이진화를 통해 필터링을 완료합니다.
  * FindLane : BirdEyedView 프레임에서 numpy 및 Sliding Windows을 통해, 흰색픽셀이라면 차선으로 인식하도록 합니다.
  * 계산된 픽셀을 근거로 2차원 함수의 계수를 구하고 구해진 함수를 통해 차선을 추정합니다.
  
 ## Source Code 목록
 
 ##### Methods
      
  * def Binary(frame, threshold):
  
  <img src = "https://user-images.githubusercontent.com/47768726/60193049-f6272e00-9871-11e9-8a64-a1c65f8e58a1.JPG" width= "45%" height = "45%"></img>
  <img src = "https://user-images.githubusercontent.com/47768726/60193054-f6272e00-9871-11e9-8e70-584bff9ed8b9.JPG" width= "45%" height = "45%"></img>
   <img src = "https://user-images.githubusercontent.com/47768726/60193056-f6bfc480-9871-11e9-91e0-277fb694180a.JPG" width= "45%" height = "45%"></img>


  * def Gray(frame):
  * def BGRtoHSV(frame):
  * def BGRtoHLS(frame):
  * def GaussianBlur(frame):
  * def Dilation(frame, iterations):
  * def BirdEyeView(frame):
  
  ![o1](https://user-images.githubusercontent.com/47768726/60192938-bf511800-9871-11e9-814c-3bf781361879.JPG)
![o2](https://user-images.githubusercontent.com/47768726/60192942-c1b37200-9871-11e9-82f7-fa4ff963a3b4.JPG)
![o3](https://user-images.githubusercontent.com/47768726/60192946-c2e49f00-9871-11e9-93bd-34956bceb52c.JPG)

  
  * def warpFrame(frame, perspective):
  * def LaneDetection(frame):
  * def FindLane(binaryFrame, prevFits = None):
    * def FindLaneBases():
  * def SearchLane(xCenter, PixelColor=None):
  * def probeNearby(prevFit, PixelColor):
  * def PolyFit(xPoints, yPoints):
  * def Qudratic(coeffs, pts):
  * def DrawPolygon(frame, points, color=[255,0,0], thickness = 5, isClosed = True):
  * def ReductionFrame(frame):
  
### Main Code
  ```c
print(cv2.__version__)
cap = cv2.VideoCapture('C:\\Users\\User\\PycharmProjects\\LaneDetection\\project_video.mp4')

while True:
    ret, frame = cap.read()
    if ret == True:
        temp = frame
        # Gaussian Blur
        frame = GaussianBlur(frame)
        cv2.imshow('LaneDetect', frame)
        # Bird Eyed View
        perspective = BirdEyeView(frame)[0]
        original = BirdEyeView(frame)[1]
        frame = warpFrame(frame, perspective)

        bird = frame

        # Binary Filter
        frame = LaneDetection(frame)

        frame = Dilation(frame, 5)
        bin = ReductionFrame(frame)
        cv2.imshow('Binary', bin)


        # Find Lane
        leftLaneInfo, rightLaneInfo = FindLane(frame)

        leftLaneWindows = leftLaneInfo[0]
        rightLaneWindows = rightLaneInfo[0]

        leftLaneX, leftLaneY = leftLaneInfo[1]
        rightLaneX, rightLaneY = rightLaneInfo[1]

        leftPoly = PolyFit(leftLaneX, leftLaneY)
        rightPoly = PolyFit(rightLaneX, rightLaneY)

        ys = np.array(range(frame.shape[0]), dtype=np.int32)
        xs_left = Qudratic(leftPoly, ys)
        xs_right = Qudratic(rightPoly, ys)

        searchAreaFrame = np.dstack((frame, frame, frame)) * 255


        for window in leftLaneWindows:
           cv2.rectangle(searchAreaFrame, window[0], window[1], color = [255, 0, 0], thickness = 5)

        for window in rightLaneWindows:
            cv2.rectangle(searchAreaFrame, window[0], window[1], color = [255, 0, 0], thickness = 5)

        left_points = [(x,y) for (x,y) in zip(xs_left, ys)]
        right_points = [(x,y) for (x,y) in zip(xs_right, ys)]
        all_points = [pt for pt in left_points]
        all_points.extend([pt for pt in reversed(right_points)])

        DrawPolygon(searchAreaFrame, left_points, color=[0, 0, 255], isClosed=False, thickness=15)
        DrawPolygon(searchAreaFrame, right_points, color=[0, 0, 255], isClosed=False, thickness=15)

        birdeyedview = cv2.addWeighted(bird, 1.0, searchAreaFrame, 1.0, 0.0)
        birdeyedview = ReductionFrame(birdeyedview)
        cv2.imshow('BirdEyedView', birdeyedview)

        unwarp = warpFrame(searchAreaFrame, original)

        #cv2.imshow('Unwarp', unwarp)

        final = cv2.addWeighted(temp, 0.5, unwarp, 1.0, 0.0)


        cv2.imshow('final', final)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
```
  
  
