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
  * HSV공간에서 노란색, RGB공간에서 흰색 차선을 검출합니다.
  * 이진화를 통해 필터링을 완료합니다.
  * FindLane : BirdEyedView 프레임에서 numpy 및 Sliding Windows을 통해, 흰색픽셀이라면 차선으로 인식하도록 합니다.
  * 계산된 픽셀을 근거로 2차원 함수의 계수를 구하고 구해진 함수를 통해 차선을 추정합니다.
  
 ## Source Code 목록
 
 ## Methods
      
  ### def Binary(frame, threshold):
  
  <img src = "https://user-images.githubusercontent.com/47768726/60193049-f6272e00-9871-11e9-8a64-a1c65f8e58a1.JPG" width= "45%" height = "45%"></img>
  <img src = "https://user-images.githubusercontent.com/47768726/60193054-f6272e00-9871-11e9-8e70-584bff9ed8b9.JPG" width= "45%" height = "45%"></img>
   <img src = "https://user-images.githubusercontent.com/47768726/60193056-f6bfc480-9871-11e9-91e0-277fb694180a.JPG" width= "45%" height = "45%"></img>
   <img src = "https://user-images.githubusercontent.com/47768726/60196559-620c9500-9878-11e9-9715-75cad5e65a71.JPG" width= "45%" height = "45%"></img>

```
이미지를 이진화 합니다.

이진화하기위해 RGB이미지를 Gray Scale로 변환합니다.

임계값(threshold)를 기준으로 임계값 이하라면 검은색, 임계값 이상이면 흰색으로 변환합니다.
```

  ### def Gray(frame):
  
   <img src = "https://user-images.githubusercontent.com/47768726/60196279-b82d0880-9877-11e9-8761-853a9eb8d1fc.JPG" width= "45%" height = "45%"></img>
   <img src = "https://user-images.githubusercontent.com/47768726/60196280-b8c59f00-9877-11e9-9fcc-9de844d5ae8b.JPG" width= "45%" height = "45%"></img>
   <img src = "https://user-images.githubusercontent.com/47768726/60196281-b8c59f00-9877-11e9-866f-fbe2b3c95945.JPG" width= "45%" height = "45%"></img>
   <img src = "https://user-images.githubusercontent.com/47768726/60196283-b8c59f00-9877-11e9-853c-3ea6d5069ae4.JPG" width= "45%" height = "45%"></img>

```
return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

RGB이미지를 Gray Scale로 변환합니다.
```

  ### def BGRtoHSV(frame):

  <img src = "https://user-images.githubusercontent.com/47768726/60197540-89646180-987a-11e9-81e9-444195bbd4ec.jpg" width="40%" height="40%"></img>
  
  ```  
  return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
  
  BGR이미지를 HSV공간으로 변환합니다.
  
  Hue : 색상   Saturation : 채도  Value : 명도
  ```

  ### def BGRtoHLS(frame):
  
   <img src = "https://user-images.githubusercontent.com/47768726/60197539-89646180-987a-11e9-8ec6-ef75c7785ee5.jpg" width="40%" height="40%"></img>  
   
  ```
  return cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
  
  BGR이미지를 HLS공간으로 변환합니다.
  
  Hue : 색상  Lightness : 명도  Saturation : 채도  
  ```
  
  ### def GaussianBlur(frame):
  
  ```
    blur = cv2.GaussianBlur(frame,(5,5), 0)
    return blur
    
    5x5 필터를 사용하며, 중앙에서 인접한 값에 근거하여 필터 중앙의 픽셀값을 재조정 합니다. 
    결과적으로 이미지의 노이즈를 제거합니다.
  ```
  
  ### def Dilation(frame, iterations):
  <img src = "https://user-images.githubusercontent.com/47768726/60200367-f4189b80-9880-11e9-8148-430563c1f217.jpg" width= "90%" height = "90%"></img>

 ``` 
  previous : Dilation 5회, last : Dilation 10회
 
  kernelRow = 3
  kernelCol = 3
  kernel = np.ones((kernelRow,kernelCol), np.uint8)

  return cv2.dilate(frame, kernel, iterations)
    
  3x3 크기의 1로 채워진 kernel을 생성합니다.
  
  이 후 이미지와 커널을 통해 팽창을 실시합니다.
  
  팽창이란 영역내의 픽셀을 최대 픽셀값(255:흰색)으로 대체하여, 어두운영역이 줄어들고 밝은 영역을 늘립니다.
  
  img: Dilation을 수행할 원본 이미지
  
  kernel: Dilation을 위한 커널
  
  iterations: Dilation 반복 횟수
  
  ```
  ### def BirdEyeView(frame):
  
   <img src = "https://user-images.githubusercontent.com/47768726/60192938-bf511800-9871-11e9-814c-3bf781361879.JPG" width= "45%" height = "45%"></img>
   <img src = "https://user-images.githubusercontent.com/47768726/60252302-fa505b80-9904-11e9-8b84-a9895c1ba5ba.JPG" width= "45%" height = "45%"></img>   
   <img src = "https://user-images.githubusercontent.com/47768726/60192942-c1b37200-9871-11e9-82f7-fa4ff963a3b4.JPG" width= "45%" height = "45%"></img>
   <img src = "https://user-images.githubusercontent.com/47768726/60252303-fa505b80-9904-11e9-83f4-73b3462760fb.JPG" width= "45%" height = "45%"></img>
   <img src = "https://user-images.githubusercontent.com/47768726/60192946-c2e49f00-9871-11e9-93bd-34956bceb52c.JPG" width= "45%" height = "45%"></img>
   <img src = "https://user-images.githubusercontent.com/47768726/60252304-fa505b80-9904-11e9-9fe4-bc30642c8fd1.JPG" width= "45%" height = "45%"></img>
  
  ```
  BirdEyedView를 통해 차선인식을 수행합니다.
  
  BirdEyedView란 이름처럼 새가 위에서 바라보는 것 같이 위에서 아래로 내려다 보는듯한 이미지로 만들어 줍니다.
  
  변환시키기 위해 4점(좌상,좌하,우상,우하) 변환을 이용합니다.
  
  변환하고 싶은 이미지를 4점을 통해서 지정하고, 각 점들이 어디로 펼쳐질것인지에 대한 목적지 점을 설정합니다.
  
  먼저 변환을 위해 getPerspectiveTransform(원본 좌표, 결과 좌표)를 사용하여 행렬을 만들어냅니다.
  
  ```
  
  * def warpFrame(frame, perspective):
  
  ```
  
  이 함수는 getPerspectiveTransform을 통해 얻어온 행렬을 이용해 직접적으로 이미지를 펼쳐주는 역할을 합니다.
  
  cv2.warpPerspective를 사용해 목적지 행렬을 토대로 이미지를 펼쳐 BirdEyedView를 생성합니다.
    
  ```
  
  * def LaneDetection(frame):
  
  ```c
    이미지에서 차선을 필터링 해주는 함수입니다.
    
    white = cv2.inRange(frame, np.array((0,200,10)), np.array((255,255,255)))
    hsv = BGRtoHSV(frame)
    yellow = cv2.inRange(hsv, np.array((10,75,75)), np.array((56,255,255)))

    both = cv2.add(yellow, white)
    return both
    
    이미지에서 먼저 BGR공간에서 흰색범위 설정을 통해 흰색차선만 보여줍니다.
    
    이후 HSV공간으로 변경후, 색상,채도,명도의 범위설정을 통해 노란색 차선을 걸러줍니다.
    
    걸러진 두 이미지를 병합하여 흰색 및 노란색 차선이 인식된 결과 반환합니다.   
    
  ```
  
  * def FindLane(binaryFrame, prevFits = None):
  
  
    ```c
    
    실질적으로 차선을 인식하는 함수입니다.
    
    BirdEyedView에서 차선을 인식하는 box형태의 SlidingWindows를 통해 차선 정보를 matrix형태로 반환합니다. 
    
    먼저 받아온 이진화 이미지 크기로 빈 img를 만듭니다

    사용할 SlidingWindow의 개수는 10개, 너비는 200, 너비의 반은 100 입니다.

    nonzero를 통해 이진화 이미지에서 값이 0(검은색)이 아닌 흰색 픽셀의 인덱스(좌표)들을 반환합니다.
    
    nonzeroY와 nonzeroX 통해 x,y 좌표로 각각 저장합니다.
    
    nonzeroY = 행, nonzeroX = 열,  
    
    예를들어 2차원배열을 표기할때 (0,0), (0,1), (1,0), (1,1) 이런식으로 표기한다면 (nonzeroY , nonzeroX) 의 형태로 대응됩니다.
    
    nonzeroY = np.nonzero(binaryFrame)[0]
    nonzeroX = np.nonzero(binaryFrame)[1]

    윈도우 내 픽셀수 임계값은 50으로 정합니다. 이 임계값은 SlidingWindow의 x좌표를 재설정하는데에 사용됩니다.
  
    ```    
    
    * def FindLaneBases():
    
    <img src = "https://user-images.githubusercontent.com/47768726/60284741-4b813f00-9947-11e9-8cbd-f0bd2fe1a01b.jpg" width = "90%" height = "90%"></img>
    
    ```
    이미지 행 기준 절반 하단부에서 차선위치를 찾습니다.
    
    이미지 행의 1/2 지점을 구합니다. int 형변형으로 홀수/2 일 경우 소수점을 무시합니다.
    
    먼저 차선 인식을 위한 Base를 찾기위해 이미지 절반 하단부에서 열 단위로 한 줄 씩 더해줍니다.
    
    해당 과정은 위 이미지의 Histogram(노란색)과 같습니다.
    
    LeftMaxIndex = np.argmax(histogram[:midCol])
    RightMaxIndex = np.argmax(histogram[midCol:]) + midCol
    
    histogram의 절반 좌측부분에서 제일 큰 값의 인덱스를 찾고,
    
    histogramd의 절반 우측부분에서 제일 큰 값의 인덱스를 찾습니다.
    
    (RightMaxIndex : midCol~끝이므로 midCol부분이 인덱스 0이 됨. 따라서 midCol을 더해줘야 실제 index값이 나옴)
    
    해당 값이 LaneBase가 되며 이 값을 반환해줍니다.
    
    ```
    
    * def SearchLane(xCenter, PixelColor=None):
    
    <img src="https://user-images.githubusercontent.com/47768726/60335023-d01b9e00-99d7-11e9-8097-ca5f42351c4f.jpg" width = "90%" height = "90%" ></img>
    
    ```
    본격적으로 SlidingWindows를 사용해 차선을 찾습니다.
    
    Window의 높이를 이미지의 높이 / 윈도우의 개수로 조정합니다.
    
    먼저 반복문을 통해 윈도우 개수만큼 반복합니다.
    
    윈도우 xLow값은 xCenter의 -100, xHigh값은 xCenter의 +100으로 조정합니다.
    
    윈도우의 높이는 이미지의 높이 / 윈도우의 개수(10개)입니다. 따라서 이미지 크기의 1/10이 됩니다.
    
    yLow의 값은 이미지의 마지막 행에서 [(i+1)*윈도우의 높이]가 됩니다. (i는 0부터 시작하며 이미지의 y좌표는 내려갈수록 커집니다)
    
    yHigh의 값은 yLow에 윈도우의 높이를 더해줍니다.
    
    해당 정보들(xLow, xHigh, yLow, yHigh)를 searchWindowCorners 리스트에 저장해줍니다.
    
    searchWindowCorners.append([(window_xLow, window_yLow),(window_xHigh, window_yHigh)])
    
    이제 특정 크기를 가진 Sliding Windows를 구현하였습니다.
    
    이 Windows를 가지고 차선을 인식해보도록 합니다.
    
    먼저 미리 저장해둔 nonzero값(이미지에서 흰색 픽셀[차선]의 좌표)중에 윈도우 안에 있다면 True, 없다면 False로 리스트를 생성합니다.
     on = ((nonzeroX >= window_xLow) & (nonzeroX <= window_xHigh)) &\
          ((nonzeroY >= window_yLow) & (nonzeroY <= window_yHigh))

    
    nonzero에서 True인 부분만 뽑아내어 차선의 좌표를 생성합니다. (윈도우에 들어와있는 흰색의 좌표 (onX, onY))
    onX = nonzeroX[on]
    onY = nonzeroY[on]

    lane_xCoordintes.append(onX)
    lane_yCoordintes.append(onY)
    
    lane_xCoordintes, lane_yCoordintes 리스트에 해당 좌표를 추가합니다
    
    if np.sum(on) >= RecenterThreshold:
    xCenter = np.int(np.mean(onX))
    
    만약 윈도우내에 픽셀수가 지정해둔 50을 넘어가면, xCenter를 흰색 픽셀들의 평균값으로 재조정해줍니다.    
    
    마지막으로  searchWindowCorners(윈도우 모서리 값), lane_xCoordintes(차선 x좌표), lane_yCoordintes(차선 y좌표)를 반환합니다.
    
    ```    
    
    ```c
    leftBaseX, rightBaseX = FindLaneBases()
    LeftLaneInfo = SearchLane(leftBaseX, PixelColor =[255,0,0])
    RightLaneInfo = SearchLane(rightBaseX, PixelColor=[0,255,0])

    return LeftLaneInfo, RightLaneInfo
    
    FindLaneBases를 통해 찾은 지점이 각자 차선에서 leftBaseX (왼쪽차선의 Xcenter), rightBaseX (오른쪽차선의 Xcenter)가 됩니다.
    
    SearchLane함수를 통해 왼쪽차선의 정보와 오른쪽 차선의 정보를 찾은 후 반환하게 됩니다.
    
    ```
    
  * def PolyFit(xPoints, yPoints):
  
   <img src = "https://user-images.githubusercontent.com/47768726/60258880-00e4d000-9911-11e9-9aa9-90ac51b553a1.jpg" width="80%" height="80%"></img>
   
 ```  
  np.polyfit를 이용하여 함수의 계수를 찾아냅니다.
  
  x값과 y값과 함수의 차원을 매개변수로 줍니다.
  
  그림과 같이 고차원의 계수부터 저차원의 계수까지 차례대로 찾을수 있게해주는 함수입니다.  
  
  ```
  * def Qudratic(coeffs, pts):
  ```
      return (coeffs[0]*pts**2 + coeffs[1] * pts + coeffs[2]).astype(np.int32)
      
      Qudratic은 계수와 pts를 받아와서 2차원 함수를 만들어 줍니다.
  ```
  
  * def DrawPolygon(frame, points, color=[255,0,0], thickness = 5, isClosed = True):
  
  ```c
    n = len(points)

    try:
        for i in range(n-1):
        cv2.line(frame, points[i], points[i+1], color, thickness)

        if isClosed == True:
        cv2.line(frame, points[n-1], points[0], color, thickness)

    except :
        print("도형을 그리는데 에러가 발생했습니다 {} {}".format(points[i], points[i+1]))
        raise
  
  n은 points의 길이를 저장합니다.
  try에서 오류체크를 해주며 except에서 에러처리를 해줍니다.
  
  n-1까지 반복하며 다각형을 그립니다.
  이전 포인트에서 다음포인트로 색상(color) 두께(thickness)를 통해 선을 그립니다.
  
  만약 닫힌 다각형(isClosed == True)을 그리고 싶다면 끝점(n-1)과 시작점(0)을 서로 이어주면 됩니다.
   
  ```  
  
  * def ReductionFrame(frame):
  
  ```
      return cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
      
      프레임의 사이즈를 1/2크기로 줄여주는 함수입니다.
  
  ```
  
### Main Code
  ```c
print(cv2.__version__)

#Video를 받아옵니다.
cap = cv2.VideoCapture('C:\\Users\\User\\PycharmProjects\\LaneDetection\\project_video.mp4')

while True:
    # 프레임을 읽습니다.
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

        # Window의 Corners값을 받아옴
        leftLaneWindows = leftLaneInfo[0]
        rightLaneWindows = rightLaneInfo[0]

        # 찾은 차선의 정보를 좌표로 받습니다.
        leftLaneX, leftLaneY = leftLaneInfo[1]
        rightLaneX, rightLaneY = rightLaneInfo[1]

        # 차선의 좌표를 토대로 이차원 함수의 계수를 찾습니다.
        leftPoly = PolyFit(leftLaneX, leftLaneY)
        rightPoly = PolyFit(rightLaneX, rightLaneY)

        # ys : 이미지의 행을 배열로 받아옵니다.
        # ys와 계수를 이용해 2차원 함수(xs_left, xs_right)를 생성합니다. 
        
        ys = np.array(range(frame.shape[0]), dtype=np.int32)
        xs_left = Qudratic(leftPoly, ys)
        xs_right = Qudratic(rightPoly, ys)

        # frame과 동일한 이미지를 생성합니다.
        searchAreaFrame = np.dstack((frame, frame, frame)) * 255

        # 아까 받아온 WindowCorner정보를 토대로 SlidingWindows를 그려줍니다.
        for window in leftLaneWindows:
           cv2.rectangle(searchAreaFrame, window[0], window[1], color = [255, 0, 0], thickness = 5)

        for window in rightLaneWindows:
            cv2.rectangle(searchAreaFrame, window[0], window[1], color = [255, 0, 0], thickness = 5)


        # zip : 김밥 슬라이스
        # 1,2,3,4,5
        # a,b,c,d,e
        # (1,a), (2,b), (3,c), (4,d), (5,e)
        # 왼쪽차선, 오른쪽차선을 좌표형태로 생성합니다.
        
        left_points = [(x,y) for (x,y) in zip(xs_left, ys)]
        right_points = [(x,y) for (x,y) in zip(xs_right, ys)]
        all_points = [pt for pt in left_points]
        all_points.extend([pt for pt in reversed(right_points)])

        # 생성한 좌표를 토대로 차선을 그려줍니다.
        DrawPolygon(searchAreaFrame, left_points, color=[0, 0, 255], isClosed=False, thickness=15)
        DrawPolygon(searchAreaFrame, right_points, color=[0, 0, 255], isClosed=False, thickness=15)

        #BirdEyedView위에 인식한 차선의 결과를 겹쳐서 보여줍니다.
        birdeyedview = cv2.addWeighted(bird, 1.0, searchAreaFrame, 1.0, 0.0)
        birdeyedview = ReductionFrame(birdeyedview)
        cv2.imshow('BirdEyedView', birdeyedview)

        # 마지막으로 BirdEyedView를 벗어나 실제 이미지에서 인식된 차선을 보여줍니다.
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
  
  
## 분석 및 평가

```
Canny & Hough 차선인식에서 제한됐던 곡선이 인식되었습니다.

BirdEyedView를 통하여 차선이 일렬로 정렬되면서, matrix를 열단위로 분석하는 알고리즘을 사용해볼수 있었습니다.

BirdEyedView를 생성 할 때 차선을 잘 정렬되도록 적절한 좌표를 설정하여야 정확도가 올라갔습니다.

Video를 받아와서 인식한 결과이며 새로운 Video에 적용 할 시, BirdEyedView 목적지 좌표를 새로 설정해줘야 제대로 동작할것입니다.

필터 Threshold 조절 또한 필요할것입니다.

Sliding Windows의 너비도 중요했는데, 너비를 너무 넓게잡으면 옆에 noise까지 인식하게될 확률이 있었습니다.

그리고 Windows의 개수가 많아지면 속도가 저하되며 오차가 더 나는 경우도 있었습니다.

```

## 개선 방안 
```

 맑은 날씨의 Video로 테스트하였으며, 그림자 및 햇빛이 드리운다면 정확도가 떨어집니다.
 
 필터링이 정확할수록 높은 성능을 보여줬습니다. 필터링의 중요성을 알 수 있었고 좀 더 정교한 필터를 필요로 할것입니다.
 
 자료롤 찾아보니 Sobel 필터링에서 magnitude & angle & hls공간에서 채널을 나눈후 필터링하는 것을 사용해 볼 수 있습니다. 
 
 BirdEyedView의 좌표를 조절하여 좀 더 정확도를 높일수 있을것입니다. 
 
 
```
