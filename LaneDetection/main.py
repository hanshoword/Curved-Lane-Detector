# All imports
from PIL import Image
from glob import glob
from tqdm import tqdm
import moviepy
import cv2
import matplotlib.pyplot as plt
import numpy as np
import imageio

from moviepy.editor import VideoFileClip
__DEBUG__= True

def Binary(frame, threshold):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    ret, bin = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    return bin

def Gray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

def BGRtoHSV(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

def BGRtoHLS(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

def GaussianBlur(frame):
    blur = cv2.GaussianBlur(frame,(5,5), 0)
    return blur

def Dilation(frame, iterations):
    kernelRow = 3
    kernelCol = 3
    kernel = np.ones((kernelRow,kernelCol), np.uint8)

    return cv2.dilate(frame, kernel, iterations)

def BirdEyeView(frame):
    height, width, channel = frame.shape
    mask = np.zeros_like(frame)

    srcPoint = np.array([[(width*3/7, height*50/80),  (width*4/7, height*50/80), (width, height*95/100), (0, height*95/100)]], dtype=np.float32)
    dstPoint = np.array([[(0,0), (width, 0), (width, height), (0,height)]], dtype = np.float32)

    matrix = cv2.getPerspectiveTransform(srcPoint, dstPoint)
    original = cv2.getPerspectiveTransform(dstPoint, srcPoint)

    return matrix, original

def warpFrame(frame, perspective):
    height, width, channel = frame.shape

    return cv2.warpPerspective(frame, perspective, (width, height))

def LaneDetection(frame):
    white = cv2.inRange(frame, np.array((0,200,10)), np.array((255,255,255)))
    hsv = BGRtoHSV(frame)
    yellow = cv2.inRange(hsv, np.array((10,75,75)), np.array((56,255,255)))

    both = cv2.add(yellow, white)
    return both

def FineLane(binaryFrame, prevFits = None):
    img = np.dstack((binaryFrame, binaryFrame, binaryFrame)) * 255

    #윈도우 검색 인수
    windowsNum = 10
    windowHalfWidth = 100

    # nonzero : 값이 0(검은색)이 아닌 흰색 픽셀의 인덱스들을 반환
    nonzeroY = np.nonzero(binaryFrame)[0]
    nonzeroX = np.nonzero(binaryFrame)[1]

    # 윈도우 내 픽셀수 임계값
    RecenterThreshold = 50

    # 이미지에서 차선으로 추정되는 위치 인덱스를 찾는 메서드
    def FindLaneBases():
        #이미지 행 기준 절반 하단부에서 차선위치를 찾는다
        # 이미지 행의 1/2 지점을 구한다, int 변형 : 홀수/2 일 경우 소수점 무시
        midRow = np.int(binaryFrame.shape[0]/2)

        # 이미지 절반 하단부에서 열 단위로 한줄씩 더한다.
        histogram = np.sum(binaryFrame[midRow:, : ], axis = 0)

        # 이미지 열의 1/2지점을 구한다, int 변형 : 홀수/2 일 경우 소수점 무시
        midCol = np.int(histogram.shape[0]/2)

        #histogram의 절반 좌측부분에서 제일 큰값의 인덱스
        #histogramd의 절반 우측부분에서 제일 큰값의 인덱스
        #RightMaxIndex : midCol~끝이므로 midCol부분이 인덱스 0이 됨. 따라서 midCol을 더해줘야 실제 index값이 나온다

        LeftMaxIndex = np.argmax(histogram[:midCol])
        RightMaxIndex = np.argmax(histogram[midCol:]) + midCol

        return LeftMaxIndex, RightMaxIndex

    def SearchLane(xCenter, PixelColor=None):
        #윈도우 높이 = 이미지의 높이 / 윈도우 개수
        windowHeight = np.int(binaryFrame.shape[0] / windowsNum)

        searchWindowCorners = []
        lane_xCoordintes = []
        lane_yCoordintes = []

        # i=0부터 윈도우 개수만큼 반복
        for i in range(windowsNum):
            # Low = xcenter - 100
            # xcenter
            # High = xcenter + 100

            window_xLow = xCenter - windowHalfWidth
            window_xHigh = xCenter + windowHalfWidth

            # Low~High로 구성된 윈도우, windowNum개수 만큼 행을 나눔
            # 상 : Low, 하 : High

            window_yLow = binaryFrame.shape[0] - (i+1)*windowHeight
            window_yHigh = window_yLow + windowHeight

            # 윈도우 코너값 저장
            searchWindowCorners.append([(window_xLow, window_yLow),(window_xHigh, window_yHigh)])

            # 윈도우 안에 흰색픽셀이 있다면 True, 없다면 False
            on = ((nonzeroX >= window_xLow) & (nonzeroX <= window_xHigh)) &\
                 ((nonzeroY >= window_yLow) & (nonzeroY <= window_yHigh))

            # 좌표를 얻음 (onX, onY) : 윈도우에 들어와있는 흰색의 좌표
            onX = nonzeroX[on]
            onY = nonzeroY[on]

            img[onY, onX, :] = PixelColor
            cv2.rectangle(img, (window_xLow, window_yLow), (window_xHigh, window_yHigh), [0,0,255],5)

            lane_xCoordintes.append(onX)
            lane_yCoordintes.append(onY)

            if np.sum(on) >= RecenterThreshold:
                xCenter = np.int(np.mean(onX))

        lane_xCoordintes = np.concatenate(lane_xCoordintes)
        lane_yCoordintes = np.concatenate(lane_yCoordintes)

        return searchWindowCorners, (lane_xCoordintes, lane_yCoordintes)

    def probeNearby(prevFit, PixelColor):
        # 2차함수로 예측값 구함
        xPredict = prevFit[0] * nonzeroY**2 + prevFit[1] * nonzeroY + prevFit[2]

        on = (nonzeroX >= xPredict - windowHalfWidth) &\
             (nonzeroX <= xPredict - windowHalfWidth)

        laneXcoordintes = nonzeroX[on]
        laneYcoordintes = nonzeroY[on]

        return laneXcoordintes, laneYcoordintes

    if prevFits == None:
        leftBaseX, rightBaseX = FindLaneBases()
        LeftLaneInfo = SearchLane(leftBaseX, PixelColor =[255,0,0])
        RightLaneInfo = SearchLane(rightBaseX, PixelColor=[0,255,0])
    else:
        LeftLaneCoords = probeNearby(prevFits[0], PixelColor = [255,0,0])
        RightLaneCoords = probeNearby(prevFits[1], PixelColor = [0,255,0])

        LeftLaneInfo = ([], LeftLaneCoords)
        RightLaneInfo = ([], RightLaneCoords)

    return LeftLaneInfo, RightLaneInfo

# 2차원의 계수를 찾아주는 함수
def PolyFit(xPoints, yPoints):
    return np.polyfit(yPoints, xPoints, 2)

# 2차원 식을 만드는 함수
def Qudratic(coeffs, pts):
    return (coeffs[0]*pts**2 + coeffs[1] * pts + coeffs[2]).astype(np.int32)

def DrawPolygon(frame, points, color=[255,0,0], thickness = 5, isClosed = True):
    n = len(points)

    try:
        for i in range(n-1):
            cv2.line(frame, points[i], points[i+1], color, thickness)

        if isClosed == True:
            cv2.line(frame, points[n-1], points[0], color, thickness)

    except :
        print("도형을 그리는데 에러가 발생했습니다 {} {}".format(points[i], points[i+1]))
        raise

def ReductionFrame(frame):
    return cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)


cap = cv2.VideoCapture('C:\\Users\\User\\PycharmProjects\\LaneDetection\\project_video.mp4')

while True:
    ret, frame = cap.read()
    if ret == True:

        temp = frame
        # Gaussian Blur
        frame = GaussianBlur(frame)

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
        leftLaneInfo, rightLaneInfo = FineLane(frame)

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