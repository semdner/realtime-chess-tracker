import streamlit as st
import cv2
import torch
import numpy as np
import av
import chess
import state
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO

piece_cls_name_mapping = {
     0: "black bishop",
     1: "black king",
     2: "black knight",
     3: "black pawn",
     4: "black queen",
     5: "black rook",
     6: "white bishop",
     7: "white king",
     8: "white knight",
     9: "white pawn",
    10: "white queen",
    11: "white rook",
}

chessboard_seg_model = YOLO("model/chessboard_segmentation.pt")
square_seg_model = YOLO("model/square_segmentation.pt")
corner_detect_model = YOLO("model/corner_detection.pt")
piece_detect_model = YOLO("model/piece_detection.pt")

# extract mask from chessboard segmentation model prediction
def get_mask(results):
    mask = None

    for result in results:
        for index, item in enumerate(result):
            mask = item.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)

    return mask

"""Calculate the contour from the mask of the chessboard"""
def get_contour(frame, results):
    contour = None

    for result in results:
        for index, item in enumerate(result):
            # create mask
            binary_mask = np.zeros(frame.shape[:2], np.uint8)
            contour_pred = item.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
            new_frame = cv2.drawContours(binary_mask, [contour_pred], -1, (255, 255, 255), cv2.FILLED)

            # create contour from mask
            contour_from_mask, hierarchy = cv2.findContours(new_frame, 1, 2)

            # reduce the points of the contour
            for cnt in contour_from_mask:
                epsilon = 0.01 * cv2.arcLength(cnt, True)
                contour = cv2.approxPolyDP(cnt, epsilon, True)

    return contour


"""Return the coordinates of the bounding boxes of predicted corner"""
def get_corners(results):
    pred_corners_bbox = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            pred_corners_bbox.append([x1, y1, x2, y2])

    return pred_corners_bbox


"""Check if a given point is within the given boudning box"""
def is_point_in_bounding_box(point, bbox):
    pt_x, pt_y = point
    bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max = bbox

    return bbox_x_min <= pt_x >= bbox_x_max and bbox_y_min <= pt_y >= bbox_y_max


"""Calculate the center point of a bounding box"""
def get_bbox_center(bbox):
    bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max = bbox

    center_x = (bbox_x_min + bbox_x_max)//2
    center_y = (bbox_y_min + bbox_y_max)//2

    return [center_x, center_y]


""""""
def get_piece_coordinates(results, piece_coordinates_mapping, piece_coordinates):
    for result in results:
        boxes = result.boxes
        for index, box in enumerate(boxes):
            # for the class number get the corresponding piece name
            piece_cls = int(box.cls[0])

            # get the keypoint where the piece is positioned
            xy_kp = result.keypoints.xy[index]
            x_kp, y_kp = xy_kp[0]
            x_kp, y_kp = int(x_kp.item()), int(y_kp.item())

            piece_coordinates_mapping[piece_cls_name_mapping[piece_cls]].append([[np.float32(x_kp), np.float32(y_kp)]])
            piece_coordinates.append((np.float32(x_kp), np.float32(y_kp)))

    return piece_coordinates_mapping, piece_coordinates


"""Calculate homography matrix"""
def get_homography_matrix(frame, points1):
    # convert the int corner points into float
    points1 = np.float32(points1)

    height = (points1[1][0]-points1[0][0])
    height = height*2

    # points to map the original points onto
    points2 = np.float32([[0, 0], [height, 0], [0, height], [height, height]])

    # create homography matrix
    M = cv2.getPerspectiveTransform(np.array(points1), points2)

    return M


"""Apply homography to frame"""
def apply_homography_to_frame(frame, points1, M):
    # convert the int corner points into float
    points1 = np.float32(points1)

    height = (points1[1][0]-points1[0][0]) * 2

    # apply homography matrix
    dst = cv2.warpPerspective(frame, M, (int(height), int(height)))

    return dst


"""Apply homography to piece coordinates"""
def apply_homography_to_pieces(M, piece_coordinates):
    # convert piece coordinates into float32 numpy array
    piece_coordinates = np.array(piece_coordinates, dtype="float32").reshape(-1, 1, 2)
    # apply homography to corner points
    piece_coordinates_transformed = cv2.perspectiveTransform(piece_coordinates, M)

    return piece_coordinates_transformed


def assign_pieces_to_squares(frame, results, piece_coordinates_mapping, piece_square_mapping):
    min_area = 2000 # detected squares smaller are removed
    height, width, _ = frame.shape
    square_size = ((height+width)//2)//8

    new_frame = frame.copy()

    for y in reversed(range(8)):
        for x in range(8):
            rank = chr(ord("a") + x)
            row = 8 - y
            square = f"{rank}{row}"

            x_min = (x * square_size)
            y_min = (y * square_size)
            x_max = (x * square_size) + square_size
            y_max = (y * square_size) + square_size
            x_cen = (x_min + x_max) / 2
            y_cen = (y_min + y_max) / 2
            x_cen = int(x_cen)
            y_cen = int(y_cen)

            for result in results:
                masks = result.masks
                for mask in masks:
                    area = int(np.sum(mask.data.cpu().numpy()))

                    if area > min_area:
                        pred_contour = mask.xy.pop().astype(np.int32)
                        x_corner, y_corner, w, h = cv2.boundingRect(pred_contour)
                        center = (x_corner + w / 2, y_corner + h / 2)

                        if center[1] > y_min and center[0] > x_min and center[1] < y_max and center[0] < x_max:
                            for piece in piece_coordinates_mapping:
                                for value in piece_coordinates_mapping[piece]:
                                    if int(value[0][0]) > x_min and int(value[0][1]) > y_min and int(value[0][0]) < x_max and int(value[0][1]) < y_max:
                                        piece_square_mapping[piece].append(square)

    return piece_square_mapping


"""Check if the detected pieces are arranged in the starting position"""
def check_starting_position(piece_coordinate_mapping):
    piece_map = {
        "pawn": chess.PAWN,
        "knight": chess.KNIGHT,
        "bishop": chess.BISHOP,
        "rook": chess.ROOK,
        "queen": chess.QUEEN,
        "king": chess.KING
    }

    starting_fen = chess.STARTING_BOARD_FEN

    board = chess.Board(None)

    for key in piece_coordinate_mapping:
        color_str, piece_str = key.split(" ")
        color = chess.WHITE if color_str == "white" else chess.BLACK
        piece = piece_map[piece_str]

        for square in piece_coordinate_mapping[key]:
            square = chess.parse_square(square)
            board.set_piece_at(square, chess.Piece(piece, color))

    if board.fen().split(" ")[0] == starting_fen:
        return True
    else:
        return False


def calibrate(frame):

    # SEGMENT THE CHESSBOARD, CREATE A CONTOUR AND REDUCE THE CONTOUR POINTS TO THE FOUR CORNER POINTS
    
    seg_chessboard_results = chessboard_seg_model.predict(frame, conf=0.9)
    
    # contour prediction based on mask
    contour = get_contour(frame, seg_chessboard_results)
    
    if contour is None:
        return False

    # get points from contour
    corners = [[point[0][0], point[0][1]] for point in contour]

    if len(corners) != 4:
        print(f"ERROR: contour must exist of exactly only 4 corners (currently {len(corners)})")
        return False

    # IMPROVE THE CORNERS IF POSSIBLE
    
    detect_corners_results = corner_detect_model.predict(frame, conf=0.6)

    # bounding box for corner prediction based on visible board corners
    pred_corners_bbox = get_corners(detect_corners_results)   

    corrected_corners = []

    # replace the contour corners with more accurate predicted corners
    if len(pred_corners_bbox) > 0:
        replaced = False
        
        for corner in corners:
            for bbox in pred_corners_bbox:
                if is_point_in_bounding_box(corner, bbox):
                    # make center points of bbox new corner point
                    center_pts = get_bbox_center(bbox)
                    corrected_corners.append(center_pts)
                    replaced = True

                    break
            
            if not replaced:
                corrected_corners.append(corner)
    else:
        return False

    corners = corrected_corners if len(corrected_corners) == 4 else corners

    if len(corners) != 4:
        print(f"ERROR: contour must exist of exactly only 4 corners (currently {len(corners)})")
        return False

    # SORT CORNERS (REWORK)

    corners = sorted(corners, key=lambda x: x[0])
    corners[2], corners[3] = corners[3], corners[2]

    # DETECT ALL VISIBLE CHESSBOARD PIECES TO CHECK IF ALL PIECES ARE PLACED CORRECTLY

    # stores the coordinates of each detected piece
    piece_coordinates_mapping = {
        "black bishop": [],
        "black king": [],
        "black knight": [],
        "black pawn": [],
        "black queen": [],
        "black rook": [],
        "white bishop": [],
        "white king": [],
        "white knight": [],
        "white pawn": [],
        "white queen": [],
        "white rook": [], 
    }

    piece_coordinates = []

    detect_pieces_results = piece_detect_model.predict(frame, conf=0.7)
    piece_coordinates_mapping, piece_coordinates = get_piece_coordinates(detect_pieces_results, piece_coordinates_mapping, piece_coordinates)

    # APPLY HOMOGRAPHY TO CHESSBOARD (FRAME) AND PIECES COORDINATES

    # calculate the homography matrix
    matrix = get_homography_matrix(frame, corners)

    if matrix is None:
        return False

    

    # apply homography to frame (display chessboard upright)
    homographed_frame = apply_homography_to_frame(frame, corners, matrix)

    if homographed_frame is None:
        return False

    # apply homography to corner points
    homographed_piece_coordinates = apply_homography_to_pieces(matrix, piece_coordinates)
    
    if homographed_piece_coordinates is None:
        return False

    # REMAP PIECES FROM NON HOMOGRAPHED COORDINATES TO HOMOGRAPHED COORDINATES

    piece_coordinates = np.array(piece_coordinates, dtype="float32").reshape(-1, 1, 2)
    piece_coordinates_before = piece_coordinates.reshape(-1, 2)
    piece_coordinates_after = homographed_piece_coordinates.reshape(-1, 2)
    piece_coordinates_combined = np.hstack((piece_coordinates_before, piece_coordinates_after))

    # loop through coordinate mapping list and update it to homographed coordinates
    for coordinates in piece_coordinates_combined:
        for piece in piece_coordinates_mapping:
            for value in piece_coordinates_mapping[piece]:
                if value[0][0] == coordinates[0] and value[0][1] == coordinates[1]:
                    value[0][0] = coordinates[2] # update x coordinate
                    value[0][1] = coordinates[3] # update y coordinate 


    # ASSIGN EACH PIECE TO A SQUARE

    piece_square_mapping = {
        "black bishop": [],
        "black king": [],
        "black knight": [],
        "black pawn": [],
        "black queen": [],
        "black rook": [],
        "white bishop": [],
        "white king": [],
        "white knight": [],
        "white pawn": [],
        "white queen": [],
        "white rook": [],
    }

    seg_squares_results = square_seg_model.predict(homographed_frame, conf=0.8)
    piece_square_mapping = assign_pieces_to_squares(homographed_frame, seg_squares_results, piece_coordinates_mapping, piece_square_mapping)

    if piece_square_mapping is None:
        return False

    # CHECK IF THE PIECES ARE CORRECTLY SETUP
    
    is_board_setup = check_starting_position(piece_square_mapping)
    
    # update cache
    if is_board_setup:
        state.corners = corners
        state.matrix = matrix

    return is_board_setup


def init_board():
    return chess.Board(chess.STARTING_FEN)