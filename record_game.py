import cv2
import torch
import numpy as np
import chess
import chess.svg
import state
from io import BytesIO
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

square_seg_model = YOLO("model/square_segmentation.pt")
piece_detect_model = YOLO("model/piece_detection.pt")

def get_number_of_pieces(results):
    for result in results:
        return len(result.boxes)

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


"""return the coordinates of the bottom keypoint for each piece"""
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


def set_board(board, piece_square_mapping):
    piece_map = {
        "pawn": chess.PAWN,
        "knight": chess.KNIGHT,
        "bishop": chess.BISHOP,
        "rook": chess.ROOK,
        "queen": chess.QUEEN,
        "king": chess.KING
    }

    for piece in piece_square_mapping:
        color_str, piece_str = piece.split(" ")
        chess_color = chess.WHITE if color_str == "white" else chess.BLACK # python chess color
        chess_piece = piece_map[piece_str] # python chess piece
        
        for square in piece_square_mapping[piece]:
            chess_square = chess.parse_square(square) # python chess square
            board.set_piece_at(chess_square, chess.Piece(chess_piece, chess_color))

    return board


def get_move(board, new_board):
    for move in board.legal_moves:
        board_copy = board.copy()
        board_copy.push(move)

        if board_copy.fen().split(" ")[0] == new_board.fen().split(" ")[0]:
            return move.uci()

    return None


def record(frame):
    # get corners, matrix and board from cache
    corners = state.corners
    matrix = state.matrix
    board = state.board

    # DETECT ALL CHESSBOARD PIECES

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

    detect_pieces_results = piece_detect_model.predict(frame, conf=0.5)

    num_of_pieces_detected = get_number_of_pieces(detect_pieces_results)
    num_of_pieces_previous = len(board.piece_map())

    if num_of_pieces_detected > num_of_pieces_previous or num_of_pieces_previous < num_of_pieces_detected-1:
        return False

    piece_coordinates_mapping, piece_coordinates = get_piece_coordinates(detect_pieces_results, piece_coordinates_mapping, piece_coordinates)

    # APPLY HOMOGRAPHY TO CHESSBOARD (FRAME) AND PIECES COORDINATES

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

    # COMPARE WHETHER THE CURRENT POSITION CAN BE REACHED FROM THE PREVIOUS POSITION

    new_board = chess.Board(None)
    new_board = set_board(new_board, piece_square_mapping)
    move = get_move(board, new_board)

    if move is None:
        return False

    board.push_uci(move)
    state.board = board

    return True


def generate_image(board):
    board = chess.Board(chess.STARTING_BOARD_FEN)
    svg_img = chess.svg.board(board, size=1000)

    png_data = cairosvg.svg2png(bytestring=svg_img.encode('utf-8'))

    image_array = np.frombuffer(png_data, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)

    img_buffer = BytesIO()
    image.save(img_buffer, format="PNG")
    img_buffer.seek(0)

    return img_buffer
