import cv2
import numpy as np 
import face_recognition as fr

video_capture = cv2.VideoCapture(0)

# Load the image and face encoding
image = fr.load_image_file("Psize_photograph.jpg")
image_face_encoding = fr.face_encodings(image)[0]
known_face_encodings = [image_face_encoding]
known_face_names = ["kareem"]

while True:
    ret, frame = video_capture.read()
    rgb_frame = frame[:, :, ::-1]

    # Find face locations and encodings in the current frame
    fc_locations = fr.face_locations(rgb_frame)
    fc_encodings = fr.face_encodings(rgb_frame, fc_locations)

    for (top, right, bottom, left), face_encoding in zip(fc_locations, fc_encodings):
        # Compare face encodings with known face encodings
        matches = fr.compare_faces(known_face_encodings, face_encoding)
        name = "unknown"

        fc_distances = fr.face_distance(known_face_encodings, face_encoding)
        match_index = np.argmin(fc_distances)

        if matches[match_index]:
            name = known_face_names[match_index]

        # Define font before using it
        font = cv2.FONT_ITALIC

        # Draw rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        # Draw the name label
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    cv2.imshow('FACE DETECTION CODECLAUSE', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
