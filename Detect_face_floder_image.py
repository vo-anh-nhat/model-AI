import face_recognition as fr
import cv2
import numpy as np
import os

train_folder = r"C:\MEARCHING LEARNING\train\\"
test_image_path = r"C:\MEARCHING LEARNING\test\nhom3.jpg"
output_image_path = r"C:\MEARCHING LEARNING\output.jpg"
# Load folder train
known_names = []
known_encodings = []

if not os.path.exists(train_folder):
    print("❌ Folder not detect!")
    exit()

train_images = os.listdir(train_folder)
if len(train_images) == 0:
    print("❌ Not have image in folder train!")
    exit()

print("📚 Loading image...")
for filename in train_images:
    try:
        image_path = os.path.join(train_folder, filename)
        image = fr.load_image_file(image_path)
        encodings = fr.face_encodings(image)
        if len(encodings) == 0:
            print(f"⚠️ Not detect face in {filename}")
            continue
        encoding = encodings[0]
        name = os.path.splitext(filename)[0].capitalize()
        known_encodings.append(encoding)
        known_names.append(name)
        print(f"✅ Added: {name}")
    except Exception as e:
        print(f"❌ Image error {filename}: {e}")

if len(known_encodings) == 0:
    print("❌ Not have face human in folder train!")
    exit()

# Load folder test
print("\n🔍 Loading folder test...")

image = cv2.imread(test_image_path)
if image is None:
    print("❌ Can not read image in folder test.")
    exit()

rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
face_locations = fr.face_locations(rgb_image)
face_encodings = fr.face_encodings(rgb_image, face_locations)

print(f"🔎 Detect {len(face_locations)} face human in folder test.")

if len(face_encodings) == 0:
    print("❌ Not detect face human to laod.")
    exit()

# Recognition and legend
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    distances = fr.face_distance(known_encodings, face_encoding)
    best_match_index = np.argmin(distances)

    name = "Unknown human"
    if distances[best_match_index] < 0.45:  # Increase accuracy
        name = known_names[best_match_index]

    # Draw frame and write name
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.rectangle(image, (left, bottom - 30), (right, bottom), (0, 255, 0), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image, name, (left + 6, bottom - 8), font, 0.8, (255, 255, 255), 1)

cv2.imshow("🔍 Result was detected", image)
cv2.imwrite(output_image_path, image)
cv2.waitKey(0)
cv2.destroyAllWindows()
