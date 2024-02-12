from flask import Flask, request, jsonify
import cv2
import os
import numpy as np
import math
from minio import Minio, S3Error
import openai
from openai import OpenAI
import base64
from dotenv import load_dotenv
from pymongo import MongoClient
from bson import ObjectId, json_util
from PIL import Image
import time
import json
from datetime import datetime
import random

def s4():
    return '{:04x}'.format(random.randint(0, 0xffff))

def guid(length):
    if length == 8:
        return s4() + s4()
    elif length == 4:
        return s4()
    elif length == 12:
        return s4() + s4() + s4()
    else:
        return s4() + s4() + s4() + s4() + s4() + s4() + hex(int(time.time()))[2:]


# from pymongo import MongoClient
# from bson import ObjectId, json_util

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# api_key = os.environ['openai_api_key']
api_key = os.getenv("OPENAI_API_KEY")

openai.api_key = api_key
client = OpenAI(
    api_key = api_key
)
mongo_client = MongoClient("mongodb://localhost:27017")

# MinIO configuration
minio_endpoint = "localhost:9000"
minio_access_key = os.getenv("MINIO_ACCESS_KEY")
minio_secret_key = os.getenv("MINIO_SECRET_KEY")
minio_bucket_name = "screenshots"

minio_client = Minio(minio_endpoint, access_key=minio_access_key, secret_key=minio_secret_key, secure=False)

# client = MongoClient("mongodb://localhost:27017")

# Helper Functions for Dashed Rectangles (unchanged)

def distance(a, b):
    x1, y1 = a
    x2, y2 = b
    # distace
    d = round(math.dist(a, b), 2)
    return d

def between(x3, x1, x2):
    if x1 <= x3 and x3 <= x2:
        return True
    else:
        return False

def intersect(a, b):
    x1, y1, h1, b1 = a
    x3, y3, h2, b2 = b

    x2, y2 = x1 + h1, y1 + b1
    x4, y4 = x3 + h2, y3 + b2

    if between(x3, x1, x2) and between(y2, y3, y4):
        return True
    if between(x3, x1, x2) and between(y4, y1, y2):
        return True
    if between(x4, x1, x2) and between(y4, y1, y2):
        return True
    if between(x4, x1, x2) and between(y2, y3, y4):
        return True
    return False

def rect_distance(a, b, verbose=False):

    if intersect(a, b):
        if verbose:
            print("intersect")
        return 0

    else:
        # points
        x1, y1, h1, b1 = a
        x3, y3, h2, b2 = b

        x2, y2 = x1 + h1, y1 + b1
        x4, y4 = x3 + h2, y3 + b2

        if verbose:
            print(x1, y1)
            print(x2, y2)
            print("")
            print(x3, y3)
            print(x4, y4)

        right = x2 <= x3
        left = x4 <= x1

        bottom = y2 <= y3
        top = y4 <= y1

        # case : I - 1st quadrant
        if right and top:
            if verbose:
                print("Ist Quadrant")
            return distance((x2, y1), (x3, y4))

        # case: II - 2nd quadrant
        if top and left:
            if verbose:
                print("IInd Quadrant")
            return distance((x1, y1), (x4, y4))

        # case: III - 3rd quadrant
        if left and bottom:
            if verbose:
                print("IIIrd Quadrant")
            return distance((x1, y2), (x4, y3))

        # case: IV - 4th quadrand
        if bottom and right:
            if verbose:
                print("IVth Quadrant")
            return distance((x2, y2), (x3, y3))

        if left:
            if verbose:
                print("left")
            return abs(x1 - x4)

        if right:
            if verbose:
                print("right")
            return abs(x2 - x3)

        if top:
            if verbose:
                print("top")
            return abs(y4 - y1)

        if bottom:
            if verbose:
                print("bottom")
            return abs(y2 - y3)

        return -1

def area(rect):
    h, b = rect[2:]
    return h * b

def drawline(img, pt1, pt2, color, thickness=1, style='dotted', gap=5):
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** .5
    pts = []
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
        y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
        p = (x, y)
        pts.append(p)

    if style == 'dotted':
        for p in pts:
            cv2.circle(img, p, thickness, color, -1)
    else:
        s = pts[0]
        e = pts[0]
        i = 0
        for p in pts:
            s = e
            e = p
            if i % 2 == 1:
                cv2.line(img, s, e, color, thickness)
            i += 1

def drawpoly(img, pts, color, thickness=1, style='dotted', ):
    s = pts[0]
    e = pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s = e
        e = p
        drawline(img, s, e, color, thickness, style)

def drawrect(img, pt1, pt2, color, thickness=1, style='dotted'):
    pts = [pt1, (pt2[0], pt1[1]), pt2, (pt1[0], pt2[1])]
    drawpoly(img, pts, color, thickness, style)


# Function for Dashed Rectangles Creation in Image (unchanged)

def process_images_dashed_rectangles(slide1_path, slide2_path):
    # Load images
    image1 = cv2.imread(slide1_path)
    image2 = cv2.imread(slide2_path)

    # Calculate the absolute difference between the two images
    difference = cv2.absdiff(image1, image2)

    # Convert the difference to grayscale
    gray_diff = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to highlight the differences
    _, thresh_diff = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print("No of changes found i.e Contours ::", len(contours))

    rects = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        rects.append([x, y, w, h])

    all_rects = rects.copy()
    th = 35

    closer_rects = []
    # do till all rects are zero
    while len(all_rects) != 0:
        i = 0
        rect = all_rects[i].copy()
        del all_rects[i]
        closers = [rect]

        for i in range(len(all_rects)):
            r = all_rects[i]
            d = rect_distance(rect, r)
            area_b = area(r)
            area_r = area(rect)

            if area_b > 250 and area_r > 250:
                if d < 15:
                    closers.append(r)
            else:
                if d < th:
                    closers.append(r)

        for r in closers[1:]:
            del all_rects[all_rects.index(r)]

        closer_rects.append(closers)

    ##
    combines = []

    for closers in closer_rects:
        allx1 = [rect[0] for rect in closers]
        ally1 = [rect[1] for rect in closers]

        x1 = min(allx1)
        y1 = min(ally1)

        x2 = max([rect[0] + rect[2] for rect in closers])
        y2 = max([rect[1] + rect[3] for rect in closers])

        x, y, w, h = x1, y1, x2 - x1, y2 - y1
        combines.append([x, y, w, h])

    #### checking if combines two are near by:
    th = 10

    all_rects = combines.copy()

    closer_rects = []
    # do till all rects are zero
    while len(all_rects) != 0:
        i = 0
        rect = all_rects[i].copy()
        del all_rects[i]
        closers = [rect]

        for i in range(len(all_rects)):
            r = all_rects[i]
            d = rect_distance(rect, r)

            if d < th:
                closers.append(r)

        for r in closers[1:]:
            del all_rects[all_rects.index(r)]

        closer_rects.append(closers)

    closure_combines = []

    for closers in closer_rects:
        allx1 = [rect[0] for rect in closers]
        ally1 = [rect[1] for rect in closers]

        x1 = min(allx1)
        y1 = min(ally1)

        x2 = max([rect[0] + rect[2] for rect in closers])
        y2 = max([rect[1] + rect[3] for rect in closers])

        x, y, w, h = x1, y1, x2 - x1, y2 - y1
        closure_combines.append([x, y, w, h])

    ##### doing once again
    all_rects = closure_combines.copy()
    th = 35

    closer_rects = []
    # do till all rects are zero
    while len(all_rects) != 0:
        i = 0
        rect = all_rects[i].copy()
        del all_rects[i]
        closers = [rect]

        for i in range(len(all_rects)):
            r = all_rects[i]
            d = rect_distance(rect, r)
            area_b = area(r)
            area_r = area(rect)

            if area_b > 250 and area_r > 250:
                if d < 15:
                    closers.append(r)
            else:
                if d < th:
                    closers.append(r)

        for r in closers[1:]:
            del all_rects[all_rects.index(r)]

        closer_rects.append(closers)

    ##
    combines = []

    for closers in closer_rects:
        allx1 = [rect[0] for rect in closers]
        ally1 = [rect[1] for rect in closers]

        x1 = min(allx1)
        y1 = min(ally1)

        x2 = max([rect[0] + rect[2] for rect in closers])
        y2 = max([rect[1] + rect[3] for rect in closers])

        x, y, w, h = x1, y1, x2 - x1, y2 - y1
        combines.append([x, y, w, h])

    #### checking if combines two are near by:
    th = 10

    all_rects = combines.copy()

    closer_rects = []
    # do till all rects are zero
    while len(all_rects) != 0:
        i = 0
        rect = all_rects[i].copy()
        del all_rects[i]
        closers = [rect]

        for i in range(len(all_rects)):
            r = all_rects[i]
            d = rect_distance(rect, r)

            if d < th:
                closers.append(r)

        for r in closers[1:]:
            del all_rects[all_rects.index(r)]

        closer_rects.append(closers)

    closure_combines = []

    for closers in closer_rects:
        allx1 = [rect[0] for rect in closers]
        ally1 = [rect[1] for rect in closers]

        x1 = min(allx1)
        y1 = min(ally1)

        x2 = max([rect[0] + rect[2] for rect in closers])
        y2 = max([rect[1] + rect[3] for rect in closers])

        x, y, w, h = x1, y1, x2 - x1, y2 - y1
        closure_combines.append([x, y, w, h])

    #### closures finding done

    image2 = cv2.imread(slide2_path)

    labels = []
    # obj: {"labelno":,"coordinates":[x,y]}

    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, rect in enumerate(closure_combines):
        x, y, w, h = rect
        #     cv2.rectangle(image2, (x-5, y-5), (x + w + 5, y + h + 5), (0,0,255), 2)
        drawrect(image2, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 0, 255), 1, '')
        cv2.putText(image2, str(i + 1), (x, y - 8), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        labels.append({
            "labelno": i,
            "coordinates": [x, y, w, h]
        })
    # Extract filename from the given path
    _, slide2_filename = os.path.split(slide2_path)

    # Create a new filename for the dashed image
    dashed_image_filename = guid(0) + '.jpg'

    out_path = os.path.join("Images", dashed_image_filename)
    # out_path = "./Images/Slide{}dashed.jpg".format(image_index)
    cv2.imwrite(out_path, image2)

    # j = json.dumps(labels)

    # out_path = r"C:\Users\saura\AI_Projects\Image_Merge\Slide9labels.json"
    # with open(out_path, 'w') as f:
    #     json.dump(labels, f)

    return dashed_image_filename

# Function to upload image to MinIO
def upload_to_minio(local_path, minio_path):
    try:
        # Get the total size of the file
        total_size = os.path.getsize(local_path)

        # Perform multipart upload with known size
        with open(local_path, "rb") as file_data:
            minio_client.put_object(
                minio_bucket_name,
                minio_path,
                file_data,
                length=total_size,  # Specify the total size of the object
                content_type="application/octet-stream",
            )

        print(f"File uploaded to minio bucket-{minio_bucket_name} successfully.")

    except S3Error as e:
        print(f"Error: {e}")
    # with open(local_path, 'rb') as data:
    #     minio_client.put_object(minio_bucket_name, minio_path, data, length=total_size, content_type='application/octet-stream')

# Function to download image from MinIO
def download_from_minio(minio_path, local_path):
    with open(local_path, 'wb') as data:
        data.write(minio_client.get_object(minio_bucket_name, minio_path).read())


def merge_images(left_image_path, right_image_path, output_path):
    # Open the left and right images
    left_image = Image.open(left_image_path)
    right_image = Image.open(right_image_path)

    # Get the dimensions of the left and right images
    left_width, left_height = left_image.size
    right_width, right_height = right_image.size

    # Calculate the width and height of the combined image
    combined_width = left_width + right_width
    combined_height = max(left_height, right_height)

    # Create a new image with the calculated dimensions
    combined_image = Image.new('RGB', (combined_width, combined_height))

    # Paste the left image onto the combined image at (0, 0)
    combined_image.paste(left_image, (0, 0))

    # Paste the right image onto the combined image at (left_width, 0)
    combined_image.paste(right_image, (left_width, 0))

    # Save the combined image
    combined_image.save(output_path)


def image_comparision(image_path):
    # print(image_path)
    res = ""
    prompt_new1 = """ You are an expert at determining differences between images. 
    I am sending you one image which is a combined or merged image of two images.the two images are screenshot of an app or dashboard.

    Image in the left is the original or baseline image. 
    Image in the right may be slightly different than the image in the left. The places where the image in the right is different have been marked with a rectangle dashed border. Each red rectangle has a reference labelled number (1, 2 etc) just above it. 
    Now for each labeled rectangular area, perform a thorough  comparison within that area in the right image with the same area in the left or baseline image,focus on the following questions. 
    Here is the checklist to follow. The word content below refers to image or text or graph or the dashboard tile.

    1. Content Location: Is there any noticeable shift in the starting point of displayed content. This content could be text, images, graphs, or tiles. 
    2. Content Changes: Are there content changes between two images. This content could be text, images, graphs, or tiles. 
    3. Content Formatting: Is there a formatting differences between two images. Formatting changes may involve aspects like background color, border color, or size. 


    For each labelled rectangle visually inspect the images and answer the above questions and provide the answer in the following JSON format. Make sure every labelled rectangular area is covered. For example, if there are 5 rectangular areas, the json need to have answers for the above questions for each of the 5 areas. If there is a change answer Yes if not answer No.  In the observation you should refer to the evidence you found in the image for the change you marked as yes.

    {
      "Differences": [
        {
          "rectangle label": 1,
          "questions": [
            {
              "question_text": "Content Location",
              "Changed": "Yes"
              "Observation": "text and the images got shited right"
            },
            {
              "question_text": "Content Change",
              "Changed": "Yes"
              "answer": "Metric value got changed"
            },
            {
              "question_text": "Content Formatting",
              "Changed": "No"
              "answer": "No formatting changes"
            },
          ]
        },
      ]
    }

    """

    # def encode_image(image_path):
    #     with open(image_path, "rb") as image_file:
    #         return base64.b64encode(image_file.read()).decode('utf-8')


    # print(image_path)
    with open(image_path, "rb") as image_file:
        base64_image=base64.b64encode(image_file.read()).decode('utf-8')
    # base64_image = encode_image(image_path)
    # print(base64_image)


    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        temperature=0,
        messages=[{"role": "system",
                   "content": "You are a helpful assistant who is expert in determining differences in two images."},
                  {"role": "user",
                   "content": [
                       {
                           "type": "text",
                           "text": prompt_new1,
                       },
                       {
                           "type": "image_url",
                           "image_url": {
                               "url": f"data:image/jpeg;base64,{base64_image}",
                           },
                       },
                   ],
                   }
                  ],
        max_tokens=2000,
    )

    res = eval(response.choices[0].json().replace("null", "0"))['message']['content']

    triple_quoted_res = f"""{res}"""
    start_index = triple_quoted_res.find("{")
    end_index = triple_quoted_res.rfind("}")

    # Extract the JSON part
    json_part = triple_quoted_res[start_index:end_index + 1]

    # Load the JSON string into a Python dictionary
    json_data = json.loads(json_part)
    # print(json_data)


        # print(res)
    return json_data



def analyze_dashboard(image_path):
    # res = ""
    prompt = """You are expert at determining if dashboard adheres to the best practices.
    I am sending you an image which is a dashboard. Visually inspect the image and answer the following questions based on what you see in the image as evidence:
    1. Border Uniformity: Is the content of each tile contained within its own border & aligned uniformly?
    2. Color Consistency: Are all the tiles the same background and border color?
    3. Spacing Symmetry: Is the horizontal and vertical spacing between the tiles the same?
    4. Font Uniformity: Are all the text titles in a dashboard tile the same font size or color?
    5. Alignment Consistency: Are metric values in all the tiles uniformly aligned both vertically/horizontally?
    6. Metric Uniformity: Do all the metric values have the same font size, color, or background color in all the tiles?
    7. Boundary Breach: Is any text, image, or graph touching the tile boundary?
    8. Missing Values: Does any metric value display as 0, NA, -, or blank?
    9. Chart Clarity: Do charts, if present, have clear axis labels, chart titles, or legends?
    10. Text Clarity: Is there any text that is overlapping or hard to read?
    Provide the response in the following JSON format
    {
        "dashboardAnalysis": [
            {
                "questionTitle": "Text Clarity",
                "Present": "No",
                "Observation": "Some text overlaps, making it hard to read in certain tiles."
            },
        ]
    }
    Only refer to the image I just provided. You should only answer the question based on evidence from the image currently provided.  Please double check your answer. In the observation you should refer to the evidence you found in the image for the issue you marked as yes."""

    # image_path_template = "./Energy_Images/Slide1.JPG"

    # def encode_image(image_path):
    #     with open(image_path, "rb") as image_file:
    #         return base64.b64encode(image_file.read()).decode('utf-8')


    with open(image_path, "rb") as image_file:
        base64_image=base64.b64encode(image_file.read()).decode('utf-8')
    # base64_image = encode_image(image_path)

    try:

        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ],
                }
            ],
            max_tokens=1500,
        )

        res = eval(response.choices[0].json().replace("null", "0"))['message']['content']
        # print(res)

        triple_quoted_res = f"""{res}"""
        start_index = triple_quoted_res.find("{")
        end_index = triple_quoted_res.rfind("}")

        # Extract the JSON part
        json_part = triple_quoted_res[start_index:end_index + 1]

        # Load the JSON string into a Python dictionary
        json_data = json.loads(json_part)
        # print(json_data)

    except:
        res=""

    return json_data




@app.route('/v1/compareimages', methods=['POST'])
def process_images_api():
    try:
        data = request.get_json()

        slide1_path = data.get('baseline_image', "")
        slide2_path = data.get('implemented_image', "")
        comparejob_id = data.get('comparejob_id', "")

        if not slide1_path or not slide2_path or not comparejob_id:
            return jsonify({'error': 'Baseline and NewImage paths are required'}), 400

        # Download images from MinIO
        local_slide1_path = 'slide1.jpg'
        local_slide2_path = 'slide2.jpg'
        download_from_minio(slide1_path, local_slide1_path)
        download_from_minio(slide2_path, local_slide2_path)

        # Process images and get the path of the processed image
        dashed_rect_image_name = process_images_dashed_rectangles(local_slide1_path, local_slide2_path)

        # Upload result image to MinIO
        dashed_rect_image_path = os.path.join("Images", dashed_rect_image_name)
        upload_to_minio(dashed_rect_image_path, dashed_rect_image_name)

        merged_image_path = "./Images/Slide_merged.jpg"

        merge_images(local_slide1_path, dashed_rect_image_path, merged_image_path)

        # time.sleep(10)

        # slide1_name = os.path.basename(local_slide1_path).split('.')[0]
        # slide2_name = os.path.basename(result_image_path).split('.')[0]
        #
        # # Creating a new name for the merged image
        # merged_image_name = f'{slide1_name}_{slide2_name}_merged.jpg'
        #
        # upload_to_minio(merged_image_path, merged_image_name)

        print("Waiting for GPT Response")

        image_comp_res=image_comparision(merged_image_path)

        best_practice_res=analyze_dashboard(local_slide2_path)

        # Adding a delay of  60 seconds (adjust as needed)
        # time.sleep(10)

        # Statement after the delay
        print("Execution Done")

        result = {
            "Baseline": slide1_path,
            "NewImage": slide2_path,
            "DashedImageFilename": dashed_rect_image_name,
            "Image_Comp_response": image_comp_res,
            "best_practice_res": best_practice_res
        }
        db = mongo_client["airegression"]
        collections = db["comparejobs"]
        print(collections)
        

        collections.update_one({"_id": ObjectId(comparejob_id)}, {
            '$set': {
                'status': 'DONE',
                'updatedAt': datetime.now(),
                'markedImage': dashed_rect_image_name,
                'bestPracticesList': best_practice_res,
                'imageComparisionList': image_comp_res
            }
        })

        # Clean up local files
        os.remove(local_slide1_path)
        os.remove(local_slide2_path)
        os.remove(dashed_rect_image_path)
        os.remove(merged_image_path)

        return jsonify(result), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5001)
