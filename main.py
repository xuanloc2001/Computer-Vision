import PySimpleGUI as sg
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import matplotlib.patches as patches
import argparse
import io
import logging
model_path = './trained_model/frozen_inference_graph.pb'

def draw_box(box, image_np):
    
    box += np.array([-(box[2] - box[0])/2, -(box[3] - box[1])/2, (box[2] - box[0])/2, (box[3] - box[1])/2]) 

    fig = plt.figure()
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    fig.add_axes(ax)

    #draw blurred boxes around box
    ax.add_patch(patches.Rectangle((0,0),box[1]*image_np.shape[1], image_np.shape[0],linewidth=0,edgecolor='none',facecolor='w',alpha=0.8))
    ax.add_patch(patches.Rectangle((box[3]*image_np.shape[1],0),image_np.shape[1], image_np.shape[0],linewidth=0,edgecolor='none',facecolor='w',alpha=0.8))
    ax.add_patch(patches.Rectangle((box[1]*image_np.shape[1],0),(box[3]-box[1])*image_np.shape[1], box[0]*image_np.shape[0],linewidth=0,edgecolor='none',facecolor='w',alpha=0.8))
    ax.add_patch(patches.Rectangle((box[1]*image_np.shape[1],box[2]*image_np.shape[0]),(box[3]-box[1])*image_np.shape[1], image_np.shape[0],linewidth=0,edgecolor='none',facecolor='w',alpha=0.8))

    return fig, ax


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def run_wally_finder(image_path):
    try:
        image_np = np.array(Image.open(image_path))
        with detection_graph.as_default():
            with tf.compat.v1.Session(graph=detection_graph) as sess:
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')

                # Actual detection.
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: np.expand_dims(image_np, axis=0)})

                if scores[0][0] < 0.1:
                    sg.popup_error('Wally not found :(', title='Wally Finder')
                    return

                print('Wally found')
                fig, ax = draw_box(boxes[0][0], image_np)
                ax.imshow(image_np)
                plt.show()
    except Exception as e:
        logging.exception("Error in run_wally_finder: %s", e)


def main():
    sg.theme('LightGrey1')

    layout = [
        [sg.Text("Choose an image")],
        [sg.InputText(key="image_path", readonly=True),
         sg.FileBrowse(file_types=(("Image files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp"),))],
        [sg.Button("Run Wally Finder"), sg.Button("Exit")],
        [sg.Image(key="-IMAGE-")],
    ]

    window = sg.Window("Wally Finder App", layout, finalize=True)

    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED or event == "Exit":
            break

        if event == "Run Wally Finder":
            image_path = values["image_path"]
            if image_path:
                run_wally_finder(image_path)
            else:
                sg.popup_warning("Please choose an image first", title="Warning")

        if event == "image_path":
            try:
                image_path = values["image_path"]
                image = Image.open(image_path)
                image.thumbnail((400, 400))
                bio = io.BytesIO()
                image.save(bio, format="PNG")
                window["-IMAGE-"].update(data=bio.getvalue())
            except Exception as e:
                print(f"Error updating image: {e}")
                logging.exception("Error updating image: %s", e)

    window.close()


if __name__ == "__main__":
    import tensorflow as tf
    import matplotlib.patches as patches

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    main()