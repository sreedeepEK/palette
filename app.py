import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import MiniBatchKMeans
import gradio as gr
import tempfile

def extract_colors(image_path, num_colors=5):
    
    
    """
    > Takes an image from the user
    > Returns dominant color based on image.
    
    """

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image.reshape(-1, 3).astype(float)

    kmeans = MiniBatchKMeans(n_clusters=num_colors, random_state=42, batch_size=1000, n_init=10)
    kmeans.fit(pixels)

    dominant_colors = kmeans.cluster_centers_.astype(int)
    return dominant_colors

def plot_palette_and_hex_code(colors):
    
    """Plot color palette and display hex codes using Matplotlib."""
    
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.axis("off")

    for i, color in enumerate(colors):
        rect = plt.Rectangle((i, 0), 1, 1, color=np.array(color) / 255.0)
        ax.add_patch(rect)

        hex_color = "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])
        ax.text(i + 0.5, -0.3, hex_color, ha='center', fontsize=12, color="black")

    ax.set_xlim(0, len(colors)) # display width
    ax.set_ylim(0, 1) # display height
    return fig

def process_image(image_path):
    
    """Extract colors, generate palette, and return the Matplotlib figure."""
   
    dominant_colors = extract_colors(image_path)

    # Save the Matplotlib figure as a temporary image file
    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig = plot_palette_and_hex_code(dominant_colors)
    fig.savefig(temp_file.name, format="png", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    return temp_file.name 

description_text = """
<div align='center'>
    <b>Palette - the range of colours used by a particular artist or in a particular picture.
Extract color palette from any image instantly. </b><br><br>
</div>
"""


# Gradio Interface
demo = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="filepath", height=400, width=700),
    outputs=gr.Image(type="filepath", height=400, width=700),  # Return image file path
    title="palette",
    description= description_text
)

demo.launch(share=True)
