from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image as PillowImage
from config.globals import manipulation_detection_model
from ezmm import Image
from third_party.TruFor.src.trufor_config import _C as config
from third_party.TruFor.src.trufor_config import update_config
from torch.nn import functional as F


# import sys
# import os
# Add the project root to sys.path
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
# if project_root not in sys.path:
#    sys.path.append(project_root)


def analyze_image(image: PillowImage.Image):
    update_config(config, None, path="third_party/TruFor/src/trufor.yaml")

    # Convert the PIL image to a tensor
    rgb = preprocess_image(image)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rgb = rgb.to(device)

    # Load the model
    model = load_model(config, manipulation_detection_model, device)

    # Perform the analysis
    model.eval()
    with torch.no_grad():
        pred, conf, det, npp = model(rgb)

    # Process the outputs
    if conf is not None:
        conf = torch.squeeze(conf, 0)
        conf = torch.sigmoid(conf)[0]
        conf = conf.cpu().numpy()

    if npp is not None:
        npp = torch.squeeze(npp, 0)[0]
        npp = npp.cpu().numpy()

    if det is not None:
        det_sig = torch.sigmoid(det).item()

    pred = torch.squeeze(pred, 0)
    pred = F.softmax(pred, dim=0)[1]
    pred = pred.cpu().numpy()

    result = {
        'map': pred,
        'score': det_sig if det is not None else None,
        'conf': conf,
        'np++': npp
    }

    # Visualize and save the result
    # save_visualization(image, result, config.OUTPUT_DIR)

    return result


def load_model(config, model_file, device):
    print('=> loading model from {}'.format(model_file))
    checkpoint = torch.load(model_file, map_location=device)

    if config.MODEL.NAME == 'detconfcmx':
        from third_party.TruFor.src.models.cmx.builder_np_conf import myEncoderDecoder as confcmx
        model = confcmx(cfg=config)
    else:
        raise NotImplementedError('Model not implemented')

    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    return model


def create_visualizations(result: dict) -> dict:
    """
    Create visualizations for the localization map, confidence map, and Noiseprint++ (if available),
    and return them as Pillow images.
    
    :param result: A dictionary containing the results of the manipulation detection.
    :return: A dictionary containing Pillow images for the localization, confidence maps, and Noiseprint++.
    """

    # Visualization for the localization map
    fig_loc, ax_loc = plt.subplots()
    ax_loc.imshow(result['map'], cmap='RdBu_r', clim=[0, 1])
    ax_loc.set_title('Localization map')
    ax_loc.axis('off')

    # Save the localization map to a buffer
    buf_loc = BytesIO()
    plt.savefig(buf_loc, format='png', bbox_inches='tight')
    buf_loc.seek(0)
    localization_map = Image(pillow_image=PillowImage.open(buf_loc).convert("RGB"))
    plt.close(fig_loc)

    # Visualization for the confidence map
    fig_conf, ax_conf = plt.subplots()
    ax_conf.imshow(result['conf'], cmap='gray', clim=[0, 1])
    ax_conf.set_title('Confidence map')
    ax_conf.axis('off')

    # Save the confidence map to a buffer
    buf_conf = BytesIO()
    plt.savefig(buf_conf, format='png', bbox_inches='tight')
    buf_conf.seek(0)
    confidence_map = Image(pillow_image=PillowImage.open(buf_conf).convert("RGB"))
    plt.close(fig_conf)

    # Visualization for the Noiseprint++ if available
    if 'np++' in result:
        noisepr = result['np++']
        fig_noise, ax_noise = plt.subplots()
        # Remove border and down-sample for better visualization
        ax_noise.imshow(noisepr[16:-16:5, 16:-16:5], cmap='gray')
        ax_noise.set_title('Noiseprint++')
        ax_noise.axis('off')

        # Save the Noiseprint++ map to a buffer
        buf_noise = BytesIO()
        plt.savefig(buf_noise, format='png', bbox_inches='tight')
        buf_noise.seek(0)
        noiseprint_map = Image(pillow_image=PillowImage.open(buf_noise).convert("RGB"))
        plt.close(fig_noise)

        result["noiseprint_map"] = noiseprint_map
    else:
        result["noiseprint_map"] = None  # No Noiseprint++ data

    # Add the created Pillow images to the result dictionary
    result["localization_map"] = localization_map
    result["confidence_map"] = confidence_map

    return result


# def create_visualizations_and_save(result: dict) -> dict:
#    """
#    Create visualizations for the localization map and confidence map, save them as files, and return Image objects.
#    
#    :param result: A dictionary containing the results of the manipulation detection.
#    :param temp_dir: A Path object specifying the directory to save the images.
#    :return: A dictionary containing Image objects for the localization and confidence maps.
#    """
#
#    # Save the localization map
#    create_visualization_image(result['map'], 'Localization map', loc_map_path)
#    result["ref_loc_map"] = 
#
#    # Save the confidence map
#    conf_map_path = temp_dir / "confidence_map.png"
#    create_visualization_image(result['conf'], 'Confidence map', conf_map_path)
#    result["ref_conf_map"] = media_registry.add(Image(conf_map_path))
#
#    return result
#
# def create_visualization_image(data, title: str, output_path: Path):
#    """
#    Create a visualization image from data and save it to a file.
#    
#    :param data: The image data to visualize.
#    :param title: Title of the plot.
#    :param output_path: Path where the image will be saved.
#    """
#    fig, ax = plt.subplots()
#    ax.imshow(data, cmap='RdBu_r' if title == 'Localization map' else 'gray', clim=[0, 1])
#    ax.set_title(title)
#    ax.axis('off')
#
#    # Save the plot as an image
#    plt.savefig(output_path, format='png', bbox_inches='tight')
#    plt.close(fig)
#

def save_visualization(image, result, output_plot_path, mask=None):
    cols = 3
    if mask is not None:
        cols += 1

    if 'np++' in result:
        cols += 1
        noisepr = result['np++']
    else:
        noisepr = None

    # Create the plot
    fig, axs = plt.subplots(1, cols)
    fig.suptitle('score: %.3f' % result['score'] if result['score'] is not None else 'No score available')

    # Disable axis for all subplots
    for ax in axs:
        ax.axis('off')

    # Plot the image
    index = 0
    ax = axs[index]
    ax.imshow(image)
    ax.set_title('Image')

    # Plot the ground truth mask if available
    if mask is not None:
        index += 1
        ax = axs[index]
        ax.imshow(mask, cmap='gray')
        ax.set_title('Ground Truth')
        ax.set_yticks([]), ax.set_xticks([]), ax.axis('on')

    # Plot the noiseprint++ if available
    if noisepr is not None:
        index += 1
        ax = axs[index]
        ax.imshow(noisepr[16:-16:5, 16:-16:5], cmap='gray')
        ax.set_title('Noiseprint++')

    # Plot the localization map
    index += 1
    ax = axs[index]
    ax.imshow(result['map'], cmap='RdBu_r', clim=[0, 1])
    ax.set_title('Localization map')

    # Plot the confidence map
    index += 1
    ax = axs[index]
    ax.imshow(result['conf'], cmap='gray', clim=[0, 1])
    ax.set_title('Confidence map')
    ax.set_yticks([]), ax.set_xticks([]), ax.axis('on')

    # Save the figure to a file instead of displaying it
    plt.savefig(output_plot_path, bbox_inches='tight')
    plt.close()


def preprocess_image(image: Image) -> torch.Tensor:
    """ Converts a PIL Image to a tensor similar to the myDataset class.
        - Converts to RGB if necessary.
        - Normalizes to the range [0, 1] by dividing by 256.0.
        - Transposes the array to match (C, H, W) format expected by the model.
    """
    img_RGB = np.array(image)  # image.convert("RGB")
    tensor = torch.tensor(img_RGB.transpose(2, 0, 1), dtype=torch.float) / 256.0
    return tensor.unsqueeze(0)  # Add batch dimension

# Example:
# image = PillowImage.open("/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/tb17xity/InFact/third_party/TruFor/test_docker/images/pristine1.jpg")
# result = analyze_image(image)

# The 'result' variable contains the numeric vectors (e.g., 'map', 'score', 'conf', 'np++')
