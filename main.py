import tensorflow as tf
import numpy as np
import PIL.Image
from tqdm import tqdm
from models.mobileNet import MobileNet
from scipy.ndimage.filters import gaussian_filter


def load_image(filename, max_size=None):
    image = PIL.Image.open(filename)

    if max_size is not None:
        # rescale appropriately
        factor = max_size / np.max(image.size)

        # Scale the image's height and width.
        size = np.array(image.size) * factor
        size = size.astype(int)
        # the size of input tensor doesnt change
        image = image.resize(size, PIL.Image.LANCZOS)

    return np.float32(image)


def save_image(image, filename):
    image = np.clip(image, 0.0, 255.0)
    image = image.astype(np.uint8)

    with open(filename, 'wb') as file:
        PIL.Image.fromarray(image).save(file, 'jpeg')


# mean square error
def mse(a, b):
    return tf.reduce_mean(tf.square(a - b))


def create_content_loss(session, model, content_image, layer_ids):
    feed_dict = model.create_feed_dict(image=content_image)
    layers = model.get_layer_tensors(layer_ids)
    values = session.run(layers, feed_dict=feed_dict)

    with model.graph.as_default():
        layer_losses = []

        for value, layer in zip(values, layers):
            value_const = tf.constant(value)
            loss = mse(layer, value_const)

            layer_losses.append(loss)

        total_loss = tf.reduce_mean(layer_losses)

    return total_loss


def gram_matrix(tensor):
    shape = tensor.get_shape()
    num_channels = int(shape[3])
    matrix = tf.reshape(tensor, shape=[-1, num_channels])
    gram = tf.matmul(tf.transpose(matrix), matrix)

    return gram


def create_style_loss(session, model, style_image, layer_ids):
    feed_dict = model.create_feed_dict(image=style_image)
    layers = model.get_layer_tensors(layer_ids)

    with model.graph.as_default():
        gram_layers = [gram_matrix(layer) for layer in layers]

        values = session.run(gram_layers, feed_dict=feed_dict)
        layer_losses = []

        for value, gram_layer in zip(values, gram_layers):
            value_const = tf.constant(value)
            loss = mse(gram_layer, value_const)
            layer_losses.append(loss)

        total_loss = tf.reduce_mean(layer_losses)

    return total_loss


def weak_style_transfer(session, model, content_image, style_image,
                        content_layer_ids, style_layer_ids,
                        weight_content=1.5, weight_style=10.0,
                        num_iterations=120, step_size=10.0, log_images=True, freqP_flay=False):
    # Create the loss-function for the content-layers and -image.
    loss_content = create_content_loss(session=session, model=model, content_image=content_image,
                                       layer_ids=content_layer_ids)

    # Create the loss-function for the style-layers and -image.
    loss_style = create_style_loss(session=session, model=model, style_image=style_image, layer_ids=style_layer_ids)

    # Create TensorFlow variables for adjusting the values of
    # the loss-functions. This is explained below.
    adj_content = tf.Variable(1e-10, name='adj_content')
    adj_style = tf.Variable(1e-10, name='adj_style')

    # Initialize the adjustment values for the loss-functions.
    session.run([adj_content.initializer, adj_style.initializer])

    update_adj_content = adj_content.assign(1.0 / (loss_content + 1e-10))
    update_adj_style = adj_style.assign(1.0 / (loss_style + 1e-10))

    loss_combined = weight_content * adj_content * loss_content + weight_style * adj_style * loss_style

    gradient = tf.gradients(loss_combined, model.input)

    # List of tensors that we will run in each optimization iteration.
    run_list = [gradient, update_adj_content, update_adj_style]

    mixed_image = content_image - 40

    for i in tqdm(range(num_iterations)):
        # Create a feed-dict with the mixed-image.
        feed_dict = model.create_feed_dict(image=mixed_image)

        grad, adj_content_val, adj_style_val = session.run(run_list, feed_dict=feed_dict)

        grad = np.squeeze(grad)

        # Scale the step-size according to the gradient-values.
        step_size_scaled = step_size / (np.std(grad) + 1e-9)

        mixed_image -= grad * step_size_scaled

        mixed_image = np.clip(mixed_image, 0.0, 255.0)
        if (log_images == True):
            if (i % 5 == 0) and freqP_flay == False:
                print("Iteration:", i)
                newIM = gaussian_filter(mixed_image, sigma=.3)
                save_image(newIM, "img/newStyle/nStyle" + str(i) + ".jpg")

            if (freqP_flay == True and i % 5 == 0):
                save_image(mixed_image, "img/newStyle/zStyle" + str(i) + ".jpg")

            newIM = gaussian_filter(mixed_image, sigma=.3)
            save_image(newIM, "img/newStyle/FinalImage.jpg")

    # Return the mixed-image.
    return mixed_image


def save_image_smoothed(image, filename, sigma):
    image = gaussian_filter(image, sigma=sigma)

    image = np.clip(image, 0.0, 255.0)

    # convert to bytes
    image = image.astype(np.uint8)

    with open(filename, 'wb') as file:
        PIL.Image.fromarray(image).save(file, 'jpeg')
    return image

if __name__ == "__main__":
    content_filename = 'img/sources/por.jpg'
    content_image = load_image(content_filename, max_size=1200)

    style_filename = 'img/styles/blackred.jpg'
    style_image = load_image(style_filename, max_size=1200)

    content_layer_ids = list([1, 2])
    style_layer_ids = list(range(8))

    model = MobileNet()

    # Create a TensorFlow-session.
    session = tf.InteractiveSession(graph=model.graph)
    img = weak_style_transfer(session=session, model=model, content_image=content_image,
                              style_image=style_image,
                              content_layer_ids=content_layer_ids,
                              style_layer_ids=style_layer_ids,
                              weight_content=2.0,
                              weight_style=50.0,
                              num_iterations=40,
                              step_size=9.0, log_images=False, freqP_flay=False)
    save_image(img, "img/newStyle/zStyle.jpg")

    # apply heavily styled and unsmoothed image onto original image
    img = weak_style_transfer(session=session, model=model, content_image=content_image,
                              style_image=img,
                              content_layer_ids=list(range(4)),
                              style_layer_ids=content_layer_ids,
                              weight_content=1.0,
                              weight_style=5.0,
                              num_iterations=40,
                              step_size=5.0, log_images=False, freqP_flay=False)

    save_image_smoothed(img, "img/newStyle/Style.jpg", 0.3)

    session.close()
