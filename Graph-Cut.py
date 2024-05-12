import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, draw
from sklearn.mixture import GaussianMixture
import maxflow
from collections import defaultdict


def rgb2yiq(rgb):
    YIQ = np.array([[0.299, 0.587, 0.114],
                    [0.59590059, -0.27455667, -0.32134392],
                    [0.21153661, -0.52273617, 0.31119955]])
    return np.dot(rgb.reshape(-1, 3), YIQ.T).reshape(rgb.shape)

class InteractiveImageSegmenter:
    def __init__(self, image_path):
        self.image = io.imread(image_path)
        if self.image.ndim == 3 and self.image.shape[2] == 4:
            self.image = color.rgba2rgb(self.image)  # Convert RGBA to RGB if necessary
        self.mask = np.zeros(self.image.shape[:2], dtype=np.uint8)  # Initialize mask for scribbles
        self.current_label = 1  # Initialize current_label to 1 (foreground)
        self.colors = {1: 'blue', 2: 'red'}  # Foreground in blue, background in red
        self.fig, self.ax = plt.subplots()
        self.ax.imshow(self.image)
        self.cid_press = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_move = self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)
        self.kid = self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.is_pressed = False  # Initialize mouse press state
        plt.show()

    def on_press(self, event):
        if event.inaxes is not None:
            self.is_pressed = True
            self.last_x, self.last_y = int(event.xdata), int(event.ydata)

    def on_release(self, event):
        self.is_pressed = False

    def on_move(self, event):
        if self.is_pressed and event.inaxes is not None:
            x, y = int(event.xdata), int(event.ydata)
            rr, cc = draw.line(self.last_y, self.last_x, y, x)
            self.mask[rr, cc] = self.current_label
            self.ax.plot([self.last_x, x], [self.last_y, y], color=self.colors[self.current_label], linewidth=5)
            self.fig.canvas.draw()
            self.last_x, self.last_y = x, y

    def on_key(self, event):
        if event.key == 'm':
            self.current_label = 3 - self.current_label  # Toggle between 1 (foreground) and 2 (background)
            label_name = 'foreground' if self.current_label == 1 else 'background'
            print(f"Switched to {label_name}")
        elif event.key == 'x':
            self.segment_image()

    def compute_pdfs(self):
        yuv = rgb2yiq(self.image)
        comps = defaultdict(lambda: np.array([]).reshape(0,3))
        for (i, j), value in np.ndenumerate(self.mask):
            if value:  # Using only marked pixels
                comps[value] = np.vstack([comps[value], yuv[i, j]])
        mu, Sigma = {}, {}
        for c in comps:
            if comps[c].size > 0:
                mu[c] = np.mean(comps[c], axis=0)
                Sigma[c] = np.cov(comps[c].T)
        return mu, Sigma

    def segment_image(self):
        mu, Sigma = self.compute_pdfs()  # Obtain mu and Sigma for each scribble type

        # Setup Gaussian Mixture Models based on computed means and covariances
        fg_gmm = GaussianMixture(n_components=1, means_init=[mu[1]],
                                  precisions_init=[np.linalg.inv(Sigma[1])])
        bg_gmm = GaussianMixture(n_components=1, means_init=[mu[2]],
                                  precisions_init=[np.linalg.inv(Sigma[2])])

        # Fit models just to formalize
        yuv_image = rgb2yiq(self.image)
        fg_gmm.fit(yuv_image[self.mask == 1].reshape(-1, 3))
        bg_gmm.fit(yuv_image[self.mask == 2].reshape(-1, 3))

        # Compute probabilities for each pixel
        fg_probs = fg_gmm.score_samples(yuv_image.reshape(-1, 3)).reshape(self.image.shape[:2])
        bg_probs = bg_gmm.score_samples(yuv_image.reshape(-1, 3)).reshape(self.image.shape[:2])

        # Setup and compute the graph cut
        g = maxflow.Graph[float]()
        nodeids = g.add_grid_nodes(self.image.shape[:2])
        structure = np.array([[0, 2, 0], [2, 0, 2], [0, 2, 0]])  # 4-connected
        g.add_grid_edges(nodeids, structure=structure, weights=50, symmetric=True)

        # Add terminal edges
        for y in range(self.image.shape[0]):
            for x in range(self.image.shape[1]):
                g.add_tedge(nodeids[y, x], -fg_probs[y, x]*25, -bg_probs[y, x]*25)

        # Find the maximum flow
        g.maxflow()
        sgm = g.get_grid_segments(nodeids)  # Get the segments of the nodes in the grid.

        # Convert result to a binary mask
        segmented = np.int_(np.logical_not(sgm))

        # Display results
        segmented_image = self.image.copy()
        segmented_image[segmented == 1] = (255, 255, 255)  # Set background pixels to black
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(self.image)
        plt.title('Original Image')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(segmented_image)
        plt.title('Segmented Image')
        plt.axis('off')
        plt.show()


if __name__ == "__main__":
    segmenter = InteractiveImageSegmenter('/Users/ziad/Downloads/LatexCurrent/img/XAI/6/6.jpg')
    #segmenter = InteractiveImageSegmenter('/Users/ziad/Desktop/rabbit.png')
