import typing
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from adjustText import adjust_text
class GridBubble():
    def __init__(self, loss_func: typing.Callable, radius = 10, center = (0, 0)) -> None:
        self.radius = radius
        self.radiuss = [radius]
        self.points = []
        self.losses = [1]
        self.min_losses = []
        self.center = center
        self.centers = [center]
        self.previous_center_index = 4
        self.loss_func = loss_func
        self.step_size = 0

    def compute_points(self):
        x, y = self.center
        self.points.clear()
        self.points.append((x - self.radius, y + self.radius))
        self.points.append((x, y + self.radius))
        self.points.append((x + self.radius, y + self.radius))
        self.points.append((x - self.radius, y))
        self.points.append((x, y))
        self.points.append((x + self.radius, y))
        self.points.append((x - self.radius, y - self.radius))
        self.points.append((x, y - self.radius))
        self.points.append((x + self.radius, y - self.radius))
        return

    def fit(self, learning_rate = 2, threshold = 10**(-5), limit_step = 1000):
        if learning_rate < 1:
            raise Exception('The learing rate must be not smaller than 1')
        self.compute_points()
        self.step_size = 0
        while(self.step_size <= limit_step):
            self.step_size += 1
            # Compute losses
            # This can be improve by only compute the loss
            # for new point, old points do not need. Hope you member late :)
            self.losses.clear()
            for x, y in self.points:
                self.losses.append(self.loss_func(x, y))
            a = np.array(self.losses)
            min_loss = a[np.isfinite(a)].min()
            # print(str(self.step_size) + ': ' + str(min_loss))

            self.min_losses.append(min_loss)
            indices_min = np.where(self.losses == min_loss)[0]
            # Update center and radius
            # == 4 min the center is not changed
            # Can we adapt the changed radius ratio (not fix at 2)?
            if 4 in indices_min:
                self.radius /= learning_rate
            else:
                self.center = self.points[indices_min[0]]
                self.radius *= learning_rate
            # Update neighboor points
            self.compute_points()
            self.centers.append(self.center)
            self.radiuss.append(self.radius)
            if (min_loss < threshold):
                return self.center
        return self.center
    
    def plot(self, path = './'):
        for i in range(0, len(self.centers)):
            plt.gca().add_patch(Rectangle((self.centers[i][0] - self.radiuss[i], self.centers[i][1] - self.radiuss[i]), 2*self.radiuss[i], 2*self.radiuss[i], edgecolor='red',
                                        facecolor='none', lw=1))
            plt.xticks(np.arange(0, np.max(self.radiuss), 5.0))
            plt.yticks(np.arange(0, np.max(self.radiuss), 5.0))
            plt.legend(loc='upper left')
            if i > 0:
                x1, y1 = self.centers[i-1]
                x2, y2 = self.centers[i]
                plt.arrow(x1, y1, x2 - x1, y2 - y1, head_length=.2, head_width=.2, length_includes_head=True)
            texts = []
            for j in range(0, i + 1):
                texts.append(plt.text(x = self.centers[j][0] + 0.3, y = self.centers[j][1] + 0.3, s = str(j)))

                plt.scatter(self.centers[j][0], self.centers[j][1], s = 1, color='black')
            plt.title(str(i + 1) + '/' + str(len(self.centers)) + ', Center: (' + str(self.centers[i][0]) + ', ' + str(self.centers[i][1]) + '), Radius: ' +  str(self.radiuss[i]) + ', Loss: ' + str(np.around(self.min_losses[i], 2)))
            adjust_text(texts, only_move={'points':'y', 'texts':'y'})
            plt.savefig(path + 'Iteration ' + str(i + 1) + '.png', format='png', dpi=600)
            plt.clf()