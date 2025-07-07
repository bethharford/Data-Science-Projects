import png
import numpy as np

#WIDTH = 3840
#HEIGHT = 2160

WIDTH = 1920
HEIGHT = 1080

offset = np.array([0.0, 0.0])
scale = 1.0
imaginary_limits = np.array(offset[1], offset[1]) + np.array([-1, 1]) * scale
real_limits = (WIDTH / HEIGHT) * np.array([-1, 1]) * scale + np.array([offset[0], offset[0]])

class Fractal:
    def __init__(self, image_width, image_height,
                 real_limits, imaginary_limits,
                 max_iterations=100, escape_threshold=2, escape_delay_constant=100):
        self.image_width = image_width
        self.image_height = image_height
        self.real_limits = real_limits
        self.imaginary_limits = imaginary_limits
        self.max_iterations = max_iterations
        self.escape_threshold = escape_threshold
        self.escape_delay_constant = escape_delay_constant
        
        self.complex_space = np.array(
            [[self.pixel_to_complex(j, i) for j in range(0, self.image_width)] for i in range(0, self.image_height)]
            ).astype(np.cdouble)
        
        self.image = None
    
    def pixel_to_complex(self, x, y):
        real = self.real_limits[0] + (x / self.image_width) * (self.real_limits[1] - self.real_limits[0])
        imaginary = self.imaginary_limits[0] + (y / self.image_height) * (self.imaginary_limits[1] - self.imaginary_limits[0])
        return real + imaginary * 1j
    
    def math_function(self, z, c):
        return np.add(np.square(np.square(z)), c)
    
    def escape_iterations(self, c):
        iterations = np.zeros((self.max_iterations, self.image_height, self.image_width))
        z = self.complex_space
        for n in range(0, self.max_iterations):
            z = self.math_function(z, c)
            iterations[n] = np.where(np.abs(z) < self.escape_threshold, 1, 0)
        #escape_delays = np.divide(np.sum(iterations, axis=0), self.escape_delay_constant)
        escape_delays = np.sum(iterations, axis=0)
        
        return escape_delays
    
    def colour_map(self, rgb):
        #r, g, b = rgb[0], rgb[1], rgb[2]
        value = rgb[0]
        
        n = 256
        v = ((value * 3) % n)
        colours = [
            [0,     0,   0],
            [0,     0, 255],
            [255, 255, 255],
            [255, 255,   0]
        ]
        colours = np.array(colours)
        
        proportion = len(colours) * (v / n)
        
        lower = int(np.floor(proportion)) % len(colours)
        upper = int(np.ceil(proportion)) % len(colours)
        
        t = proportion - lower
        
        colour = t * colours[upper] + (1 - t) * colours[lower]
        
        return colour
    
    def create_image(self, escape_delays):
        v = np.repeat(escape_delays[:, :, np.newaxis], 3, axis=2)
        
        colours = np.apply_along_axis(self.colour_map, 2, v)
        colours = np.multiply(1, colours).astype(int)
        #self.image = colours.reshape((self.image_height, self.image_width * 3)).tolist()
        self.image = colours
    
    def get_uint8_image(self):
        return self.image.astype(np.uint8)
    
    def get_png_ready_image(self):
        output = self.image
        return output.reshape((self.image_height, self.image_width * 3)).tolist()
    
    def save_image(self):
        with open('fractal.png', 'wb') as fractal_image_file:
            w = png.Writer(self.image_width, self.image_height, greyscale=False)
            w.write(fractal_image_file, self.get_png_ready_image())

if __name__ == "__main__":
    c = -0.78-0.13j
    
    fractal = Fractal(WIDTH, HEIGHT, real_limits, imaginary_limits)
    escape_delays = fractal.escape_iterations(c)
    fractal.create_image(escape_delays)
    fractal.save_image()
